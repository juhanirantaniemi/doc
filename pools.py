# -*- coding: utf-8 -*-
"""
Created on 13 feb. 2017

@author: juhani.rantaniemi
"""

# Packages
import threading
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import time as t
import timeit
import logging
from datetime import datetime, timedelta, timezone
from requests.exceptions import HTTPError
import base64
from prometheus_client import Gauge

# Dependencies
from tenacity import retry, wait_random, stop_after_attempt

from commandrecording.commandrecord import CommandRecord
from commandrecording.commandrecording import record_new_commands, CommandRecordingException
from connections.sendmessages import (
    send_cmds,
    change_setting_many,
    get_restapi,
    post_restapi,
    get_power_many,
    get_latest_measurement_v2,
)
from databases.sql import insert_mysql, select_mysql
from databases.queries import (
    insert_tsovalue,
    select_lastcmds,
    gs2_breakpoint_msg,
    insert_cmds,
    replace_lastcmd,
    insert_totalpower,
    select_activetimetargets,
    select_theoretical_activetimes,
    replace_cmdduration,
    select_reservetest_freqtimes,
    select_last_cmd_success_and_successrate,
    insert_frequency,
    insert_unavailability,
    select_theoretical_chargetimes,
    select_session_chargetime_targets,
)
from config import Config

# Create logger only for commands
logcmds = logging.getLogger("logcmds")

# Create logger only for runtimes
logruntimes = logging.getLogger("logrt")

# Prometheus metrics
vpp_delays = Gauge("vpp_runtimes", "VPP controlling runtimes", ["vpp", "market", "type"])


class ControlPool(threading.Thread):
    def __init__(
        self,
        pooltype,
        vppid,
        marketid,
        unitarray,
        allunitmodes,
        cmddict,
        unitsettingarray,
        startdt,
        enddt,
        marketdata,
        ph,
        pfarray,
        setupvariablesdict,
        setupnames,
        dryrun=False,
    ):
        threading.Thread.__init__(self)
        self.log = logging.getLogger("vpp.pool_{}".format(vppid))
        self.pooltype = pooltype
        self.vpp = vppid
        self.market = marketid
        self.unitarray = unitarray
        self.allunitmodes = allunitmodes
        self.startdt = startdt
        self.enddt = enddt
        self.prioanalysis = marketdata["PriorityAnalysis"][0]
        self.tsoverification = marketdata["TsoVerification"][0]
        self.runcycle = marketdata["RunCycleSec"][0]
        self.capacityfactor = marketdata["CapacityFactor"][0]
        self.maxcapacityMW = marketdata["MaxCapacityMW"][0]
        self.tsoverifcyclesec = marketdata["TsoVerifCycleSec"][0]
        self.followpowercycle = marketdata["FollowPowerCycleSec"][0]
        self.reference_capacity = marketdata["ReferenceCapacity"][0]
        self.reference_power = marketdata["ReferencePower"][0]
        self.freqpath = marketdata["FreqPath"][0]
        self.poweranalysistype = marketdata["PowerAnalysisType"][0]
        self.cmddict = cmddict
        self.cmdsuccessdict = {}
        self.unitsettingarray = unitsettingarray
        self.ph = ph
        self.pfarray = pfarray
        self.unitset = set(self.unitarray["PowerUnitId"])
        self.originalunits = tuple(self.unitset)
        self.changesettingscycle = marketdata["ChangeSettingsCycleMinute"][0]
        self.config = Config().config
        self.calculatetargetc = marketdata["CalculateTargetc"][0]
        self.timezone = marketdata["TimeZone"][0]
        self.bidratio, self.extrapowerboolean = self.return_bidratio()
        self.setting_overwrite_minutes = min(
            self.changesettingscycle - 1, 50
        )  # Do not overwrite settings until 50 minutes has passed or if frequent setting update is wanted
        self.setupvariablesdict = setupvariablesdict
        self.setupnames = setupnames
        self.lost_reservepower = 0  # W
        self.nextpenaltyinserts = {}
        self.dryrun = dryrun
        if "simulation" in self.freqpath.lower():
            self.freqlock = threading.RLock()
            self.simulated_frequency = 50
            sfc = threading.Thread(target=self.create_simulated_frequencies)
            sfc.daemon = True
            sfc.start()

        # Metrics
        self.pool_units_gauge = ph.pool_units_gauge
        self.vpp_cycle_duration = ph.vpp_cycle_duration
        self.vpp_total_power = ph.vpp_total_power

    def run(self):
        try:
            self.log.info(
                "VPP: {} Market: {} started for Time: {} - {} - poolsize: {}".format(
                    self.vpp, self.market, self.startdt, self.enddt, self.unitarray.size
                )
            )

            # Update cmdsuccesses
            self.get_cmdsuccesses()

            # Start sending capacity values to Tso
            if self.tsoverification:
                # Follow what is the reserve pool size (same object)
                tsot = threading.Thread(target=self.send_realtimecapacity)
                tsot.daemon = True
                tsot.start()

            # Start following power of the pool.
            if self.unitarray.size > 0:
                ppt = threading.Thread(target=self.follow_poolpower)
                ppt.daemon = True
                ppt.start()

            # Initial variables
            cmdallseconds = 0
            lastchangesettings = 1000
            lasttargetc = 1000.0
            lastchosenrefresh = 1000.0
            self.newunits = True

            while datetime.utcnow() < self.enddt:
                """
                # Steps in the decision making busy loop
                1. Update the units in the pool e.g remove units opted out by user
                2. Get the grid frequency
                3. Calculate the power target rate (power to consume as a function of the grid frequency)
                4. Decide device modes (on/off) from the database (MySQL) (Algorithms kick in here)
                5. Send settings to the devices (sending cycle, fail-safe)
                6. Send the device mode decided on in step 5
                """
                try:
                    start_time = timeit.default_timer()

                    # See the pullunits through if there are units to be removed:
                    self.iterate_pullunits()
                    self.pool_units_gauge.labels(
                        vpp="vpp_{}".format(self.vpp), market="mkt_{}".format(self.market)
                    ).set(len(self.unitset))

                    # If no units - stop the pool
                    if self.unitarray.size == 0:
                        self.log.info(
                            "VPP: {} Market: {} started for Time: {} - {} - was empty, shutting down".format(
                                self.vpp, self.market, self.startdt, self.enddt
                            )
                        )
                        break

                    # Send right settings in the beginning of the period and every wanted N minutes
                    if lastchangesettings >= self.changesettingscycle and self.dryrun == False:
                        lastchangesettings = 0
                        change_setting_many(self.unitsettingarray, self.setting_overwrite_minutes)

                    # Get powertarget according to the poweranalysistype (frequency, trading, none)
                    try:
                        Ptargetrate, freq = getattr(self, self.poweranalysistype)()
                    except Exception as e:
                        logcmds.info(e)
                        Ptargetrate, freq = 1, -1

                    # Get last commandsmodes and create dictionary out of them
                    lastcmds = np.array(
                        select_mysql(select_lastcmds(tuple(self.unitarray["PowerUnitId"]))),
                        dtype=[
                            ("PowerUnitId", "U50"),
                            ("LastCmdDateTime", datetime),
                            ("ModeId", "int"),
                            ("CmdValue", "float"),
                        ],
                    )
                    self.lastcmddict = {row["PowerUnitId"]: (row["LastCmdDateTime"], row["ModeId"]) for row in lastcmds}

                    # Get the c value for the units:
                    if self.calculatetargetc and lasttargetc >= 10:
                        lasttargetc = 0
                        self.update_lastcmd_duration(self.unitset)
                        self.calculate_targetc()

                    # Modescores are calculated
                    if self.prioanalysis or self.newunits or cmdallseconds == 0:
                        # Calculate scores for modes - if wanted and on first round
                        self.calculate_modescores(lastchosenrefresh)
                        lastchosenrefresh = (
                            min(15.0, lastchosenrefresh) % 15
                        )  # set back to 0 if greater than or equal to 30

                    # Decide the unit commands.
                    cmds = self.get_commands(Ptargetrate)

                    decide_cmds_runtime = timeit.default_timer() - start_time
                    vpp_delays.labels(
                        vpp="vpp_{}".format(self.vpp), market="mkt_{}".format(self.market), type="decide_cmd"
                    ).set(round(decide_cmds_runtime, 1))

                    # Command units to keep the watchdog timers alive
                    recmd_units = self.unitarray["PowerUnitId"][
                        np.where(cmdallseconds % ((self.unitarray["CmdAllSec"] // self.runcycle) * self.runcycle) == 0)
                    ]
                    if len(recmd_units) > 0 and self.dryrun is False:
                        cmds_start_time = timeit.default_timer()
                        # Select default command for units that don't have a command in the list.
                        recmds = cmds[np.isin(cmds["PowerUnitId"], recmd_units)]
                        default_units = recmd_units[np.isin(recmd_units, cmds["PowerUnitId"], invert=True)]
                        default_cmds = self.allunitmodes[["PowerUnitId", "ModeId", "ModeTypeId", "CmdValue"]][
                            (np.isin(self.allunitmodes["PowerUnitId"], default_units))
                            & (self.allunitmodes["CmdValue"] == 0)
                        ]
                        send_cmds(np.concatenate((recmds, default_cmds)), self.cmddict)
                        cmds_runtime = timeit.default_timer() - cmds_start_time
                        vpp_delays.labels(
                            vpp="vpp_{}".format(self.vpp), market="mkt_{}".format(self.market), type="cmd_all"
                        ).set(round(cmds_runtime, 1))

                    cmds = cmds[
                        np.logical_not(
                            np.in1d(
                                cmds[["PowerUnitId", "ModeId", "CmdValue"]],
                                lastcmds[["PowerUnitId", "ModeId", "CmdValue"]],
                            )
                        )
                    ]

                    if cmds.size > 0 and self.dryrun == False:

                        # Send commands to Tingcore
                        send_cmds(cmds[["PowerUnitId", "ModeId"]], self.cmddict)

                        insert_db_start_time = timeit.default_timer()
                        # Update the duration of the lastcmd
                        dtnow = datetime.now(tz=timezone.utc)
                        self.update_lastcmd_duration(cmds["PowerUnitId"])

                        command_records = _to_command_records(dtnow, cmds)

                        try:
                            record_new_commands(command_records)
                        except CommandRecordingException as e:
                            logcmds.error("Failed to record new commands to the database.", exc_info=e)

                        insert_db_runtime = timeit.default_timer() - insert_db_start_time
                        vpp_delays.labels(
                            vpp="vpp_{}".format(self.vpp), market="mkt_{}".format(self.market), type="insert_db"
                        ).set(round(insert_db_runtime, 1))

                        unique, counts = np.unique(cmds["ModeTypeId"], return_counts=True)
                        logcmds.info(
                            "VPP:{} Market:{} Freq:{} Cmds: {}".format(
                                self.vpp, self.market, freq, dict(zip(unique, counts))
                            )
                        )

                    runtime = timeit.default_timer() - start_time
                    self.vpp_cycle_duration.labels(
                        vpp="vpp_{}".format(self.vpp), market="mkt_{}".format(self.market)
                    ).observe(runtime)
                    if runtime > 5:
                        logruntimes.info(
                            "Run-time: {} seconds VPP: {} Market {}".format(round(runtime, 1), self.vpp, self.market)
                        )
                    else:
                        logruntimes.debug(
                            "Run-time: {} seconds VPP: {} Market {}".format(round(runtime, 1), self.vpp, self.market)
                        )
                    t.sleep(max(0, self.runcycle - runtime))
                    cmdallseconds = cmdallseconds + self.runcycle
                    lastchangesettings = lastchangesettings + self.runcycle / 60
                    lasttargetc = lasttargetc + self.runcycle / 60
                    lastchosenrefresh = lastchosenrefresh + self.runcycle / 60
                except Exception as e:
                    self.log.exception("PoolRun Error: {} with VPP: {} Market: {}".format(e, self.vpp, self.market))
                    t.sleep(60)
            self.log.info(
                "VPP: {} Market: {} ended for Time: {} - {} - poolsize: {}".format(
                    self.vpp, self.market, self.startdt, self.enddt, self.unitarray.size
                )
            )

            # If there is a nonVPP market that can end before hour changes - the units are put for the rest of the hour on failSafe-mode
            if self.pooltype != "VPP" and "PROD_WATER_HEATER" in self.setupnames["DeviceType"]:
                dtnow = datetime.utcnow()
                self.ph.gather_single_pool(
                    "VPP",
                    self.originalunits,
                    dtnow,
                    dtnow.replace(microsecond=0, second=0, minute=0) + timedelta(hours=1),
                    999,
                )
        except Exception:
            self.log.exception(
                "Fatal error in VPP: {} Market: {} poolsize: {}".format(self.vpp, self.market, self.unitarray.size)
            )

    def calculate_modescores(self, lastchosenrefresh=0):
        # Calculate modescores by setup
        for setupid, devicetype in self.setupnames:
            setupunits = self.unitarray[self.unitarray["SetupId"] == setupid][
                ["PowerUnitId", "CmdSanctNullTimeMin", "CmdSanctStart", "Targetc", "TotalPower", "PowerMin"]
            ]
            for unit, cmdsanctnulltime, cmdstartsanct, targetc, totalpower, totalminpower in setupunits:
                # Get unitmodes of the units on ascending power order. Sorting is done in the SQL-query
                moderownumbers = np.where(self.allunitmodes["PowerUnitId"] == unit)
                unitmodes = self.allunitmodes[moderownumbers]

                # Calculate the mode scores for each mode
                scores = []
                chosens = sum(unitmodes["Chosen"])
                targetpower = totalminpower + totalpower * targetc
                max_modecost = max(unitmodes["ModeCost"])

                # Do not calculate scores if only one mode is found
                if len(unitmodes) == 1:
                    scores.append(0)

                # Calculate Setup spesific scores
                elif devicetype in ["ELENIA_BATTERY", "UPS", "ETTEPLAN_UPS", "ALFEN_BATTERY"]:
                    # Form quadratic mode score function s(P_m) = a*P_m^2 + b*P_m + c for batteries (P_m is mode power)
                    # score diff of WH that corresponds to score diff of battery modes right outside normal range
                    controlconfigs = self.config["battery_controlling"][devicetype]
                    WH_diff = controlconfigs["critical_wh_score_diff"]
                    # radius of normal range of battery powers
                    normal_threshold = controlconfigs["normal_threshold"] * totalpower / 2

                    # Parameters a, b, and c for quadratic mode score function s(P_m)
                    a = -WH_diff / (2 * np.mean(np.diff(sorted(unitmodes["PowerMax"]))) * normal_threshold)
                    b = -2 * a * targetpower  # greatest value is at target power
                    c = -min(
                        a * np.max(unitmodes["PowerMax"]) ** 2 + b * np.max(unitmodes["PowerMax"]),
                        a * np.min(unitmodes["PowerMax"]) ** 2 + b * np.min(unitmodes["PowerMax"]),
                    )  # all mode scores >= 0

                    for idx, row in enumerate(unitmodes[["PowerMax", "ModeTypeId", "ModeCost"]]):
                        modepower, modetypeid, modecost = row
                        # Calculate common modescores based on 1. Targetpower 2. ModeCost
                        score = a * modepower ** 2 + b * modepower + c + max_modecost
                        score -= modecost

                        self.allunitmodes[moderownumbers[0][idx]]["Score"] = int(round(score, 0))
                        scores.append(score)

                elif devicetype in ("PROD_WATER_HEATER", "PLUGSURFING"):  # Water heater and plugsurfing mode scores
                    cmdsuccess = self.cmdsuccessdict.get(unit, {"LastCmdSuccess": 1})["LastCmdSuccess"]

                    lastcmdtime, lastmodetype = self.lastcmddict[unit]
                    for idx, row in enumerate(unitmodes[["PowerMax", "ModeTypeId", "ModeCost"]]):
                        modepower, modetypeid, modecost = row
                        # Calculate common modescores based on 1. Targetpower 2. CmdSuccess 3. LastCmd 4. ModeCost
                        if targetpower >= 0 and targetpower <= totalpower:  # TargetPower
                            score = (
                                max(0, (totalpower - abs(targetpower - modepower)) / totalpower * 100) + max_modecost
                            )  # mode scores >= 0
                        elif targetpower > totalpower:
                            score = (
                                max(0, (targetpower + 2 * modepower - 2 * totalpower) / totalpower * 100) + max_modecost
                            )
                        else:
                            score = max(0, (totalpower - targetpower - 2 * modepower) / totalpower * 100) + max_modecost

                        if cmdsuccess == modetypeid == 0:  # CmdSuccess
                            score += 5000  # last cmd failed, add score for OFF-mode

                        if (
                            devicetype == "PROD_WATER_HEATER"
                            and (lastmodetype == modetypeid)
                            and cmdsanctnulltime > 0
                            and cmdstartsanct > 0
                        ):  # LastCmd
                            score += max(
                                0,
                                (
                                    cmdstartsanct
                                    - (
                                        (cmdstartsanct / cmdsanctnulltime)
                                        * ((datetime.utcnow() - lastcmdtime).total_seconds() / 60.0)
                                    )
                                ),
                            )
                        score -= modecost  # ModeCost

                        self.allunitmodes[moderownumbers[0][idx]]["Score"] = int(round(score, 0))
                        scores.append(score)
                elif devicetype == "NETLED_VERTICAL_FARM":
                    for idx, row in enumerate(unitmodes[["PowerMax", "ModeTypeId", "ModeCost"]]):
                        modepower, modetypeid, modecost = row
                        score = 100 - modecost + 5 * (modepower == 0)  # dummy score to start with
                        self.allunitmodes[moderownumbers[0][idx]]["Score"] = int(round(score, 0))
                        scores.append(score)

                # Calculate DeltaS and DeltaPs compared to the highest score
                if (
                    chosens != 1 or lastchosenrefresh >= 15
                ):  # In the first loop or every 15 minutes select a default chosen
                    maxscoreidx = moderownumbers[0][np.argmax(scores)]
                    self.allunitmodes["Chosen"][self.allunitmodes["PowerUnitId"] == unit] = 0
                    self.allunitmodes["Chosen"][maxscoreidx] = 1
                else:
                    maxscoreidx = moderownumbers[0][np.where(self.allunitmodes["Chosen"][moderownumbers] == 1)[0]][0]

                self.allunitmodes["DeltaS"][moderownumbers] = (
                    self.allunitmodes["Score"][moderownumbers] - self.allunitmodes["Score"][maxscoreidx]
                )
                self.allunitmodes["DeltaP"][moderownumbers] = (
                    self.allunitmodes["PowerMax"][moderownumbers] - self.allunitmodes["PowerMax"][maxscoreidx]
                )
        self.log.debug(
            "VPP: {} Market: {} ModeScoreSum {}".format(self.vpp, self.market, sum(self.allunitmodes["Score"]))
        )

    def get_commands(self, Ptargetrate):
        # Calculate Target Power
        Ptotalmin = np.sum(self.unitarray["PowerMin"])
        Ptotalmax = np.sum(self.unitarray["PowerMax"])
        Ptotal = Ptotalmax - Ptotalmin
        Ptarget = Ptotalmin + (
            self.bidratio * Ptargetrate * Ptotal + self.extrapowerboolean * (1 - self.bidratio) * Ptotal
        )
        Pchosen = np.sum(self.allunitmodes["PowerMax"][self.allunitmodes["Chosen"] == 1])
        while True:

            # See if power has to increase or decrease
            if Pchosen < Ptarget:
                valid_idx = np.where(np.logical_and(self.allunitmodes["DeltaP"] > 0, self.allunitmodes["Score"] >= 0))[
                    0
                ]
            else:
                valid_idx = np.where(np.logical_and(self.allunitmodes["DeltaP"] < 0, self.allunitmodes["Score"] >= 0))[
                    0
                ]

            # Break if maximum or minimum power has been achieved or the power change would take us further from the target
            if len(valid_idx) < 1:
                break

            bestind = valid_idx[self.allunitmodes[valid_idx]["DeltaS"].argmax()]
            Pdelta, unit, bestmodepower = self.allunitmodes[bestind][["DeltaP", "PowerUnitId", "PowerMax"]]

            if abs(Ptarget - (Pchosen + Pdelta)) >= abs(Ptarget - Pchosen):
                break

            self.allunitmodes["Chosen"][self.allunitmodes["PowerUnitId"] == unit] = 0
            self.allunitmodes["Chosen"][bestind] = 1
            self.allunitmodes["DeltaS"][self.allunitmodes["PowerUnitId"] == unit] -= self.allunitmodes[bestind][
                "DeltaS"
            ]
            self.allunitmodes["DeltaP"][self.allunitmodes["PowerUnitId"] == unit] -= self.allunitmodes[bestind][
                "DeltaP"
            ]
            Pchosen += Pdelta
        return self.allunitmodes[["PowerUnitId", "ModeId", "ModeTypeId", "CmdValue"]][
            (self.allunitmodes["Chosen"] == 1) & (self.allunitmodes["Allowed"] == 1)
        ]

    def calculate_targetc(self):
        dtnow = datetime.utcnow()
        # hoursleft = max(0.1,(self.enddt - dtnow).seconds/3600)
        hoursleft = 1  # constant multiplier now
        sumact = 0
        sumacttheor = 0
        # Calculate the target c for each setup separtely
        for setupid, devicetype in self.setupnames:
            setuprows = np.where(self.unitarray["SetupId"] == setupid)[0]
            setupunits = self.unitarray[setuprows]["PowerUnitId"]
            if devicetype == "PROD_WATER_HEATER":
                # Get the activetime targets and realized for the unit until end of this hour
                activetimetargets = dict(select_mysql(select_activetimetargets(tuple(setupunits), dtnow)))
                activetimestheoretical = dict(select_mysql(select_theoretical_activetimes(tuple(setupunits), dtnow)))
                for idx in setuprows:
                    unit = self.unitarray[idx]["PowerUnitId"]
                    # The TargetC can be less than zero as well as greater than 1
                    targetc = round(
                        (activetimetargets.get(unit, 0) - activetimestheoretical.get(unit, 0)) / hoursleft, 2
                    )
                    sumact += activetimetargets.get(unit, 0)
                    sumacttheor += activetimestheoretical.get(unit, 0)
                    self.unitarray[idx]["Targetc"] = targetc
            elif devicetype in ["ELENIA_BATTERY", "UPS", "ETTEPLAN_UPS", "ALFEN_BATTERY"]:
                batterydict = self.setupvariablesdict[devicetype]
                for idx in setuprows:
                    unit, minpower, maxpower = self.unitarray[idx][["PowerUnitId", "PowerMin", "PowerMax"]]
                    variables = batterydict[unit]
                    cmdsuccessrate = self.cmdsuccessdict.get(unit, {"CmdSuccessRate": 1})["CmdSuccessRate"]
                    successrate_limit = self.config["battery_controlling"][devicetype]["cmdsuccessrate_limit"]
                    socloss = 0
                    connectionloss = 0
                    # Get the SOC-values
                    currentSOC = self.ph.get_latest_state(
                        unit=unit,
                        devicetype=devicetype,
                        variables=variables,
                        startdt=dtnow - timedelta(minutes=20),
                        enddt=dtnow,
                        defaultvalue=variables["targetSOC"],
                    )
                    targetpower = (variables["targetSOC"] - currentSOC) / 100 * variables["CapacityWh"] / hoursleft
                    targetc = round((targetpower - minpower) / (maxpower - minpower), 2)
                    # If battery is empty or full, the available power is marked to be zero and controlling is limited
                    if variables["minSOC"] <= currentSOC <= variables["maxSOC"]:
                        self.unitarray[idx]["AvailablePower"] = self.unitarray[idx]["TotalPower"]
                        self.allunitmodes["Allowed"][self.allunitmodes["PowerUnitId"] == unit] = 1
                    else:
                        forbidden_power_direction = 1 if currentSOC > variables["maxSOC"] else -1
                        self.allunitmodes["Allowed"][
                            (self.allunitmodes["PowerUnitId"] == unit)
                            & (self.allunitmodes["CmdValue"] * forbidden_power_direction > 0)
                        ] = 0
                        self.unitarray[idx]["AvailablePower"] = 0
                        socloss = 0.5 * (maxpower - minpower)
                    # If previous commands have not been successful the battery is marked unavailable due to connection
                    if cmdsuccessrate < successrate_limit:
                        self.unitarray[idx]["AvailablePower"] = 0
                        connectionloss = maxpower - minpower
                    # Mark reason for the lost capacity to the database
                    if (socloss or connectionloss) and self.nextpenaltyinserts.get(unit, dtnow) <= dtnow:
                        insert_unavailability(unit, dtnow, socloss, connectionloss)
                        self.nextpenaltyinserts[unit] = dtnow + timedelta(minutes=5)

                    self.unitarray[idx]["Targetc"] = targetc
            elif devicetype == "PLUGSURFING":
                # calculate chargetime targets until end of current hour from unitschedule
                chargetimetargets = dict(select_session_chargetime_targets(tuple(setupunits)))
                # calculate commanded charge time of session(session energy could be used as well in later phase)
                chargetimestheoretical = dict(
                    select_theoretical_chargetimes(tuple(setupunits), dtnow - timedelta(hours=10), dtnow)
                )
                for idx in setuprows:
                    unit = self.unitarray[idx]["PowerUnitId"]
                    # The TargetC can be less than zero as well as greater than 1
                    targetc = round(
                        (chargetimetargets.get(unit, 0) - chargetimestheoretical.get(unit, 0)) / hoursleft, 2
                    )
                    self.unitarray[idx]["Targetc"] = targetc
            elif devicetype == "NETLED_VERTICAL_FARM":
                # The targetc should stay constant but if power is lost, the available power is updated
                enddt = datetime.utcnow()
                startdt = enddt - timedelta(minutes=20)
                for idx in setuprows:
                    unit = self.unitarray[idx]["PowerUnitId"]
                    minpower, maxpower = self.ph.get_latest_state(
                        unit=unit,
                        devicetype=devicetype,
                        startdt=startdt,
                        enddt=enddt,
                        defaultvalue=tuple(self.unitarray[idx][["PowerMin", "PowerMax"]]),
                    )
                    self.unitarray[idx]["AvailablePower"] = maxpower - minpower
        self.log.debug(
            "VPP: {} Market: {} TargetCSum {} SumActiveTarget {} SumActiveTheor {} HoursLeft {}".format(
                self.vpp, self.market, sum(self.unitarray["Targetc"]), sumact, sumacttheor, hoursleft
            )
        )

    def get_cmdsuccesses(self):
        try:
            # Search for cmdhistory from the previous 1 hours
            enddt = self.startdt
            startdt = enddt - timedelta(hours=2)

            last_cmdsuccess = np.array(
                select_last_cmd_success_and_successrate(tuple(self.unitarray["PowerUnitId"]), startdt, enddt),
                dtype=[("PowerUnitId", "U50"), ("CmdSuccess", int), ("CmdSuccessRate", float)],
            )
            for row in last_cmdsuccess:
                self.cmdsuccessdict[row["PowerUnitId"]] = {
                    "LastCmdSuccess": row["CmdSuccess"],
                    "CmdSuccessRate": row["CmdSuccessRate"],
                }

        except Exception as e:
            self.log.error("Failed to update command successes: ", e)

    def update_lastcmd_duration(self, units):
        cmddurations = []
        dtnow = datetime.utcnow()
        for unit in units:
            try:
                lastcmdtime = self.lastcmddict[unit][0]
                diff = dtnow - lastcmdtime
                hourdiff = round(diff.days * 24 + diff.seconds / 3600, 2)
                cmddurations.append((unit, lastcmdtime, hourdiff))
            except Exception as e:
                self.log.error("Error in updating lastcmd duration: %s", e)
        insert_mysql(replace_cmdduration(), cmddurations)

    def iterate_pullunits(self):
        # See which units have to be pulled from the pool
        pullunitset = self.ph.return_pullunits()
        pullflag = 0
        self.newunits = False

        # In pooltype="VPP" remove units that are on the list
        if (self.pooltype == "VPP") and (len(pullunitset) > 0):
            newunits = self.unitset - pullunitset
            pullflag = 1

        # In poolType="nonVPP" remove units that are not on the list
        elif self.pooltype != "VPP":
            newunits = self.unitset & pullunitset
            pullflag = 1

        # If there has been changes - update unitarray and unitset
        if (pullflag == 1) and (len(newunits) != self.unitarray.size):
            self.unitarray = self.unitarray[np.in1d(self.unitarray["PowerUnitId"], list(newunits))]
            self.allunitmodes = self.allunitmodes[np.in1d(self.allunitmodes["PowerUnitId"], list(newunits))]
            self.unitset = newunits
            self.newunits = True

    def follow_poolpower(self):
        while (datetime.utcnow() < self.enddt) and (self.unitarray.size > 0) and self.dryrun == False:
            try:
                # Get poolpower
                totalpower = get_power_many(self.unitarray["PowerUnitId"])
                # Save poolpower
                insert_totalpower(totalpower, self.vpp, self.market)
                self.vpp_total_power.labels(vpp="vpp_{}".format(self.vpp)).set(totalpower)
            except Exception as e:
                self.log.error("Error when getting power: %s", e)
            if self.reference_power != "None":
                self.send_realtime_tso_value(reference=self.reference_power, powerMW=totalpower / 1000000)
            t.sleep(self.followpowercycle)

    def send_realtimecapacity(self):
        while (datetime.utcnow() < self.enddt) and (self.unitarray.size > 0) and self.dryrun == False:
            try:
                try:
                    activepower = np.sum(self.unitarray["AvailablePower"]) * self.bidratio
                except Exception as e:
                    self.log.error("Error when sending realtime capacity: %s", e)
                    activepower = 0
                powerMW = min(self.maxcapacityMW, activepower / 1000000 * self.capacityfactor)
                self.send_realtime_tso_value(reference=self.reference_capacity, powerMW=powerMW)
            except Exception as e:
                # Catch ALL exceptions so as not to kill the thread prematurely
                self.log.error("Uncaught error when sending realtime capacity: %s", e)
            t.sleep(self.tsoverifcyclesec)

        # Send capacity with zero at the end of hour.
        self.send_realtime_tso_value(reference=self.reference_capacity, powerMW=0.0)

    def send_realtime_tso_value(self, reference, powerMW):
        """
        Private implementation of the actual sending and
        storage of realtime capacity data
        """
        capacity_data, utcnow = gs2_breakpoint_msg(reference, powerMW)
        data = {"type": "capacity", "reference": reference, "data": capacity_data, "timestamp": utcnow.timestamp()}
        resp = post_restapi(self.config["external_apis"]["tao_data"]["real_time_capacity"], data)
        try:
            resp.raise_for_status()
        except HTTPError as err:
            self.log.error("Error sending realtime capacity data: %s", err)
        insert_mysql(insert_tsovalue(), ((reference, self.vpp, powerMW),))
        self.log.debug("TSO realtime capacity reported: %f MW", powerMW)

    def frequency_poweranalysis(self):
        # Get frequency
        if "simulation" in self.freqpath.lower():
            with self.freqlock:
                freq = self.simulated_frequency
        else:
            success, timetag, freq = get_frequency(self.freqpath)

        # Get target powerTargetRate
        return get_Ptargetrate(freq, self.pfarray), freq

    def fixed_target(self):
        return get_Ptargetrate(50, self.pfarray), 0

    def return_bidratio(self):
        # Bidratio decreases if there is a missmatch between the expected c and the actual average c
        try:
            averagec = float(np.average(self.unitarray["Targetc"], weights=self.unitarray["TotalPower"]))
            if self.capacityfactor < averagec:
                bidratio, extrapowerboolean = round((averagec - 1) / float(self.capacityfactor - 1), 2), 1
            else:
                bidratio, extrapowerboolean = round(averagec / float(self.capacityfactor), 2), 0
        except Exception as e:
            bidratio, extrapowerboolean = 0, 0
            self.log.error("Vpp: {}, Market: {}, error in returning bidratio: {}".format(self.vpp, self.market, e))
        self.log.info(
            "Vpp: {}, Market: {}, Bidratio: {} Extrapowerboolean: {}".format(
                self.vpp, self.market, bidratio, extrapowerboolean
            )
        )
        return bidratio, extrapowerboolean

    def create_simulated_frequencies(self):
        # Get time and frequency values from database.
        freqtime = select_mysql(select_reservetest_freqtimes(self.market))
        # Save frequency value to database and restore value
        for freq, waittimesec in freqtime:
            with self.freqlock:
                self.simulated_frequency = freq
            insert_mysql(
                insert_frequency(),
                [(self.simulated_frequency, "Reservetest-" + str(self.vpp) + "-" + str(self.market))],
            )
            t.sleep(waittimesec)


# ------- Support functions -------------
def get_frequency(path):  # Read frequency from Simpson
    """
    Request frequency data from Simpson (system hosted by TAO)

    Example response would be like below
    {
        "source": "FIN",
        "timestamp": 1530792191.980747,
        "formatted_timestamp": "2018.07.05 15.03.11",
        "frequency": 50.0400009155273
    }
    """

    # Network paths need to be encoded before making the http request
    # TODO: Remove this bit once VPP moves to AWS
    src = base64.urlsafe_b64encode(path.encode()).decode()
    url = "{}?source={}".format(Config().config["external_apis"]["tao_data"]["grid_frequency"], src)

    for i in range(0, 10):
        try:
            resp = get_restapi(url=url)
            # Raise an exception if there was a non 2xx HTTP error code response
            resp.raise_for_status()
            result = resp.json()
            freq = result["frequency"]
            # TODO: Consider timezones. In AWS servers are in UTC
            timetag = datetime.strptime(result["formatted_timestamp"], "%Y.%m.%d %H.%M.%S")
            success = True
            break
        except Exception as err:
            if i == 9:
                success = False
        t.sleep(0.1)
    if success == False or timetag < datetime.utcnow() - timedelta(minutes=5) or freq < 40 or freq > 60:
        # Get frequency from backup source
        try:
            backupfreq = Config().config["vpp"]["backup_frequency"]
            timetag, freq = get_latest_measurement_v2(
                backupfreq["id"], backupfreq["measurement"], datetime.utcnow() - timedelta(minutes=5), datetime.utcnow()
            )
            timetag = datetime.strptime(timetag, "%Y-%m-%d %H:%M:%S")
            success = True
        except Exception as err:
            timetag = datetime(1900, 1, 1, 0, 0, 0)
            freq = 50.0
            logging.getLogger("get_frequency").warning("Frequency reading failed: {0}".format(err))
            success = False
    return success, timetag, round(freq, 2)


def get_Ptargetrate(freq, pfarray):
    # Calculate using the powerrate vs. frequency array the correct powerrate.
    if len(pfarray) == 1:
        Ptargetrate = pfarray[0][0]
    else:
        freqcomparison = np.append(pfarray, [(0, 0), (1, 100), (200, freq)], axis=0)

        # sorted according to frequency (asc)
        freqcomparison = freqcomparison[freqcomparison[:, 1].argsort()]

        # Index of freq in the freq_comparison array
        freqindex = np.argmax(freqcomparison, axis=0)[0]
        f1 = freqcomparison[freqindex - 1][1]
        p1 = freqcomparison[freqindex - 1][0]
        f2 = freqcomparison[freqindex + 1][1]
        p2 = freqcomparison[freqindex + 1][0]

        # Target according to the reference curve determined by the points
        Ptargetrate = (freq - f1) / (f2 - f1) * (p2 - p1) + p1
    return min(max(Ptargetrate, 0), 1)


def _to_command_records(dtnow, cmds):
    command_dicts = [dict(zip(cmds.dtype.names, values)) for values in cmds]

    command_records = [
        CommandRecord(
            time=dtnow,
            powerunit_id=command["PowerUnitId"],
            mode_type=command["ModeId"],
            command_value=command["CmdValue"],
        )
        for command in command_dicts
    ]
    return command_records
