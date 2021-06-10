from random import random
import math
import numpy as np
import time as t
import networkx as nx
import matplotlib.pyplot as plt

def random_nodes(node_count, ared_width, area_height):
    nodes = []
    pos = {}
    n = 0
    while len(nodes) < node_count:
        x, y = int(random()*ared_width), int(random()*area_height)
        if (x, y) not in pos:
            nodes.append([n, math.sqrt(x**2 + y**2), math.atan2(y,x)])
            pos[(x, y)] = n
            n += 1
    pos = {n: coord for coord, n in pos.items()}
    return nodes, pos

def calculate_distance(d1, d2, r1, r2):
    #Uses cosine formula
    radius = min(abs(r1-r2), 2 * math.pi - abs(r1-r2))
    return math.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * math.cos(radius))

def direct_distances(nodes):
    W = np.zeros((node_count, node_count))
    for i, d1, r1  in nodes:
        for j, d2, r2 in nodes[i+1:]:
            W[i,j] = W[j][i] =  round(calculate_distance(d1, d2, r1, r2), 1)
    return W
    
def capacity_factors(node_count, prod_count):
    F = np.zeros((prod_count, node_count))
    for i in range(0, prod_count):
        F[i,:] = 1 - (i/prod_count)
    return F

def consumptions(node_count, prod_count):
    EC = np.zeros((node_count, node_count))
    for i in range(prod_count, node_count):
        EC[:,i] = random()*10
    return EC

#Do prims algorithm and store distances and paths between node
def distances_edges_paths(W, node_count):
    maxw = np.max(W)
    E = np.zeros((node_count, node_count)) #Edges: TRUE / FALSE
    P = np.empty((node_count,node_count),dtype=object) #Edge paths between nodes. Matrice operations dont work with the 
    #STEPS = np.zeros((node_count, node_count))
    for i,j in np.ndindex(P.shape): #Initialize paths with lists
        P[i,j] = []
    graph_start = int(random()*node_count)
    E[graph_start][graph_start] = True
    D = np.ones((node_count, node_count)) * np.inf #Distances betwee nodes: INF / NUM
    D[graph_start][graph_start] = 0
    ones = np.ones((node_count, node_count))
    while np.sum(D) == np.inf:
        #Get from rows where edges alraedy exist, next min: row, colum, value
        #Pass the parent rows for the next new row&column+ min_weight
        #Set the new weight
        #Create mask with Trues on rows and replace False rows with INF
        forbidded_parent_nodes = ((np.nanmax(E, axis=1) == 0).T * ones).T #The picked row indicates to known node and column for the new
        forbidded_child_nodes = ((np.nanmax(E, axis=1) == 1).T * ones)
        W_next = W + ((forbidded_parent_nodes + forbidded_child_nodes) * maxw)
        minw = np.min(W_next)
        known_n, new_n = np.argwhere(W_next== minw)[0]
        D[new_n,:] = D[known_n,:] + minw #Update the row to the found new node
        D[:,new_n] = D[:,known_n] + minw #update the column to the found new node
        D[new_n][new_n] = 0
        E[known_n][new_n] = E[new_n][known_n] = E[new_n][new_n] = True
        for n in np.argwhere(D[known_n,:] < np.inf):
            n=n[0]
            if n != new_n:
                P[new_n,n] = [[new_n, known_n]] + P[known_n,n]
                P[n, new_n] = [[n2, n1] for n1, n2 in list(reversed(P[new_n,n]))]
                #STEPS[new_n,:] = STEPS[known_n,:] + 1 #Update the row to the found new node
                #STEPS[:,new_n] = STEPS[:,known_n] + 1 #update the column to the found new node
    return D, E, P

#Update the new node with old node info + ([new, old])
#Pass the reverse info for the "mirror" location

def calculate_c():
    #Ep[j,:] = i.append[i]Ep
    np.fill_diagonal(E, 0)
    np.fill_diagonal(D, np.inf)#Make sure that the minimum distance is not zero

    #Assistin variables
    Dpc = D[:prod_count,prod_count:] # Select only production rows and consumption columns.
    Fpc = F[:prod_count,prod_count:]
    all_ns = np.arange(node_count-prod_count)

    #Loss Optimal
    #Calculate the nodes which are used for Phi and distances.
    n_mininDs = np.argmin(Dpc, axis=0)#[:prod_count,prod_count:]
    ZigmaF_T_loss = np.sum(Fpc[n_mininDs, all_ns])
    Zigma_w_l_loss = np.sum(Dpc[n_mininDs, all_ns])*loss

    #Site optimal (n=0 is optimal)
    optimal_prod_node = 0
    ZigmaF_site = np.sum(Fpc[0, all_ns])
    Zigma_w_l_site = np.sum(Dpc[0,all_ns]*loss)

    #Total Optimal:
    Epc = Fpc- Fpc * Dpc * loss
    n_min_total = np.argmax(Epc, axis=0)
    ZigmaF_T_tot = np.sum(Fpc[n_min_total, all_ns])
    Zigma_w_l_tot = np.sum(Dpc[n_min_total, all_ns])*loss

    C = (ZigmaF_T_tot*(1-Zigma_w_l_tot)) / (ZigmaF_site * (1 - Zigma_w_l_site))
    print(C)
    (ZigmaF_T_tot-ZigmaF_T_tot* Zigma_w_l_tot) / (ZigmaF_T_tot-ZigmaF_T_tot* Zigma_w_l_site)

def transmission_optimal_production_sites():
    return 0

def transmission_energy_network_1d(EC, P, node_count):
    #Iterate path is P matrice and create Energy Edges matrice
    EE = np.zeros((node_count, node_count)) #TODO add the loss energy
    for i in range(0, EC.shape[0]):
        if EC[i] > 0:
            for ii, jj in P[i]:
                EE[ii, jj] += EC[i]
    return EE




def generation_optimal_production_sites(F, D, EC, P, node_count, prod_count):
    optimal_prod_node = 0
    D = D[optimal_prod_node,prod_count:]
    F = F[optimal_prod_node,prod_count:]
    EC = EC[optimal_prod_node,prod_count:]
    P = P[optimal_prod_node,prod_count:]
    all_ns = np.arange(node_count-prod_count)
    #Return production sites with highest Capacity Factor, so that the capacity matches the requirement
    
    #Calculate how much capacity is required to match all losses
    Cpt = np.sum(EC) / ( np.sum(F) * ( 1 - np.sum(D) * loss ) )
    Dt = np.dot(D, EC)
    EE = transmission_energy_network_1d(EC, P, node_count)
    return Cpt, Dt, EE #Production capacity, Distances tranmitted, Transmitted powers

def create_shortcuts(W, D, E, P):
    #Calculate Connected node
    np.fill_diagonal(D, 0)
    #print(D)
    while True:
        i0, j0 = np.argwhere(D - W == np.nanmax(D - W))[0]
        if D[i0,j0] < shortcutx * W[i0,j0]:
            #Stop iteration if shortcut is less than N x
            break
        iterable_nodes = [(i0,j0),(j0,i0)] #First value i with original distances, Second value j with "new" potentially shorter paths
        E[i0,j0] = E[j0,i0] = True
        #D[i0,j0] = D[j0,i0] = W[i0,j0]
        for orig_n, shortcut_n in iterable_nodes:
            Dprev = D
            d = W[orig_n,shortcut_n]
            mask = D[orig_n,:] > (D[shortcut_n,:] + d)
            #print(mask)
            if sum(mask) > 0:
                D[orig_n, mask] = D[mask, orig_n] = D[shortcut_n,mask] + d
                iterable_nodes += [(child[0], orig_n) for child in np.argwhere(E[orig_n,:] == True) if child[0] != shortcut_n]#Prevent that wont return back to potential new
                for n in np.argwhere(mask == True):
                    n=n[0]
                    if n != orig_n:
                        P[orig_n,n] = [[orig_n, shortcut_n]] + P[shortcut_n,n]
                        P[n, orig_n] = [[n2, n1] for n1, n2 in list(reversed(P[orig_n,n]))]
    return D, E, P
            
def print_graph(E, pos):
    G = nx.from_numpy_matrix(E)
    nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
    plt.show()




print("started")
t0 = t.time()
#Variables
node_count = 10
prod_count = 3
loss = 6 / 100 / 1000 # %/1000km
shortcutx = 3
area_width = int(random()*1000)
area_height = int(random()*1000)

#Algorithms
nodes, pos = random_nodes(node_count, area_width, area_height)
W = direct_distances(nodes)
EC = consumptions(node_count, prod_count)
F = capacity_factors(node_count, prod_count)
D, E, P = distances_edges_paths(W, node_count)
#print_graph(E, pos)
D, E, P = create_shortcuts(W, D, E, P)
#print_graph(E, pos)
transmission_optimal_production_sites()
generation_optimal_production_sites(F, D, EC, P, node_count, prod_count)
print(t.time() - t0)

#https://stackoverflow.com/questions/44360084/multiplying-numpy-2d-array-with-1d-array
#lossess: https://library.e.abb.com/public/56aef360ec16ff59c1256fda004aeaec/04MP0274%20Rev.%2000.pdf

