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
    #Uses cosine formula to calculate direct distances between nodes
    radius = min(abs(r1-r2), 2 * math.pi - abs(r1-r2))
    return math.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * math.cos(radius))

def direct_distances(nodes, node_count):
    W = np.zeros((node_count, node_count))
    for i, d1, r1  in nodes:
        for j, d2, r2 in nodes[i+1:]:
            W[i,j] = W[j][i] =  round(calculate_distance(d1, d2, r1, r2), 1)
    return W
    
def capacity_factors(node_count, prod_count):
    F = np.zeros((prod_count, node_count-prod_count))
    for i in range(0, prod_count):
        F[i,:] = 1 - (i/(prod_count*10))
    return F

def consumptions(node_count, prod_count):
    EC = np.zeros((node_count, node_count))
    for i in range(prod_count, node_count):
        EC[:,i] = random()*10
    return EC

#Do prims algorithm and store distances and paths between all nodes
def distances_edges_paths(W, node_count):
    maxw = np.max(W)
    D = np.ones((node_count, node_count)) * np.inf #Distances betwee nodes
    E = np.zeros((node_count, node_count)) #Edges: TRUE / FALSE
    P = np.empty((node_count,node_count),dtype=object) #Paths between nodes
    #STEPS = np.zeros((node_count, node_count))
    for i,j in np.ndindex(P.shape): #Initialize paths with lists - Matrice operations dont work with the lists in numpy unfortunately
        P[i,j] = []
    graph_start = int(random()*node_count)
    E[graph_start][graph_start] = True
    D[graph_start][graph_start] = 0
    ones = np.ones((node_count, node_count)) # Helper matrice
    while np.sum(D) == np.inf:
        #Find nodes that the next edge is forbidden
            #Nodes that are alraedy part of the grid
            #Edges where either of the nodes is part of the grid
        forbidded_off_grid_edges = ((np.nanmax(E, axis=1) == 0).T * ones).T #The picked row indicates to known node and column for the new
        forbidded_on_grid_edges = ((np.nanmax(E, axis=1) == 1).T * ones)
        W_next = W + ((forbidded_off_grid_edges + forbidded_on_grid_edges) * maxw)
        minw = np.min(W_next)
        known_n, new_n = np.argwhere(W_next== minw)[0]
        D[new_n,:] = D[known_n,:] + minw #Update the distance row to the found new node
        D[:,new_n] = D[:,known_n] + minw #update the distance column to the found new node
        D[new_n][new_n] = 0 # Overwrite the zero distance
        E[known_n][new_n] = E[new_n][known_n] = E[new_n][new_n] = True
        #Update the paths from and toward the new node
        for n in np.argwhere(D[known_n,:] < np.inf):
            n=n[0]
            if n != new_n:
                #Update the path from the new node
                P[new_n,n] = [[new_n, known_n]] + P[known_n,n]
                #Reserve the list as path toward the node
                P[n, new_n] = [[n2, n1] for n1, n2 in list(reversed(P[new_n,n]))]
                #STEPS[new_n,:] = STEPS[known_n,:] + 1
                #STEPS[:,new_n] = STEPS[:,known_n] + 1
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


def transmission_optimal_production_sites(F, D, EC, P, node_count, prod_count, loss):
    optimal_prod_node = 0
    D = D[:prod_count,prod_count:]
    EC = EC[:prod_count,prod_count:]
    P = P[:prod_count,prod_count:]
    all_ns = np.arange(node_count-prod_count)
    #Return production sites with highest Capacity Factor, so that the capacity matches the requirement
    
    #Calculate how much capacity is required to match all losses
    EperC = F - F * D * loss #E / C c=1
    prod_n_selected = np.argmax(EperC, axis=0)
    F = F[prod_n_selected, all_ns]
    D = D[prod_n_selected, all_ns]
    EC_1d = EC[prod_n_selected, all_ns] #Becomes one 
    C_per_E = np.sum( EC_1d / (F * ( 1 - D * loss ) ) ) / np.sum(EC_1d)
    d_per_E = np.dot(D, EC_1d) / np.sum(EC_1d)
    EE = transmission_energy_network_2d(EC, P, node_count, prod_n_selected, all_ns)
    return C_per_E, d_per_E, EE, set(prod_n_selected) #Production capacity, Distances tranmitted, Transmitted powers

def transmission_energy_network_1d(EC, P, node_count):
    #Iterate path is P matrice and create Energy Edges matrice
    EE = np.zeros((node_count, node_count)) #TODO add the loss energy
    for i in range(0, EC.shape[0]):
        if EC[i] > 0:
            for ii, jj in P[i]:
                EE[ii, jj] += EC[i]
    return EE

def transmission_energy_network_2d(EC, P, node_count, n_min_total, all_ns):
    #Iterate path is P matrice and create Energy Edges matrice
    EE = np.zeros((node_count, node_count)) #TODO add the loss energy
    for j in all_ns:
        i = n_min_total[j]
        if EC[i, j] > 0:
            for ii, jj in P[i, j]:
                EE[ii, jj] += EC[i, j]
    return EE



def generation_optimal_production_sites(F, D, EC, P, node_count, prod_count, loss):
    optimal_prod_node = 0
    D = D[optimal_prod_node,prod_count:]
    F = F[optimal_prod_node,:]
    EC = EC[optimal_prod_node,prod_count:]
    P = P[optimal_prod_node,prod_count:]
    #Return production sites with highest Capacity Factor, so that the capacity matches the requirement
    #Calculate how much capacity is required to match all losses.
    #Capacity / Energy * 8760
    C_per_E = np.sum( EC / (F * ( 1 - D * loss ) ) ) / np.sum(EC)
    #Weighted average on distance
    d_per_E = np.dot(D, EC) / np.sum(EC)
    EE = transmission_energy_network_1d(EC, P, node_count)
    return C_per_E, d_per_E, EE, {0} #Production capacity, Distances tranmitted, Transmitted powers

def create_shortcuts(W, D, E, P, shortcutx):
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
        #Do breadth first search to update the distances and paths until it is not a shortcut
        for orig_n, shortcut_n in iterable_nodes:
            d = W[orig_n,shortcut_n]
            mask = D[orig_n,:] > (D[shortcut_n,:] + d)
            
            if sum(mask) > 0:
                #Update the distance to the node with the information provided by the shortcut node
                D[orig_n, mask] = D[mask, orig_n] = D[shortcut_n,mask] + d
                #Add child nodes of this node to iterate (orig_n becomes shortcut n)
                iterable_nodes += [(child[0], orig_n) for child in np.argwhere(E[orig_n,:] == True) if child[0] != shortcut_n] #Prevent that we wont return back toward the shortcut node
                for n in np.argwhere(mask == True):
                    n=n[0]
                    if n != orig_n:
                        P[orig_n,n] = [[orig_n, shortcut_n]] + P[shortcut_n,n]
                        P[n, orig_n] = [[n2, n1] for n1, n2 in list(reversed(P[orig_n,n]))]
    return D, E, P
            
def print_graph(E, pos, node_count, prod_count, prod_n_selected):
    # The producer node is orange if there is production and grey if no production
    prod_node_colors = [(255/255, 165/255, 0) if n in prod_n_selected else (128/255, 128/255, 128/255) for n in range(0,prod_count)]
    # The consumer node color is green
    cons_node_colors = [(2/255, 100/255, 64/255)]* (node_count-prod_count)


    G = nx.from_numpy_matrix(E)
    nx.draw(G, pos=pos, with_labels=True, font_weight='bold', node_color=prod_node_colors+cons_node_colors)
    plt.show()



def calculation(loss):
    #Variables
    node_count = 150
    prod_count = 15
    shortcutx = 3
    area_width = int(random()*1500)
    area_height = int(random()*1500)

    #Algorithms
    nodes, pos = random_nodes(node_count, area_width, area_height)
    W = direct_distances(nodes, node_count)
    EC = consumptions(node_count, prod_count)
    F = capacity_factors(node_count, prod_count)
    D, E, P = distances_edges_paths(W, node_count)
    #print_graph(E, pos)
    D, E, P = create_shortcuts(W, D, E, P, shortcutx)
    
    C_per_E_zonal, d_per_E_zonal, EE_zonal, prod_n_selected = generation_optimal_production_sites(F, D, EC, P, node_count, prod_count, loss)
    print_graph(E, pos, node_count, prod_count, prod_n_selected)
    C_per_E_nodal, d_per_E_nodal, EE_nodal, prod_n_selected = transmission_optimal_production_sites(F, D, EC, P, node_count, prod_count, loss)
    print_graph(E, pos, node_count, prod_count, prod_n_selected)   
    return (area_width, area_height, prod_count, node_count, C_per_E_zonal, d_per_E_zonal, np.std(EE_zonal), C_per_E_nodal, d_per_E_nodal, np.std(EE_nodal))
    


#https://stackoverflow.com/questions/44360084/multiplying-numpy-2d-array-with-1d-array
#lossess: https://library.e.abb.com/public/56aef360ec16ff59c1256fda004aeaec/04MP0274%20Rev.%2000.pdf

print("started")
t0 = t.time()
loss = 6 / 100 / 1000 # %/1000km
for _ in range (0,4):
    results = [result_tuple for result_tuple in calculation(loss)]
print(t.time() - t0)
