import random
import math
import numpy as np
import time as t
import networkx as nx
import matplotlib.pyplot as plt

print("started")
t0 = t.time()
#Create random nodes
formulas = []
pos = {}# {n: ( distance, radius)}
node_count = 100
prod_count = 20
capacity_factors = {n: 1 - (n/prod_count) for n in range(0, prod_count)}
n = 0
loss = 6 / 100 / 1000 # %/1000km

while len(pos) < node_count:
    x, y = int(random.random()*1000), int(random.random()*1000)
    if (x, y) not in pos:
        formulas.append([n, math.sqrt(x**2 + y**2), math.atan2(y,x)])
        pos[(x, y)] = n
        n += 1
    
pos = {n: coord for coord, n in pos.items()}

#Create matrix between distances of nodes

def return_distance(d1, d2, r1, r2):
    #Uses cosine formula
    radius = min(abs(r1-r2), 2 * math.pi - abs(r1-r2))
    return math.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * math.cos(radius))


#Create weights and distances
W = np.zeros((node_count, node_count))
F = np.zeros((node_count, node_count))
for i, d1, r1  in formulas:
    for j, d2, r2 in formulas[i+1:]:
        W[i,j] = W[j][i] =  round(return_distance(d1, d2, r1, r2), 1)
    if i < prod_count:
        F[:,i] = F[i,:] = capacity_factors[i]
maxw = np.max(W)

#Do prims algorithm and store distances between node
E = np.zeros((node_count, node_count)) #Edges: TRUE / FALSE
E[0][0] = True
D = np.ones((node_count, node_count)) * np.inf #Distances betwee nodes: INF / NUM
D[0][0] = 0
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
    i, j = np.argwhere(W_next== minw)[0]
    D[j,:] = D[i,:] + minw #Update the row to the found new node
    D[:,j] = D[:,i] + minw #update the column to the found new node
    D[j][j] = 0
    E[i][j] = E[j][i] = E[j][j] = True
print(t.time() - t0)
np.fill_diagonal(D, np.inf)#Make sure that the minimum distance is not zero

#Assistin variables
Dpc = D[:prod_count,prod_count:]
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

G = nx.from_numpy_matrix(E)
nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
plt.show()



# Calculate status quo: Optimal location
    # Select the maximum node
    # Calculate consumption and lossess
    # 
# Calculate loss-aware optimum
    # Select minimum per row for E - W*lossess
    # Calculate consumption and lossess

#Find percentage where loss-aware and location-only are the same.



#https://stackoverflow.com/questions/44360084/multiplying-numpy-2d-array-with-1d-array

#lossess: https://library.e.abb.com/public/56aef360ec16ff59c1256fda004aeaec/04MP0274%20Rev.%2000.pdf