
import random
import math
import numpy as np
import time as t

print("started")
t0 = t.time()
#Create random nodes
coordinates = {}
coords = [] # {n: ((x, y), distance, radius)}
node_count = 1000
d = {}
n = 0
#Section is a bit slow
while len(d) < node_count:
    x, y = int(random.random()*50), int(random.random()*50)
    if (x, y) not in d:
        coords.append([n, (x, y), math.sqrt(x**2 + y**2), math.atan2(y,x)])
        d[(x, y)] = True
        n += 1

#Create matrix between distances of nodes

def return_distance(d1, d2, r1, r2):
    #Uses cosine formula
    radius = min(abs(r1-r2), 2 * math.pi - abs(r1-r2))
    return math.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * math.cos(radius))


#Create weights and distances
W = np.zeros((node_count, node_count))
for i, coord, d1, r1  in coords:
    for j, coord, d2, r2 in coords[i+1:]:
        W[i,j] = W[j][i] =  round(return_distance(d1, d2, r1, r2), 1)
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
    no_go_rows = ((np.nanmax(E, axis=1) == 0).T * ones).T #The picked row indicates to known node and column for the new
    W_next = W + ((E + no_go_rows) * maxw) #Make sure that already picked ones and not available rows are not the min
    minw = np.min(W_next) 
    i, j = np.argwhere(W_next== minw)[0]
    D[j,:] = D[i,:] + minw #Update the row to the found new node
    D[:,j] = D[:,i] + minw #update the column to the found new node
    D[j][j] = 0
    E[i][j] = E[j][i] = E[j][j] = True
print(W)
print(D)
print(E)
print(t.time() - t0)


#https://stackoverflow.com/questions/44360084/multiplying-numpy-2d-array-with-1d-array