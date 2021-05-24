

import random
import math
import numpy as np

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
W = np.ones((node_count, node_count)) * np.inf
for i, coord, d1, r1  in coords:
    for j, coord, d2, r2 in coords[i+1:]:
        W[i,j] = W[j][i] =  return_distance(d1, d2, r1, r2)

#Do prims algorithm and store distances between node
E = np.ones((node_count, node_count)) * False #TRUE / FALSE
E[0][0] = True
D = np.ones((node_count, node_count)) * np.inf # NP.INF / NUM

while np.sum(distances) != np.inf:
    #Get from rows where edges alraedy exist, next min: row, colum, value
    #Pass the parent rows for the next new row&column+ min_weight
    #Set the new weight

    #Create mask with Trues on rows and replace False rows with INF
    next_E_rows = W*([np.nanmax(edges, axis=1) == 1] * node_count)
    #Create mask which gives true for potential next edges and np.nan for used
    minw = numpy.amin(next_E_rows)
    i, j = np.where(next_E_rows == minw)
    D[j][:] = D[i][:] + minw
    D[i][j] = D[j][i] = minw
    E[i][j] = E[j][i] = True
