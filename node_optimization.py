import networkx as nx
import matplotlib.pyplot as plt


#Add nodes
G = nx.Graph()
pos = {0: (400, 20), 1: (20, 30), 2: (40, 30), 3: (30, 10)} 
for n, p in pos.items():
    G.add_node(n,pos=p)


#Add edge (link)
G.add_edge(0,1)

#Draw
nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
plt.show()
print("done")

#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path_length.html#networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path_length

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
A = np.array([[1, 1, 1], [5, 11, 55], [1, 1221, 0]])
G = nx.from_numpy_matrix(A)
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()