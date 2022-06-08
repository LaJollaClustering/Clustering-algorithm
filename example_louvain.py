import cluster_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

# load the karate club graph
G = nx.karate_club_graph()

# The initial cluster
# In this example, we use the default initial cluster
initial_cluster = {}
for i, v in enumerate(G.nodes()):
    initial_cluster[v] = i

# compute the best leveled cluster
l_cluster = cluster_louvain.best_leveled_cluster(G, initial_cluster)

# draw the graph
pos = nx.spring_layout(G)

# color the nodes according to their leveled cluster
cmap = cm.get_cmap('viridis', max(l_cluster.values()) + 1)
nx.draw_networkx_nodes(G, pos, l_cluster.keys(), node_size=40, 
                       cmap=cmap, node_color=list(l_cluster.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()