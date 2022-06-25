from __future__ import print_function

import array

import numbers
import warnings

import networkx as nx
import numpy as np

from cluster_status import Status

__author__ = """Thomas Aynaud (src), customized by Miryam Huang"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.
#    some functions are added by Miryam Huang

'''
you can set MAX_PASS on your own, 
it represents how many iterations would you like to do in each call of _one_level.
Since some times the user don't want to find the modularity maximum, they want few iterations.
I seldom encounter this situation so I don't really know why the original author set this parameter.
'''
MAX_PASS = -1
_MIN = 0.0000001


def check_random_s(seed):
    if seed is None: 
        return np.random.mtrand._rand
    if seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def leveled_cluster(dendrogram, level):
    # Return the cluster of the nodes (clustering result) at the given level i.e. leveled_cluster (l_cluster)

    l_cluster = dendrogram[0].copy()
    for i in range(1, level + 1):
        for node, cluster in l_cluster.items():
            l_cluster[node] = dendrogram[i][cluster]
    return l_cluster


def modularity(l_cluster, graph, weight='weight'):
    #Compute the modularity of a leveled_cluster.
    #Modularity computation:

    #References
    #Newman, M.E.J. & Girvan, M. Finding and evaluating community
    #structure in networks. Physical Review E 69, 26113(2004).
    
    if graph.is_directed():
        raise TypeError("Graph type error, use only undirected graph")

    inc = dict([]) #intra-cluster degree(indegree): how many intracluster edges of each cluster
    deg = dict([]) #cluster's total degree: in-degree + how many intercluster edges(out-degree)
    edges = graph.size(weight=weight)
    if edges == 0:
        raise ValueError("Undefined modularity")

    for u in graph: #u: node
        cluster = l_cluster[u]#the community that 'node' this vertix belongs to
        #Compute the degree of the cluster
        #same as deg[com]=deg[com]+graph.degree but if deg[com] is empty, set it to 0. i.e., deg.get(cluster, 0.)
        deg[cluster] = deg.get(cluster, 0.) + graph.degree(u, weight=weight)
        for v, datas in graph[u].items(): #v: neighbor node of u
            edge_weight = datas.get(weight, 1)
            if l_cluster[v] == cluster: #same cluster: add in-degree
                if v == u:
                    inc[cluster] = inc.get(cluster, 0.) + float(edge_weight)
                else:
                    inc[cluster] = inc.get(cluster, 0.) + float(edge_weight) / 2.
    #computer modularity
    res = 0.
    for cluster in set(l_cluster.values()):
        res += (inc.get(cluster, 0.) / edges) - \
               (deg.get(cluster, 0.) / (2. * edges)) ** 2
    return res


def best_leveled_cluster(graph,
                   l_cluster=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_s=None,
                   mode=0,
                   is_confident=None):
    
    """Compute the leveled cluster of the graph nodes which maximises the modularity by the Louvain heuristices
    
    resolution :
        changing the size of the clusters, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    randomize :
        In each iteration, randomizing the node evaluation order and the cluster evaluation
        order to get different level clusters.
    random_state :
        If int, random_s is the seed used by the random number generator;
        If RandomState instance, random_s is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    #References
    #Blondel, V.D. et al. Fast unfolding of communities in
    #large networks. J. Stat. Mech 10008, 1-12(2008).

    #the final level_cluster is the best clustering result
    dendo = generate_dendrogram(graph,
                                l_cluster,
                                weight,
                                resolution,
                                randomize,
                                random_s,
                                mode,
                                is_confident)
    return leveled_cluster(dendo, len(dendo) - 1)


def generate_dendrogram(graph,
                        l_cluster_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_s=None,
                        mode=0,
                        is_confident=None):

    #Find clusters in the graph and return the associated dendrogram

    if graph.is_directed():
        raise TypeError("Graph type error, use only undirected graph")

    # Properly handle random state
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If it shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_s = 0

    if randomize and random_s is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_s = check_random_s(random_s)

    # When there is no edge
    # every node is its cluster
    if graph.number_of_edges() == 0:
        clus = dict([])
        for i, node in enumerate(graph.nodes()):
            clus[node] = i
        return [clus]
    #run once
    current_graph = graph.copy() #copy graph and use it as current graph
    status = Status() #the graph structure is maitained in status
    status.initialization(current_graph, weight, l_cluster_init)
    status_list = list() #handle the level_cluster, the whole list is the dendrogram i.e. clustering history
    #do customized one_level louvain
    __customized_one_level(current_graph, status, weight, resolution, random_s, mode,
                   is_confident)
    #compute modularity and set as new_modularity, it will be compared later on
    new_modularity = __modularity(status, resolution)
    #l_cluster is the clustering result: which node belongs to which cluster
    l_cluster = __renumber(status.node_to_cluster)
    #append 1st leveled_cluster (1st level clustering result)
    status_list.append(l_cluster)
    mod = new_modularity
    #Produce induced_graph: compress the clusters to nodes of the new graph
    current_graph = induced_graph(l_cluster, current_graph, weight)
    #Do initialization again and then run same steps as above in the while loop
    status.initialization(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_s)
        new_modularity = __modularity(status, resolution)
        #if finding the optimal culustering (maxmimum modularity), break
        if new_modularity - mod < _MIN:
            break
        l_cluster = __renumber(status.node_to_cluster)
        status_list.append(l_cluster)
        mod = new_modularity
        current_graph = induced_graph(l_cluster, current_graph, weight)
        status.initialization(current_graph, weight)
    return status_list[:]


def induced_graph(l_cluster, graph, weight="weight"):
    #compress clusters to nodes and make new graph for the next level

    ret = nx.Graph() #create a new blank graph
    ret.add_nodes_from(l_cluster.values()) #assing clusters to nodes: cl_1->n_1 cl_2->n_2...
    
    

    #connect edges (inter-cluster edges weights btw clusters in original graph -> edges weights btw nodes)
    '''   
    Trace step: 
    say node_a,node_b have edge(a,b) with weight w(a,b)=3;
    say node_a,node_c have edge(a,c) with weight w(a,c)=8;
    a in cluster_1 c_1;   b, c in cluster_2 c_2;
    Then, in this new graph, our two nodes c1,c2(representing the cluster_1 and cluster_2)
    don't have edges yet. So, w(c1,c2)=0, initially.

    in the for loop, alg originally choose node a, b;
    ret.add_edge updates w(c1,c2) to 0+3=3.
    so next trial alg go to node a, c;
    ret.add_edge updates w(c1,c2) to 3+8=11.
    
    keep repeating this procedure,until all inter-cluster edges weight of c_1 and c_2 are counted
    '''
    for node1, node2, datas in graph.edges(data=True): #pick two nodes (they are indeed nodes in the original clusters)
        edge_weight = datas.get(weight, 1) #get edge weights from those two nodes(clusters) 
        clu1 = l_cluster[node1] #see cluster that node1 belongs to
        clu2 = l_cluster[node2] #see cluster that node2 belongs to
        w_prec = ret.get_edge_data(clu1, clu2, {weight: 0}).get(weight, 1)
        ret.add_edge(clu1, clu2, **{weight: w_prec + edge_weight}) #update edge weight

    return ret


def __renumber(dictionary):
    #example: after clustering, there might be #1, #4, #7 clusters, just simply rename them to #1, #2, #3
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret

'''
I am not familiar with this part.
'''
def load_binary(data):
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __customized_one_level(graph, status, weight_key, resolution, random_s, mode,
                   is_confident):
    #compute custimized one level of clusters
    if mode > 3:
        raise ValueError("no such mode: %d" % mode)
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution) #compute modularity
    new_mod = cur_mod #set modularity

    while modified and nb_pass_done != MAX_PASS:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_s): #randomly pick a node and start from the node
            clu_node = status.node_to_cluster[node] #the cluster that the node belongs to
            #mode 2 CS as initial cluster and remaining nodes do nothing.
            if mode == 2 and is_confident[clu_node] == True:
                continue
            #mode 3 CS as initial cluster and do Louvain, remaining nodes do nothing but add to the CS who did Louvain
            if mode == 3 and is_confident[clu_node] == False:
                continue
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.) #the degree of each node/2m
            neigh_clusters = __neighborcluster(node, graph, status, weight_key) #neighbor cluster
            #remove_cost: wikipedia modularity difference (remove node from its mother cluster)
            remove_cost = - neigh_clusters.get(clu_node,0) + \
                resolution * (status.degrees.get(clu_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            #remove the node
            __remove(node, clu_node,
                     neigh_clusters.get(clu_node, 0.), status)
            best_clu = clu_node
            best_increase = 0
            for clu, dnc in __randomize(neigh_clusters.items(), random_s): #neighbor cluster of current node
                #mode 1: only process CS<->CS, remainging_node<->remaining_node
                if mode == 1 and is_confident[clu_node] != is_confident[clu]:
                    continue
                #mode 2: only process if the neighbor cluster is a confident set
                if mode == 2 and is_confident[clu] == False:
                    continue
                #mode 3: only process if the neighbor cluster is a confident set
                if mode == 3 and is_confident[clu] == False:
                    continue
                #incr: wikipedia modularity difference (add node to its neighbor cluster)
                incr = remove_cost + dnc - \
                       resolution * status.degrees.get(clu, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_clu = clu
            #insert node to the neighbor cluster
            __insert(node, best_clu,
                     neigh_clusters.get(best_clu, 0.), status)
            if best_clu != clu_node:
                modified = True
        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < _MIN: #which means current modularity is the optimal, so break
            break
        if mode == 3:
            __customized_one_level(graph, status, weight_key, resolution, random_s, 2, is_confident)

#normal_one_level
def __one_level(graph, status, weight_key, resolution, random_s):
    #compute one level of clusters

    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution) #compute modularity
    new_mod = cur_mod #set modularity

    while modified and nb_pass_done != MAX_PASS:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_s): #randomly pick a node and start from the node
            clu_node = status.node_to_cluster[node] #the cluster that the node belongs to
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.) #the degree of each node/2m
            neigh_clusters = __neighborcluster(node, graph, status, weight_key) #neighbor cluster
            #remove_cost: wikipedia modularity difference (remove node from its mother cluster)
            remove_cost = - neigh_clusters.get(clu_node,0) + \
                resolution * (status.degrees.get(clu_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            #remove the node
            __remove(node, clu_node,
                     neigh_clusters.get(clu_node, 0.), status)
            best_clu = clu_node
            best_increase = 0
            for clu, dnc in __randomize(neigh_clusters.items(), random_s): #neighbor cluster of current node
                #incr: wikipedia modularity difference (add node to its neighbor cluster)
                incr = remove_cost + dnc - \
                       resolution * status.degrees.get(clu, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_clu = clu
            #insert node to the neighbor cluster
            __insert(node, best_clu,
                     neigh_clusters.get(best_clu, 0.), status)
            if best_clu != clu_node:
                modified = True
        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < _MIN: #which means current modularity is the optimal, so break
            break


def __neighborcluster(node, graph, status, weight_key):

    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborclu = status.node_to_cluster[neighbor]
            weights[neighborclu] = edge_weight + weights.get(neighborclu, 0) 

    return weights


def __remove(node, com, weight, status):
    #Remove node from cluster com and modify status
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node_to_cluster[node] = -1


def __insert(node, clu, weight, status):
    #Insert node into cluster and modify status
    status.node_to_cluster[node] = clu
    status.degrees[clu] = (status.degrees.get(clu, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[clu] = float(status.internals.get(clu, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status, resolution):
    
    #Fast compute the modularity of the leveled cluster of the graph using precomputed status
    
    edges = float(status.total_weight)
    result = 0.
    for cluster in set(status.node_to_cluster.values()):
        in_degree = status.internals.get(cluster, 0.)
        degree = status.degrees.get(cluster, 0.)
        if edges > 0:
            result += in_degree * resolution / edges -  ((degree / (2. * edges)) ** 2)
    return result


def __randomize(items, random_s):
    #Returns a list containing a random permutation of items
    randomized_items = list(items)
    random_s.shuffle(randomized_items)
    return randomized_items
