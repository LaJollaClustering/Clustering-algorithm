from __future__ import print_function

import array

import numbers
import warnings

import networkx as nx
import numpy as np

from cluster_status import Status

__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.
#    some problems were fixed by Miryam Huang

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
    # Return the cluster of the nodes at the given level i.e. leveled_cluster (l_cluster)

    l_cluster = dendrogram[0].copy()
    for i in range(1, level + 1):
        for node, cluster in l_cluster.items():
            l_cluster[node] = dendrogram[i][cluster]
    return l_cluster


def modularity(l_cluster, graph, weight='weight'):
    #Compute the modularity of a leveled cluster of a graph

    #References
    #Newman, M.E.J. & Girvan, M. Finding and evaluating community
    #structure in networks. Physical Review E 69, 26113(2004).
    
    if graph.is_directed():
        raise TypeError("Graph type error, use only undirected graph")

    inc = dict([])
    deg = dict([])
    edges = graph.size(weight=weight)
    if edges == 0:
        raise ValueError("Undefined modularity")

    for node in graph:
        clu = l_cluster[node]
        deg[clu] = deg.get(clu, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if l_cluster[neighbor] == clu:
                if neighbor == node:
                    inc[clu] = inc.get(clu, 0.) + float(edge_weight)
                else:
                    inc[clu] = inc.get(clu, 0.) + float(edge_weight) / 2.

    res = 0.
    for clu in set(l_cluster.values()):
        res += (inc.get(clu, 0.) / edges) - \
               (deg.get(clu, 0.) / (2. * edges)) ** 2
    return res


def best_leveled_cluster(graph,
                   l_cluster=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_s=None):
    
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

    dendo = generate_dendrogram(graph,
                                l_cluster,
                                weight,
                                resolution,
                                randomize,
                                random_s)
    return leveled_cluster(dendo, len(dendo) - 1)


def generate_dendrogram(graph,
                        l_cluster_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_s=None):

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

    # special case, when there is no edge
    # every node is its cluster
    if graph.number_of_edges() == 0:
        clus = dict([])
        for i, node in enumerate(graph.nodes()):
            clus[node] = i
        return [clus]

    current_graph = graph.copy()
    status = Status()
    status.initialization(current_graph, weight, l_cluster_init)
    status_list = list()
    __one_level(current_graph, status, weight, resolution, random_s)
    new_modularity = __modularity(status, resolution)
    l_cluster = __renumber(status.node_to_cluster)
    status_list.append(l_cluster)
    mod = new_modularity
    current_graph = induced_graph(l_cluster, current_graph, weight)
    status.initialization(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_s)
        new_modularity = __modularity(status, resolution)
        if new_modularity - mod < _MIN:
            break
        l_cluster = __renumber(status.node_to_cluster)
        status_list.append(l_cluster)
        mod = new_modularity
        current_graph = induced_graph(l_cluster, current_graph, weight)
        status.initialization(current_graph, weight)
    return status_list[:]


def induced_graph(l_cluster, graph, weight="weight"):
    #Produce the graph where nodes are the clusters

    ret = nx.Graph()
    ret.add_nodes_from(l_cluster.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        clu1 = l_cluster[node1]
        clu2 = l_cluster[node2]
        w_prec = ret.get_edge_data(clu1, clu2, {weight: 0}).get(weight, 1)
        ret.add_edge(clu1, clu2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):

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


def __one_level(graph, status, weight_key, resolution, random_s):
    #compute one level of clusters

    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != MAX_PASS:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_s):
            clu_node = status.node_to_cluster[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.) 
            neigh_clusters = __neighborcluster(node, graph, status, weight_key)
            remove_cost = - neigh_clusters.get(clu_node,0) + \
                resolution * (status.degrees.get(clu_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            __remove(node, clu_node,
                     neigh_clusters.get(clu_node, 0.), status)
            best_clu = clu_node
            best_increase = 0
            for clu, dnc in __randomize(neigh_clusters.items(), random_s):
                incr = remove_cost + dnc - \
                       resolution * status.degrees.get(clu, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_clu = clu
            __insert(node, best_clu,
                     neigh_clusters.get(best_clu, 0.), status)
            if best_clu != clu_node:
                modified = True
        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < _MIN:
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
