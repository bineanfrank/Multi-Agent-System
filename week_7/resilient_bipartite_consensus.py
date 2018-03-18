# resilient_bipartite_consensus.py

import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


def sign(num):
    if num < 0:
        return -1.0
    elif num > 0:
        return 1.0
    else:
        return 0.0


def get_limited_neighbors(node_no, graph, neighbors, time_step, cur_value, F):
    """
    get neighbors, F larger or F smaller will be removed.
    :param graph:
    :param neighbors:
    :param time_step:
    :param cur_value:
    :param F: F-local
    :return:
    """

    # key value pairs
    neighbor_values = []

    for i in neighbors:
        neighbor_values.append(
            {'node': i, 'value': graph.node[i]['value'][time_step]})

    # print(neighbor_values)
    neighbor_values = sorted(neighbor_values, key=lambda node: node['value'])

    # print(neighbor_values)

    index_front = 0
    while index_front < F:
        # if abs(neighbor_values[index_front]['value']) < abs(cur_value):
        if neighbor_values[index_front]['value'] < cur_value:
            index_front += 1
        else:
            break

    index = 0
    index_end = len(neighbor_values) - 1
    while index < F:
        #if abs(neighbor_values[index_end]['value']) > abs(cur_value):
        if neighbor_values[index_end]['value'] > cur_value:
            index_end -= 1
            index += 1
        else:
            break

    final_neighbors = []
    for item in neighbor_values[index_front:index_end + 1]:
        final_neighbors.append(item['node'])

    return final_neighbors


def create_graph(data_path, nodes_num):
    """
    create a graph and return
    :param data_path: data source
    :param weighted: if it is a weighted graph
    :return:
    """
    graph = nx.DiGraph()

    with open(data_path) as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            edges = []
            for i in range(2, len(tmp_input), 2):
                edge = (int(tmp_input[0]), int(
                    tmp_input[i]), float(tmp_input[i + 1]))
                edges.append(edge)
            graph.add_weighted_edges_from(edges)
    return graph


def get_adj_mat(graph):
    """
    get the adjacency matrix
    :param graph:
    :return:
    """
    size = len(graph.nodes())
    matrix = np.zeros(shape=(size, size), dtype=np.float)

    for edge in graph.edges(data=True):
        matrix[edge[0] - 1][edge[1] - 1] = edge[2]['weight']
        matrix[edge[1] - 1][edge[0] - 1] = edge[2]['weight']

    return matrix


def get_in_neighbors(node, in_edge):
    in_neighbors = []
    for edge in in_edge:
        in_neighbors.append(edge[1] if edge[0] == node else edge[0])
    return in_neighbors


def draw_graph(fg, name):
    """
    (2, 2)-robustness => (2, 1)-robustness => 2-robustness
    :param fg:
    :param name:
    :return:
    """
    pos = dict()
    pos.setdefault(1, [1, 3])
    pos.setdefault(2, [5, 5])
    pos.setdefault(3, [9, 5])
    pos.setdefault(4, [13, 5])
    pos.setdefault(5, [18, 5])
    pos.setdefault(6, [20, 3])
    pos.setdefault(7, [16, 3])
    pos.setdefault(8, [18, 1])
    pos.setdefault(9, [13, 1])
    pos.setdefault(10, [9, 1])
    pos.setdefault(11, [5, 1])
    pos.setdefault(12, [5, 3])
    pos.setdefault(13, [9, 3])
    pos.setdefault(14, [13, 3])
    nx.draw_networkx_nodes(fg, pos, node_size=300)
    nx.draw_networkx_edges(fg, pos)
    nx.draw_networkx_labels(fg, pos)
    nx.draw_networkx_edge_labels(
        fg, pos, edge_labels=nx.get_edge_attributes(fg, 'weight'), label_pos=0.3)
    plt.savefig("./pngs/graph_" + name + ".png")
    plt.show()


def modulus_consensus(data_path, fig_name, fig_path, graph_name):
    a = 0.6
    graph = create_graph(data_path=data_path, nodes_num=14)
    # graph = create_graph(data_path=data_path)
    matrix = get_adj_mat(graph=graph)

    draw_graph(graph, graph_name)

    print(matrix)
    for time_step in range(1000):
        for i in graph.nodes():
            current_value = graph.node[i]['value'][time_step]
            in_edges = graph.in_edges(i)
            in_neighbors = get_in_neighbors(i, in_edges)
            delta = 0.0
            for j in in_neighbors:
                current_neighbor_value = graph.node[j]['value'][time_step]
                delta += abs(matrix[j - 1][i - 1]) * (current_neighbor_value * sign(matrix[j - 1][i - 1]) - current_value)
            final_result = current_value + 0.01 * delta
            graph.node[i]['value'].append(final_result)

    for i in graph.nodes():
        print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(1001)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig(fig_path + "/" + fig_name + '.png')
    plt.show()


def attack(cur_value, time_step, attack_mode):
    if attack_mode == 1:
        '''
        F-local attack
        '''
        if time_step == 1:
            '''First sharp change'''
            return -3
        else:
            '''Then, change steadily'''
            return cur_value + 0.01
    elif attack_mode == 0:
        return random.uniform(-2, 2)
    else:
        return 9 + math.sin(time_step * 0.09)


def resilient_bipartite_consensus(data_path, fig_path, fig_name, malicious_node, attack_mode):
    a = 0.6
    graph = create_graph(data_path=data_path, nodes_num=14)
    matrix = get_adj_mat(graph=graph)
    print(matrix)

    for time_step in range(3000):
        for i in graph.nodes():
            current_value = graph.node[i]['value'][time_step]
            in_edges = graph.in_edges(i)
            in_neighbors = get_in_neighbors(i, in_edges)
            if i == malicious_node:
                graph.node[i]['value'].append(
                    attack(cur_value=current_value, time_step=time_step, attack_mode=attack_mode))
                continue
            neighbors = get_limited_neighbors(node_no=i, graph=graph, neighbors=in_neighbors, time_step=time_step,
                                              cur_value=current_value, F=1)
            delta = 0.0
            for j in neighbors:
                current_neighbor_value = graph.node[j]['value'][time_step]
                delta += abs(matrix[j - 1][i - 1]) * (current_neighbor_value * sign(matrix[j - 1][i - 1]) - current_value)
            final_result = current_value + 0.1 * delta
            graph.node[i]['value'].append(final_result)

    # for i in graph.nodes():
    #     print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(3001)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'], label='node ' + str(i))
    # plt.legend(loc='bottom right', bbox_to_anchor=(0.1, 1.05), ncol=5)
    # plt.legend(loc='best', ncol=5)
    plt.savefig(
        fig_path + "/" + fig_name + '_malicious_node_' + str(malicious_node) + '_attack_' + str(attack_mode) + '.png')
    plt.show()


if __name__ == '__main__':
    resilient_bipartite_consensus(data_path="./data/data_balanced_7_7_nodes.in",
                                  fig_path="./pngs",
                                  fig_name="resilient_bipartite_consensus",
                                  malicious_node=1,
                                  attack_mode=0)
    modulus_consensus(data_path="./data/data_balanced_7_7_nodes.in",
                      fig_path="./pngs",
                      fig_name="modulus_consensus",
                      graph_name="balanced_7_7_nodes")
