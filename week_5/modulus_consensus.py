# Finite-Time Consensus for Multiagent Systems With Cooperative and Antagonistic Interactions
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


def get_limited_neighbors(graph, neighbors, time_step, cur_value, F):
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
        neighbor_values.append({'node': i, 'value': graph.node[i]['value'][time_step]})

    # print(neighbor_values)

    neighbor_values = sorted(neighbor_values, key=lambda node: node['value'])

    # print(neighbor_values)

    index_front = 0
    while index_front < F:
        if neighbor_values[index_front]['value'] < cur_value:
            index_front += 1
        else:
            break

    index = 0
    index_end = len(neighbor_values) - 1
    while index < F:
        if neighbor_values[index_end]['value'] > cur_value:
            index_end -= 1
            index += 1
        else:
            break

    final_neighbors = []
    for item in neighbor_values[index_front:index_end + 1]:
        final_neighbors.append(item['node'])

    return final_neighbors


def create_graph(data_path, weighted=False):
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

            if weighted:
                flag = True
                for num in range(2, len(tmp_input)):
                    if flag:
                        graph.add_edge(int(tmp_input[0]), int(tmp_input[num]), weight=0.0)
                        flag = False
                    else:
                        graph.add_edge(int(tmp_input[0]), int(tmp_input[num - 1]), weight=float(tmp_input[num]))
                        flag = True
            else:
                for num in range(2, len(tmp_input)):
                    graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
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


def drawGraph(fg, name):
    """
    (2, 2)-robustness => (2, 1)-robustness => 2-robustness
    :param fg: 
    :param name: 
    :return: 
    """
    # pos = dict()
    # pos.setdefault(1, [1, 3])
    # pos.setdefault(2, [3, 5])
    # pos.setdefault(3, [5, 5])
    # pos.setdefault(4, [7, 5])
    # pos.setdefault(5, [11, 5])
    # pos.setdefault(6, [13, 3])
    # pos.setdefault(7, [9, 3])
    # pos.setdefault(8, [11, 1])
    # pos.setdefault(9, [7, 1])
    # pos.setdefault(10, [5, 1])
    # pos.setdefault(11, [3, 1])
    # pos.setdefault(12, [3, 3])
    # pos.setdefault(13, [5, 3])
    # pos.setdefault(14, [7, 3])

    pos = dict()
    pos.setdefault(1, [1, 2])
    pos.setdefault(2, [3, 2])
    pos.setdefault(3, [3, 1])
    pos.setdefault(4, [1, 1])
    pos.setdefault(5, [5, 2])
    pos.setdefault(6, [7, 2])
    pos.setdefault(7, [7, 1])
    pos.setdefault(8, [5, 1])

    nx.draw_networkx_nodes(fg, pos, node_size=450)
    nx.draw_networkx_edges(fg, pos)
    nx.draw_networkx_labels(fg, pos)
    nx.draw_networkx_edge_labels(fg, pos, edge_labels=nx.get_edge_attributes(fg, 'weight'), label_pos=0.3)
    plt.savefig("./pngs/" + name + ".png")
    plt.show()


def ftc_cai(data_path, fig_name, fig_path, graph_name):
    a = 0.6
    graph = create_graph(data_path=data_path, weighted=True)
    # graph = create_graph(data_path=data_path)
    matrix = get_adj_mat(graph=graph)

    drawGraph(graph, graph_name)

    print(matrix)
    for time_step in range(10000):
        for i in range(1, len(graph.nodes()) + 1):
            neighbors = graph.neighbors(i)
            sum = 0.0
            for j in neighbors:
                sum += (matrix[i - 1][j - 1] * (
                    graph.node[j]['value'][time_step] - sign(matrix[i - 1][j - 1]) * graph.node[i]['value'][time_step]))
            final_result = graph.node[i]['value'][time_step] + 0.01 * sign(sum) * (abs(sum) ** a)
            graph.node[i]['value'].append(final_result)

    for i in graph.nodes():
        print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(10001)
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
            return cur_value + 0.001
    elif attack_mode == 0:
        if time_step % 2 == 0:
            return random.uniform(26, 27)
        else:
            return random.uniform(24, 25)
    else:
        return 15 + math.sin(time_step * 0.005)


def ftc_cai_f_local(data_path, fig_path, fig_name, malicious_node, attack_mode):
    a = 0.6
    graph = create_graph(data_path=data_path, weighted=True)
    matrix = get_adj_mat(graph=graph)
    print(matrix)

    for time_step in range(15000):
        for i in range(1, len(graph.nodes()) + 1):
            if i == malicious_node:
                graph.node[i]['value'].append(
                    attack(cur_value=graph.node[i]['value'][time_step], time_step=time_step, attack_mode=attack_mode))
                continue
            neighbors = get_limited_neighbors(graph=graph, neighbors=graph.neighbors(i), time_step=time_step,
                                              cur_value=graph.node[i]['value'][time_step], F=1)
            sum = 0.0
            for j in neighbors:
                sum += (matrix[i - 1][j - 1] * (
                    graph.node[j]['value'][time_step] - sign(matrix[i - 1][j - 1]) * graph.node[i]['value'][time_step]))
            final_result = graph.node[i]['value'][time_step] + 0.001 * sign(sum) * (abs(sum) ** a)
            graph.node[i]['value'].append(final_result)

    # for i in graph.nodes():
    #     print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(15001)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'], label='node ' + str(i))
    # plt.legend(loc='bottom right', bbox_to_anchor=(0.1, 1.05), ncol=5)
    # plt.legend(loc='best', ncol=5)
    plt.savefig(
        fig_path + "/" + fig_name + '_malicious_node_' + str(malicious_node) + '_attack_' + str(attack_mode) + '.png')
    plt.show()


if __name__ == '__main__':
    # Balanced and unbalanced cases
    # ftc_cai(data_path="./data/data-balanced-with-7_7-nodes.in", fig_path="./pngs",
    #         fig_name="modulus_consensus_balanced_7_7_nodes", graph_name="graph_balanced_7_7_nodes")
    # ftc_cai(data_path="./data/data-unbalanced-with-7_7-nodes.in", fig_path="./pngs",
    #         fig_name="modulus_consensus_unbalanced_7_7_nodes", graph_name="graph_unbalanced_7_7_nodes")

    ftc_cai(data_path="./data/data-balanced-with-4_4-nodes.in", fig_path="./pngs",
            fig_name="modulus_consensus_balanced_4_4_nodes", graph_name="graph_balanced_4_4_nodes")
    # add F-local attack with F = 1
    # ftc_cai_f_local(data_path="./data/data-balanced-with-7_7-nodes.in", fig_path="./pngs",
    #                 fig_name="modulus_consensus_unbalanced_7_7_nodes", malicious_node=4, attack_mode=1)
    # ftc_cai_f_local(data_path="./data/data-connected-with-7_7-nodes.in", fig_path="./pngs",
    #                 fig_name="Finit-Time-Consensus-Balanced-With-7_7-Nodes", malicious_node=i, attack_mode=1)
    # ftc_cai_f_local(data_path="./data/data-connected-with-7_7-nodes.in", fig_path="./pngs",
    #                 fig_name="Finit-Time-Consensus-Balanced-With-7_7-Nodes", malicious_node=i, attack_mode=2)
