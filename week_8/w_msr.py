import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import itertools


def sign(num):
    if num < 0:
        return -1.0
    elif num > 0:
        return 1.0
    else:
        return 0.0


def create_graph_by_adjacency_matrix(from_file, n, left_side):

    graph = nx.DiGraph()

    for i in range(1, n + 1):
        graph.add_node(i, value=[])

    matrix = np.identity(n)
    i = 0
    j = 0
    flag = True
    with open(from_file) as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            j = 0
            print(tmp_input)
            for tmp in tmp_input:
                if flag:
                    graph.node[j + 1]['value'].append(float(tmp))
                else:
                    matrix[i][j] = int(tmp)
                j += 1
            if not flag:
                i += 1
            flag = False

    # import robustness_checker
    # r,s = robustness_checker.determine_robustness_multi_process(matrix)

    # print(r, s)

    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                if i + 1 in left_side:
                    if j + 1 in left_side:
                        sign = 1
                    else:
                        sign = -1
            else:
                sign = 0


            if not graph.has_edge(i + 1, j + 1):
                graph.add_edge(i + 1, j + 1, sign=sign)  # True 代表正， False代表负

    import robustness_checker
    a, b = robustness_checker.determine_robustness_multi_process(matrix)
    print(a, b)

    return graph


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
        # if abs(neighbor_values[index_end]['value']) > abs(cur_value):
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
        fg, pos, edge_labels=nx.get_edge_attributes(fg, 'sign'), label_pos=0.3)
    plt.savefig("./pngs/graph_" + name + ".png")
    plt.show()


"""
领接矩阵每一行绝对值相加等于1，使得|A|是一个stochastic matrix
"""


def w_msr():
    graph = create_graph_by_adjacency_matrix(
        "./data/data_2_robust_7_7_nodes.in", 14, left_side=[1, 2, 12, 11, 3, 13, 10])

    draw_graph(graph, "2_robust_7_7_nodes")
    print(graph.edges(data=True))
    for time_step in range(1000):
        for i in graph.nodes():
            # if i == 4:
            #     if time_step == 1:
            #         graph.node[i]['value'].append(graph.node[i]['value'][time_step] - 4)
            #     else:
            #         graph.node[i]['value'].append(graph.node[i]['value'][time_step] + 0.01)
            #     continue
            current_value = graph.node[i]['value'][time_step]
            in_edges = graph.in_edges(i)

            # print("in_edges:", in_edges)

            in_neighbors = get_in_neighbors(i, in_edges)
            delta = 0.0
            weight = 1.0 / len(in_neighbors)
            # print("neighbors:", in_neighbors)
            for j in in_neighbors:
                current_neighbor_value = graph.node[j]['value'][time_step]
                sgn = graph[j][i]['sign']
                print("sgn:", sgn)
                delta += weight * (current_neighbor_value *
                                   sign(sgn) - current_value)
            final_result = current_value + 0.01 * delta
            graph.node[i]['value'].append(final_result)

    plt.xlabel("time-step")
    plt.ylabel("values")

    # print("shape" + str(len(graph.node[]['value'])))

    x_axis = range(1001)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig("./pngs/w_msr_2_robust_7_7_nodes.png")
    plt.show()


if __name__ == '__main__':
    w_msr()
