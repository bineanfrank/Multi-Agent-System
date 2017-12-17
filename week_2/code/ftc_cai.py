# Finite-Time Consensus for Multiagent Systems With Cooperative and Antagonistic Interactions

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def sign(num):
    if num < 0:
        return -1.0
    elif num > 0:
        return 1.0
    else:
        return 0.0


def create_graph(data_path, weighted=False):
    """
    create a graph and return
    :param data_path: data source
    :param weighted: if it is a weighted graph
    :return: 
    """
    graph = nx.Graph()
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
                        graph.edge[int(tmp_input[0])][int(tmp_input[num - 1])]['weight'] = float(tmp_input[num])
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
    pos = dict()
    pos.setdefault(1, [1, 2])
    pos.setdefault(2, [3, 2])
    pos.setdefault(3, [3, 1])
    pos.setdefault(4, [1, 1])
    pos.setdefault(5, [5, 2])
    pos.setdefault(6, [7, 2])
    pos.setdefault(7, [7, 1])
    pos.setdefault(8, [5, 1])

    nx.draw_networkx_nodes(fg, pos, node_size=300)
    nx.draw_networkx_edges(fg, pos)
    nx.draw_networkx_labels(fg, pos)
    nx.draw_networkx_edge_labels(fg, pos, edge_labels=nx.get_edge_attributes(fg, 'weight'))
    plt.savefig("./pngs/Graph" + name + ".png")
    plt.show()


def ftc_cai(data_path, fig_name, fig_path, graph_name):
    a = 0.6
    graph = create_graph(data_path=data_path, weighted=True)
    matrix = get_adj_mat(graph=graph)

    drawGraph(graph, graph_name)

    print(matrix)
    for time_step in range(1000):
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
    x_axis = range(1001)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig(fig_path + "/" + fig_name + '.png')
    plt.show()


def ftc_cai_f_total(data_path, fig_path, fig_name, malicious_node):
    a = 0.6
    graph = create_graph(data_path=data_path, weighted=True)
    matrix = get_adj_mat(graph=graph)
    print(matrix)

    for time_step in range(1000):
        for i in range(1, len(graph.nodes()) + 1):
            if i == malicious_node:
                if time_step == 1:
                    graph.node[i]['value'].append(4)
                else:
                    graph.node[i]['value'].append(graph.node[i]['value'][time_step] + 0.01)
                continue

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
    x_axis = range(1001)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig(fig_path + "/" + fig_name + '-Malicious-Node-' + str(malicious_node) + '.png')
    plt.show()


def ftc_cai_no_delay(graph):
    matrix = np.array([0, -1, 0, 1, -1, 0, -1, 0, 0, -1, 0, 1, 1, 0, 1, 0], dtype=np.float).reshape(4, 4)
    for time_step in range(1000):
        for i in range(1, len(graph.nodes()) + 1):
            neighbors = graph.neighbors(i)
            sum = 0.0
            for j in neighbors:
                sum += abs(matrix[i - 1][j - 1]) * (
                    graph.node[i]['value'][time_step] - sign(matrix[i - 1][j - 1]) * graph.node[j]['value'][time_step])

            final_result = graph.node[i]['value'][time_step] - 0.01 * sum
            graph.node[i]['value'].append(final_result)

    for i in graph.nodes():
        print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(1001)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig('./pngs/Finite-Time-Consensus-No-Delay.png')
    plt.show()


if __name__ == '__main__':
    # Balanced and unbalanced cases
    ftc_cai(data_path="./data/data-balanced-with-4_4-nodes.in", fig_path="./pngs",
            fig_name="Finit-Time-Consensus-Balanced-With-4_4-Nodes", graph_name="-Balanced")
    ftc_cai(data_path="./data/data-unbalanced-with-4_4-nodes.in", fig_path="./pngs",
            fig_name="Finit-Time-Consensus-Unbalanced-With-4_4-Nodes", graph_name="-Unbalanced")

    # add F-total attack with F = 1
    # ftc_cai_f_total(data_path="./data/data-balanced-with-4_4-nodes.in", fig_path="./pngs",
    #                 fig_name="Finit-Time-Consensus-Balanced-With-4_4-Nodes", malicious_node=8)
    # ftc_cai_f_total(data_path="./data/data-balanced-with-4_4-nodes.in", fig_path="./pngs",
    #                 fig_name="Finit-Time-Consensus-Balanced-With-4_4-Nodes", malicious_node=4)
