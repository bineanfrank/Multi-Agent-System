# Finite-Time Consensus for Multiagent Systems With Cooperative and Antagonistic Interactions

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math


def sign(num):
    if num < 0:
        return -1.0
    elif num > 0:
        return 1.0
    else:
        return 0.0


def ftc_cai_unbalanced(graph, num):
    a = 0.6
    # adj_mat = nx.adjacency_matrix(graph)
    # matrix = np.array(adj_mat.todense(), dtype=np.float)
    # print(matrix)
    matrix = np.array([0, 1, -2, 0, 1, 0, 4, 0, -2, 4, 0, 3, 0, 0, 3, 0], dtype=np.float).reshape(4, 4)

    print(matrix)

    for time_step in range(1500):
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
    x_axis = range(1501)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'])
    if num == 1:
        plt.savefig('./pngs/Finite-Time-Consensus-Unbalanced_1.png')
    else:
        plt.savefig('./pngs/Finite-Time-Consensus-Unbalanced_2.png')
    plt.show()


def ftc_cai_balanced(graph, num):
    a = 0.6
    # adj_mat = nx.adjacency_matrix(graph)
    # matrix = np.array(adj_mat.todense(), dtype=np.float)
    # print(matrix)
    matrix = np.array([0, 1, -2, 0, 1, 0, -4, 0, -2, -4, 0, 3, 0, 0, 3, 0], dtype=np.float).reshape(4, 4)

    print(matrix)

    for time_step in range(600):
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
    x_axis = range(601)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'])
    if num == 1:
        plt.savefig('./pngs/Finite-Time-Consensus-Balanced_1.png')
    else:
        plt.savefig('./pngs/Finite-Time-Consensus-Balanced_2.png')
    plt.show()


def ftc_cai_no_delay(graph):
    # adj_mat = nx.adjacency_matrix(graph, weight='weight')
    # matrix = np.array(adj_mat.todense(), dtype=np.float)
    # print("matrix = ")
    # print(matrix)
    matrix = np.array([0, -1, 0, 1, -1, 0, -1, 0, 0, -1, 0, 1, 1, 0, 1, 0], dtype=np.float).reshape(4, 4)
    for time_step in range(1000):
        for i in range(1, len(graph.nodes()) + 1):
            neighbors = graph.neighbors(i)
            print("neighbors")
            print(neighbors)

            sum = 0.0
            for j in neighbors:
                print("matrix[i - 1][j - 1] = %d" % matrix[i - 1][j - 1])

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

    graph = nx.Graph()
    with open("./data/data-balanced1.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            flag = True
            for num in range(2, len(tmp_input)):
                if flag:
                    graph.add_edge(int(tmp_input[0]), int(tmp_input[num]), weight=0.0)
                    flag = False
                else:
                    graph.edge[int(tmp_input[0])][int(tmp_input[num - 1])]['weight'] = float(tmp_input[num])
                    flag = True
    ftc_cai_balanced(graph=graph, num=1)

    graph = nx.Graph()
    with open("./data/data-balanced2.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            flag = True
            for num in range(2, len(tmp_input)):
                if flag:
                    graph.add_edge(int(tmp_input[0]), int(tmp_input[num]), weight=0.0)
                    flag = False
                else:
                    graph.edge[int(tmp_input[0])][int(tmp_input[num - 1])]['weight'] = float(tmp_input[num])
                    flag = True
    ftc_cai_balanced(graph=graph, num=2)

    graph = nx.Graph()
    with open("./data/data-unbalanced1.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            flag = True
            for num in range(2, len(tmp_input)):
                if flag:
                    graph.add_edge(int(tmp_input[0]), int(tmp_input[num]), weight=0.0)
                    flag = False
                else:
                    graph.edge[int(tmp_input[0])][int(tmp_input[num - 1])]['weight'] = float(tmp_input[num])
                    flag = True
    ftc_cai_unbalanced(graph=graph, num=1)

    graph = nx.Graph()
    with open("./data/data-unbalanced2.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            flag = True
            for num in range(2, len(tmp_input)):
                if flag:
                    graph.add_edge(int(tmp_input[0]), int(tmp_input[num]), weight=0.0)
                    flag = False
                else:
                    graph.edge[int(tmp_input[0])][int(tmp_input[num - 1])]['weight'] = float(tmp_input[num])
                    flag = True
    ftc_cai_unbalanced(graph=graph, num=2)

    # graph = nx.Graph()
    # with open("./data/data-cn.in") as f:
    #     for line in f.readlines():
    #         tmp_input = line.strip('\n').split(' ')
    #         graph.add_node(int(tmp_input[0]), value=[])
    #         graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
    #         flag = True
    #         for num in range(2, len(tmp_input)):
    #             if flag:
    #                 graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    #                 flag = False
    #             else:
    #                 graph.edge[int(tmp_input[0])][int(tmp_input[num - 1])]['weight'] = float(tmp_input[num])
    #                 flag = True
    # # nx.draw(graph, with_labels=True, weight='weight')
    # # plt.show()
    # print(graph.nodes(data=True))
    # print(graph.edges(data=True))
    # ftc_cai_no_delay(graph=graph)
