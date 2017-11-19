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


def ftc_cai(graph):
    a = 0.6
    adj_mat = nx.adjacency_matrix(graph)
    matrix = np.array(adj_mat.todense(), dtype=np.float)
    print(matrix)
    for time_step in range(1000):
        # print("start")
        # for x in range(1, 5):
        #     print(graph.node[x]['value'])
        # print("end")

        for i in range(1, len(graph.nodes()) + 1):
            neighbors = graph.neighbors(i)
            # print("neighbors: %s" % neighbors)
            sum = 0.0
            for j in neighbors:
                # print("i = %d, j = %d, matrix[i][j]:%d" % (i, j, matrix[i][j]))
                # print("current_node_value = %d" % graph.node[i]['value'][time_step])
                # print("neighbor_value = %d" % graph.node[j]['value'][time_step])
                sum += (matrix[i - 1][j - 1] * (
                    graph.node[j]['value'][time_step] - sign(matrix[i - 1][j - 1]) * graph.node[i]['value'][time_step]))

            # print("sum = %s" % str(sum))
            print("abs sum = %s" % str(abs(sum)))
            print("a = %f" % a)
            print("abs(sum) ** a = %f" % (abs(sum) ** a))
            print("math.pow(abs(sum), a) = %f" % (math.pow(abs(sum), a)))
            final_result = graph.node[i]['value'][time_step] + sign(sum) * (abs(sum) ** a)
            # final_result = -sum
            # print("final_sum = %s" % str(final_result))
            graph.node[i]['value'].append(final_result)

    for i in graph.nodes():
        print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(1001)
    for i in graph.nodes():
        print(graph.node[i]['value'])
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig('./pngs/Finite-Time-Consensus.png')
    plt.show()


def ftc_cai_no_delay(graph):
    adj_mat = nx.adjacency_matrix(graph)
    matrix = np.array(adj_mat.todense(), dtype=np.float)
    # add zero columns and rows to the first column and row.
    print(matrix)
    for time_step in range(10):
        for i in range(1, len(graph.nodes()) + 1):
            neighbors = graph.neighbors(i)
            sum = 0.0
            for j in neighbors:
                sum += abs(matrix[i - 1][j - 1]) * (
                    graph.node[i]['value'][time_step] - sign(matrix[i - 1][j - 1]) * graph.node[j]['value'][time_step])

            final_result = graph.node[i]['value'][time_step] - sum
            graph.node[i]['value'].append(final_result)

    for i in graph.nodes():
        print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(11)
    for i in graph.nodes():
        print(graph.node[i]['value'])
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig('./pngs/Finite-Time-Consensus-No-Delay.png')
    plt.show()


if __name__ == '__main__':

    graph = nx.Graph()
    # with open("./data/data-balanced.in") as f:
    #     for line in f.readlines():
    #         tmp_input = line.strip('\n').split(' ')
    #         graph.add_node(int(tmp_input[0]), value=[])
    #         graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
    #         flag = True
    #         for num in range(2, len(tmp_input)):
    #             if flag:
    #                 graph.add_edge(int(tmp_input[0]), int(tmp_input[num]), weight=0.0)
    #                 flag = False
    #             else:
    #                 graph.edge[int(tmp_input[0])][int(tmp_input[num - 1])]['weight'] = float(tmp_input[num])
    #                 flag = True
    # ftc_cai(graph=graph)

    with open("./data/data-cn.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            flag = True
            for num in range(2, len(tmp_input)):
                if flag:
                    graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
                    flag = False
                else:
                    graph.edge[int(tmp_input[0])][int(tmp_input[num - 1])]['weight'] = float(tmp_input[num])
                    flag = True
    ftc_cai_no_delay(graph=graph)
