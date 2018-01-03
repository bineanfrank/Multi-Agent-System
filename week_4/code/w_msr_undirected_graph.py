# w_msr_undirected_graph.py
# W-MSR.py
# A implementation of algorithms of paper "Resilient Asymptotic Consensus in Robust Networks"

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def get_neighbor_values(graph, neighbors, time_step, cur_value, F):
    """
    get neighbors that F larger or F smaller will be removed.
    :param graph: 
    :param neighbors: 
    :param time_step: 
    :param cur_value: 
    :param F: F-total
    :return: 
    """
    neighbor_values = []
    for i in neighbors:
        neighbor_values.append(graph.node[i]['value'][time_step])
    neighbor_values.sort()

    index_front = 0
    while index_front < F:
        if neighbor_values[index_front] < cur_value:
            index_front += 1
        else:
            break

    index = 0
    index_end = len(neighbor_values) - 1
    while index < F:
        if neighbor_values[index_end] > cur_value:
            index_end -= 1
            index += 1
        else:
            break

    return neighbor_values[index_front:index_end + 1]


def w_msr(graph):
    """
    time varying
    :param graph: (2, 2)-robust, 1-Total
    :return: 
    """
    for time_step in range(200):
        for i in range(1, len(graph.nodes()) + 1):  # 编号从1开始
            if i == 14:
                if time_step == 1:
                    graph.node[i]['value'].append(0)
                else:
                    graph.node[i]['value'].append(graph.node[i]['value'][time_step] + 0.01)
                continue
            neighbors = graph.neighbors(i)
            cur_value = graph.node[i]['value'][time_step]
            neighbor_values = get_neighbor_values(graph=graph, neighbors=neighbors, time_step=time_step,
                                                  cur_value=cur_value, F=1)
            weighted_sum = cur_value * (1.0 / (len(neighbor_values) + 1))
            for j in neighbor_values:
                t = 1.0 / (len(neighbor_values) + 1)
                weighted_sum += t * j
            graph.node[i]['value'].append(weighted_sum)

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(201)
    handle1 = 1
    handle2 = 2
    for i in range(1, 15):
        if i == 14:
            handle1, = plt.plot(x_axis, graph.node[i]['value'], 'g--')
        else:
            handle2, = plt.plot(x_axis, graph.node[i]['value'], 'r-')
    plt.legend(handles=[handle1, handle2], labels=['Malicious', 'Normal'], loc="best")
    plt.savefig('./pngs/Weighted-Mean-Subsequence-Reduced.png')
    plt.show()


if __name__ == '__main__':
    graph = nx.Graph()
    with open("./data/data.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            for num in range(2, len(tmp_input)):
                graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    w_msr(graph=graph)