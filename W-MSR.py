# W-MSR.py
# A implementation of algorithms of paper "Resilient Asymptotic Consensus in Robust Networks"

import networkx as nx
import matplotlib.pyplot as plt
import random


def lcp(graph):
    for time_step in range(200):
        for i in range(1, len(graph.nodes()) + 1):  # 编号从1开始
            if i == 14:
                if time_step == 1:
                    graph.node[i]['value'].append(0)
                else:
                    graph.node[i]['value'].append(graph.node[i]['value'][time_step] + 0.01)
                continue
            neighbors = graph.neighbors(i)
            weighted_sum = graph.node[i]['value'][time_step] * (1.0 / (len(neighbors) + 1))
            for j in neighbors:
                t = 1.0 / (len(neighbors) + 1)
                weighted_sum += t * graph.node[j]['value'][time_step]
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
    plt.legend(handles=[handle1, handle2], labels=['Malicious', 'Normal'], loc="bottom right")
    plt.savefig('Linear-Consensus-Algorithm.png')
    plt.show()


def lcp_by_guang(graph):
    for time_step in range(200):
        for i in range(1, len(graph.nodes()) + 1):  # 编号从1开始
            if i == 14:
                graph.node[i]['value'].append(graph.node[i]['value'][0])
                continue
            neighbors = graph.neighbors(i)
            weight_sum = 0.0
            neighbors_sum = 0.0
            cur_value = graph.node[i]['value'][time_step]
            for j in neighbors:
                t = random.uniform(1 - (8.0 / 9.0), (8.0 / 9.0) / len(neighbors))
                neighbors_sum += t * graph.node[j]['value'][time_step]
                weight_sum += t
            graph.node[i]['value'].append(cur_value * (-weight_sum) + neighbors_sum + cur_value)

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(201)
    for i in range(1, 15):
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig('Linear-Consensus-Algorithm-By-Guang.png')
    plt.show()


def get_neighbor_values(graph, neighbors, time_step, cur_value, F):
    neighbor_values = []
    for i in neighbors:
        neighbor_values.append(graph.node[i]['value'][time_step])
    neighbor_values.sort()

    print("cur_value = %s" % cur_value)

    print("Before:")
    print(neighbor_values)

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
    plt.savefig('Weighted-Mean-Subsequence-Reduced.png')
    plt.show()


if __name__ == '__main__':
    graph = nx.Graph()
    with open("data.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            for num in range(2, len(tmp_input)):
                graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    lcp(graph=graph)
    #
    # graph = nx.Graph()
    # with open("data.in") as f:
    #     for line in f.readlines():
    #         tmp_input = line.strip('\n').split(' ')
    #         graph.add_node(int(tmp_input[0]), value=[])
    #         graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
    #         for num in range(2, len(tmp_input)):
    #             graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    # lcp_by_guang(graph=graph)

    graph = nx.Graph()
    with open("data.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            for num in range(2, len(tmp_input)):
                graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    w_msr(graph=graph)
