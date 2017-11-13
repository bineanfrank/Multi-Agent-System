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


def lcp(graph):
    """
    time invariant
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
    plt.legend(handles=[handle1, handle2], labels=['Malicious', 'Normal'], loc="best")
    plt.savefig('./pngs/Linear-Consensus-Algorithm.png')
    plt.show()


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
    for i in range(1, 9):
        if i == 14:
            handle1, = plt.plot(x_axis, graph.node[i]['value'], 'g--')
        else:
            handle2, = plt.plot(x_axis, graph.node[i]['value'], 'r-')
    plt.legend(handles=[handle1, handle2], labels=['Malicious', 'Normal'], loc="best")
    plt.savefig('./pngs/Weighted-Mean-Subsequence-Reduced.png')
    plt.show()


def bernoulli_remove_edges(graph):

    edges = graph.edges()

    bernoulli_flips = np.random.binomial(n=1, p=.5, size=len(edges))
    for index, flag in enumerate(bernoulli_flips):
        edge = edges[index]
        if flag == 1:
            graph.edge[edge[0]][edge[1]]['state'] = 0
        else:
            graph.edge[edge[0]][edge[1]]['state'] = 1


def lcp_time_varying(graph):
    """
    time-varying
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
            if time_step % 10 != 0:  # 如果是模十，需要进行伯努利模拟删除一般的边
                bernoulli_remove_edges(graph=graph)
                neighbors = [value for value in neighbors if graph.edge[i][value]['state'] == 1]

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
    plt.legend(handles=[handle1, handle2], labels=['Malicious', 'Normal'], loc="best")
    plt.savefig('./pngs/Linear-Consensus-Algorithm-Time-Varying.png')
    plt.show()


def w_msr_time_varying(graph):
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

            if time_step % 10 != 0:  # 如果是模十，需要进行伯努利模拟删除一般的边
                bernoulli_remove_edges(graph=graph)
                neighbors = [value for value in neighbors if graph.edge[i][value]['state'] == 1]

            cur_value = graph.node[i]['value'][time_step]

            if len(neighbors) > 0:
                neighbor_values = get_neighbor_values(graph=graph, neighbors=neighbors, time_step=time_step,
                                                      cur_value=cur_value, F=1)
            weighted_sum = cur_value * (1.0 / (len(neighbor_values) + 1))
            if len(neighbor_values) > 0:
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
    plt.savefig('./pngs/Weighted-Mean-Subsequence-Reduced-Time-Varying.png')
    plt.show()


def w_msr_prop_1(graph):
    """
    time varying
    :param graph: (2, 2)-robust, 1-Total
    :return: 
    """
    for time_step in range(20):
        for i in range(1, len(graph.nodes()) + 1):  # 编号从1开始
            neighbors = graph.neighbors(i)
            cur_value = graph.node[i]['value'][time_step]
            neighbor_values = get_neighbor_values(graph=graph, neighbors=neighbors, time_step=time_step,
                                                  cur_value=cur_value, F=2)
            weighted_sum = cur_value * (1.0 / (len(neighbor_values) + 1))
            for j in neighbor_values:
                t = 1.0 / (len(neighbor_values) + 1)
                weighted_sum += t * j
            graph.node[i]['value'].append(weighted_sum)

    plt.xlabel("time-step")
    plt.ylabel("values")
    plt.legend()
    x_axis = range(21)
    for i in range(1, 9):
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig('./pngs/Weighted-Mean-Subsequence-Reduced-Prop1.png')
    plt.show()


if __name__ == '__main__':
    # graph = nx.Graph()
    # with open("./data/data-balanced.in") as f:
    #     for line in f.readlines():
    #         tmp_input = line.strip('\n').split(' ')
    #         graph.add_node(int(tmp_input[0]), value=[])
    #         graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
    #         for num in range(2, len(tmp_input)):
    #             graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    # lcp(graph=graph)

    # graph = nx.Graph()
    # with open("./data/data-balanced.in") as f:
    #     for line in f.readlines():
    #         tmp_input = line.strip('\n').split(' ')
    #         graph.add_node(int(tmp_input[0]), value=[])
    #         graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
    #         for num in range(2, len(tmp_input)):
    #             graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    # lcp_by_guang(graph=graph)

    # graph = nx.Graph()
    # with open("./data/data_prop_1.in") as f:
    #     for line in f.readlines():
    #         tmp_input = line.strip('\n').split(' ')
    #         graph.add_node(int(tmp_input[0]), value=[])
    #         graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
    #         for num in range(2, len(tmp_input)):
    #             graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    #
    # nx.draw(graph, with_labels=True)
    # plt.savefig("./pngs/prop1.png")
    # plt.draw()
    # plt.show()
    # w_msr_prop_1(graph=graph)

    # graph = nx.Graph()
    # with open("./data/data-balanced.in") as f:
    #     for line in f.readlines():
    #         tmp_input = line.strip('\n').split(' ')
    #         graph.add_node(int(tmp_input[0]), value=[])
    #         graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
    #         for num in range(2, len(tmp_input)):
    #             graph.add_edge(int(tmp_input[0]), int(tmp_input[num]))
    # # w_msr_by_guang(graph=graph)
    #
    graph = nx.DiGraph()
    with open("./data/data.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            for num in range(2, len(tmp_input)):
                graph.add_edge(int(tmp_input[0]), int(tmp_input[num]), state=1)
    lcp_time_varying(graph=graph)

    graph = nx.DiGraph()
    with open("./data/data.in") as f:
        for line in f.readlines():
            tmp_input = line.strip('\n').split(' ')
            graph.add_node(int(tmp_input[0]), value=[])
            graph.node[int(tmp_input[0])]['value'].append(float(tmp_input[1]))
            for num in range(2, len(tmp_input)):
                graph.add_edge(int(tmp_input[0]), int(tmp_input[num]), state=1)
    w_msr_time_varying(graph=graph)
