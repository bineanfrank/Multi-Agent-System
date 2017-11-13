# Finite-Time Consensus for Multiagent Systems With Cooperative and Antagonistic Interactions

import matplotlib.pyplot as plt
import networkx as nx
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
    for time_step in range(7):
        print("start")
        for x in range(1, 5):
            print(graph.node[x]['value'])
        print("end")

        for i in range(1, len(graph.nodes()) + 1):
            neighbors = graph.neighbors(i)
            print("neighbors: %s" % neighbors)
            sum = 0.0
            current_node_value = graph.node[i]['value'][time_step]
            for j in neighbors:
                neighbor_value = graph.node[j]['value'][time_step]
                sum += (matrix[i - 1][j - 1] * (neighbor_value - sign(matrix[i - 1][j - 1]) * current_node_value))
            print("sum = %s" % str(sum))
            final_sum = sign(sum) * (abs(sum) ** a)
            graph.node[i]['value'].append(final_sum)

    for i in graph.nodes():
        print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(8)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig('./pngs/Finite-Time-Consensus.png')
    plt.show()


if __name__ == '__main__':

    graph = nx.Graph()

    with open("./data/data-balanced.in") as f:
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
    ftc_cai(graph=graph)
