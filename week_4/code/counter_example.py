# numerical_example_bidirectional.py
# Behaviors of networks with antagonistic interactions and switching topologies

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

A = np.matrix([
    [1, 0, 0],
    [1 / 3, 1 / 3, 1 / 3],
    [-1 / 2, 0, 1 / 2]
])


def initialize():
    # init node values
    node_values = np.array([1, 0, -1])

    global graph
    graph = nx.Graph()

    # initial graph
    for i in range(len(A)):
        graph.add_node(i, values=[node_values[i]])


def get_neighbors(i):
    neighbors = []
    global graph
    for j in graph.nodes():
        e = (j, i)
        if graph.has_edge(*e):
            neighbors.append(j)
    return neighbors


def is_square(n):
    i = 1
    while n > 0:
        n -= i
        i += 2
    if n == 0:
        return True
    else:
        return False


def simulate():
    initialize()
    # value of u
    u = 0.01
    for time_step in range(1, 500):
        for i in graph.nodes():
            neighbors = get_neighbors(i)
            # neighbors = graph.neighbors(i)
            cur_value_i = graph.node[i]['values'][time_step - 1]
            result = 0
            for j in neighbors:
                cur_value_j = graph.node[j]['values'][time_step - 1]
                result = result + A[i - 1, j - 1] * cur_value_j
            graph.node[i]['values'].append(cur_value_i - u * result)

    # for i in graph.nodes():
    #     print(graph.node[i]['values'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(500)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['values'], label='node ' + str(i))
    # # plt.legend(loc='bottom right', bbox_to_anchor=(0.1, 1.05), ncol=5)
    # plt.legend(loc='best', ncol=5)
    plt.savefig("./pngs/counter_example.png")
    plt.show()


if __name__ == '__main__':
    simulate()
