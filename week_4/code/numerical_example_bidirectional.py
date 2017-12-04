# numerical_example_bidirectional.py
# Behaviors of networks with antagonistic interactions and switching topologies

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

ebunch = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

global graph
global A4
global A5
global current_graph
global flag


def get_graph(to_graph):
    global graph
    global current_graph
    graph.remove_edges_from(ebunch)
    if to_graph == 4:
        graph.add_edge(0, 2)
        graph.add_edge(2, 0)
        current_graph = 4
    else:
        graph.add_edge(1, 2)
        graph.add_edge(2, 1)
        current_graph = 5
    return graph


def initialize():

    # init node values
    node_values = np.array([-1.5, 1.0, 0.])

    global graph
    graph = nx.DiGraph()

    # three status of graph
    global A4
    A4 = np.matrix([
        [0.5, 0., -0.5],
        [0., 1.0, 0.],
        [-0.5, 0., -0.5]
    ])
    global A5
    A5 = np.matrix([
        [1.0, 0., 0.],
        [0., 0.5, 0.5],
        [0., 0.5, 0.5]
    ])

    # initial graph
    for i in range(len(A4)):
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
    global graph
    global current_graph
    global flag
    flag = False
    for time_step in range(1, 50000):
        # change topology every step
        if is_square(time_step):
            graph = get_graph(to_graph=5)
            flag = True
        elif flag:
        	flag = False
        	graph = get_graph(to_graph=5)
        else:
            graph = get_graph(to_graph=4)
        for i in graph.nodes():
            neighbors = get_neighbors(i)
            # neighbors = graph.neighbors(i)
            cur_value_i = graph.node[i]['values'][time_step - 1]
            result = 0
            for j in neighbors:
                cur_value_j = graph.node[j]['values'][time_step - 1]
                if current_graph == 4:
                    R = 1 if A4[i - 1, j - 1] > 0 else -1
                else:
                    R = 1 if A5[i - 1, j - 1] > 0 else -1
                result = result + math.sin(cur_value_i - R * cur_value_j)
            graph.node[i]['values'].append(cur_value_i - u * result)
        
    # for i in graph.nodes():
    #     print(graph.node[i]['values'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(50000)
    for i in graph.nodes():
        plt.plot(x_axis, graph.node[i]['values'], label='node ' + str(i))
    # plt.legend(loc='bottom right', bbox_to_anchor=(0.1, 1.05), ncol=5)
    plt.legend(loc='best', ncol=5)
    plt.savefig("./pngs/dynamics_bidirectional.png")
    plt.show()


if __name__ == '__main__':
    simulate()
