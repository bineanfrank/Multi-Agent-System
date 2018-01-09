# Signed consensus problems on networks of agents with fixed and switching topologies
import matplotlib.pyplot as plt
import networkx as nx


def sign(num):
    if num < 0:
        return -1.0
    elif num > 0:
        return 1.0
    else:
        return 0.0


# for fix topology
global graph
global D
global X
global y

# for switching topology
global g1, g2, g3, g4
global y1, y2, y3, y4


def get_in_neighbors(node, in_edge):
    in_neighbors = []
    for edge in in_edge:
        in_neighbors.append(edge[1] if edge[0] == node else edge[0])
    return in_neighbors


def fix_topology():
    '''init'''
    global graph
    global D
    global X
    global y
    edges = [(1, 2), (1, 8), (3, 4), (4, 5), (6, 4), (7, 1), (7, 3), (7, 6), (8, 7)]
    graph = nx.DiGraph(edges)

    D = [1, -1, -1, -1, 1, 1, -1, -1]
    X = [[7], [-10], [-3], [2], [-8], [15], [10], [12]]
    y = 0.45

    '''start interation'''
    for time_step in range(20):
        for i in range(len(X)):
            current_value = X[i][time_step]
            in_edges = graph.in_edges(i + 1)
            in_neighbors = get_in_neighbors(i + 1, in_edges)
            delta = 0
            for neighbor in in_neighbors:
                delta += (D[neighbor - 1] * X[neighbor - 1][time_step] - D[i] * current_value)
            next_value = X[i][time_step] + y * delta * D[i]
            X[i].append(next_value)
    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(21)
    for i in graph.nodes():
        plt.plot(x_axis, X[i - 1])
    plt.savefig("./pngs/fixed_signed_consensus.png")
    plt.show()


def switching_topology():
    '''init'''
    global D
    global X
    global g1, g2, g3, g4
    global y1, y2, y3, y4

    D = [1, -1, -1, -1, 1, 1, -1, -1]
    X = [[7], [-10], [-3], [2], [-8], [15], [10], [12]]

    g1 = nx.DiGraph()
    g2 = nx.DiGraph()
    g3 = nx.DiGraph()
    g4 = nx.DiGraph()

    for i in range(1, 9):
        g1.add_node(i)
        g2.add_node(i)
        g3.add_node(i)
        g4.add_node(i)

    edges1 = [(1, 8), (4, 5)]
    edges2 = [(7, 1), (3, 4)]
    edges3 = [(8, 7), (7, 6), (7, 3)]
    edges4 = [(1, 2), (6, 4)]

    g1.add_edges_from(edges1)
    g2.add_edges_from(edges2)
    g3.add_edges_from(edges3)
    g4.add_edges_from(edges4)

    y1 = 0.35
    y2 = 0.9
    y3 = 0.7
    y4 = 0.8

    current_graph = g1
    current_y = y1
    flag = True
    '''start interation'''
    for time_step in range(100):
        if current_graph is g1:
            if flag:
                flag = False
            else:
                current_graph = g2
                current_y = y2
                flag = True
        elif current_graph is g2:
            if flag:
                flag = False
            else:
                current_graph = g3
                current_y = y3
                flag = True
        elif current_graph is g3:
            if flag:
                flag = False
            else:
                current_graph = g4
                current_y = y4
                flag = True
        else:
            if flag:
                flag = False
            else:
                current_graph = g1
                current_y = y1
                flag = True
        for i in range(len(X)):
            current_value = X[i][time_step]
            in_edges = current_graph.in_edges(i + 1)
            in_neighbors = get_in_neighbors(i + 1, in_edges)
            delta = 0
            for neighbor in in_neighbors:
                delta += (D[neighbor - 1] * X[neighbor - 1][time_step] - D[i] * current_value)

            next_value = X[i][time_step] + current_y * delta * D[i]
            X[i].append(next_value)
    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(101)
    for i in g1.nodes():
        plt.plot(x_axis, X[i - 1])
    plt.savefig("./pngs/switching_signed_consensus.png")
    plt.show()


if __name__ == '__main__':
    fix_topology()
    switching_topology()
