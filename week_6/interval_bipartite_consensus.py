# interval_bipartite_consensus.py
# Interval Bipartite Consensus of Networked Agents Associated With Signed
# Digraphs
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# for fix topology
global graph
global X
global y


def sign(num):
    if num < 0:
        return -1.0
    elif num > 0:
        return 1.0
    else:
        return 0.0


def get_in_neighbors(node, in_edge):
    in_neighbors = []
    for edge in in_edge:
        in_neighbors.append(edge[1] if edge[0] == node else edge[0])
    return in_neighbors

# 获取一个图的邻接矩阵
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


def interval_bipartite_consensus(graph_no):
    '''init'''
    global graph
    global D
    global X
    global y

    edges1 = [(1, 3, 5.0), (2, 1, -3.0), (2, 5, 4.0), (2, 4, 2.0), (3, 2, -1.0),
              (3, 6, 6.0), (4, 5, 8.0), (5, 6, -7.0)]
    edges2 = [(1, 3, 5.0), (2, 1, -3.0), (2, 5, -4.0), (2, 4, 2.0), (3, 2, 1.0),
              (3, 6, -6.0), (4, 5, 8.0), (5, 6, 7.0)]
    edges3 = [(2, 1, -3.0), (1, 3, 5.0), (1, 2, -1.0), (2, 4, 2.0),
              (3, 4, -9.0), (3, 5, 4.0), (3, 6, 6.0), (4, 5, 8.0), (5, 6, 7.0)]
    edges4 = [(2, 1, -3.0), (1, 3, 5.0), (1, 2, -1.0), (2, 4, 2.0),
              (3, 4, 9.0), (3, 5, 4.0), (4, 5, 8.0), (5, 6, -7.0), (6, 3, 6.0)]

    graph = nx.DiGraph()

    for i in range(1, 7):
        graph.add_node(i)

    if graph_no == 1:
        graph.add_weighted_edges_from(edges1)
    elif graph_no == 2:
        graph.add_weighted_edges_from(edges2)
    elif graph_no == 3:
        graph.add_weighted_edges_from(edges3)
    else:
        graph.add_weighted_edges_from(edges4)

    X = [[1], [2], [3], [-2], [-1], [-6]]
    y = 0.02

    adj_graph_1 = get_adj_mat(graph)
    diagnal_matrix = []
    for i in range(len(adj_graph_1)):
        sum = 0
        for j in range(len(adj_graph_1)):
            sum += abs(adj_graph_1[i][j])
        diagnal_matrix.append(sum)
    diagnal_matrix = np.mat(np.diag(diagnal_matrix))

    L = diagnal_matrix - adj_graph_1
    print("Linear")
    print(np.linalg.eig(L))
    print("Linear")

    '''start interaction'''
    for time_step in range(200):
        for i in range(len(X)):
            current_value = X[i][time_step]
            in_edges = graph.in_edges(i + 1)
            in_neighbors = get_in_neighbors(i + 1, in_edges)
            delta = 0
            for neighbor in in_neighbors:
                delta += graph[neighbor][i + 1]['weight'] * (X[neighbor - 1][time_step] - sign(
                    graph[neighbor][i + 1]['weight']) * X[i][time_step])
            next_value = X[i][time_step] + y * delta
            X[i].append(next_value)
    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(201)

    tmp = np.array(X)
    print(tmp[:, -1])
    for i in graph.nodes():
        plt.plot(x_axis, X[i - 1], label='node ' + str(i))
    plt.legend(loc='best', ncol=1)
    plt.savefig("./pngs/interval_bipartite_consensus_" +
                str(graph_no) + ".png")
    plt.show()


if __name__ == '__main__':
        # for i in range(1, 5):
        # 	interval_bipartite_consensus(graph_no=i
    interval_bipartite_consensus(graph_no=1)
