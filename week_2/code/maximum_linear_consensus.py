import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def maximum_linear_consensus(graph):
    sigma = 0.1
    a = 0.0
    k = 0

    adj_mat = nx.adjacency_matrix(graph)
    matrix = np.array(adj_mat.todense(), dtype=np.float)
    # add zero columns and rows to the first column and row.
    matrix = np.row_stack(([0 for _ in range(matrix.shape[0])], matrix))
    matrix = np.column_stack(([0 for _ in range(matrix.shape[0])], matrix))

    print(matrix)
    for time_step in range(10):
        for i in range(1, len(graph.nodes()) + 1):
            neighbors = graph.neighbors(i)

            # find max x_k
            max_x_k = graph.node[neighbors[0]]['value'][time_step]
            k = neighbors[0]
            for j in neighbors[1:]:
                if graph.node[j]['value'][time_step] > max_x_k:
                    max_x_k = graph.node[j]['value'][time_step]
                    k = j

            # determine a
            if graph.node[i]['value'][time_step] <= max_x_k:
                a = 1.0
            else:
                a = 0.0

            # determine part one
            part_one = (1 - sigma) * (max_x_k - graph.node[i]['value'][time_step])

            # determine part two
            part_two = 0.0
            for j in neighbors:
                if j != k:
                    part_two += (matrix[i][j] * (graph.node[j]['value'][time_step] - graph.node[i]['value'][time_step]))
            part_two = sigma * part_two

            # result
            result = graph.node[i]['value'][time_step] + a * (part_one + part_two)
            graph.node[i]['value'].append(result)

    for i in graph.nodes():
        print(graph.node[i]['value'])

    plt.xlabel("time-step")
    plt.ylabel("values")
    x_axis = range(11)
    for i in graph.nodes():
        print(graph.node[i]['value'])
        plt.plot(x_axis, graph.node[i]['value'])
    plt.savefig('./pngs/maximum_linear_consensus_1.png')
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
    print(graph.nodes(data=True))
    maximum_linear_consensus(graph=graph)
