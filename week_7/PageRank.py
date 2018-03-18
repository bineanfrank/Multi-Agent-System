# PageRank.py

import numpy as np
import matplotlib.pyplot as plt

P = np.mat(np.array([0.2, 0.2, 0.2, 0.2]).reshape(4, 1))
W = np.mat(np.array([
    [0, 0.5, 0, 0.5],
    [0.333, 0, 0, 0.5],
    [0.333, 0.5, 0, 0],
    [0.333, 0, 1.0, 0]
]).reshape(4, 4))

Ps = [[1/4], [1/4], [1/4], [1/4]]

for i in range(50):
    # P = W * P
    P = (1 - 0.2) * W * P + np.ones(4).reshape(4, 1) * 0.2 / 4
    print(P)
    for i in range(4):
    	Ps[i].append(P[i, 0])

plt.xlabel("time-step")
plt.ylabel("values")
x_axis = range(51)
for i in [1, 2, 3, 4]:
    plt.plot(x_axis, Ps[i - 1])
plt.savefig("./pngs/pagerank.png")
plt.show()
