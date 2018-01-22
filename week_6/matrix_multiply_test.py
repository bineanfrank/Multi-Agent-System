# matrix multiply tests
import numpy as np
import random
import matplotlib.pyplot as plt

L = np.mat(np.array([1, -1, 0, 0, 1.5, -1.5, -2, 0, 2]).reshape(3, 3))
X = np.mat(np.array([4, 2, 1]).reshape(3, 1))

X1 = [[4], [2], [1]]

for i in range(30):
    X = X + 0.1 * (-L) * X
    for i in range(3):
        X1[i].append(X[i])
x_axis = range(31)
plt.xlabel("time-step")
plt.ylabel("values")
for i in range(3):
    plt.plot(x_axis, X1[i - 1])
plt.savefig("./pngs/matrix_multiply_test.png")
plt.show()