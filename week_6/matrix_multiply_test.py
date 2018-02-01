# matrix multiply tests
import numpy as np
import random
import matplotlib.pyplot as plt

# L = np.mat(np.array([1, -1, 0, 0, 1.5, -1.5, -2, 0, 2]).reshape(3, 3))
L = np.mat(np.array([1, 0.6, -0.4, 1, 1, 0, -0.2, 0.8, 1]).reshape(3, 3))
X = np.mat(np.array([0.6, 0.4, 0.2]).reshape(3, 1))

print(L)

a = np.round(np.linalg.eigvals(L))

print(a)

X1 = [[0.6], [0.4], [0.2]]

for i in range(300):
    X = X + 0.02 * (-L) * X
    for i in range(3):
        X1[i].append(X[i])

# print(X1)
x_axis = range(301)
plt.xlabel("time-step")
plt.ylabel("values")
for i in range(3):
    plt.plot(x_axis, X1[i - 1])
plt.savefig("./pngs/matrix_multiply_test.png")
plt.show()
