import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

nums = np.array([-9, -8, 0, -1, 3])
sum = 0
for i in range(len(nums)):
	sum += (nums[i] * 1.0 / len(nums) * 1.0)
print(sum)