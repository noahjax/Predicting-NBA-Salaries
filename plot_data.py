import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import util

train_x, train_y, dev_x, dev_y, test_x, test_y = util.loadTreeData()

n, bins, patches = plt.hist(train_y, 50, density=True)
plt.xlabel('Salary')
plt.ylabel('Count')
plt.show()

