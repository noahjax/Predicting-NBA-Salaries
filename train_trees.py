import numpy as np 
import pandas as pd
import util
from decision_tree import DecisionTree

train_x, train_y, dev_x, dev_y, test_x, test_y,  = util.loadTreeData()

'''Play with DT hyperparams'''
max_depths = [5, 10, 15, 20, 25]
min_leaf_sizes = [5, 10, 20, 50, 100]

results = []

for leaf_size in min_leaf_sizes:
    for depth in max_depths:
        cur_result = []
        cur_result.append(leaf_size)
        cur_result.append(depth)

        print('Fitting Tree with leaf size of', leaf_size, 'and depth', depth)
        DT = DecisionTree(min_leaf_size=leaf_size, max_depth=depth)
        DT.fit(train_x, train_y)

        preds = DT.predict(train_x)
        rmse = util.findRMSE(preds, train_y)
        cur_result.append(rmse)

        preds = DT.predict(dev_x)
        rmse = util.findRMSE(preds, dev_y)
        cur_result.append(rmse)
        results.append(cur_result)

results.sort(key=lambda x: x[3])
for result in results:
    print('Leaf Size: \t', result[0], 'Depth: \t', result[1], 'Train RMSE: \t', result[2], 'Dev RMSE: \t', result[3])

np.savetxt('Output/tree_dev_rmse.csv', results, delimiter=',')

'''End DT'''

def predictAndEval(DT, x, y, scale=False, scale_factor=None, min_year=None):
    preds = DT.predict(x)
    if scale: 
        scaled_preds = util.inflationUnscale(x[:,1], preds, scale_factor, min_year)
        scaled_y = util.inflationUnscale(x[:,1], y, scale_factor, min_year)
    else:
        scaled_preds = preds
        scaled_y = y
    rmse = util.findRMSE(scaled_preds, scaled_y)
    return rmse
