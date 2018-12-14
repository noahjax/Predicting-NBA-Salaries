import numpy as np 
import pandas as pd
import util
from random_forest import RandomForest

train_x, train_y, dev_x, dev_y, test_x, test_y = util.loadTreeData()

'''Random forest hyperparams'''
#Leaf size 10, depth 15 seams suitable for RF
# Play with tree params
num_trees = [1, 5, 10, 25, 50, 100, 200]
min_leaf_sizes = [20]
depths = [20]

results = []

for tree_num in num_trees:
    for leaf_size in min_leaf_sizes:
        for depth in depths:
            print('Fitting Forest with leaf size of', leaf_size, 'depth', depth, 'and', tree_num, 'trees')
            cur_result = []
            cur_result.append(leaf_size)
            cur_result.append(depth)
            cur_result.append(tree_num)

            RF = RandomForest(num_trees=tree_num, min_leaf_size=leaf_size, max_depth=depth) 
            RF.fit(train_x, train_y)

            preds = RF.predict(train_x)
            rmse = util.findRMSE(preds, train_y)
            cur_result.append(rmse)

            preds = RF.predict(test_x)
            rmse = util.findRMSE(preds, test_y)
            cur_result.append(rmse)
            results.append(cur_result)

results.sort(key=lambda x: x[4])
for result in results:
    print('Leaf Size: \t', result[0], 'Depth: \t', result[1], 'Num Trees: \t', result[2], 'Train RMSE: \t', result[3], 'Test RMSE: \t', result[4])

np.savetxt('Output/forest_test_rmse.csv', results, delimiter=',')
