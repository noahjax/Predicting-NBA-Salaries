import numpy as np 
import pandas as pd
import util
from linreg import LinearRegression
from decision_tree import DecisionTree
from random_forest import RandomForest
from neural_net import NeuralNet

'''Run Linreg, Decision Tree, Random Forest, and Neural Net to compare performance'''

#Predict Train Average
def testWorstCase():
    train_x, train_y, dev_x, dev_y, test_x, test_y = util.loadTreeData()
    ave = np.full(len(train_y), np.mean(train_y))
    train_worst_rmse = util.findRMSE(ave, train_y)    
    ave = np.full(len(test_y), np.mean(train_y))
    test_worst_rmse = util.findRMSE(ave, test_y)
    print('Worst RMSE: \t', test_worst_rmse)
    return train_worst_rmse, test_worst_rmse

#Linear Regression
def testLinreg():
    train_x, train_y, dev_x, dev_y, test_x, test_y, x_max, y_max = util.loadLinRegData()
    LR = LinearRegression()
    LR.fit_stochastic(train_x, train_y, eta=.01, eps=10**-12, max_iters=10**8)
    preds = LR.predict(train_x)
    train_linreg_rmse = util.findRMSE(preds, train_y) * y_max[0]
    preds = LR.predict(test_x)
    test_linreg_rmse = util.findRMSE(preds, test_y) * y_max[0]
    print('Linreg RMSE: \t', test_linreg_rmse)
    return train_linreg_rmse, test_linreg_rmse

#Decision Tree
def testDT():
    train_x, train_y, dev_x, dev_y, test_x, test_y,  = util.loadTreeData()
    DT = DecisionTree(max_depth=10, min_leaf_size=50)
    DT.fit(train_x, train_y)
    preds = DT.predict(train_x)
    train_dt_rmse = util.findRMSE(preds, train_y)
    preds = DT.predict(test_x)
    test_dt_rmse = util.findRMSE(preds, test_y)
    print('DT RMSE: \t', test_dt_rmse)
    DT.printTree(DT.root)
    return train_dt_rmse, test_dt_rmse

#Random Forest
def testRF():
    train_x, train_y, dev_x, dev_y, test_x, test_y,  = util.loadTreeData()
    RF = RandomForest(min_leaf_size=50, max_depth=10, num_trees=10)
    RF.fit(train_x, train_y)
    preds = RF.predict(train_x)
    train_rf_rmse = util.findRMSE(preds, train_y)
    preds = RF.predict(test_x)
    test_rf_rmse = util.findRMSE(preds, test_y)
    print('RF RMSE: \t', test_rf_rmse)
    return train_rf_rmse, test_rf_rmse

#Neural Net
def testNN():
    train_x, train_y, dev_x, dev_y, test_x, test_y, x_max, y_max = util.loadLinRegData(pad=False)
    NN = NeuralNet(eps=10**-8, layer_sizes=[9,7,5,3,1])
    NN.fit(train_x, train_y, y_max[0], test_x, test_y)
    preds = NN.predict(train_x)
    train_nn_rmse = util.findRMSE(preds, train_y)*y_max[0]
    preds = NN.predict(test_x)
    test_nn_rmse = util.findRMSE(preds, test_y)*y_max[0]
    print('NN RMSE:', test_nn_rmse)
    return train_nn_rmse, test_nn_rmse

rf_rmse = testRF()
print(rf_rmse)
