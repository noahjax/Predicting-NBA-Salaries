import numpy as np 
import pandas as pd
import util
from neural_net import NeuralNet

train_x, train_y, dev_x, dev_y, test_x, test_y, x_max, y_max = util.loadLinRegData()

'''Play with Neural Net hyperparams'''

layer_architectures = [[15,3,1], [19, 7, 3, 1],[9,7,5,3,1]]


results = []

for arch in layer_architectures:
    cur_result = []
    cur_result.append(arch)

    print("Fitting NN with architecture", arch)
    NN = NeuralNet(layer_sizes=arch)
    NN.fit(train_x, train_y, y_max[0])

    preds = NN.predict(train_x)
    rmse = util.findRMSE(preds, train_y)*y_max[0]
    cur_result.append(rmse)

    preds = NN.predict(dev_x)
    rmse = util.findRMSE(preds, dev_y)*y_max[0]
    cur_result.append(rmse)
    results.append(cur_result)
    print('Train rmse', cur_result[1], 'Dev rmse', cur_result[2])

results.sort(key=lambda x: x[2])
for result in results:
    print('Architecture: \t', result[0], 'Train RMSE: \t', result[1], 'Dev RMSE: \t', result[2])

np.savetxt('Output/nn_rmse_various_architectures.csv', results)