import numpy as np 
import pandas as pd
import util
from linreg import LinearRegression

train_x, train_y, dev_x, dev_y, test_x, test_y, x_max, y_max = util.loadLinRegData()

#Play with different eps values
eps_vals = [10**-11, 10**-12, 10**-13]
eta_vals = [10**-4, 10**-3, 10**-2]
max_iters = [10**8, 10**7, 10**6]

results = []

for eps_val in eps_vals:
    for eta_val in eta_vals:
        for max_iter in max_iters:
                print('Fitting regression with eta', eta_val, 'eps', eps_val, 'max iter', max_iter)
                cur_result = []
                cur_result.append(eps_val)
                cur_result.append(eta_val)
                cur_result.append(max_iter)
                
                LR = LinearRegression()
                LR.fit_stochastic(train_x, train_y, eta=eta_val, eps=eps_val, max_iters=max_iter)

                preds = LR.predict(train_x)
                rmse = util.findRMSE(preds, train_y)
                cur_result.append(rmse*y_max[0])
                
                preds = LR.predict(dev_x)
                rmse = util.findRMSE(preds, dev_y)
                cur_result.append(rmse*y_max[0])
                results.append(cur_result)

results.sort(key=lambda x: x[4])
for result in results:
    print('eps:\t', result[0], 'eta:\t', result[1], 'max iter:\t', result[2], 'Train RMSE:\t', result[3], 'Test RMSE:\t', result[4])

np.savetxt('Output/linreg_dev_rmse.csv' , results, delimiter=',')
