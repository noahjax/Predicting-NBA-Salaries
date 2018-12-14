import numpy as np 
import pandas as pd
import util

#Class that runs linear regression
class LinearRegression():

    #Initialize class
    def __init__(self, theta_0=None, use_theta_0=False):
        self.theta = theta_0
        self.use_theta_0 = use_theta_0

    #Fit using mini-batch descent
    def fit(self, train_x, train_y, eta=10**-3, eps=10**-11, max_iters=10**8, batch_size=100):
        m,n = train_x.shape
        if batch_size > m: batch_size = m
        if not self.use_theta_0:
            self.theta = np.zeros(n)

        i = 0
        loc = 0
        while True:
            #Define mini-batch to consider
            if loc + batch_size > m:
                cur_xs = np.append(train_x[loc:,:], train_x[:(loc+batch_size)%m,:], axis=0)
                cur_ys = np.append(train_y[loc:], train_y[:(loc+batch_size)%m])
            else:
                cur_xs = train_x[loc:loc+batch_size]
                cur_ys = train_y[loc:loc+batch_size]
            loc = (loc + batch_size) % m

            #Find miss and direction to update towards
            preds = np.dot(cur_xs, self.theta)
            misses = cur_ys[:,0] - preds

            #Update theta
            new_theta = self.theta + eta * (1/batch_size) * np.dot(cur_xs.T, misses)
            if np.linalg.norm(self.theta-new_theta) < eps or i > max_iters:
                # print(i, np.linalg.norm(self.theta-new_theta))
                self.theta = new_theta[:]
                break

            i += 1
            if i % 100 == 0:
                # print('magnitude theta', np.linalg.norm(new_theta))
                # print(i, np.linalg.norm(self.theta-new_theta))
                pass

            self.theta = new_theta[:]

    #Fit model with stochastic gradient descent 
    def fit_stochastic(self, train_x, train_y, eta=10**-3, eps=10**-11, max_iters=10**7):
        m,n = train_x.shape
        if not self.use_theta_0:
            self.theta = np.zeros(n)

        #Run stochastic gradient descent
        i = 0
        while True:
            cur_x = train_x[i % m,:]
            cur_y = train_y[i % m]
            pred = self.theta.dot(cur_x)
            miss = cur_y - pred

            #Update theta, see if you have converged 
            new_theta = self.theta + eta * miss * cur_x
            conv = np.linalg.norm(self.theta-new_theta)
            if conv < eps or i > max_iters:
                # print(i, conv)
                self.theta = new_theta[:]
                break
            
            i += 1
                
            self.theta = new_theta[:]

    #Makes predictions on some test input
    def predict(self, test_x):
        m,n = test_x.shape
        preds = np.zeros(m)

        for i, cur_x in enumerate(test_x):
            preds[i] = self.theta.dot(cur_x)

        return preds
        
    
 


