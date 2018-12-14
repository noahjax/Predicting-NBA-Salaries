import numpy as np 
import pandas as pd
import util

#Basic 2 layer NN
    # 5 neurons on first layer, 1 on second
class NeuralNet():
    
    def __init__(self, eta=10**-3, layer_sizes=[11,5,1], eps=10**-8, max_iters=10**6, batch_size=10):
        self.num_layers = len(layer_sizes)
        self.batch_size = batch_size
        self.eta = eta
        self.eps = eps
        self.max_iters = max_iters
        self.layer_sizes = layer_sizes
        self.layers = []
        self.intercepts = []

    #Wrapper class that points to the current activation func
    def activation(self, x):
        return self.leakyRelu(x)

    def dActivation(self, x):
        return self.dleakyRelu(x)

    def relu(self, x):
        return x * (x > 0)

    def drelu(self, x):
        return 1. * (x > 0)

    def leakyRelu(self, x, alpha=.01):
        return np.where(x > 0, x, x * alpha)

    def dleakyRelu(self, x, alpha=.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    def initWeights(self, m, n):
        for layer in range(self.num_layers):
            if layer == 0:
                layer_in = n
            else:
                layer_in = self.layer_sizes[layer-1]
            layer_out = self.layer_sizes[layer]
            weights = np.random.randn(layer_in, layer_out) * np.sqrt(2/layer_in)
            self.layers.append(weights)
            self.intercepts.append(np.random.randn(self.layer_sizes[layer]))

    def computePreds(self, x):
        z_vals = []
        a_vals = []
        #Compute predicted values
        cur_a = x
        for layer in range(self.num_layers):
            cur_weights = self.layers[layer]
            cur_z = cur_a.dot(cur_weights) + self.intercepts[layer].reshape(self.layer_sizes[layer])
            z_vals.append(cur_z)
            if layer < self.num_layers-1:
                cur_a = self.activation(cur_z)
            else: 
                cur_a = cur_z
            a_vals.append(cur_a)
        return cur_a, z_vals, a_vals

    #Update the weights for each layer
    def updateWeights(self, x, y, y_hat, z_vals, a_vals):
        m,n = x.shape
        #Init derivatives
        dl_dz = (y_hat.reshape((m,1))-y.reshape((m,1)))/m
        for layer in range(self.num_layers-1, -1, -1):
            #Update layer weights and intercepts
            cur_weights = self.layers[layer]
            cur_intercepts = self.intercepts[layer]
            cur_a = a_vals[layer-1] if layer >= 1 else x
            cur_weights -= self.eta * cur_a.T.dot(dl_dz)
            cur_intercepts -= self.eta * np.mean(dl_dz)
            #Update dl_dz for the next layer
            if layer > 0:
                dl_dz = dl_dz.dot(cur_weights.T) * self.dActivation(z_vals[layer-1])

    def fit(self, train_x, train_y, y_max=1, test_x=None, test_y=None):
        m,n = train_x.shape
        #Initialize weights
        self.initWeights(m,n)

        #Update until convergence        
        it = 0
        prev_rmse, rmse = None, None
        # while it < self.max_iters and (prev_mse == None or prev_mse - mse > self.eps):
        while it < self.max_iters:
            #Mini-batch
            for i in range(0,m,self.batch_size):
                if i+self.batch_size < m: 
                    cur_x = train_x[i:i+self.batch_size,:]
                    cur_y = train_y[i:i+self.batch_size]
                else:
                    cur_x = train_x[i:,:]
                    cur_y = train_y[i:]
                #Compute predicted values
                y_hat, z_vals, a_vals = self.computePreds(cur_x)
                
                #Update weights
                self.updateWeights(cur_x, cur_y, y_hat, z_vals, a_vals)

            if it % 100 == 0:
                y_hat, z_vals, a_vals = self.computePreds(train_x)
                self.updateWeights(train_x, train_y, y_hat, z_vals, a_vals)
                prev_rmse = rmse
                rmse = util.findRMSE(y_hat, train_y)
                #Print some stuff to test how things are going
                if it % 1000 == 0: 
                    if np.any(test_y != None):
                        preds = self.predict(test_x)
                        test_rmse = util.findRMSE(preds, test_y)*y_max
                        print(it, rmse*y_max, test_rmse)
                    else: print(it, rmse*y_max)
                #Check for convergence
                if prev_rmse != None and prev_rmse - rmse < self.eps: 
                    if self.eta > 10**-5: 
                        self.eta *= .1 
                        print('Eta drop')
                    else:   
                        print('Converged at iteration', it, 'with rmse', rmse*y_max)
                        return
            it += 1
        
        print('Max iterations reached, rmse is', rmse*y_max)

    def predict(self, x):
        return self.computePreds(x)[0]

