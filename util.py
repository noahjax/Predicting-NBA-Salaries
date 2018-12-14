import numpy as np
import pandas as pd

#Loads data for DT's
def loadTreeData():
    #Train data
    train_x = pd.read_csv('Data/train_x.csv').values
    train_x = train_x[:,1:]
    train_y = pd.read_csv('Data/train_y.csv', index_col=0).values
    #Dev data
    dev_x = pd.read_csv('Data/dev_x.csv').values
    dev_x = dev_x[:,1:]
    dev_y = pd.read_csv('Data/dev_y.csv', index_col=0).values
    #Test data
    test_x = pd.read_csv('Data/test_x.csv')
    test_x = test_x.values[:,1:]      
    test_y = pd.read_csv('Data/test_y.csv', index_col=0).values

    return train_x, train_y, dev_x, dev_y, test_x, test_y

def loadMinYearScale():
    data = pd.read_csv('Data/min_year_scale_factor.csv', header=None).values
    min_year = data[0][0]
    scale_factor = data[1][0]
    return min_year, scale_factor

#Loads data for linreg (adds intercept and max-normalizes)
def loadLinRegData(pad=True):
    #Train data
    train_x = pd.read_csv('Data/train_x.csv').values
    train_x, x_max = max_normalize(train_x)
    if pad: train_x[:,0] = 1
    else: train_x = train_x[:,1:]
    train_y = pd.read_csv('Data/train_y.csv', index_col=0).values
    train_y, y_max = max_normalize(train_y) 
    #Dev data
    dev_x = pd.read_csv('Data/dev_x.csv').values
    dev_x,_ = max_normalize(dev_x, x_max)
    if pad: dev_x[:,0] = 1
    else: dev_x = dev_x[:,1:]
    dev_y = pd.read_csv('Data/dev_y.csv', index_col=0).values
    dev_y,_ = max_normalize(dev_y, y_max)
    #Test data
    test_x = pd.read_csv('Data/test_x.csv').values
    test_x,_= max_normalize(test_x, x_max)
    if pad: test_x[:,0] = 1
    else: test_x = test_x[:,1:]
    test_y = pd.read_csv('Data/test_y.csv', index_col=0).values
    test_y,_ = max_normalize(test_y, y_max)

    return train_x, train_y, dev_x, dev_y, test_x, test_y, x_max, y_max

#Helper to normalize a data frame
def max_normalize(data, data_max=None):
    if np.any(data_max) == None: data_max = np.max(data, axis=0)
    data = data/data_max
    return data, data_max

def inflationScale(years, data, scale_factor, min_year):
    scaled = np.zeros(data.shape)
    for i, row in enumerate(data):
        diff = years[i] - min_year
        scaled[i] = data[i] / (scale_factor**diff)
    return scaled 

def inflationUnscale(years, data, scale_factor, min_year):
    scaled = np.zeros(data.shape)
    for i, row in enumerate(data):
        diff = years[i] - min_year
        scaled[i] = data[i] * (scale_factor**diff)
    return scaled 

#Evaluate MSE
def findMSE(preds, y):
    err = preds.reshape(y.shape) - y
    return np.mean(err**2)

#Evaluate RMSE
def findRMSE(preds, y):
    return np.sqrt(findMSE(preds, y))
