import numpy as np 
import pandas as pd
import util
from decision_tree import DecisionTree

class RandomForest():

    def __init__(self, num_trees=100, max_depth=10, min_leaf_size=10, feature_factor=3):
        self.trees = []
        self.num_trees = num_trees
        self.feature_factor = feature_factor
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

    def buildTrees(self, train_x, train_y):
        m,n = train_x.shape
        feature_vec = np.array([i for i in range(n)])
        num_features = int(n/self.feature_factor)
        
        for i in range(self.num_trees):
            sample_indices = np.random.randint(0, m, size=m)
            cur_x = train_x[sample_indices]
            cur_y = train_y[sample_indices]
            np.random.shuffle(feature_vec)
            cur_features = feature_vec[:num_features]

            curDT = DecisionTree()
            curDT.fit(cur_x, cur_y, True, cur_features)
            self.trees.append(curDT)
            print('Done with tree', i)

    def fit(self, train_x, train_y):
        self.buildTrees(train_x, train_y)

    def findPred(self, x):
        tree_preds = np.zeros(self.num_trees)
        for i, tree in enumerate(self.trees):
            tree_preds[i] = tree.findPred(x)

        return np.mean(tree_preds)

    def predict(self, test_x):
        m,n = test_x.shape
        preds = np.zeros(m)
        for i,x in enumerate(test_x):
            preds[i] = self.findPred(x)

        return preds

