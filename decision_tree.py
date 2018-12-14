import numpy as np 
import pandas as pd
import util
from linreg import LinearRegression

#Class to hold decision nodes
class DTreeNode():
    def __init__(self, level, feature, split_val, left, right, ave, loss, size):
        self.level = level
        self.feature = feature
        self.split_val = split_val
        self.ave = ave
        self.left = left
        self.right = right
        self.loss = loss
        self.size = size

#Class to build a decision tree
class DecisionTree():

    def __init__(self, root=None, max_depth=10, min_leaf_size=10):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.root = root    #Do I need to setup right here?

    #Evaluates the quality of a split using standard deviation reduction as loss
    def splitVAR_old(self, labels, index):
        m = len(labels)
        left_var = np.var(labels[:index]) * (index/m)
        right_var = np.var(labels[index:]) * ((m-index)/m)
        return left_var + right_var

    #Evaluates a split using mean squared error reduction as loss
    def splitMSE(self, labels, index):
        m = len(labels)
        left_mse = util.findMSE(np.mean(labels[:index]), labels[:index])
        print(left_mse, np.var(labels[:index]))
        right_mse = util.findMSE(np.mean(labels[index:]), labels[index:])
        return left_mse + right_mse

    #Uses variance reduction to evaluate the loss of a split
    def splitLoss(self, left, right, Y):
        if len(left) <= 0 or len(right) <= 0: return np.inf
        left_var = np.var(Y[left]) * len(left) 
        right_var = np.var(Y[right]) * len(right) 
        return (left_var + right_var)/(len(left)+len(right))
    
    #Evaluates the quality of a split
    def evaluateSplit(self, split_val, X_feature, Y, node_indices):
        left, right = [], []

        for i in node_indices:
            if X_feature[i] < split_val:
                left.append(i)
            else: 
                right.append(i)

        if len(left) < self.min_leaf_size or len(right) < self.min_leaf_size:
            split_loss = np.inf
        else:
            split_loss = self.splitLoss(left, right, Y)
        return split_loss, left, right

    #Find the best possible split on a given feature
    def featuresSplit(self, feature, X_feature, Y, node_indices):
        feature_vals = set(X_feature[node_indices])
        min_loss = np.var(Y[node_indices])
        best_split, best_left, best_right = None, None, None

        for split_val in feature_vals:
            cur_loss, left, right = self.evaluateSplit(split_val, X_feature, Y, node_indices)
            if cur_loss < min_loss:
                min_loss = cur_loss
                best_left, best_right = left, right
                best_split = split_val

        return min_loss, best_split, best_left, best_right

    #Find where to split a node
    def getNodeSplit(self, X, Y, node_indices, feature_indices, level):
        m,n = X.shape

        best_feature, best_split, best_left, best_right = None, None, None, None
        #Check if node is leaf or not before splitting
        if level >= self.max_depth or m <= 2 * self.min_leaf_size:
            min_loss = np.var(Y[node_indices])
            node_ave = np.mean(Y[node_indices])
        else:
            min_loss = np.inf
            node_ave = None

            #Loop over all possible features
            for feature in feature_indices:
                #Find the best loss you can achieve with this feature
                cur_loss, split_val, left, right = self.featuresSplit(feature, X[:, feature], Y, node_indices)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    best_feature, best_split, best_left, best_right = feature, split_val, left, right
            
            #No good fits were found, make this node a leaf
            if best_split == None: 
                min_loss = np.var(Y[node_indices])
                node_ave = np.mean(Y[node_indices])

        return best_feature, best_split, best_left, best_right, node_ave, min_loss    

    def buildTree(self, train_x, train_y, node_indices, feature_indices, level=0):
        
        feature, split_val, left, right, node_ave, node_loss = self.getNodeSplit(train_x, train_y, node_indices, feature_indices, level)
        node = DTreeNode(level, feature, split_val, left, right, node_ave, node_loss, len(node_indices))

        #Keep splitting
        if not node.ave:
            node.left = self.buildTree(train_x, train_y, left, feature_indices, level=level+1)
            node.right = self.buildTree(train_x, train_y, right, feature_indices, level=level+1)
        
        return node

    def fit(self, train_x, train_y, limit_tree=False, feature_indices=None):
        m,n = train_x.shape
        if not limit_tree: feature_indices = [i for i in range(n)]
        node_indices = [i for i in range(m)]
        self.root = self.buildTree(train_x, train_y, node_indices, feature_indices)

    #Find prediction for a single x value
    def findPred(self, x):
        cur_node = self.root
        #Walk down to bottom of tree
        while not cur_node.ave:
            feature_val = x[cur_node.feature]
            if feature_val < cur_node.split_val:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
        
        return cur_node.ave

    def predict(self, test_x):
        m,n = test_x.shape
        preds = np.zeros(m)

        for row in range(m):
            preds[row] = self.findPred(test_x[row,:])

        return preds

    def printTree(self, node, pad=''):

        if node.left: self.printTree(node.left, pad=pad+'\t')
        
        print_str = 'Split on feature ' + str(node.feature) + ' at value ' + str(node.split_val) + ' Size: ' + str(node.size) + " Pred: " + str(node.ave)
        if node.ave: 
            print_str = "Leaf, predict: " + str(node.ave) + ' Size: ' + str(node.size)
            pad += '-> '
        print(pad, 'Level:', node.level, print_str,  'Loss:', node.loss)
        
        if node.right: self.printTree(node.right, pad=pad+'\t')

    def printNode(self, node):
        print("Level: ", node.level, 'Pred: ', node.ave, "Left: ", node.left, 'Right: ', node.right, '\n')

   