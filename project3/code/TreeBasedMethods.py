import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import collections
import random
import matplotlib.pyplot as plt

def gini_index(y):
    proportions = np.unique(y,return_counts=True)[1]/ len(y)
    gini = np.sum([p * (1-p) for p in proportions if p > 0])
    return gini

def entropy(y):
    proportions = np.unique(y,return_counts=True)[1]/ len(y)
    entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
    return entropy

def mse(y):
    return np.sum([(i-np.mean(y))**2 for i in y])

class Node:

    # implements a node in a classification tree
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):

        # contains feature of split and threshold and references to children nodes if not lead
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        
        # contains predicted value if leaf
        self.value = value
    
    def is_leaf(self):
        # boolean if it's a leaf or not
        return self.value is not None

class DecisionTreeClass:

    # implements the tree

    def __init__(self, classes=[-1,1], max_depth=100, min_samples_split=2):
        self.classes = classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _is_finished(self, depth):

        # has finished growing the tree if reached max_depth of num_samples in each split

        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _build_tree(self, X, y, depth=0):

        # builds tree recursively by getting the best split

        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            if len(np.unique(y)) == 1:
                return Node(value=y[0])
            else:
                most_common_Label = np.argmax(np.unique(y,return_counts=True)[1])
                return Node(value=self.classes[most_common_Label])
        else:
            # get best split
            rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
            best_feat, best_thresh = self._best_split(X, y, rnd_feats)

            left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
            left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
            right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
            return Node(best_feat, best_thresh, left_child, right_child)
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _create_split(self, X, thresh):

        # splits data given a feature and threshold

        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _score(self, X, y, thresh, index = gini_index):

        # computes gini index loss (or other score) for a split

        parent_loss = index(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * index(y[left_idx]) + (n_right / n) * index(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):

        # finds the best split by looking at all features and possible thresholds

        split = {'score': -1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._score(X_feat, y, thresh)
                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

class RandomForest:
    
    # class the implements random forest

    def __init__(self, num_trees=25, min_samples_split=2, max_depth=5):

        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # list to store the trained decision trees
        self.decision_trees = []
        
    def _sample(X, y):

        # function to create bootstrap samples

        n_rows, n_cols = X.shape
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[samples], y[samples]
        
    def fit(self, X, y):

        # trains the classifier

        # reset the trees
        if len(self.decision_trees) > 0:
            self.decision_trees = []
            
        # build each tree of the forest
        num_built = 0
        while num_built < self.num_trees:
            clf = DecisionTreeClass(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth
            )
            # create sample
            _X, _y = self._sample(X, y)
            # Train
            clf.fit(_X, _y)

            self.decision_trees.append(clf)
            num_built += 1
    
    def predict(self, X):
        
        # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        
        y = np.swapaxes(a=y, axis1=0, axis2=1)
        
        # Use majority voting for the final prediction
        predictions = []
        for preds in y:
            counter = collections.Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        return predictions

def compute_error(y, y_pred, w_i):

    # compute weighted misclassification error

    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error):

    # compute voting weight for AdaBoost weak classifier

    return np.log((1 - error) / error)

def update_weights(w_i, alpha, y, y_pred):

    # update individual weights w_i after a boosting iteration

    weights = w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

    return weights/sum(weights)
    
class AdaBoost:
    
    def __init__(self):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, y, M = 100):


        self.alphas = [] 
        self.training_errors = []
        self.M = M

        # iterate over M weak classifiers
        for m in range(0, M):
            
            # set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # initially weights are all equal to 1 / N
            else:
                w_i = update_weights(w_i, alpha_m, y, y_pred)
            
            #  fit weak classifier and predict labels
            G_m = DecisionTreeClass(max_depth = 1)  

            sample = random.choices(np.arange(len(y)), w_i, k=len(y))

            X_ = X[sample]
            y_ = y[sample]

            G_m.fit(X_,y_)
            y_pred = G_m.predict(X)
            
            self.G_M.append(G_m)

            # compute error and alpha
            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)

    def predict(self, X):

        # predict using fitted model
       
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        # predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        # calculate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred