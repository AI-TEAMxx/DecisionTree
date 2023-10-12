import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import copy
from matplotlib import pyplot as plt

class BaseDecisionTree:
    def __init__(self, impurity_measure='entropy', prune=False):
        self.impurity_measure = impurity_measure
        self.tree = None
        self.prune = prune

    def calculate_impurity(self, y):
        # Calculate impurity (either entropy or gini) of a set
        if self.impurity_measure == 'entropy':
            # Implement entropy calculation logic
            unique_labels, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            entropy = -np.sum(probs * np.log2(probs))
            return entropy
        elif self.impurity_measure == 'gini':
            # Implement Gini impurity calculation logic
            unique_labels, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            gini = 1 - np.sum(probs ** 2)
            return gini

    def calculate_information_gain(self, X, y, feature_index, threshold):
        # Calculate information gain based on impurity (either entropy or gini)
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        left_impurity = self.calculate_impurity(y[left_mask])
        right_impurity = self.calculate_impurity(y[right_mask])

        total_samples = len(y)
        weighted_impurity = (len(y[left_mask]) / total_samples) * left_impurity + (
                len(y[right_mask]) / total_samples) * right_impurity

        parent_impurity = self.calculate_impurity(y)
        information_gain = parent_impurity - weighted_impurity
        return information_gain

    def find_best_split(self, X, y):
        # Find the best feature and threshold to split on
        num_features = X.shape[1]
        best_feature_index = None
        best_threshold = None
        best_info_gain = -1

        for feature_index in range(num_features):
            unique_values = np.unique(X[:, feature_index])
            for threshold in unique_values:
                info_gain = self.calculate_information_gain(X, y, feature_index, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]  # Leaf node with a single class label

        if X.shape[0] == 0 or np.all(X == X[0, :], axis=0).all():
            return np.unique(y)[np.argmax(np.bincount(y))]  # Leaf node with the most common label

        best_feature_index, best_threshold = self.find_best_split(X, y)
        if best_feature_index is None:
            return np.unique(y)[np.argmax(np.bincount(y))]  # Leaf node with the most common label

        left_mask = X[:, best_feature_index] <= best_threshold
        right_mask = X[:, best_feature_index] > best_threshold

        left_subtree = self.build_tree(X[left_mask], y[left_mask])
        right_subtree = self.build_tree(X[right_mask], y[right_mask])

        return (best_feature_index, best_threshold, left_subtree, right_subtree)

    def prune_tree(self, tree, X_train, X_val, y_train, y_val):
        if not isinstance(tree, tuple):
            return
        feature_index, threshold, left_subtree, right_subtree = tree

        self.prune_tree(left_subtree, X_train, X_val, y_train, y_val)
        self.prune_tree(right_subtree, X_train, X_val, y_train, y_val)

        y_val_pred = self.predict(X_val)
        acc_before = accuracy_score(y_val, y_val_pred)

        tree_copy = copy.deepcopy(tree)
        tree = np.unique(y_train)[np.argmax(np.bincount(y_train))]

        y_val_pred_pruned = self.predict(X_val)
        acc_after = accuracy_score(y_val, y_val_pred_pruned)

        if acc_after >= acc_before:
            return tree
        else:
            return tree_copy

    def learn(self, X, y):
        X_train_val, X_val, y_train_val, y_val = train_test_split(X, y, test_size=0.2)
        self.tree = self.build_tree(X_train_val, y_train_val)

        if self.prune:
            self.prune_tree(self.tree, X_train_val, X_val, y_train_val, y_val)

    def plot_tree(self, node, depth=0, parent_x=0.5, parent_y=1.0):
        if not isinstance(node, tuple):
            # 如果节点是叶子节点，绘制叶子节点
            x = parent_x
            y = 1.0 - depth * 0.1
            plt.text(x, y, str(node),
                     size=6, ha="center", va="center",
                     bbox=dict(boxstyle="circle", ec=(0.5, 0.5, 0.5), fc=(0.5, 0.8, 0.8)))
        else:
            feature_index, threshold, left_subtree, right_subtree = node
            feature_name = self.feature_names[feature_index]
            x = parent_x
            y = 1.0 - depth * 0.1
            plt.text(x, y, f"{feature_name} <= {threshold}",
                     size=6, ha="center", va="center",
                     bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
            left_x = parent_x - 0.1
            right_x = parent_x + 0.1
            left_y = 1.0 - (depth + 1) * 0.1 # 修改这里，计算右子树的y坐标
            plt.plot([parent_x, left_x], [parent_y, left_y], c="black")
            plt.plot([parent_x, right_x], [parent_y, 1.0-(depth+1)*0.1], c="black")  # 注意这里，将右子树的y坐标传递给右子树
            self.plot_tree(left_subtree, depth + 1, left_x, left_y)
            self.plot_tree(right_subtree, depth + 1, right_x, 1.0-(depth+1)*0.1)

    def visualize(self, feature_names, titlename=None):
        self.feature_names = feature_names
        plt.figure(figsize=(9, 12))
        plt.axis("off")
        plt.title(titlename)
        self.plot_tree(self.tree)
        plt.show()

    def predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_index, threshold, left_subtree, right_subtree = tree
        if x[feature_index] <= threshold:
            return self.predict_one(x, left_subtree)
        else:
            return self.predict_one(x, right_subtree)

    def predict(self, X):
        predictions = []
        for x in X:
            predicted_class = self.predict_one(x, self.tree)
            predictions.append(predicted_class)
        return np.array(predictions)

    def print_tree(self, tree=None, indent=""):
        if tree is None:
            tree = self.tree
        if not isinstance(tree, tuple):
            print(indent + "Class:", tree)
        else:
            feature_index, threshold, left_subtree, right_subtree = tree
            print(indent + "Feature", feature_index, "<=", threshold)
            print(indent + "--> Left:")
            self.print_tree(left_subtree, indent + "   ")
            print(indent + "--> Right:")
            self.print_tree(right_subtree, indent + "   ")
