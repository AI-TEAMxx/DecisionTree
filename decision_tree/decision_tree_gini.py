from decision_tree.tree import BaseDecisionTree

class GiniDecisionTree(BaseDecisionTree):
    def __init__(self, prune=False):
        super().__init__(impurity_measure='gini', prune=False)
