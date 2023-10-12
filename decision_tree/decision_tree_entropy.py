from decision_tree.tree import BaseDecisionTree

class EntropyDecisionTree(BaseDecisionTree):
    def __init__(self, prune=False):
        super().__init__(impurity_measure='entropy', prune=False)
