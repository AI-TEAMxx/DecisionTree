# Decision Tree Classification with Gini and Entropy Impurity Measures

This repository contains Python code for training and evaluating decision tree classifiers using two different impurity measures: Gini impurity and Entropy impurity. Decision trees are a popular machine learning algorithm used for both classification and regression tasks.

## Getting Started

These instructions will help you understand how to use the provided code to train and evaluate decision tree classifiers based on Gini and Entropy impurity measures. 

### Prerequisites

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn (for visualization)

You can install the required packages using pip:

`pip install pandas numpy scikit-learn matplotlib seaborn`

Run the decision tree classifiers:
Run the following command to train and evaluate the decision tree classifiers:

`python main.py`

This will output the accuracy of both Gini-based and Entropy-based decision tree classifiers on your dataset.

Visualize decision boundaries (if applicable):

If your dataset is two-dimensional, the code will also generate visualizations of the decision boundaries for both classifiers. These visualizations can help you understand how each classifier separates different classes in your data.

## Customization
You can customize the decision tree classifiers by modifying the GiniDecisionTree and EntropyDecisionTree classes in gini_decision_tree.py and entropy_decision_tree.py. For example, you can change the pruning behavior or add additional methods specific to your needs.

If you want to compare this decision tree with existing libraries, you can use the` python ohthertree.py`

## Contributions(mainly)
Xuanming Zhang:Decision tree algorithm, Code, Error Analysis

Xiaoxue Wang: post-pruning algorithm, visualization,code, write the report