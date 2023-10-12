from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from decision_tree.decision_tree_gini import GiniDecisionTree
from decision_tree.decision_tree_entropy import EntropyDecisionTree
from decision_tree.utils import load_csv
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

if __name__ == "__main__":
    # Load data and perform training and testing
    filename = 'wine_dataset.csv'
    X, y = load_csv(filename)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree1 = EntropyDecisionTree()
    tree1.learn(X_train_val, y_train_val)
    tree1_prune = EntropyDecisionTree(prune=True)
    tree1_prune.learn(X_train_val, y_train_val)
    tree2 = GiniDecisionTree()
    tree2.learn(X_train_val, y_train_val)
    tree2_prune = GiniDecisionTree(prune=True)
    tree2_prune.learn(X_train_val, y_train_val)

    predictions1 = tree1.predict(X_test)
    predictions1_prune = tree1_prune.predict(X_test)
    predictions2 = tree2.predict(X_test)
    predictions2_prune = tree2_prune.predict(X_test)

    accuracy1 = accuracy_score(y_test, predictions1)
    accuracy1_prune = accuracy_score(y_test, predictions1_prune)
    accuracy2 = accuracy_score(y_test, predictions2)
    accuracy2_prune = accuracy_score(y_test, predictions2_prune)

    print("Accuracy by entropy:", accuracy1)
    print("Accuracy by entropy with pruning:", accuracy1_prune)
    print("Accuracy by gini:", accuracy2)
    print("Accuracy by gini with pruning:", accuracy2_prune)

    # Visualize the decision boundary (for 2D data only)
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary")
        plt.show()

feature_names = ['citric acid',	'residual sugar',	'pH',	'sulphates',	'alcohol']  # 替换为您的特征名称列表
class_names = ['white', 'red']  # 替换为您的类别名称列表
tree2.visualize(feature_names=feature_names, titlename="Gini")
tree2_prune.visualize(feature_names,titlename="Gini with pruning")

# 计算混淆矩阵
confusion = confusion_matrix(y_test, predictions2)


# 生成分类报告
report = classification_report(y_test, predictions2)
print("Classification Report:")
print(report)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix about gini')
plt.show()