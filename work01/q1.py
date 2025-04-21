import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix


# 预处理封装
def pre_process(filepath):
    data = pd.read_csv(filepath, sep=' ')
    data = np.array(data)
    # 移除最后一列的nan
    data = data[:, :-1]  # 去掉最后nan

    # 分离特征和标签
    X = data[:, :256]  # 前256列是特征
    Y = data[:, 256:]  # 后10列是one-hot标签

    return X, Y


# 自定义的欧氏距离计算函数
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# 我的自定义的 kNN 模型
def my_knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_sample in X_test:
        # 计算测试样本和所有训练样本的距离
        distances = [euclidean_distance(test_sample, train_sample) for train_sample in X_train]

        # 获取最近的 k 个样本的索引
        k_indices = np.argsort(distances)[:k]

        # 获取这 k 个样本的标签
        k_nearest_labels = [y_train[i] for i in k_indices]

        # 多数投票，选择出现最多的标签作为预测结果
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])

    return predictions


def call_my_knn(X, y, k):
    # 预测列表和标签列表
    predictions = []
    true_labels = []

    for i in range(X.shape[0]):
        # 留一法：选择当前样本为测试集，其他样本为训练集
        X_loo = np.delete(X, i, axis=0)
        y_loo = np.delete(y, i, axis=0)

        # 当前样本为测试集
        # 注意kNN的输入需要一个二维数组，X[i]是一个形状为 (256,) 的一维向量
        # 需要将其reshape为批数为1的二维数组
        # reshape(1, -1)是对X[i]的重塑操作
        # 1表示将这个样本作为一个批次的数据
        # -1表示自动计算剩余的维度

        X_test_loo = X[i].reshape(1, -1)
        y_test_loo = y[i]

        # 调用自定义 kNN 模型
        y_pred_loo = my_knn_predict(X_loo, y_loo, X_test_loo, k=k)

        # 记录预测结果
        predictions.append(y_pred_loo[0])
        true_labels.append(y_test_loo)

    # 计算准确率，nmi,cen
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


# 计算不同 k 值下的精确度并绘图
def print_accuracies(X, y_labels, k_values):
    accuracies = []

    for k in k_values:
        accuracy = call_my_knn(X, y_labels, k)
        accuracies.append(accuracy)
        print(f"k = {k}, accuracy = {accuracy}")


if __name__ == '__main__':
    # 导入实验数据集
    mypath = 'D:/28301/Desktop/机器学习/作业1/semeion.data'
    # 数据预处理，分离样本特征和样本标签，并将标签由独热码转化为数字
    X_flat, y = pre_process(mypath)
    y_labels = np.argmax(y, axis=1)  # 将 one-hot 编码转换为数字标签
    print(X_flat.shape, y.shape)
    # 测试 k 从 1 到 13 的精确度
    k_values = [5, 9, 13]
    print_accuracies(X_flat, y_labels, k_values)
