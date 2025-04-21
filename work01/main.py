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


# 定义简单的 CEN (混淆熵) 实现
def ConfusionEntropy(cm):
    cen = 0
    total = np.sum(cm)

    for i in range(len(cm)):
        row_sum = np.sum(cm[i])  # 每一行的和
        for j in range(len(cm[i])):
            if cm[i][j] > 0:
                pij = cm[i][j] / total  # 类别 i 被预测为 j 的概率
                rij = cm[i][j] / row_sum  # 类别 i 被正确预测的概率
                cen += pij * (1 - rij)  # 混乱度

    return cen


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
    nmi = normalized_mutual_info_score(true_labels, predictions)
    cen = ConfusionEntropy(confusion_matrix(true_labels, predictions))
    return accuracy, nmi, cen


def call_ML_package_knn(X, y, k=3):
    # kNN 参数设置
    knn = KNeighborsClassifier(n_neighbors=k)

    # 预测列表和标签列表
    predictions = []
    true_labels = []

    for i in range(X.shape[0]):
        # 留一法：选择当前样本为测试集，其他样本为训练集
        X_loo = np.delete(X, i, axis=0)
        y_loo = np.delete(y, i, axis=0)

        # 当前样本为测试集
        X_test_loo = X[i].reshape(1, -1)
        y_test_loo = y[i]

        # 训练 kNN 模型
        knn.fit(X_loo, y_loo)
        # 预测
        y_pred = knn.predict(X_test_loo)

        # 记录预测结果
        predictions.append(y_pred[0])
        true_labels.append(y_test_loo)

    # 计算准确率，nmi,cen
    accuracy = accuracy_score(true_labels, predictions)
    nmi = normalized_mutual_info_score(true_labels, predictions)
    cen = ConfusionEntropy(confusion_matrix(true_labels, predictions))
    return accuracy, nmi, cen


# 绘制 Accuracy, NMI 和 CEN 比较
def plot_knn_comparison(X, y, k_values):
    accuracies_ml = []
    accuracies_custom = []
    nmis_ml = []
    nmis_custom = []
    cen_ml = []
    cen_custom = []

    for k in k_values:
        # 调用机器学习包的 kNN 模型
        ac1, nmi1, cen1 = call_ML_package_knn(X, y, k)
        accuracies_ml.append(ac1)
        nmis_ml.append(nmi1)
        cen_ml.append(cen1)

        # 调用自定义的 kNN 模型
        ac2, nmi2, cen2 = call_my_knn(X, y, k)
        accuracies_custom.append(ac2)
        nmis_custom.append(nmi2)
        cen_custom.append(cen2)

    # 绘制 Accuracy 对比图
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(k_values, accuracies_ml, label='use_package kNN Accuracy', marker='o')
    plt.plot(k_values, accuracies_custom, label='my kNN Accuracy', marker='x')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()

    # 绘制 NMI 对比图
    plt.subplot(3, 1, 2)
    plt.plot(k_values, nmis_ml, label='use_package kNN NMI', marker='o')
    plt.plot(k_values, nmis_custom, label='my kNN NMI', marker='x')
    plt.xlabel('k')
    plt.ylabel('NMI')
    plt.title('NMI Comparison')
    plt.legend()

    # 绘制 CEN 对比图
    plt.subplot(3, 1, 3)
    plt.plot(k_values, cen_ml, label='use_package kNN CEN', marker='o')
    plt.plot(k_values, cen_custom, label='my kNN CEN', marker='x')
    plt.xlabel('k')
    plt.ylabel('CEN')
    plt.title('CEN Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 导入实验数据集
    mypath = 'D:/28301/Desktop/机器学习/作业1/semeion.data'
    # 数据预处理，分离样本特征和样本标签，并将标签由独热码转化为数字
    X_flat, y = pre_process(mypath)
    y_labels = np.argmax(y, axis=1)  # 将 one-hot 编码转换为数字标签

    k_values = range(1, 14)  # 测试 k 从 1 到 13

    # 运行绘图函数
    plot_knn_comparison(X_flat, y_labels, k_values)
