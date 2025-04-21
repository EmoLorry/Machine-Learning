import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据集
data = pd.read_csv("winequality-white.csv")

# 划分特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 为训练集和测试集添加偏置项 (x0 = 1)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# OLS最小二乘法的实现
def ordinary_least_squares(X, y):
    # 计算 (X^T X)^(-1) X^T y
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return beta

# 训练OLS模型
beta_hat = ordinary_least_squares(X_train, y_train)

# 预测函数
def predict(X, beta):
    return np.dot(X, beta)
# 在训练集上进行预测
y_train_pred = predict(X_train, beta_hat)

# 在测试集上进行预测
y_test_pred = predict(X_test, beta_hat)

# 计算训练集 MSE
mse_train = np.mean((y_train_pred - y_train) ** 2)

# 计算测试集 MSE
mse_test = np.mean((y_test_pred - y_test) ** 2)

# 输出结果
print(f'Train MSE: {mse_train}')
print(f'Test MSE: {mse_test}')

# 打印回归系数
print("Estimated coefficients (beta):")
print(beta_hat)