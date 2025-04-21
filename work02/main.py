import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("winequality-white.csv")
# （4898，12）

# print(data)
# 转化为numpy数组
data = np.array(data)

# 划分特征和标签值
X = data[:, :-1]
y = data[:, -1]

# 数据分层划分，按照4:1划分为训练集和测试集
# train_test_split 中的 stratify=y 参数可以确保分层采样。
# stratify 的作用是根据目标变量 y 中各个类别的比例，
# 按相同比例划分数据集，从而在训练集和测试集中保持数据的类别分布一致。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 特征标准化
# StandardScaler 是一种常用的标准化方法，
# 它通过将数据进行均值为 0、方差为 1 的处理来消除特征量纲的差异。
scaler = StandardScaler()

# fit 阶段:计算每个特征在训练集上的均值 μ 和标准差 σ。
# 这一步只对训练集进行，因为测试集的数据是未知的，不能提前用它来影响标准化过程。
# 这样可以确保模型在实际应用中只依赖于训练集数据，符合机器学习的原则。
# transform 阶段:
# 利用训练集计算得到的均值和标准差对训练集进行标准化
X_train = scaler.fit_transform(X_train)
# 为了向量相乘方便，在训练集X左侧添加全为1的一列
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# 对测试集进行标准化，但 不重新计算 均值和标准差，而是使用在训练集上计算得到的均值 μ 和标准差 σ
# 这样做的目的是保持测试集与训练集在相同的标准化尺度上，确保模型在训练和测试阶段的一致性。
X_test = scaler.transform(X_test)
# 在测试集X左侧也添加全为1的一列
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])


# 批量梯度下降（BGD）与随机梯度下降（SGD）优化
# 线性回归模型定义
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, method='batch'):
        self.alpha = learning_rate
        self.epochs = epochs
        self.method = method  # 'batch' for BGD, 'stochastic' for SGD

    def fit(self, X_train, y_train, X_test, y_test):
        self.m, self.n = X_train.shape
        # 随机初始化回归系数
        self.theta = np.random.randn(self.n)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # 历史记录 MSE
        self.train_history = []
        self.test_history = []

        for epoch in range(self.epochs):
            if self.method == 'batch':
                self.batch_gradient_descent()
            elif self.method == 'stochastic':
                self.stochastic_gradient_descent()

            # 记录每次迭代训练集和测试集的 MSE
            train_mse = self.mean_squared_error(self.predict(self.X_train), self.y_train)
            test_mse = self.mean_squared_error(self.predict(self.X_test), self.y_test)
            self.train_history.append(train_mse)
            self.test_history.append(test_mse)

    def batch_gradient_descent(self):
        # gradients= m/1* XT (Xθ−y)
        gradients = 1 / self.m * self.X_train.T.dot(self.X_train.dot(self.theta) - self.y_train)
        # print(self.X.T.shape)
        # print(self.theta.shape)
        # print(self.y.shape)
        # print(gradients.shape)
        self.theta -= self.alpha * gradients

    def stochastic_gradient_descent(self):
        for i in range(self.m):
            random_idx = np.random.randint(self.m)
            xi = self.X_train[random_idx:random_idx + 1]
            yi = self.y_train[random_idx:random_idx + 1]
            # print(xi.shape)
            # print(yi.shape)
            gradients = -2 * xi.T.dot(yi - xi.dot(self.theta))
            self.theta -= self.alpha * gradients

    def predict(self, X):
        # print(np.dot(X, self.theta))
        return np.dot(X, self.theta)

    def mean_squared_error(self, predictions, y):
        # print(predictions)
        # print(y)
        return np.mean((predictions - y) ** 2)


# # 实例化线性回归模型，分别使用 BGD 和 SGD 进行训练
# model_bgd = LinearRegression(learning_rate=0.01, epochs=1000, method='batch')
# model_bgd.fit(X_train, y_train, X_test, y_test)
#
# model_sgd = LinearRegression(learning_rate=0.0001, epochs=100, method='stochastic')
# model_sgd.fit(X_train, y_train, X_test, y_test)
#
# # 绘制 MSE 收敛曲线
# plt.figure(figsize=(12, 6))
#
# # BGD 收敛曲线
# plt.subplot(1, 2, 1)
# plt.plot(model_bgd.train_history, label='Training MSE')
# plt.plot(model_bgd.test_history, label='Testing MSE')
# plt.title('BGD Convergence Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.legend()
#
# # SGD 收敛曲线
# plt.subplot(1, 1, 1)
# plt.plot(model_sgd.train_history, label='Training MSE')
# plt.plot(model_sgd.test_history, label='Testing MSE')
# plt.title('SGD Convergence Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Squared Error (MSE)')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

# 学习率列表
learning_rates = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# 用于存储不同学习率下的 MSE 收敛曲线和最终MSE
history_list = []
final_mse_list = []

# 使用不同学习率训练模型并记录历史和最终的MSE
for lr in learning_rates:
    model = LinearRegression(learning_rate=lr, epochs=1000, method='batch')
    model.fit(X_train, y_train, X_test, y_test)
    history_list.append(model.test_history)  # 保存每个学习率的收敛曲线

    # 记录每个学习率下训练结束时的 MSE
    final_mse = model.test_history[-1]
    final_mse_list.append(final_mse)
    print(f'Learning rate = {lr}, Final Test MSE = {final_mse}')

# 绘制不同学习率下的收敛曲线
plt.figure(figsize=(10, 6))

for i, lr in enumerate(learning_rates):
    plt.plot(history_list[i], label=f'Learning rate = {lr}')

plt.title('SGD Convergence with Different Learning Rates')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()
