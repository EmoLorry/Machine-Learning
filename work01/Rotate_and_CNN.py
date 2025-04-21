from main import *

# 使用pytorch实现CNN图片识别
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix



def plot_one(X, y, num_index):
    plt.figure(figsize=(4, 4))  # 调整为合适的图像大小
    plt.imshow(X[num_index].reshape(16, 16), cmap='gray')  # 展示16x16的图像
    plt.title(f"Label: {np.argmax(y[num_index])}")  # 标签为one-hot编码的最大值索引
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 图片左上旋转左下旋转，进行数据增强
def get_reinforce_data(input_x, input_y, n):
    X_plus = []
    Y_plus = []

    for k in range(input_x.shape[0]):
        # 初始化旋转矩阵
        rotated_matrix = np.zeros((n, n), dtype=input_x.dtype)
        # 手动进行左上90度旋转 (顺时针)
        for i in range(n):
            for j in range(n):
                rotated_matrix[j][n - 1 - i] = input_x[k][i][j]
        X_plus.append(np.copy(rotated_matrix))  # 追加矩阵的副本
        # 在这个操作中，X_plus实际上包含的是同一个rotated_matrix
        # 的不同引用（它们指向同一个对象）。所以，为了避免后面的旋转覆盖之前的旋转结果
        # 故使用深拷贝

        Y_plus.append(input_y[k])

        # 手动进行左下90度旋转 (逆时针)
        for i in range(n):
            for j in range(n):
                rotated_matrix[n - 1 - j][i] = input_x[k][i][j]
        X_plus.append(rotated_matrix)  # 追加矩阵的副本
        Y_plus.append(input_y[k])

    return np.array(X_plus), np.array(Y_plus)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


# 训练和评估函数
def train_cnn_loo(X, Y, epochs=3, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre = []
    true = []

    for i in range(X.shape[0]):
        # 留一法：选择当前样本为测试集，其他样本为训练集
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(Y, i, axis=0)

        # 当前样本为测试集
        X_test = X[i]
        y_test = np.argmax(Y[i])  # Convert one-hot encoded label to class index
        true.append(y_test)

        # 将数据转为PyTorch张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
        y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
            device)  # Add batch and channel dimensions

        # 构建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        for epoch in range(epochs):
            model.train()
            for data, target in train_loader:
                # print(data.shape, data.dtype)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 评估模型
        model.eval()
        with torch.no_grad():
            output = model(X_test_tensor)
            y_pred = torch.argmax(output, dim=1).cpu().numpy()
            pre.append(y_pred.item())

    # 输出评估指标
    avg_acc = accuracy_score(true, y_pred)
    avg_nmi = normalized_mutual_info_score(true, y_pred)
    avg_cen = ConfusionEntropy(confusion_matrix(true, y_pred))

    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average NMI: {avg_nmi:.4f}")
    print(f"Average CEN: {avg_cen:.4f}")


if __name__ == '__main__':
    # 导入实验数据集
    mypath = 'D:/28301/Desktop/机器学习/作业1/semeion.data'
    # 数据预处理，分离样本特征和样本标签，并将标签由独热码转化为数字
    X_flat, Y = pre_process(mypath)
    X = X_flat.reshape(-1, 16, 16)
    # 通过旋转进行数据增强，得到新样本和标签对应的向量
    X_p, Y_p = get_reinforce_data(X, Y, 16)

    print(X_p.shape, Y_p.shape)
    print(X.shape, Y.shape)
    plot_one(X_flat, Y,100)
    plot_one(X_p, Y_p, 200)
    plot_one(X_p, Y_p, 201)
    # # 将原始数据与增强数据合并
    # X_pro = np.concatenate((X, X_p), axis=0)  # 按样本（行）合并
    # Y_pro = np.concatenate((Y, Y_p), axis=0)  # 按样本（行）合并
    # print(X_pro.shape, Y_pro.shape)
    #
    # train_cnn_loo(X_pro, Y_pro, 3)
