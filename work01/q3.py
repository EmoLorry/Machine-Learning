from main import *
# 使用pytorch实现CNN图片识别

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix


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

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        # Activation layers
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 经过三轮卷积、池化、激活
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(-1, 128 * 2 * 2)  #

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return self.fc3(x)


# 训练和评估函数
def train_cnn(X, Y, epochs=10, batch_size=64, val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 划分训练集和验证集
    dataset_size = X.shape[0]
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    Y_tensor = torch.tensor(np.argmax(Y, axis=1), dtype=torch.long).to(device)  # Convert one-hot to label

    dataset = TensorDataset(X_tensor, Y_tensor)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 开始训练
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 训练阶段
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                preds = torch.argmax(output, dim=1).cpu().numpy()
                labels = target.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        # 计算评估指标
        acc = accuracy_score(all_labels, all_preds)
        nmi = normalized_mutual_info_score(all_labels, all_preds)
        cen = ConfusionEntropy(confusion_matrix(all_labels, all_preds))

        # 打印结果
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {running_loss / len(train_loader):.4f}, Accuracy: {acc:.4f}, NMI: {nmi:.4f}, CEN: {cen:.4f}")

    torch.save(model, './LeNet.pkl')


if __name__ == '__main__':
    # 导入实验数据集
    mypath = 'D:/28301/Desktop/机器学习/作业1/semeion.data'
    # 数据预处理，分离样本特征和样本标签，并将标签由独热码转化为数字
    X_flat, Y = pre_process(mypath)
    X = X_flat.reshape(-1, 16, 16)
    # 通过旋转进行数据增强，得到新样本和标签对应的向量
    X_p, Y_p = get_reinforce_data(X, Y, 16)

    # 将原始数据与增强数据合并
    X_pro = np.concatenate((X, X_p), axis=0)  # 按样本（行）合并
    Y_pro = np.concatenate((Y, Y_p), axis=0)  # 按样本（行）合并
    print(X_pro.shape, Y_pro.shape)

    # 训练并评估CNN模型
    train_cnn(X_pro, Y_pro, epochs=50, val_split=0.1)
