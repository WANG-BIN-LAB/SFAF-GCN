import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random
# 定义特征数据文件夹路径
feature_folder = 'H:/sMRI+SVM/MCI_NC_328_10K/k10-test1feature'
# seed = 42   # 设定随机数种子
# seed = int(seed)
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# 加载训练集特征数据
train_files = os.listdir(os.path.join(feature_folder, 'train'))
X_train = []
for file in train_files:
    feature_data = np.load(os.path.join(feature_folder, 'train', file))
    X_train.append(feature_data)
X_train = np.array(X_train)

# 加载测试集特征数据
test_files = os.listdir(os.path.join(feature_folder, 'test'))
X_test = []
for file in test_files:
    feature_data = np.load(os.path.join(feature_folder, 'test', file))
    X_test.append(feature_data)
X_test = np.array(X_test)

# 加载训练集和测试集标签数据（与之前一样）
train_labels = pd.read_csv('H:/sMRI+SVM/MCI_NC_328_10K/train_fold_2.csv')  # 训练集标签数据，包含一个名为"label"的列
test_labels = pd.read_csv('H:/sMRI+SVM/MCI_NC_328_10K/test_fold_2.csv')  # 测试集标签数据，包含一个名为"label"的列
y_train = train_labels['label'].values
y_test = test_labels['label'].values
# 进行特征降维
n_components = 150  # 设置主成分个数
pca = PCA(n_components=n_components)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # 创建 StandardScaler 对象
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))  # 对训练数据进行标准化
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))  # 对测试数据进行标准化
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

# 使用 imputer 对象对数据进行填充
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.fit_transform(X_test_scaled)
X_train = pca.fit_transform(X_train_scaled)
X_test = pca.transform(X_test_scaled)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 初始化模型、损失函数和优化器
model = SimpleNN(input_size=X_train.shape[1], hidden_size=512, num_classes=2)
# model = SimpleNN(input_size=np.prod(X_train.shape[-2:]), hidden_size=64, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        tensor_shape = inputs.size()
        outputs = model(inputs)
        # outputs = model(inputs.view(-1, tensor_shape[-2] * tensor_shape[-1]))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 模型评估
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        # outputs = model(inputs.view(-1, tensor_shape[-2] * tensor_shape[-1]))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
# print(predicted)
# print(labels)
accuracy = correct / total
print("Accuracy:", accuracy)
# 计算敏感性
# recall = recall_score(y_test, y_pred)

# # 计算特异性
# specificity = recall_score(y_test, y_pred, pos_label=0)

# # 计算AUC
# auc = roc_auc_score(y_test, y_pred)
# accuracy = np.mean(y_pred == y_test)