# import torch
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import random
# # 定义全连接层分类器
# class Classifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x

# def main():
#     # 设置输入特征维度、隐藏层维度和输出类别数
#     input_size = 2700  # 假设输入特征维度为10
#     hidden_size = 512  # 设定隐藏层维度为5
#     output_size = 2 # 假设有3个类别需要分类
#     # net_weighted = np.load('G:/BNGNN/Data/npy_AD_NC1/'+ 'net_wighted_AD_NC.npy',allow_pickle=True).reshape(82, -1)
#     # net_label = np.load('G:/BNGNN/Data/npy_AD_NC1/' + 'net_label_AD_NC.npy',allow_pickle=True)
#     # 创建模型实例
#     model = Classifier(input_size, hidden_size, output_size)

#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
#     optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器
    
#     # 准备训练数据

    # num_net = len(net_label)
    # all_idx = [i for i in range(num_net)]
    # random.shuffle(all_idx)
    # train_idx = all_idx[:65]
    # test_idx = all_idx[65:]
#     input = torch.randn(5, input_size)  # 生成5个样本的输入数据
#     labels = torch.tensor([1, 0, 2, 1, 2])  # 真实标签

#     # 训练模型
#     for epoch in range(100):
#         # labels = np.array(np.array(env.net_label[indexes][:,1], dtype=np.float64))
# 		# labels = torch.LongTensor(labels)
#         optimizer.zero_grad()
#         # labels = np.array(net_label[train_idx][:,1],dtype=np.float64)
#         # labels = torch.LongTensor(labels)
#         # input=np.array(net_weighted[train_idx],dtype=np.float64)
#         # input = torch.LongTensor(input)
#         output = model(input)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#         if epoch % 10 == 0:
#             print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

#     # 测试模型
#     test_input = torch.randn(1, input_size)  # 生成一个测试样本的输入数据
#     # test_input = net_weighted[test_idx] # 生成一个测试样本的输入数据
#     with torch.no_grad():
#         output = model(test_input)
#         predicted_label = torch.argmax(output, dim=1)
#         print('Predicted label:', predicted_label.item())

# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve,auc
net_weighted = np.load('G:/BNGNN/Data/npy_AD_NC1/'+ 'net_wighted_AD_NC.npy',allow_pickle=True).reshape(82, -1)
net_label = np.load('G:/BNGNN/Data/npy_AD_NC1/' + 'net_label_AD_NC.npy',allow_pickle=True)
num_net = len(net_label)
all_idx = [i for i in range(num_net)]
random.shuffle(all_idx)
train_idx = all_idx[:65]
test_idx = all_idx[65:]
# 定义全连接层分类器模型
class Classifier(nn.Module):
    # def __init__(self, input_size, hidden_size, num_classes):
    #     super(Classifier, self).__init__()
    #     self.fc1 = nn.Linear(input_size, hidden_size)
    #     self.fc2 = nn.Linear(hidden_size, num_classes)
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     out = self.fc1(x)
    #     out = self.relu(out)
    #     out = self.fc2(out)
    #     return out
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # 新添加的全连接层
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)  # 将数据传递到新添加的全连接层
        out = self.relu(out)  # 应用激活函数
        out = self.fc3(out)
        return out
num_epochs=600
# 模拟输入数据和标签
# input_data = torch.randn(10, 5)  # 假设有10个样本，每个样本有5个特征
input=np.array(net_weighted[train_idx],dtype=np.float32)
input = torch.tensor(input)
# labels = torch.LongTensor([0, 1, 0, 1, 1, 0, 2, 2, 1, 0])  # 对应的标签
labels = np.array(net_label[train_idx][:,1],dtype=np.float64)
labels = torch.LongTensor(labels)
# 创建分类器模型
input_size = 2700  # 输入特征维度为5
hidden_size1 = 512  # 隐藏层维度为10
hidden_size2 = 128
num_classes = 2  # 分类的类别数为2
model = Classifier(input_size, hidden_size1,  hidden_size2,num_classes)
max_val_acc = 0.
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
accuracies = []
# 训练模型
for epoch in range(num_epochs):
    # Forward pass
    random.shuffle(all_idx)
    train_idx = all_idx[:65]
    test_idx = all_idx[65:]
    input=np.array(net_weighted[train_idx],dtype=np.float32)
    input = torch.tensor(input)
# labels = torch.LongTensor([0, 1, 0, 1, 1, 0, 2, 2, 1, 0])  # 对应的标签
    labels = np.array(net_label[train_idx][:,1],dtype=np.float64)
    labels = torch.LongTensor(labels)
    outputs = model(input)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 600, loss.item()))
    # 使用训练好的模型进行预测
   
    test=np.array(net_weighted[test_idx],dtype=np.float32)
    test = torch.tensor(test)
    predicted = model(test).argmax(dim=1)
    # print('Predicted:', predicted)
    labels_test = np.array(net_label[test_idx][:,1],dtype=np.float64)
    # print('true:', labels_test)
    val_acc = accuracy_score(labels_test, predicted)
    # print(val_acc)
    accuracies.append(val_acc)
    if val_acc > max_val_acc:
        max_val_acc = val_acc
				# print("Epoch: {}".format(epoch), " Train_Acc: {:.5f}".format(train_acc), " Val_Acc: {:.5f}".format(max_val_acc), " Test_Acc: {:.5f}".format(max_test_acc))
print(max_val_acc)
epochs = range(1, num_epochs + 1)
plt.plot(epochs, accuracies, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.show()