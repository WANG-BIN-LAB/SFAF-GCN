import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        adj = adj.unsqueeze(0)  
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1) # 双线性层 x_1 W x_2 + b, 输出 batch * 1 维度，相当于输出表示两个输入之间的关系？

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None): #c应该与h_pl接近，与h_mi远离？

        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)

        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
class DGI(nn.Module):
    # ft_size, hid_units, nonlinearity
    def __init__(self, ft_size, hid_units, nonlinearity):
        super(DGI, self).__init__()
        self.fc1 = nn.Linear(ft_size, hid_units)
        self.fc2 = nn.Linear(hid_units, 1)
        self.nonlinearity = nonlinearity()
    # def __init__(self, n_in, n_h, activation):
    #     super(DGI, self).__init__()
    #     self.gcn = GCN(n_in, n_h, activation)
    #     self.read = AvgReadout() #读出函数，其实这里就是所有节点表示的均值

    #     self.sigm = nn.Sigmoid()

    #     self.disc = Discriminator(n_h) #判别器，定义为一个双线性函数bilinear
    def forward(self, features, shuf_fts, adj, sparse, *args):
        h = self.nonlinearity(self.fc1(features))
        logits = self.fc2(h)
        return logits
    # def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2): # msk: None, samp_bias1: None, samp_bias2: None,

    #     h_1 = self.gcn(seq1, adj, sparse)

    #     c = self.read(h_1, msk)
    #     c = self.sigm(c) # c表示全图信息

    #     h_2 = self.gcn(seq2, adj, sparse)

    #     ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2) #计算c-h_1,c_h_2的双线性判别器的结果

    #     return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach() #将tensor从计算图中分离出来，不参与反向传播


# 定义函数用于加载.mat文件并提取邻接矩阵
def load_adjacency_matrices(folder_path):
    adjacency_matrices = []
    
    # 遍历文件夹中的所有.mat文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            # 构造文件的完整路径
            file_path = os.path.join(folder_path, filename)
            # 加载.mat文件
            mat_data = sio.loadmat(file_path)

            # 假设邻接矩阵存储在名为'adjacency'的变量中
            if 'FC' in mat_data:
                adj_matrix = mat_data['FC']
                
                # 确保邻接矩阵是二维数组
                if isinstance(adj_matrix, np.ndarray) and adj_matrix.ndim == 2:
                    adjacency_matrices.append(adj_matrix)
                else:
                    print(f"Warning: '{filename}' does not contain a valid adjacency matrix.")
            else:
                print(f"Warning: '{filename}' does not contain an adjacency matrix.")

    return adjacency_matrices

# 使用示例
folder_path = 'H:/研究生/数据/FC_360/FC'  # 替换为你的文件夹路径
adjacency_matrices = load_adjacency_matrices(folder_path)

# 超参数设置
ft_size = adjacency_matrices[0].shape[0]  # 假设特征数等于邻接矩阵的节点数
hid_units = 32
batch_size = 360
nonlinearity = nn.ReLU
# 创建 DGI 模型
model = DGI(ft_size, hid_units, nonlinearity)

# 定义损失函数
b_xent = nn.BCEWithLogitsLoss()

# 示例优化器设置
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100  # 增加训练轮次
for epoch in range(num_epochs):
    for i, adj in enumerate(adjacency_matrices):
        features = torch.tensor(adj, dtype=torch.float32)  
        shuf_fts = features[torch.randperm(360)] 
        lbl = torch.randint(0, 2, (batch_size, 1)).float()

        optimizer.zero_grad()  # 梯度清零
        logits = model(features, shuf_fts, adj, sparse=False) 
        
        loss = b_xent(logits, lbl)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if i % 10 == 0:  # 每训练10个批次输出损失
            print(f'Epoch [{epoch + 1}/{num_epochs}], File {i + 1}, Loss: {loss.item()}')
# # 遍历每个邻接矩阵并进行前向传播
# for i, adj in enumerate(adjacency_matrices):
#     features = torch.rand(batch_size, ft_size)  # 假设随机生成特征
#     shuf_fts = torch.rand(batch_size, ft_size)  # 随机生成打乱特征
#     lbl = torch.randint(0, 2, (batch_size, 1)).float()  

#     # 前向传播
#     logits = model(features, shuf_fts, adj, sparse=False)
    
#     # 计算损失
#     loss = b_xent(logits, lbl)

#     # 输出结果
#     print(f'File {i + 1} - Logits:\n{logits}')
#     print(f'File {i + 1} - Labels:\n{lbl}')
#     print(f'File {i + 1} - Loss: {loss.item()}')

#     # 反向传播和优化
#     optimizer.zero_grad()  # 梯度清零
#     loss.backward()        # 反向传播
#     optimizer.step()      # 更新参数