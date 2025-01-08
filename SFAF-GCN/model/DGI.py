import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.io as sio
import numpy as np
import pandas as pd
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # 简化的图卷积计算
        out = torch.matmul(adj, x)
        return self.linear(out)

class DGI(nn.Module):
    def __init__(self, input_dim, hid_units):
        super(DGI, self).__init__()
        self.graph_conv1 = GraphConvLayer(input_dim, hid_units)  # 第一层图卷积
        self.graph_conv2 = GraphConvLayer(hid_units, hid_units)   # 第二层图卷积
        self.fc = nn.Linear(hid_units, 3)  # 最终降维到 3

    def forward(self, x, adj):
        z = F.relu(self.graph_conv1(x, adj))
        z = F.relu(self.graph_conv2(z, adj))
        z = self.fc(z)  # 降维到 3
        return z

# 假设原始数据和邻接矩阵
mat_data = sio.loadmat('H:/研究生/数据/FC_360/FC/FC_time_100610.mat')
if 'FC' in mat_data:
    adj_matrix = mat_data['FC']
input_data =  torch.tensor(adj_matrix, dtype=torch.float32)    # 输入特征
adjacency_matrix =  torch.tensor(adj_matrix, dtype=torch.float32)    # 简化的邻接矩阵（自连接）

# 创建 DGI 模型，将 hid_units 设置为 8 （可以根据需要调整）
model = DGI(input_dim=360, hid_units=64)

# 进行降维
reduced_data = model(input_data, adjacency_matrix)

# 输出结果的形状
print("Reduced Data Shape:", reduced_data.shape)  # 应该是 (360, 3)
print(reduced_data)
reduced_data_np = reduced_data.detach().numpy()
reduced_data_df = pd.DataFrame(reduced_data_np, columns=[f'Component_{i+1}' for i in range(reduced_data.shape[1])])

# 保存为 Excel 文件
output_file = 'H:/研究生/数据/FC_360/reduced_data.csv'
reduced_data_df.to_csv(output_file, index=False)

print(f'Reduced data saved to {output_file}')
# 计算皮尔逊相关系数
correlation_matrix = np.corrcoef(reduced_data_np, rowvar=False)

# 输出相关性矩阵
print("Correlation Matrix:")
print(correlation_matrix)