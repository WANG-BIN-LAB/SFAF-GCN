import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy
import sys,os
import torch.nn.init as init
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import StratifiedShuffleSplit
from torch_scatter import scatter_add,scatter_mean
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import BallTree
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
from pygcn.layers import GraphConvolution,GINNet,ChebConv,GPRGNN
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from pygcn.layers import GraphSAGEConv
from torch_geometric.data import Data 
from torch_geometric.loader import DataLoader  
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter,add_self_loops,degree
from torch_geometric.data import Batch 
from einops import rearrange, repeat
from scipy.stats import pearsonr
import torch_geometric.utils
class MLPClassifier(nn.Module):
	def __init__(self, input_dim, hidden_dim1,hidden_dim2,output_dim):
		super(MLPClassifier, self).__init__()
		self.hidden_layers = nn.Sequential(
			nn.Linear(input_dim, hidden_dim1),
			nn.ReLU(),
			nn.Linear(hidden_dim1, hidden_dim2),
			nn.ReLU()
		)
		self.classifier = nn.Linear(hidden_dim2, output_dim)
	def forward(self, x):
		x = self.hidden_layers(x)
		x = self.classifier(x)
		return x
class ModuleSERO(nn.Module):
	def __init__(self, hidden_dim, input_dim, dropout=0.5, upscale=1):
		super().__init__()
		self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU()) 
		# self.attend = nn.Linear(round(upscale*hidden_dim), 1) 
		self.attend = nn.Linear(round(upscale*hidden_dim), input_dim) 
		self.dropout = nn.Dropout(dropout) 


	def forward(self, x, node_axis=-1):
		# assumes shape [... x node x ... x feature] x：294,90,32

		x_readout = x.mean(dim=-1)#x_readout：294，90
		x_shape = x_readout.shape#x_shape：294，90
		
		x_embed = self.embed(x.mean(dim=-1))#294，90
		# x_embed = self.embed(x_readout.reshape(-1,x_shape[-1])) #x_embed：294，32
		# x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1) #x_graphattention:294，90
		x_graphattention = torch.sigmoid(self.attend(x_embed)).squeeze(-1)  #294，90
		permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))#维度排列 (permute_idx)
		x_graphattention = x_graphattention.permute(permute_idx)  #x_graphattention:294，90
		# x_graphattention = self.dropout(x_graphattention) #294
		# x_weighted  = x * x_graphattention.unsqueeze(-1)
		# output = x_weighted.mean(dim=node_axis)#294,32
		# return(output,x_graphattention)
		return (x * self.dropout(x_graphattention.unsqueeze(-1))), x_graphattention
		# return (torch.matmul(rearrange(x, 'b n c -> b c n') , self.dropout(x_graphattention.unsqueeze(-1)))), x_graphattention
	# self.dropout(x_graphattention.unsqueeze(-1)):torch.Size([294, 64, 1]);x:torch.Size([294, 90, 64])  相乘结果为[294, 90, 64]
class ModuleGARO(nn.Module):
	def __init__(self, hidden_dim, dropout=0.3, upscale=1.0, **kwargs):
		super().__init__()
		self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
		self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
		self.dropout = nn.Dropout(dropout)


	def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
		x_q = self.embed_query(x.mean(dim=1, keepdims=True))
		x_k = self.embed_key(x)
		x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 'b n c -> b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
		return(torch.matmul(rearrange(x, 'b n c -> b c n'), rearrange(x_graphattention, 'b n c -> b c n'))).mean(-1),x_graphattention

		# return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(-1), x_graphattention
class GCN(nn.Module):
	# def __init__(self,features, adjs,hid_dim1,hid_dim2,hid_dim3,out_dim, drop, slope):
	def __init__(self,features0, features1,adjs,k, hid_dim1,hid_dim2,hid_dim3,out_dim, drop, slope):
		super(GCN, self).__init__()
		self.features0 = torch.FloatTensor(features0)
		self.features0 = nn.Parameter(self.features0, requires_grad=False)#82,90,90
		self.features1 = torch.FloatTensor(features1)
		self.features1 = nn.Parameter(self.features1, requires_grad=False)#82,90,30
		self.adjs = torch.FloatTensor(adjs)#82,90,90
		
		self.adjs = self.adj_process(self.adjs)
		# self.adjs = self.adj_process(self.adjs,k)
		
		self.adjs = nn.Parameter(self.adjs, requires_grad=False)#82,90,90
		_, _, self.dim_in = features0.shape
		self.gate_nn = nn.Linear(90*hid_dim2, 1)
		
		self.fc1 = GraphConvolution(self.dim_in, hid_dim1,self.adjs.size()[0])
		self.fc0 = GraphConvolution(30, hid_dim1,self.adjs.size()[0])
		#更改最后分类器
		self.fc2 = GraphConvolution(64,hid_dim2,self.adjs.size()[0])
		# self.fc2 = ChebConv(94,hid_dim2,3)
		# self.fc4 = GPRGNN(self.dim_in,hid_dim1,hid_dim1,'GPR_prop','PPR',0.5,0.5)
		self.fc3 = GraphConvolution(hid_dim2, hid_dim3,self.adjs.size()[0])
		self.fc4 = ChebConv(self.dim_in,hid_dim1,3)
		# self.fc4 = GINNet(self.dim_in, hid_dim1,self.adjs.size()[0])
		self.sero_layer = ModuleSERO(hidden_dim=90, input_dim=90, dropout=drop, upscale=1.0)
		self.garo_layer = ModuleGARO(hidden_dim=hid_dim2, dropout=drop, upscale=1.0)
		# self.fc1 = GraphSAGEConv(self.dim_in, hid_dim1,self.adjs.size()[0])
		
		# self.fc2 = GraphSAGEConv(hid_dim1, hid_dim2,self.adjs.size()[0])
		
		# self.fc3 = GraphSAGEConv(hid_dim2, hid_dim3,self.adjs.size()[0])
		# self.classifier1= GraphConvolution(hid_dim2, out_dim,self.adjs.size()[0])
		self.classifier2 = nn.Linear(hid_dim2*90, out_dim, bias=True)
		# self.classifier3 = MLPClassifier(hid_dim2, 128, 64,out_dim)
		# self.classifier2 = nn.Linear(hid_dim1*90*2, out_dim, bias=True)
		self.classifier3 = MLPClassifier(hid_dim2*90, 32, 16,out_dim)
		self.leaky_relu = nn.LeakyReLU(slope)
		self.dropout = nn.Dropout(drop)
		self.loss_function = nn.CrossEntropyLoss()
	def adj_to_edge_indices(self,adjs):  
		# 确保输入是三维张量  
		assert adjs.dim() == 3, "Input tensor must be 3D"  
		# 初始化边索引列表  
		edge_indices_list = []  
	
		# 遍历每个图的邻接矩阵  
		for i in range(adjs.size(0)):  
			adj_matrix = adjs[i]  # 提取第i个图的邻接矩阵  
	
			# 找到非零元素的位置（即边的存在）  
			rows, cols = torch.nonzero(adj_matrix, as_tuple=True)  
	
			# 对于无向图，只保留上半（或下半）三角矩阵中的边  
			# 注意：在PyTorch中，通常不需要这样做，因为你可以通过后续处理来避免重复  
			# 但如果你确实需要这样做，可以添加适当的条件判断  
	
			# 创建一个包含边索引的二维张量  
			edge_indices = torch.stack((rows, cols), dim=0)  
	
			# 将边索引添加到列表中  
			edge_indices_list.append(edge_indices)  
	
		return edge_indices_list 
	# def coor(adjs):
	# 	# 初始化一个用于存储结果的张量  
	# 	corr_matrix = torch.empty(294, 90, 90)  
	# 	# 遍历每个样本  
	# 	for i in range(adjs.size(0)):  
   	# 	# 获取当前样本的所有二维切片  
	# 		slices = adjs[i]  
    # 	# 遍历每个二维切片（除了自身）  
	# 		for j in range(slices.size(0)):  
	# 			for k in range(slices.size(0)):  
	# 				if j != k:  # 排除自身与自身的比较  
    #             # 计算两个二维切片之间的皮尔逊相关性  
    #             # 将torch张量转换为numpy数组进行计算  
	# 					corr, _ = pearsonr(slices[j].numpy().ravel(), slices[k].numpy().ravel())  
                  
    #             # 将结果存储在对应的位置  
	# 					corr_matrix[i, j, k] = corr  
	# 				else:  
    #             # 对于对角线元素，可以设置为1（完美相关性）或NaN（表示无意义）  
	# 					corr_matrix[i, j, k] = float('1.0')  # 或者 1.0，如果你想要表示完美相关性 
	# 	return corr_matrix
	def adj_process(self, adjs):
		net_num, n_num, n_num = adjs.shape  # 获取邻接矩阵的维度信息
		adjs = adjs.detach()  # 将输入的邻接矩阵转换为不需要梯度的形式
		for i in range(net_num):  # 遍历每个邻接矩阵
			adjs[i] += torch.eye(n_num)  # 对角线上加上自环
			adjs[i][adjs[i]>0.5] = 1.  # 邻接矩阵二值化处理
			adjs[i][adjs[i]<0.5] = 0.
			degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)  # 计算度矩阵
			degree_matrix = torch.pow(degree_matrix, -1/2)  # 度矩阵元素取倒数的平方
			degree_matrix[degree_matrix == float("inf")] = 0.  # 处理度为0的情况
			degree_matrix = torch.diag(degree_matrix)  # 转换为对角矩阵
			adjs[i] = torch.mm(degree_matrix, adjs[i])  # 左乘度矩阵
			adjs[i] = torch.mm(adjs[i], degree_matrix)  # 右乘度矩阵，完成对称归一化
		return adjs  # 返回处理后的邻接矩阵	

	# def adj_process(self, adjs,k):
	# 	net_num, n_num, n_num = adjs.shape  # 获取邻接矩阵的维度信息
	# 	adjs = adjs.detach()  # 将输入的邻接矩阵转换为不需要梯度的形式
	# 	for i in range(net_num):  # 遍历每个邻接矩阵
	# 		# 计算距离矩阵
	# 		dist_matrix = 1 - adjs[i]
	# 		# 构建KNN图
	# 		knn_graph = kneighbors_graph(dist_matrix, k+1, mode='connectivity', include_self=False)
	# 		# 将KNN图转换为邻接矩阵
	# 		adjs[i] = torch.from_numpy(knn_graph.toarray())
	# 		adjs[i] = (adjs[i] + adjs[i].t()) / 2
	# 		# 去除自连接
	# 		adjs[i].fill_diagonal_(0)
	# 		degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)  # 计算度矩阵
	# 		degree_matrix = torch.pow(degree_matrix, -1/2)  # 度矩阵元素取倒数的平方
	# 		degree_matrix[degree_matrix == float("inf")] = 0.  # 处理度为0的情况
	# 		degree_matrix = torch.diag(degree_matrix)  # 转换为对角矩阵
	# 		adjs[i] = torch.mm(degree_matrix, adjs[i])  # 左乘度矩阵
	# 		adjs[i] = torch.mm(adjs[i], degree_matrix)  # 右乘度矩阵，完成对称归一化
	# 	return adjs  # 返回处理后的邻接矩阵	
 
	# def adj_process(self, adjs,k):
	# 	net_num, n_num, n_num = adjs.shape
	# 	adjs = adjs.detach().clone()
	# 	for i in range(net_num):
	# 		dist_matrix = 1 - adjs[i]
	# 		tree = BallTree(dist_matrix)
	# 		_, ind = tree.query(dist_matrix, k=k+1)
	# 		knn_graph = np.zeros_like(dist_matrix)
	# 		for j in range(n_num):
	# 			knn_graph[j, ind[j, 1:]] = 1
	# 			knn_graph[ind[j, 1:], j] = 1
	# 		adjs[i] = torch.from_numpy(knn_graph)
	# 		adjs[i] = (adjs[i] + adjs[i].t()) / 2
	# 		adjs[i].fill_diagonal_(0)
	# 		degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)  # 计算度矩阵
	# 		degree_matrix = torch.pow(degree_matrix, -1/2)  # 度矩阵元素取倒数的平方
	# 		degree_matrix[degree_matrix == float("inf")] = 0.  # 处理度为0的情况
	# 		degree_matrix = torch.diag(degree_matrix)  # 转换为对角矩阵
	# 		adjs[i] = torch.mm(degree_matrix, adjs[i])  # 左乘度矩阵
	# 		adjs[i] = torch.mm(adjs[i], degree_matrix)  # 右乘度矩阵，完成对称归一化
	# 	return adjs
	def forward(self, input):
		action, index = input
		predict_array = np.empty((len(index), 18))
		features0 = self.features0[index]#294,90.90fmri的初始特征
		features1 = self.features1[index]#294,90.30smri的初始特征
		adj = self.adjs[index]#fgraph
		
		features0 = self.fc4(features0,adj)#对FC进行卷积
		features0 = self.leaky_relu(features0)
		features0 = self.dropout(features0)#换成功能以后得到一个卷积层后的功能特征，后续将该功能特征与结构特征融合进而输入到下一个卷积层294,90,64
        #将卷积得到的两个特征进行拼接
		# features_all = torch.cat((features0, features1),axis=-1)#294,90,94(64+30)
		if action == 1:
			features = self.fc2(features0,adj)
			features = self.leaky_relu(features)
			features = self.dropout(features)#294，90，32
		
			if len(features.shape) < 3:
			
				predict_array.append(predict)
			else:
				
				predict = np.reshape(features.detach().numpy(), (len(index), -1))

				predict= torch.Tensor(predict)
            	# individual_names = ['individual_' + str(idx) for idx in index]  # 假设个体名称为'individual_' + 索引
				# for i, name in enumerate(individual_names):
				# 	file_path = os.path.join('./result/', name + '.npy')
				# 	np.save(file_path, predict[i])
				# predict = torch.as_tensor(predict, dtype=torch.float32)

				# predict = features.view(-1)
				# predict = torch.mean(features, dim=1)
				
				# np.save('./result/'+ str(len(index))+'.npy', predict.detach().numpy())
				# predict_array.append(predict)
				# Standardization 标准化
			predict1 = self.classifier2(predict)
            # # features2 = predict1
			predict1 = F.log_softmax(predict1, dim=1)#二维数组
		return features,predict1
	
# class GlobalMeanPool(torch.nn.Module):  
#     def forward(self, x, batch):  
#         return torch_scatter.scatter_mean(x, batch, dim=0)
##新GCN
# class GCN(torch.nn.Module):

# 	def __init__(self,features, adjs,k, hid_dim1,hid_dim2,hid_dim3,out_dim, drop, slope):
# 		super(GCN, self).__init__()
# 		self.num_graphs, self.num_nodes, self.num_features = features.shape
# 		self.features = torch.FloatTensor(features)
# 		self.adjs = torch.FloatTensor(adjs)#82,90,90
# 		self.processed_adjs = self.adj_process(self.adjs,k)  # 预处理邻接矩阵
# 		#GAT
# 		self.gat1 = GATConv(self.num_features, hid_dim1, heads=4, concat=False)
# 		self.gat2 = GATConv(hid_dim1, hid_dim2, heads=1, concat=False)


# 		# self.adjs = nn.Parameter(self.adjs, requires_grad=False)#82,90,90
# 		self.sage1 = SAGEConv(self.num_features, hid_dim1)  # 定义两层GraphSAGE层
# 		self.sage2 = SAGEConv(hid_dim1, hid_dim2)
# 		self.sage3 = SAGEConv(hid_dim2, hid_dim3)
# 		self.leaky_relu = nn.LeakyReLU(slope)
# 		self.dropout = nn.Dropout(drop)
# 		self.loss_function = nn.CrossEntropyLoss()
# 		self.classifier2 = nn.Linear(hid_dim2*self.num_nodes , out_dim, bias=True)
# 		self.classifier3 = MLPClassifier(hid_dim2*self.num_nodes, 128, out_dim)
# 		# self.classifier2 = nn.Linear(hid_dim2 , out_dim, bias=True)
# 		# self.classifier3 = MLPClassifier(hid_dim2, 128, out_dim)
# 	def adj_process(self, adjs,k):
# 		net_num, n_num, n_num = adjs.shape
# 		adjs = adjs.detach().clone()
# 		for i in range(net_num):
# 			dist_matrix = 1 - adjs[i]
# 			tree = BallTree(dist_matrix)
# 			_, ind = tree.query(dist_matrix, k=k+1)
# 			knn_graph = np.zeros_like(dist_matrix)
# 			for j in range(n_num):
# 				knn_graph[j, ind[j, 1:]] = 1
# 				knn_graph[ind[j, 1:], j] = 1
# 			adjs[i] = torch.from_numpy(knn_graph)
# 			adjs[i] = (adjs[i] + adjs[i].t()) / 2
# 			adjs[i].fill_diagonal_(0)
# 			# degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)  # 计算度矩阵
# 			# degree_matrix = torch.pow(degree_matrix, -1/2)  # 度矩阵元素取倒数的平方
# 			# degree_matrix[degree_matrix == float("inf")] = 0.  # 处理度为0的情况
# 			# degree_matrix = torch.diag(degree_matrix)  # 转换为对角矩阵
# 			# adjs[i] = torch.mm(degree_matrix, adjs[i])  # 左乘度矩阵
# 			# adjs[i] = torch.mm(adjs[i], degree_matrix)  # 右乘度矩阵，完成对称归一化
# 		return adjs
# 	# def adj_process(self, adjs):
# 	# 	net_num, n_num, n_num = adjs.shape  # 获取邻接矩阵的维度信息
# 	# 	adjs = adjs.detach()  # 将输入的邻接矩阵转换为不需要梯度的形式
# 	# 	for i in range(net_num):  # 遍历每个邻接矩阵
# 	# 		adjs[i] += torch.eye(n_num)  # 对角线上加上自环
# 	# 		adjs[i][adjs[i]>0.5] = 1.  # 邻接矩阵二值化处理
# 	# 		adjs[i][adjs[i]<0.5] = 0.
# 			# degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)  # 计算度矩阵
# 			# degree_matrix = torch.pow(degree_matrix, -1/2)  # 度矩阵元素取倒数的平方
# 			# degree_matrix[degree_matrix == float("inf")] = 0.  # 处理度为0的情况
# 			# degree_matrix = torch.diag(degree_matrix)  # 转换为对角矩阵
# 			# adjs[i] = torch.mm(degree_matrix, adjs[i])  # 左乘度矩阵
# 			# adjs[i] = torch.mm(adjs[i], degree_matrix)  # 右乘度矩阵，完成对称归一化
# 		# return adjs  # 返回处理后的邻接矩阵	
# 	def forward(self, input):
# 		action, index = input
# 		features = self.features[index]
# 		adj = self.processed_adjs[index]
# 		all_row = []  
# 		all_col = [] 
# 		net_num, n_num, n_num = adj.shape
# 		num_node_features = 30
# 		features_2d = features.view(-1, num_node_features)  # 形状变为 (num_subjects * num_nodes, num_node_features) 
# 		edge_index_list = [] 
# 		for i in range(net_num): 
# 			adj_i = adj[i]
# 			row, col = torch.nonzero(adj_i, as_tuple=False).t()
# 			edge_index = torch.stack([row, col], dim=0)
# 			edge_index_list.append(edge_index)  
# 		data_list = []
# 		for i in range(net_num):
# 			node_features = features_2d[i * n_num:(i + 1) * n_num] 
# 			edge_index = edge_index_list[i]
# 			data = Data(x=node_features, edge_index=edge_index)
# 			data_list.append(data)
# 		# edge_index = torch.stack([row, col], dim=0)
# 		num_edges = edge_index.size(1)  
# 		edge_attr = torch.ones(num_edges, 1)
# 		predictions = [] 
# 		graph_representations = [] 
# 		#将batch_size改为1
# 		# data_loader = DataLoader(data_list, batch_size=1, shuffle=False)  
# 		data_loader = DataLoader(data_list, batch_size=net_num, shuffle=False) 
# 		for i in range(net_num): 
# 			 # 获取当前图的特征和邻接矩阵  
# 			node_features = features_2d[i * n_num:(i + 1) * n_num] 
# 			adj_i = adj[i]
# 			# 获取边的索引 
# 			row, col = torch.nonzero(adj_i, as_tuple=False).t()  
# 			edge_index = torch.stack([row, col], dim=0)
# 			edge_attr = torch.ones(edge_index.size(1), 1)
# 			data = Data(x=node_features, edge_index=edge_index)
# 			x = self.sage1(data.x, data.edge_index)
# 			# data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
# 			# x = self.gat1(data.x, data.edge_index, data.edge_attr)
# 			x = self.leaky_relu(x)
# 			x = self.dropout(x)

# 			if action == 1:
# 				x = self.sage2(x,edge_index)
# 				# x = self.gat2(x,data.edge_index, data.edge_attr)
# 				features0 = x
# 				x = self.leaky_relu(x)
# 				x = self.dropout(x)
# 				# 最大汇总
# 				# x = torch.max(x, dim=0).values
# 				# 加权平均汇总
# 				# weights = torch.tensor([0.5, 0.5, ...])  # 按节点重要性定义权重
# 				# graph_representation = torch.matmul(nodes.T, weights)
# 				# 拉直汇总
# 				graph_representations.append(x.view(-1))
# 				# 平均汇总
# 				# x = scatter(x, data.batch, dim=0,reduce='mean')
# 				# graph_representations.append(x)
# 				# x = torch.tensor(np.reshape(x.detach().numpy(), (len(index), -1)))
# 				# x = self.classifier2(x)
# 				predictions.append(x)
# 		# 聚合所有图的节点表示到图级别的表示
# 		graph_level_representations = torch.stack(graph_representations, dim=0)  
#       	# 将图级别的表示输入到分类器中  
# 		graph_predictions = self.classifier2(graph_level_representations)
#   		# 遍历 DataLoader 中的每个批次（在这种情况下，每个批次是一个图） 
# 		return features0,F.log_softmax(graph_predictions, dim=1)
		# for batch in data_loader: 
		# 	x, edge_index = batch.x, batch.edge_index  
		# 	edge_attr = torch.ones(edge_index.size(1), 1)  # 为当前图生成 edge_attr 
		# 	x = self.gat1(x,edge_index,edge_attr)
			
		# 	# x = self.sage1(x,edge_index)
		# 	if action == 1:
		# 		# x = self.sage2(x,edge_index)
		# 		x = self.gat2(x,edge_index,edge_attr)
		# 		features0 = x
		# 		x = self.leaky_relu(x)
		# 		x = self.dropout(x)
		# 		# x = scatter(x, data.batch, dim=0,reduce='mean')
		# 		# graph_representations.append(x)
		# 		x = torch.tensor(np.reshape(x.detach().numpy(), (len(index), -1)))
		# 		x = self.classifier2(x)
		# 		predictions.append(x)
		# 	elif action == 2:
		# 		x = self.sage2(x,edge_index)
		# 		features0 = x
		# 		x = self.leaky_relu(x)
		# 		x = self.dropout(x)
		# 		x = self.sage3(x,edge_index)
		# 		x = self.leaky_relu(x)
		# 		x = self.dropout(x)
		# 		# x = torch.mean(x, dim=1)
		# 		x = torch.tensor(np.reshape(x.detach().numpy(), (len(index), -1)))
		# 		x = self.classifier2(x)
		# 		predictions.append(x)
		# # graph_representations = torch.stack(graph_representations, dim=0)  
        # # 应用图分类器得到最终的预测  
		# # predictions = self.classifier2(graph_representations)  
		# predictions = torch.cat(predictions, dim=0)
		# # x = self.gat1(features,edge_index,edge_attr)
		# # # x = self.sage1(features,edge_index)#乘了以后features变成NAN
		# # x = self.leaky_relu(x)
		# # x = self.dropout(x)
		# # if action == 1:
		# # 	# x = self.sage2(x,edge_index)
		# # 	x = self.gat2(x,edge_index,edge_attr)
		# # 	features0 = x
		# # 	x = self.leaky_relu(x)
		# # 	x = self.dropout(x)
		# # x = torch.tensor(np.reshape(x.detach().numpy(), (len(index), -1)))
		# # # x = x.mean(dim=1)  
		# # # x = self.classifier2(x) 
		# # x = self.classifier3(x) 
		# 	# predict = features.detach().numpy().reshape(-1, features.shape[-1])
		# # predict= torch.Tensor(predict)
		# # predict1 = self.classifier2(predict)
		# # predict1 = F.log_softmax(predict1, dim=1)#二维数组
  
		# return features0,F.log_softmax(predictions, dim=1)
class gnn_env(object):
	# def __init__(self, dataset, view, max_layer,hid_dim1, hid_dim2,hid_dim3,out_dim, drop, slope, lr, weight_decay, gnn_type, device, policy, benchmark_num,repeat):
	def __init__(self, dataset, view, max_layer, k, hid_dim1, hid_dim2,hid_dim3,out_dim, drop, slope, lr, weight_decay, gnn_type, device, policy, benchmark_num,repeat):
		self.dataset = dataset
		self.view = view
		self.max_layer = max_layer
		self.k = k
		self.action_num = max_layer
		self.device = device
		self.policy = policy
		self.benchmark_num = benchmark_num
		self.repeat = repeat
		# self.train_idx_list = train_idx_list
		# self.test_idx_list = test_idx_list
		self.load_dataset()
		if gnn_type == 'GCN':
			# self.model = GCN(self.init_net_feat, self.net_brain_adj,  hid_dim1, hid_dim2,hid_dim3, out_dim, drop, slope).to(device)
			self.model = GCN(self.init_net_feat0, self.init_net_feat1,self.net_brain_adj, k, hid_dim1, hid_dim2,hid_dim3, out_dim, drop, slope).to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
		self.batch_size_qdn = self.num_train
		self.state_shape = self.reset().shape#到reset，返回state维度为90
		self.past_performance = [0.]
	# def stratified_sampling(self, X, y):
	# 	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	# 	train_idx, valid_idx = next(sss.split(X, y))
	# 	return train_idx, valid_idx

	# def stratified_sampling(self,X, y):
	# 	train_valid_idx = list(range(0,len(X)-18))
	# 	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	# 	train_idx, valid_idx = next(sss.split(X[train_valid_idx], y[train_valid_idx]))
	# 	return train_idx, valid_idx

	def load_dataset(self):
		self.dataset == 'ADNI'
		ratio = [259, 0, 30]
		# ratio = [294, 0, 34]
		# ratio = [279, 0, 49]
		# ratio = [50,15,17]
		net_label, net_brain_adj,net_weighted0,net_weighted1,net_subject_adj = self.load_ADNI()
		self.num_net = len(net_label)
		self.net_label = net_label
		all_idx = [i for i in range(self.num_net)]
		# test_idx = list(range(len(all_idx)-18, len(all_idx)))
		# train_valid_idx= list(range(0, 102))
		# for idx in test_idx:
		# 	all_idx.remove(idx)
		# random.shuffle(all_idx)
		# train_idx =  self.train_idx_list[self.repeat]
		train_valid_idx = all_idx[:ratio[0] + ratio[1]]	
		test_idx = all_idx[ratio[0] + ratio[1]:]
		# random.shuffle(train_valid_idx)
		train_idx = train_valid_idx[:ratio[0]]
		val_idx = train_valid_idx[ratio[0]:ratio[0] + ratio[1]]
		# train_idx, val_idx= self.stratified_sampling(train_valid_idx, net_label[0:102,1])

		self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
		self.num_train, self.num_val, self.num_test = len(self.train_idx), len(self.val_idx), len(self.test_idx)
		self.init_net_feat0 = net_weighted0
		self.init_net_feat1 = net_weighted1
		self.net_brain_adj = net_brain_adj
		self.transition_adj = []
		self.transition_adj.append(net_subject_adj)
		tmp_subject_adj = net_subject_adj
		for _ in range(1, self.action_num):
			net_subject_adj = np.matmul(net_subject_adj, tmp_subject_adj)
			net_subject_adj[net_subject_adj>0.5] = 1
			self.transition_adj.append(net_subject_adj)

	def load_ADNI(self):
		# net_label = np.load('./Data/'+'npy_MCI_NC_328/' + 'net_label.npy',allow_pickle=True)
		# net_brain_adj = np.load('./Data/'+'npy_MCI_NC_328/'+ 'net_brain_adj_MCI_NC.npy',allow_pickle=True)
		# net_weighted = np.load('./Data/'+'npy_MCI_NC_328/'+ 'net_brain_adj_MCI_NC.npy',allow_pickle=True)
		# # net_weighted = np.load('./Data/'+'npy_MCI_NC_194/'+ 'net_wighted_MCI_NC.npy',allow_pickle=True)
		# net_subject_adj = np.load('./Data/'+'npy_MCI_NC_328/'+ 'net_subject_adj_MCI_NC.npy',allow_pickle=True)
		net_label = np.load('./Data/'+'AD_NC_K3/' + 'net_label.npy',allow_pickle=True)
		net_brain_adj = np.load('./Data/'+'AD_NC_K3/'+ 'net_brain_adj.npy',allow_pickle=True)
		net_weighted0 = np.load('./Data/'+'AD_NC_K3/'+ 'net_brain_adj.npy',allow_pickle=True)
		net_weighted1 = np.load('./Data/'+'AD_NC_K3/'+ 'net_wighted.npy',allow_pickle=True)
		net_subject_adj = np.load('./Data/'+'AD_NC_K3/'+ 'net_subject_adj.npy',allow_pickle=True)
		# net_label = np.load('./Data/'+'MCI_NC_K10/' + 'net_label.npy',allow_pickle=True)
		# net_brain_adj = np.load('./Data/'+'MCI_NC_K10/'+ 'net_brain_adj.npy',allow_pickle=True)
		# net_weighted0 = np.load('./Data/'+'MCI_NC_K10/'+ 'net_brain_adj.npy',allow_pickle=True)
		# net_weighted1 = np.load('./Data/'+'MCI_NC_K10/'+ 'net_wighted.npy',allow_pickle=True)
		# net_subject_adj = np.load('./Data/'+'MCI_NC_K10/'+ 'net_subject_adj.npy',allow_pickle=True)
		return net_label, net_brain_adj,net_weighted0, net_weighted1,net_subject_adj

	def reset(self):
		state = np.mean(self.net_brain_adj,axis=1)
		state = state[self.train_idx[0]]
		self.optimizer.zero_grad()
		return state
	def transition(self, action, index):
		neighbors = np.nonzero(self.transition_adj[action][index])[0]
		legal_neighbors = np.array(self.train_idx)
		neighbors = np.intersect1d(neighbors,legal_neighbors)
		next_index = np.random.choice(neighbors,1)[0]
		next_state = np.mean(self.net_brain_adj, axis=1)
		next_state = next_state[next_index]
		return next_state, next_index
	def step(self, action, index):
		self.model.train()
		self.optimizer.zero_grad()
		self.train(action, index)
		next_state, next_index = self.transition(action, index)
		val_acc = self.eval()
		benchmark = np.mean(np.array(self.past_performance[-self.benchmark_num:]))
		self.past_performance.append(val_acc)
		reward = val_acc - benchmark
		return next_state, next_index, reward, val_acc
	def train(self, act, index):
		self.model.train()
		pred = self.model((act, index))
		label = np.array([self.net_label[index][1]], dtype=np.float64)
		label = torch.LongTensor(label).to(self.device)
		F.nll_loss(pred, label).backward()
		self.optimizer.step()
	def eval(self):
		self.model.eval()
		batch_dict = {}
		val_indexes = self.val_idx
		val_states = np.mean(self.net_brain_adj, axis=1)
		val_states = val_states[val_indexes]
		val_actions = self.policy.eval_step(val_states)
		for act, idx in zip(val_actions, val_indexes):
			if act not in batch_dict.keys():
				batch_dict[act] = []
			batch_dict[act].append(idx)
		val_acc = 0.
		for act in batch_dict.keys():
			indexes = batch_dict[act]
			if len(indexes) > 0:
				preds = self.model((act, indexes))#二维数组
				preds = preds.max(1)[1]#变为标签
				labels = torch.LongTensor(np.array(self.net_label[indexes][:,1], dtype=np.float64)).to(self.device)
				val_acc += preds.eq(labels).sum().item()
		return val_acc/len(val_indexes)
