import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.cross_decomposition import CCA
import sys,os
import torch.nn.init as init
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import kneighbors_graph
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
from pygcn.layers import GraphConvolution
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x
class GCN(nn.Module):
	def __init__(self,features, adjs, net_smri, hid_dim1,hid_dim2,hid_dim3,out_dim, drop, slope):
	# def __init__(self,features, adjs,k, hid_dim1,hid_dim2,hid_dim3,out_dim, drop, slope):
		super(GCN, self).__init__()
		self.features = torch.FloatTensor(features)
		self.features = nn.Parameter(self.features, requires_grad=False)#82,90,30
		self.adjs = torch.FloatTensor(adjs)#82,90,90
		# self.adjs = self.adj_process(self.adjs)
		# self.adjs = self.adj_process(self.adjs,k)
		self.adjs = nn.Parameter(self.adjs, requires_grad=False)#82,90,90
		self.net_smri = torch.FloatTensor(net_smri)
		self.net_smri = nn.Parameter(self.net_smri, requires_grad=False)#82,90,30
		_, _, self.dim_in = features.shape
		self.fc1 = GraphConvolution(self.dim_in, hid_dim1,self.adjs.size()[0])
		
		self.fc2 = GraphConvolution(hid_dim1, hid_dim2,self.adjs.size()[0])
		
		self.fc3 = GraphConvolution(hid_dim2, hid_dim3,self.adjs.size()[0])
		
		self.classifier1= GraphConvolution(hid_dim1, out_dim,self.adjs.size()[0])
		self.classifier2 = nn.Linear(22, out_dim, bias=True)
		self.classifier3 = MLPClassifier(22, 16, out_dim)
		self.leaky_relu = nn.LeakyReLU(slope)
		self.dropout = nn.Dropout(drop)
		self.loss_function = nn.CrossEntropyLoss()

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
	# 		# 将距离矩阵转换为向量
	# 		# dist_vector = squareform(dist_matrix)
	# 		# dist_vector = dist_vector.reshape(-1, 1)
	# 		# 构建KNN图
	# 		knn_graph = kneighbors_graph(dist_matrix, k+1, mode='connectivity', include_self=False)
	# 		# 将KNN图转换为邻接矩阵
	# 		adjs[i] = torch.from_numpy(knn_graph.toarray())

	# 		# adjs[i] = knn_graph.toarray()
	# 		# 对称化处理
	# 		adjs[i] = (adjs[i] + adjs[i].t()) / 2
	# 		# 去除自连接
	# 		adjs[i].fill_diagonal_(0)
	# 		# np.fill_diagonal(adjs[i], 0)
	# # 		adjs[i] += torch.eye(n_num)  # 对角线上加上自环
	# # 		adjs[i][adjs[i]>0.5] = 1.  # 邻接矩阵二值化处理
	# # 		adjs[i][adjs[i]<0.5] = 0.
	# # 		degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)  # 计算度矩阵
	# # 		degree_matrix = torch.pow(degree_matrix, -1/2)  # 度矩阵元素取倒数的平方
	# # 		degree_matrix[degree_matrix == float("inf")] = 0.  # 处理度为0的情况
	# # 		degree_matrix = torch.diag(degree_matrix)  # 转换为对角矩阵
	# # 		adjs[i] = torch.mm(degree_matrix, adjs[i])  # 左乘度矩阵
	# # 		adjs[i] = torch.mm(adjs[i], degree_matrix)  # 右乘度矩阵，完成对称归一化
	# 	return adjs  # 返回处理后的邻接矩阵	


	# def forward(self, x, adj):
	# 	x = F.relu(self.gc1(x, adj))
	# 	x = F.dropout(x, self.dropout, training=self.training)
	# 	x = self.gc2(x, adj)
	# 	return F.log_softmax(x, dim=1)

	def forward(self, input):
		action, index = input
		predict_array = np.empty((len(index), 18))
		features = self.features[index]
		# sMRI = np.mean(self.net_smri[index], axis=1)
		sMRI = torch.mean(self.net_smri[index], dim=1)
		adj = self.adjs[index]
		features = self.fc1(features,adj)#乘了以后features变成NAN
		features = self.leaky_relu(features)
		features = self.dropout(features)
		if action == 1:
			features = self.fc2(features,adj)
			features = self.leaky_relu(features)
			features = self.dropout(features)
		elif action == 2:
			features = self.fc2(features,adj)
			features = self. leaky_relu(features)
			features = self.dropout(features)
			features = self.fc3(features,adj)
			features = self.leaky_relu(features)
			features = self.dropout(features)
		if len(features.shape) < 3:
			predict1 = torch.mean(features, dim=1)
			kcca = CCA(n_components=11)
			kcca.fit(predict1, sMRI)
			predict1_new, sMRI_new = kcca.transform(predict1, sMRI)
			fusion = np.hstack((predict1_new, sMRI_new))
		else:
			# predict = self.classifier( features,adj)
			predict1 = torch.mean(features, dim=1)
			predict1 = predict1.detach()
			kcca = CCA(n_components=11)
			kcca.fit(predict1, sMRI)
			predict1_new, sMRI_new = kcca.transform(predict1, sMRI)
			fusion = np.hstack((predict1_new, sMRI_new))
			# np.save('./result/'+ str(len(index))+'.npy', predict.detach().numpy())
			# predict_array.append(predict)
			# self.features = torch.FloatTensor(features)
			fusion_tensor = torch.FloatTensor(fusion)
		predict = self.classifier3(fusion_tensor)
		predict = F.log_softmax(predict, dim=1)#二维数组
		return predict

class gnn_env(object):
	def __init__(self, dataset, view, max_layer,hid_dim1, hid_dim2,hid_dim3,out_dim, drop, slope, lr, weight_decay, gnn_type, device, policy, benchmark_num):
	# def __init__(self, dataset, view, max_layer,k,hid_dim1, hid_dim2,hid_dim3,out_dim, drop, slope, lr, weight_decay, gnn_type, device, policy, benchmark_num):
		self.dataset = dataset
		self.view = view
		self.max_layer = max_layer
		# self.k = k
		self.action_num = max_layer
		self.device = device
		self.policy = policy
		self.benchmark_num = benchmark_num
		self.load_dataset()
		if gnn_type == 'GCN':
			self.model = GCN(self.init_net_feat, self.net_brain_adj, self.net_smri, hid_dim1, hid_dim2,hid_dim3, out_dim, drop, slope).to(device)
			# self.model = GCN(self.init_net_feat, self.net_brain_adj, k, hid_dim1, hid_dim2,hid_dim3, out_dim, drop, slope).to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
		self.batch_size_qdn = self.num_train
		self.state_shape = self.reset().shape#到reset，返回state维度为90
		self.past_performance = [0.]
	def load_dataset(self):
		self.dataset == 'ADNI'
		ratio = [60, 12, 11]
		#ratio = [50,15,17]
		net_label, net_brain_adj,net_weighted, net_subject_adj ,net_smri = self.load_ADNI()
		self.num_net = len(net_label)
		self.net_label = net_label
		self.net_smri = net_smri
		all_idx = [i for i in range(self.num_net)]
		# random.shuffle(all_idx)
		train_valid_idx = all_idx[:ratio[0] + ratio[1]]	
		test_idx = all_idx[ratio[0] + ratio[1]:]
		random.shuffle(train_valid_idx)
		train_idx = train_valid_idx[:ratio[0]]
		val_idx = train_valid_idx[ratio[0]:ratio[0] + ratio[1]]
		self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
		self.num_train, self.num_val, self.num_test = len(self.train_idx), len(self.val_idx), len(self.test_idx)
		self.init_net_feat = net_weighted
		self.net_brain_adj = net_brain_adj
		self.transition_adj = []
		self.transition_adj.append(net_subject_adj)
		tmp_subject_adj = net_subject_adj
		for _ in range(1, self.action_num):
			net_subject_adj = np.matmul(net_subject_adj, tmp_subject_adj)
			net_subject_adj[net_subject_adj>0.5] = 1
			self.transition_adj.append(net_subject_adj)

	def load_ADNI(self):
		net_label = np.load('./Data/'+'npy_MCI_NC_train/' + 'net_label_MCI_NC.npy',allow_pickle=True)
		net_brain_adj = np.load('./Data/'+'npy_MCI_NC_train/'+ 'net_brain_adj_MCI_NC.npy',allow_pickle=True)
		net_weighted = np.load('./Data/'+'npy_MCI_NC_train/'+ 'net_brain_adj_MCI_NC.npy',allow_pickle=True)
		net_subject_adj = np.load('./Data/'+'npy_MCI_NC_train/'+ 'net_subject_adj_MCI_NC.npy',allow_pickle=True)
		net_smri = np.load('./Data/'+'npy_MCI_NC_train/'+ 'net_wighted_MCI_NC.npy',allow_pickle=True)
		return net_label, net_brain_adj,net_weighted, net_subject_adj, net_smri

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
