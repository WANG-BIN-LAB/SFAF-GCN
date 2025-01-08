import torch
import argparse
# from model.GAT import gnn_env, GCN
from model.gnn import gnn_env, GCN
from model.dqn import QAgent
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_curve,auc
import random
from scipy import interp
import os
from matplotlib import pyplot as plt
parser = argparse.ArgumentParser(description='SFAF-GCN')
parser.add_argument('--dataset', type=str, default="ADNI")
parser.add_argument('--view', type=str, default="DTI")
parser.add_argument('--action_num', type=int, default=2)
parser.add_argument('--hid_dim1', type=int, default=64)
parser.add_argument('--hid_dim2', type=int, default=32)
parser.add_argument('--hid_dim3', type=int, default=16)
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--slope', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.005) 
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--gnn_type', type=str, default='GCN')
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--max_timesteps', type=int, default=2000)
parser.add_argument('--epoch_num', type=int, default=1000)
parser.add_argument('--discount_factor', type=float, default=0.95)
parser.add_argument('--epsilon_start', type=float, default=1.0) 
parser.add_argument('--epsilon_end', type=float, default=0.05)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--epsilon_decay_steps', type=int, default=20)
parser.add_argument('--benchmark_num', type=int, default=20)#20一个历史记录窗口的大小，以此来评估模型性能的趋势。
parser.add_argument('--replay_memory_size', type=int, default=10000)
parser.add_argument('--replay_memory_init_size', type=int, default=500)
parser.add_argument('--memory_batch_size', type=int, default=20)
parser.add_argument('--update_target_estimator_every', type=int, default=1)
parser.add_argument('--norm_step', type=int, default=100)
parser.add_argument('--mlp_layers', type=list, default=[128, 64, 32])
args = parser.parse_args()
#args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = torch.device('cpu')
def main():
	acc_records = []
	acc_records1 = []
	auc_records = []
	specificity_list = []
	sensitivity_list = []
	print("begin")

	for repeat in range(args.repeats):
		seed =42# 设定随机数种子
		seed = int(seed)
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.enabled = False
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%初始化env和agent
#   k=args.k,
		env = gnn_env(dataset = args.dataset,
					  view = args.view,
					  max_layer = args.action_num,
					  k=args.k,
					  hid_dim1 = args.hid_dim1,
					  hid_dim2 = args.hid_dim2,
					  hid_dim3 = args.hid_dim3,
					  out_dim = args.out_dim,
					  drop = args.dropout,
					  slope = args.slope,
					  lr = args.lr,
					  weight_decay = args.weight_decay,
					  gnn_type = args.gnn_type,
					  device = args.device,
					  policy = "",
					  benchmark_num = args.benchmark_num,
					  repeat = repeat,
					  )
		agent = QAgent(replay_memory_size = args.replay_memory_size,
					   replay_memory_init_size = args.replay_memory_init_size,
					   update_target_estimator_every = args.update_target_estimator_every,
					   discount_factor = args.discount_factor,
					   epsilon_start = args.epsilon_start,
					   epsilon_end = args.epsilon_end,
					   epsilon_decay_steps = args.epsilon_decay_steps,
					   lr=args.lr,
					   batch_size=args.memory_batch_size,
					   num_net = env.num_net,#82
					   action_num=env.action_num,#3
					   norm_step=args.norm_step,
					   mlp_layers=args.mlp_layers,
					   state_shape=env.state_shape,#82,90
					   device=args.device)
		env.policy = agent
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%初始化GNN模型和优化器
		# GNN2 = GCN(env.init_net_feat, env.net_brain_adj,  args.hid_dim1,args.hid_dim2,args.hid_dim3,args.out_dim, args.dropout, args.slope).to(args.device)
		# GNN2 = GCN(env.init_net_feat, env.net_brain_adj, env.net_smri, args.hid_dim1,args.hid_dim2,args.hid_dim3,args.out_dim, args.dropout, args.slope).to(args.device)
		GNN2 = GCN(env.init_net_feat0, env.init_net_feat1,env.net_brain_adj,env.k,args.hid_dim1,args.hid_dim2,args.hid_dim3,args.out_dim, args.dropout, args.slope).to(args.device)
		# GNN2 = GCN(env.init_net_feat, env.init_net_feat, args.hid_dim, args.out_dim, args.dropout, args.slope).to(args.device)
		GNN_lr = 0.005
		GNN_wd = 0.001
		# torch.nn.utils.clip_grad_norm_(GNN2.parameters(), max_norm)
		Optimizer = torch.optim.Adam(GNN2.parameters(), lr=GNN_lr, weight_decay=GNN_wd)
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%准备数据，将网络结构数据经过处理，得到训练集，验证集和测试集的节点状态以及他们对应的动作。
		train_gnn_buffer = defaultdict(list)
		val_gnn_buffer = defaultdict(list)
		test_gnn_buffer = defaultdict(list)
		acc_s = []  # 存储每个 epoch 的准确率
		loss_s = []  # 存储每个 epoch 的损失值
		states = np.mean(env.net_brain_adj, axis=1)#net_brain_adj(82,90,90) states(82,90)
		train_states = states[env.train_idx]#train_idx list =[1,2,3,4,...] train_states(train_num,90)
		val_states = states[env.val_idx]#val_states(val_num,90)
		test_states = states[env.test_idx]#test_states(test_num,90)
		train_actions = [1] *259
		val_actions = [1]* 0
		test_actions= [1]* 30
		# train_actions = [1] *294
		# val_actions = [1]* 0
		# test_actions= [1]* 34
		for act, idx in zip(train_actions, env.train_idx):
			train_gnn_buffer[act].append(idx)
		for act, idx in zip(val_actions, env.val_idx):
			val_gnn_buffer[act].append(idx)
		for act, idx in zip(test_actions, env.test_idx):
			test_gnn_buffer[act].append(idx)
		max_train_acc = 0.
		max_val_acc = 0.
		max_test_acc =0.
		max_test_auc =0.
		
		for epoch in range(0, args.epoch_num):
			train_loss = 0.
			val_loss = 0.
			test_loss = 0.
			train_pred_list = []
			train_true_list = []
			val_pred_list = []
			val_true_list = []
			test_pred_list = []
			test_true_list = []
			test_prob_list = []
			GNN2.train()
			#对每个动作进行训练，遍历所有动作
			for act in range(args.action_num):
				indexes = train_gnn_buffer[act]
				if len(indexes) > 1:
					predict,preds = GNN2((act, indexes))
					labels = np.array(np.  array(env.net_label[indexes][:,1], dtype=np.float64))
					#np.array(self.net_label[indexes][:,1], dtype=np.float64)
					#label = np.array([self.net_label[index][1]], dtype=np.float64)
					labels = torch.LongTensor(labels).to(args.device)
					loss = F.nll_loss(preds, labels)
					train_loss += loss
					preds = preds.max(1)[1]
					train_pred_list.extend(preds.to('cpu').numpy())
					train_true_list.extend(labels.to('cpu').numpy())
					Optimizer.zero_grad()  # 清除梯度
					loss.backward()
					torch.nn.utils.clip_grad_norm_(GNN2.parameters(), 1)  #梯度裁剪
					Optimizer.step()
					
			GNN2.eval()
			with torch.no_grad():
				for act in range(args.action_num):
					indexes = val_gnn_buffer[act]
					if len(indexes) > 1:
						predict,preds = GNN2((act, indexes))
						#labels = np.array(env.net_label[indexes])
						labels = np.array(np.array(env.net_label[indexes][:,1], dtype=np.float64))
						labels = torch.LongTensor(labels).to(args.device)
						loss = F.nll_loss(preds, labels)
						val_loss += loss
						preds = preds.max(1)[1]
						val_pred_list.extend(preds.to('cpu').numpy())
						val_true_list.extend(labels.to('cpu').numpy())
			GNN2.eval()
			with torch.no_grad():
				for act in range(args.action_num):
					indexes = test_gnn_buffer[act]
					if len(indexes) > 1:
						predict,preds = GNN2((act, indexes))
						labels = np.array(np.array(env.net_label[indexes][:,1], dtype=np.float64))
						labels = torch.LongTensor(labels).to(args.device)
						loss = F.nll_loss(preds, labels)
						test_loss += loss
						y_score = preds[:, 1] 
						test_prob_list.extend(y_score.to('cpu').detach().numpy())
						preds = preds.max(1)[1]
						test_pred_list.extend(preds.to('cpu').numpy())
						test_true_list.extend(labels.to('cpu').numpy())
			train_pred_list = np.array(train_pred_list)
			train_true_list = np.array(train_true_list)
			train_fpr, train_tpr, _ = roc_curve(train_true_list, train_pred_list.ravel())
			train_acc = accuracy_score(train_true_list, train_pred_list)
			print("训练集{:.5f}".format(train_acc))
			# train_pred_list = np.array(train_pred_list)
			# train_true_list = np.array(train_true_list)
			# train_acc = accuracy_score(train_true_list, train_pred_list)
			# val_pred_list = np.array(val_pred_list)
			# val_true_list = np.array(val_true_list)
			# val_acc = accuracy_score(val_true_list, val_pred_list)
			test_prob_list = np.array(test_prob_list)
			test_loss /= len(indexes)
			test_pred_list = np.array(test_pred_list)
			test_true_list = np.array(test_true_list)
			test_acc = accuracy_score(test_true_list, test_pred_list)
			TP = np.sum((test_true_list == 1) & (test_pred_list == 1))
			FN = np.sum((test_true_list == 1) & (test_pred_list == 0))
			FP = np.sum((test_true_list == 0) & (test_pred_list == 1))
			TN = np.sum((test_true_list == 0) & (test_pred_list == 0))
			print("测试集{:.5f}".format(test_acc))
			# 计算特异性和敏感性
			specificity = TN / (TN + FP)
			sensitivity = TP / (TP + FN)
			acc_s.append(test_acc)
			loss_s.append(test_loss)
    # 将特异性和敏感性添加到列表中
			# specificity_list.append(specificity)
			# sensitivity_list.append(sensitivity)
			#%%%%%%%%
			#test_true_list=[1,1,0,1,0,0]
			#%%%%%%%
			# if train_acc > max_train_acc:
			# 	max_train_acc = train_acc
			# test_fpr, test_tpr, _ = roc_curve(test_true_list, test_prob_list.ravel())
			test_fpr, test_tpr, _ = roc_curve(test_true_list, test_pred_list.ravel())
			test_auc = auc(test_fpr, test_tpr)
			if test_acc > max_test_acc:
				max_test_acc = test_acc
				max_test_auc = test_auc
				max_sensitivity = sensitivity
				max_specificity = specificity
				max_fpr = test_fpr
				max_tpr = test_tpr
				best_epoch = epoch
				# 保存模型参数
				# best_model =torch.save(GNN2.state_dict(), 'best_model.pth')
				# save_model_parameters(best_model, 'best_model.pth')
				# file_path='./result/MCI_NC_K10_linear/'
				file_path = os.path.join('./result/AD_NC_K4_linear/', "test4" + '.npy')
				np.save(file_path, predict)

		print(test_pred_list)
		print(test_true_list)
		acc_records1.append(max_val_acc)
		# acc_records2.append(max_train_acc)
		# auc_records.append(max_test_auc)
		print("Test_Acc: {:.5f}".format(max_test_acc))
		print("Test_AUC: {:.5f}".format(max_test_auc))
		print("Test_sen: {:.5f}".format(max_sensitivity))
		print("Test_spe: {:.5f}".format(max_specificity))
		print("Test_fpr:" ,max_fpr)
		print("Test_tpr:",max_tpr)
		specificity_list.append(max_specificity)
		sensitivity_list.append(max_sensitivity)
		acc_records.append(max_test_acc)
		auc_records.append(max_test_auc)
		print('----------------------------------------------')
		print('----------------------------------------------')
	mean_acc1 = np.mean(np.array(acc_records1))
	std_acc1 = np.std(np.array(acc_records1))

	mean_acc = np.mean(np.array(acc_records))
	std_acc = np.std(np.array(acc_records))
	print("testAcc: {:.5f}".format(mean_acc),'± {:.5f}'.format(std_acc))
	mean_auc = np.mean(np.array(auc_records))
	std_auc = np.std(np.array(auc_records))
	print("AUC: {:.5f}".format(mean_auc),'± {:.5f}'.format(std_auc))
	# mean_acc = np.mean(np.array(acc_records))
	# mean_acc2 = np.mean(np.array(acc_records2))
	# std_acc2 = np.std(np.array(acc_records2))
	# std_acc = np.std(np.array(acc_records))
	mean_specificity = np.mean(np.array(specificity_list))
	std_specificity = np.std(np.array(specificity_list))
	print("Specificity: {:.5f}".format(mean_specificity), '± {:.5f}'.format(std_specificity))

	mean_sensitivity = np.mean(np.array(sensitivity_list))
	std_sensitivity = np.std(np.array(sensitivity_list))
	print("Sensitivity: {:.5f}".format(mean_sensitivity), '± {:.5f}'.format(std_sensitivity))


# 创建子图，分别用于绘制准确率和损失值
	plt.subplot(1, 2, 1)
	plt.plot(range(0, args.epoch_num), acc_s)
	plt.xlabel('epoch')
	plt.ylabel('accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(range(0, args.epoch_num), loss_s)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	# 插值操作
	tpr_interp = np.linspace(0, 1, 100)  # 生成新的 TPR 数据点
	fpr_interp = np.interp(tpr_interp, max_tpr, max_fpr)  # 使用插值函数估算新的 FPR 数据点

# 计算插值后的 AUC
	roc_auc_interp = auc(fpr_interp, tpr_interp)
	plt.show()  # 显示绘制的图形
	print("Done!")  # 训练结束
	plt.plot(fpr_interp, tpr_interp, color='yellow',  label='GCN ROC curve (area = %0.2f)' % np.mean(roc_auc_interp))
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic (ROC) curve')
	plt.legend(loc="lower right")
	plt.show()
if __name__ == '__main__':

	for i in range(1):
		main()
