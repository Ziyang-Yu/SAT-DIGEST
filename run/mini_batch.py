import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from digest import (get_data, metis, permute,
                    SubgraphLoader, EvalSubgraphLoader,
                    models, compute_micro_f1, dropout)
from digest.data import get_ppi
from math import ceil
import time
from torch_sparse import SparseTensor
import numpy as np
import dill as pickle
import os
from torch_geometric.utils import is_undirected
import random

from digest.auxiliary_models.lstm_gcn import LSTM_GCN
from digest.history_series import History_Series

random.seed(30)
torch.manual_seed(123)
np.random.seed(123)


def change_density(data, density_level):
	num_one = int(data.num_nodes * data.num_nodes * density_level)
	row_index = np.sort(np.random.choice(data.num_nodes, num_one, replace=True))
	col_index = np.sort(np.random.choice(data.num_nodes, num_one, replace=True))
	row = torch.from_numpy(row_index)
	col = torch.from_numpy(col_index)
	out = SparseTensor(rowptr=None, row=row, col=col, value=None, sparse_sizes=(data.num_nodes, data.num_nodes),
	                   is_sorted=True)
	data.adj_t = out
	print(out)
	return data


def change_features(data):
	shifted = torch.roll(data.x, shifts=(1, 0), dims=(1, 0))
	new_tensor = shifted + data.x
	# data.x = torch.cat((data.x, new_tensor), 1)
	new_tensor_2 = shifted - data.x
	new_tensor_3 = shifted * data.x
	new_tensor_4 = shifted + 2 * data.x
	new_tensor_5 = 2 * shifted + data.x
	new_tensor_6 = (shifted + data.x) / 2
	new_tensor_7 = shifted + 1.5 * data.x
	new_tensor_8 = shifted + 0.8 * data.x
	new_tensor_9 = 2 * shifted + 1.7 * data.x
	data.x = torch.cat((data.x, new_tensor, new_tensor_2, new_tensor_3, new_tensor_4, new_tensor_5, new_tensor_6,
	                    new_tensor_7, new_tensor_8, new_tensor_9), 1)
	return data


class Minibatch:
	def __init__(self, rank, conf, params):
		self.rank, self.conf, self.params = rank, conf, params
		self.criterion = None
		self.model = None
		self.optimizer = None
		self.train_loader = None
		self.eval_loader = None
		self.max_steps = self.params.max_steps
		self.grad_norm = None
		self.edge_dropout = None
		self.batch_num = ceil(self.params.num_parts / self.params.batch_size)
		# self.batch_num = 1
		# self.pull_data = None
		
		self.data = None
		self.val_data = None
		self.test_data = None
		self.device = None
		self.mini_batch = None
		self.x = None
		self.adj_t = None
		self.y = None
		self.batch_size = None
		self.train_mask = None
		self.args = None
		self.local_epoch = 3
		self.his_data = None
		self.n_ids = None
		self.subgraph_adj = None
		# store parameters of each mini-batch
		self.index_parameters = dict()
		self.auxiliary_model = None
		self.hist_embs = None
		self.init_all_params()

		self.liner = False
		self.subset_mini_batch_pull_data = None
		self.corrected_embs = None
		self.aux_optimizer = torch.optim.Adam(params=self.auxiliary_model.parameters(), lr=self.params.lr)
	
	def init_all_params(self):
		try:
			self.edge_dropout = self.params.edge_dropout
		except:  # noqa
			self.edge_dropout = 0.0
		self.grad_norm = None if isinstance(self.params.grad_norm, str) else self.params.grad_norm
		# self.device = torch.device('cpu') if self.rank != 0 else torch.device(f'cuda:{self.rank}')
		self.device = torch.device(f'cuda:{self.rank}')
		print("self.device is ", self.device)
		t = time.perf_counter()
		print('Loading data...', end=' ', flush=True)
		# data, in_channels, out_channels = get_data(self.conf.root, self.conf.dataset.name)
		file_name = os.path.join(self.conf.dataset.name, str(self.params.num_parts), 'nodes_in_out.pickle')
		print(file_name)
		f = open(file_name, 'rb')
		print('1')
		nodes_in_out = pickle.load(f)
		f.close()
		# with open(file_name, 'rb') as handle:
		# 	nodes_in_out = pickle.load(handle)
		# 	handle.close()
		num_nodes, in_channels, out_channels = nodes_in_out
		print(f'Done! [{time.perf_counter() - t:.2f}s]')
		if self.rank == 0:
			with open(os.path.join(self.conf.dataset.name, str(self.params.num_parts), 'data.pickle'),
			          'rb') as handle:
				self.data = pickle.load(handle)
		
		# if self.data.y.dim() == 1:
		# 	self.criterion = torch.nn.CrossEntropyLoss()
		# else:
		# 	self.criterion = torch.nn.BCEWithLogitsLoss()
		self.criterion = torch.nn.CrossEntropyLoss()
		
		if self.rank == 0:
			for i in range(self.params.num_parts):
				file_name = os.path.join(self.conf.dataset.name, str(self.params.num_parts),
				                         'part_' + str(i) + '.pickle')
				with open(file_name, 'rb') as handle:
					part_data = pickle.load(handle)
				self.index_parameters[i] = dict()
				self.index_parameters[i]['x_size'] = part_data['x'].size()
				self.index_parameters[i]['n_id'] = part_data['args'][0]
				self.index_parameters[i]['offset'] = part_data['args'][1]
				self.index_parameters[i]['count'] = part_data['args'][2]
				self.index_parameters[i]['batch_size'] = part_data['batch_size']
			
			with open(os.path.join(self.conf.dataset.name, str(self.params.num_parts), 'eval_loader.pickle'),
			          'rb') as handle:
				self.eval_loader = pickle.load(handle)
				handle.close()
			
			t = time.perf_counter()
			print('Calculating buffer size...', end=' ', flush=True)
			# We reserve a much larger buffer size than what is actually needed for
			# training in order to perform efficient history accesses during inference.
			buffer_size = max([n_id.numel() for _, _, n_id, _, _, _ in self.eval_loader])
			print(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')
		
		if self.conf.dataset.name == 'ppi':
			self.val_data, _, _ = get_ppi(self.conf.root, split='val')
			self.test_data, _, _ = get_ppi(self.conf.root, split='test')
			if self.conf.model.loop:
				self.val_data.adj_t = self.val_data.adj_t.set_diag()
				self.test_data.adj_t = self.test_data.adj_t.set_diag()
			if self.conf.model.norm:
				self.val_data.adj_t = gcn_norm(self.val_data.adj_t, add_self_loops=False)
				self.test_data.adj_t = gcn_norm(self.test_data.adj_t, add_self_loops=False)
		
		if self.conf.dataset.name == 'pm25':
			self.liner = False
			# self.criterion = torch.nn.MSELoss()
			self.criterion = torch.nn.L1Loss()
		else:
			self.liner = False
		
		kwargs = {}
		if self.conf.model.name[:3] == 'PNA':
			kwargs['deg'] = self.data.adj_t.storage.rowcount()
		
		GNN = getattr(models, self.conf.model.name)
		if self.rank == 0:
			self.model = GNN(
				num_nodes=num_nodes,
				in_channels=in_channels,
				out_channels=out_channels,
				pool_size=self.params.pool_size,
				buffer_size=buffer_size,
				linear=self.liner,
				**self.params.architecture,
				**kwargs,
			).to(self.device)
		else:
			self.model = GNN(
				num_nodes=num_nodes,
				in_channels=in_channels,
				out_channels=out_channels,
				pool_size=self.params.pool_size,
				# buffer_size=buffer_size,
				buffer_size=None,
				linear=self.liner,
				**self.params.architecture,
				**kwargs,
			).to(self.device)
		
		self.optimizer = torch.optim.Adam([
			dict(params=self.model.reg_modules.parameters(), weight_decay=self.params.reg_weight_decay),
			dict(params=self.model.nonreg_modules.parameters(), weight_decay=self.params.nonreg_weight_decay)
		], lr=self.params.lr)
		
		if self.rank == 0:
			t = time.perf_counter()
			print('Fill history...', end=' ', flush=True)
			# self.mini_test(self.model, self.eval_loader)
			self.mini_test()
			print(f'Done! [{time.perf_counter() - t:.2f}s]')
		
		# get mini batch data
		file_name = os.path.join(self.conf.dataset.name, str(self.params.num_parts),
		                         'part_' + str(self.rank) + '.pickle')
		print("load...")
		with open(file_name, 'rb') as handle:
			part_data = pickle.load(handle)
			handle.close()
		print("finish load...")
		self.model.to(self.device)

		self.x = part_data['x'].to(self.device)
		self.adj_t = part_data['adj_t'].to(self.device)
		self.y = part_data['y'].to(self.device)
		self.train_mask = part_data['train_mask'].to(self.device)
		self.adj_t = dropout(self.adj_t, p=self.edge_dropout)
		self.batch_size = part_data['batch_size']
		self.args = part_data['args']
		self.n_ids = part_data['n_ids']

		self.subgraph_adj = part_data['subgraph_adj']

		self.auxiliary_model = LSTM_GCN(self.params.architecture.hidden_channels, self.params.architecture.hidden_channels, 2)
		self.hist_embs = torch.nn.ModuleList([History_Series(num_nodes, self.params.architecture.hidden_channels) 
								for _ in range(self.params.architecture.num_layers)])

		self.loss_func = torch.nn.MSELoss()


		del part_data
		
		if self.rank != 0:
			del self.data
	
	@torch.no_grad()
	def full_test(self):
		self.model.eval()
		return self.model(self.data.x.to(self.model.device), self.data.adj_t.to(self.model.device)).cpu()
	
	def compute_mse_train(self, out):
		return torch.nn.functional.l1_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
	
	def compute_mse_val(self, out):
		return torch.nn.functional.l1_loss(out[self.data.val_mask], self.data.y[self.data.val_mask])
	
	def compute_mse_test(self, out):
		return torch.nn.functional.l1_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
	
	@torch.no_grad()
	def mini_test(self):
		self.model.eval()
		return self.model.evaluate(loader=self.eval_loader)
	
	def micro_f1_train(self, out):
		return compute_micro_f1(out, self.data.y, self.data.train_mask)
	
	def micro_f1_val(self, out):
		return compute_micro_f1(out, self.data.y, self.data.val_mask)
	
	def micro_f1_test(self, out):
		return compute_micro_f1(out, self.data.y, self.data.test_mask)
	
	def micro_f1_val_ppi(self):
		return compute_micro_f1(self.full_test(self.model, self.val_data), self.val_data.y)
	
	def micro_f1_test_ppi(self):
		return compute_micro_f1(self.full_test(self.model, self.test_data), self.test_data.y)
	
	def set_weights(self, state_dict):
		self.model.load_state_dict(state_dict)
	
	def process_sigle_push(self, client_index, local_push_data):
		for local_push, his in zip(local_push_data, self.model.histories):
			# print("********")
			# print(len(self.index_parameters[client_index]['n_id'][:self.index_parameters[client_index]['batch_size']]))
			# print(local_push.size())
			# print(self.index_parameters[client_index]['offset'])
			# print(self.index_parameters[client_index]['count'])
			his.push(local_push,
			         self.index_parameters[client_index]['n_id'][:self.index_parameters[client_index]['batch_size']],
			         self.index_parameters[client_index]['offset'],
			         self.index_parameters[client_index]['count'])
	
	# def aggregate_local_push(self, local_push_data):
	# 	aggregated_local_push = dict()
	# 	n_ids = []
	# 	for client_index, client_push_data in enumerate(local_push_data):
	# 		ids = self.index_parameters[client_index]['n_id'][:self.index_parameters[client_index]['batch_size']]
	# 		for his_index, client_layer_push in enumerate(client_push_data):
	# 			if client_index == 0:
	# 				aggregated_local_push[his_index] = client_layer_push
	# 			else:
	
	def process_local_push(self, local_push_data):
		for client_index, client_push_data in enumerate(local_push_data):
			self.process_sigle_push(client_index, client_push_data)
	
	# for local_push, his in zip(local_push_data, self.model.histories):
	# 	for client_index, local_push_client in enumerate(local_push):
	# 		print(client_index)
	# 		his.push(local_push_client,
	# 				 self.index_parameters[client_index]['n_id'][
	# 				 :self.index_parameters[client_index]['batch_size']],
	# 				 self.index_parameters[client_index]['offset'],
	# 				 self.index_parameters[client_index]['count']
	# 				 )
	
	def get_pull_data(self, client_index):
		pull_data_per_his = []
		for i, his in enumerate(self.model.histories):
			pull_data_per_his.append(his.pull(self.index_parameters[client_index]['n_id'][
			                                  self.index_parameters[client_index]['batch_size']:]).to(
				torch.device('cpu')))
		return pull_data_per_his
	
	# def get_pull_data(self, client_index, his_index):
	# 	return self.model.histories[his_index].pull(self.index_parameters[client_index]['n_id'][
	# 										self.index_parameters[client_index]['batch_size']:]).to(torch.device('cpu'))
	#
	def get_all_pull_data(self):
		print("Start to get all pull data!")
		all_pull_data = []
		for i, his in enumerate(self.model.histories):
			layer_his = []
			print("Current layer is {}".format(i))
			for client_index in range(self.batch_num):
				print("Current mini-batch is {}".format(client_index))
				pull_data = his.pull(self.index_parameters[client_index]['n_id'][
				                     self.index_parameters[client_index]['batch_size']:]).to(torch.device('cpu'))
				layer_his.append(pull_data)
				print("The size of pull data is {}".format(pull_data.size()))
			all_pull_data.append(layer_his)
		return all_pull_data
	
	def train(self, his_data, if_old_his_data, if_train_aux_model):
		print("Start local training!")
		# if self.subset_mini_batch_pull_data is None:
		# 	print(len(self.n_ids[self.batch_size:]))
		# 	print(len(self.n_ids))
		# 	print(self.batch_size)
		# 	self.subset_mini_batch_pull_data = random.sample(range(len(self.n_ids[self.batch_size:])), num_neighbour_to_selected)
		# print(self.subset_mini_batch_pull_data)
		if if_train_aux_model:
			self.auxiliary_model.to(self.device)
			self.auxiliary_model.train()

			"""
			get size
			
			print(self.corrected_embs[0].size())
			print(his_data[0].size())
			print(self.adj_t)
			print(self.batch_size)
			"""
			label = torch.stack(his_data).detach().to(self.device)
			pred = torch.stack(self.corrected_embs).to(self.device)
			loss = self.loss_func(label, pred)
			loss.backward()
			self.aux_optimizer.step()
		
			self.auxiliary_model = self.auxiliary_model.cpu()
			del label, pred, loss
		


		self.model.train()
		if not if_old_his_data:
			self.his_data = [i.to(self.device) for i in his_data]
		for i in range(self.local_epoch):
			self.optimizer.zero_grad()
			# print(self.x.get_device())
			out, local_push_data, corrected_embs = self.model(self.x, self.adj_t, self.batch_size, self.his_data, 
											self.subgraph_adj, self.auxiliary_model, self.hist_embs, self.rank, *self.args)
			# print(out[self.train_mask], self.y[self.train_mask])
			loss = self.criterion(out[self.train_mask], self.y[self.train_mask])
			loss.backward()
			if self.grad_norm is not None:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
			self.optimizer.step()
			total_loss = float(loss) * int(self.train_mask.sum())
			total_examples = int(self.train_mask.sum())
		self.corrected_embs = corrected_embs
		return total_loss / total_examples, local_push_data
	
	def move_to_cuda(self, device, *args):
		for arg in args:
			arg = arg.to(device)
			print(arg.device)
		return args


if __name__ == '__main__':
	pass
