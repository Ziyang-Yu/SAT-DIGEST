import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from digest import (get_data, metis, permute,
                    SubgraphLoader, EvalSubgraphLoader,
                    models, compute_micro_f1, dropout)
import time
import numpy as np
import random
import hydra
import os
import dill as pickle
from hydra import compose, initialize
from omegaconf import OmegaConf
import argparse

random.seed(30)
torch.manual_seed(123)
np.random.seed(123)


def read_conf():
	initialize(config_path="conf", job_name="test_app")
	conf = compose(config_name="config")
	conf.model.params = conf.model.params[conf.dataset.name]
	params = conf.model.params
	print(OmegaConf.to_yaml(conf))
	return conf, params


# @hydra.main(config_path='conf', config_name='config')
def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-n', '--num_parts', default=4, type=int, help='number of parts')
	args = parser.parse_args()
	print("start to read config file")
	conf, params = read_conf()
	# num_parts = conf.num_parts
	num_parts = args.num_parts
	
	# conf.model.params = conf.model.params[conf.dataset.name]
	# params = conf.model.params
	
	print('config file read successfully!')
	files_path = os.path.join(conf.dataset.name, str(num_parts))
	if not os.path.exists(files_path):
		os.makedirs(files_path)
	
	t = time.perf_counter()
	print('Loading data...', end=' ', flush=True)
	data, in_channels, out_channels = get_data(conf.root, conf.dataset.name)
	with open(os.path.join(files_path, 'nodes_in_out.pickle'), 'wb') as handle:
		pickle.dump([data.num_nodes, in_channels, out_channels], handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	print(f'Done! [{time.perf_counter() - t:.2f}s]')
	perm, ptr = metis(data.adj_t, num_parts=num_parts, log=True)
	data = permute(data, perm, log=True)
	if conf.model.loop:
		t = time.perf_counter()
		print('Adding self-loops...', end=' ', flush=True)
		data.adj_t = data.adj_t.set_diag()
		print(f'Done! [{time.perf_counter() - t:.2f}s]')
	if conf.model.norm:
		t = time.perf_counter()
		print('Normalizing data...', end=' ', flush=True)
		data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
		print(f'Done! [{time.perf_counter() - t:.2f}s]')
	
	train_loader = SubgraphLoader(data, ptr, batch_size=params.batch_size,
	                              shuffle=False, num_workers=params.num_workers,
	                              persistent_workers=params.num_workers > 0)
	eval_loader = EvalSubgraphLoader(data, ptr, batch_size=params['batch_size'])
	
	with open(os.path.join(files_path, 'eval_loader.pickle'), 'wb') as handle:
		pickle.dump(eval_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	with open(os.path.join(files_path, 'data.pickle'), 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	for i, (batch, batch_size, *args) in enumerate(train_loader):
		x = batch.x
		adj_t = batch.adj_t
		y = batch.y[:batch_size]
		train_mask = batch.train_mask[:batch_size]
		# adj_t = dropout(adj_t, p=edge_dropout)
		batch_size = batch_size
		args = args
		n_ids = args[0]
		subgraph_adj = args[3]
		#print(adj_t)
		#print(subgraph_adj)
		save_data = {'x': x, 'adj_t': adj_t, 'y': y, 'train_mask': train_mask, 'batch_size': batch_size, 'args': args,
		             'n_ids': n_ids, 'subgraph_adj': subgraph_adj}
		with open(os.path.join(files_path, 'part_' + str(i) + '.pickle'), 'wb') as handle:
			pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	main()
