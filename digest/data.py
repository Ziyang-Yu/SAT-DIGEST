from typing import Tuple
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon,
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI, AmazonProducts)
from ogb.nodeproppred import PygNodePropPredDataset

from .utils import index2mask, gen_masks
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset
import os
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2, radians
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as f
import math


class PM25(InMemoryDataset):
	def __init__(self, root, transform=None):
		super(PM25, self).__init__('.', transform, None, None)
		df = pd.read_csv(os.path.join(root, "all_stations.csv"))
		self.root = root
		x, y, distance_df = self.process_data(df)
		distance_df = pd.read_csv(os.path.join(root, "distance.csv"))
		
		'''
		density 100%
		'''
		# row = torch.from_numpy(np.array([i for i in range(x.shape[0]) for j in range(x.shape[0])])).to(torch.long)
		# col = torch.from_numpy(np.array([j for i in range(x.shape[0]) for j in range(x.shape[0])])).to(torch.long)
		# edge_index = torch.stack([row, col], dim=0)
		# data = Data(edge_index=edge_index)
		# data.adj_t = SparseTensor(row=row, col=col, value=torch.flatten(torch.tensor(distance_df.values)).float())
		#
		'''
		change density
		'''
		target_density = 0.0005
		all_values = torch.tensor(distance_df.values)
		closest_station = torch.topk(all_values, int(all_values.size()[1] * target_density), dim=1)
		col = closest_station.indices.flatten().to(torch.long)
		row = torch.from_numpy(np.array([i for i in range(all_values.size()[0]) for j in range(int(all_values.size()[1] * target_density))])).to(torch.long)
		
		edge_index = torch.stack([row, col], dim=0)
		data = Data()
		softmax = torch.nn.Softmax(dim=1)
		closest_station_values = softmax(closest_station.values)
		value = closest_station_values.flatten().float()
		data.adj_t = SparseTensor(row=row, col=col, value=value)
		print(data.adj_t)
		
		
		
		data.num_nodes = x.shape[0]
		distance_df = None
		data.x = x.type(torch.float32)
		y = y.type(torch.float32)
		data.y = y.clone().detach()
		data.num_classes = 2
		data.num_features = data.x.shape[1]
		X_train, X_test, y_train, y_test = train_test_split(pd.Series(range(data.num_nodes)),
		                                                    pd.Series(range(data.num_nodes)), test_size=0.2,
		                                                    random_state=1)
		
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
		                                                  random_state=1)  # 0.25 x 0.8 = 0.2
		
		# create train and test masks for data
		train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
		test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
		val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
		
		train_mask[X_train.index] = True
		test_mask[X_test.index] = True
		val_mask[X_val.index] = True
		data.train_mask = train_mask
		data.test_mask = test_mask
		data.val_mask = val_mask
		
		# print(data.train_mask.size())
		# print(data.val_mask.size())
		# print(data.test_mask.size())
		
		self.data, self.slices = self.collate([data])
	
	def process_data(self, df):
		# df = df.iloc[:500]
		df = df.sort_values(by=['id'])
		df = df.dropna(subset=['lat', 'lon'])
		df2 = df[['lat', 'lon']]
		df = df[['location_type', 'pm_2.5', 'temp_f', 'temp_c', "humidity", 'pressure', '10min_avg',
		         '30min_avg', '1hour_avg', '6hour_avg', '1day_avg', '1week_avg']]
		
		df['location_type'] = df['location_type'].replace({'outside': 0, 'inside': 1, np.nan: 2})
		
		# apply normalization techniques
		for column in df.columns:
			if column != 'pm_2.5':
				df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
		
		df = df.fillna(0)
		# x = df.loc[:, df.columns != 'pm_2.5'].to_numpy()
		x = torch.tensor(df.loc[:, df.columns != 'pm_2.5'].values)
		y = df.loc[:, df.columns == 'pm_2.5']
		# import statistics
		# print(statistics.median(y['pm_2.5'].tolist()))
		# y.loc[df['pm_2.5'] < 10, 'pm_2.5'] = 0
		# y.loc[df['pm_2.5'] >= 10,'pm_2.5'] = 1
		# y = y.to_numpy()
		# y = np.array(y['pm_2.5'].tolist())
		# y = torch.from_numpy(y)
		y = torch.tensor(y.values)
		
		# process distance
		# distance_df = self.get_distance_matrix(df2)
		distance_df = None
		return x, y, distance_df
	
	def get_distance_matrix(self, df):
		distances = pdist(df.values, metric=self.dist)
		points = [f'point_{i}' for i in range(1, len(df) + 1)]
		result = pd.DataFrame(squareform(distances), columns=points)
		result = 1 / (result + 1)
		np.fill_diagonal(result.values, 0)
		distance_df = result.div(result.sum(axis=1), axis=0)
		distance_df.to_csv(os.path.join(self.root, "distance.csv"), index=False)
		print("Save adj matrix succ!")
		return distance_df
	
	def dist(self, x, y):
		"""Function to compute the distance between two points x, y"""
		
		lat1 = radians(x[0])
		lon1 = radians(x[1])
		lat2 = radians(y[0])
		lon2 = radians(y[1])
		
		R = 6373.0
		
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		
		a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))
		
		distance = R * c
		if math.isnan(distance):
			print(x, y)
		return round(distance, 4)
	
	def _download(self):
		return
	
	def _process(self):
		return
	
	def __repr__(self):
		return '{}()'.format(self.__class__.__name__)


def get_planetoid(root: str, name: str) -> Tuple[Data, int, int]:
	transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
	dataset = Planetoid(f'{root}/Planetoid', name, transform=transform)
	return dataset[0], dataset.num_features, dataset.num_classes


def get_wikics(root: str) -> Tuple[Data, int, int]:
	dataset = WikiCS(f'{root}/WIKICS', transform=T.ToSparseTensor())
	data = dataset[0]
	data.adj_t = data.adj_t.to_symmetric()
	data.val_mask = data.stopping_mask
	data.stopping_mask = None
	return data, dataset.num_features, dataset.num_classes


def get_coauthor(root: str, name: str) -> Tuple[Data, int, int]:
	dataset = Coauthor(f'{root}/Coauthor', name, transform=T.ToSparseTensor())
	data = dataset[0]
	torch.manual_seed(12345)
	data.train_mask, data.val_mask, data.test_mask = gen_masks(
		data.y, 20, 30, 20)
	return data, dataset.num_features, dataset.num_classes


def get_amazon(root: str, name: str) -> Tuple[Data, int, int]:
	dataset = Amazon(f'{root}/Amazon', name, transform=T.ToSparseTensor())
	data = dataset[0]
	torch.manual_seed(12345)
	data.train_mask, data.val_mask, data.test_mask = gen_masks(
		data.y, 20, 30, 20)
	return data, dataset.num_features, dataset.num_classes


def get_amazon_products(root: str) -> Tuple[Data, int, int]:
	dataset = AmazonProducts(f'{root}/AmazonProducts', pre_transform=T.ToSparseTensor())
	data = dataset[0]
	return data, dataset.num_features, dataset.num_classes


def get_arxiv(root: str) -> Tuple[Data, int, int]:
	dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB',
	                                 pre_transform=T.ToSparseTensor())
	data = dataset[0]
	data.adj_t = data.adj_t.to_symmetric()
	data.node_year = None
	data.y = data.y.view(-1)
	split_idx = dataset.get_idx_split()
	data.train_mask = index2mask(split_idx['train'], data.num_nodes)
	data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
	data.test_mask = index2mask(split_idx['test'], data.num_nodes)
	return data, dataset.num_features, dataset.num_classes


def get_products(root: str) -> Tuple[Data, int, int]:
	dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB',
	                                 pre_transform=T.ToSparseTensor())
	data = dataset[0]
	data.y = data.y.view(-1)
	split_idx = dataset.get_idx_split()
	data.train_mask = index2mask(split_idx['train'], data.num_nodes)
	data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
	data.test_mask = index2mask(split_idx['test'], data.num_nodes)
	return data, dataset.num_features, dataset.num_classes


def get_papers100m(root: str) -> Tuple[Data, int, int]:
	dataset = PygNodePropPredDataset('ogbn-papers100M', f'{root}/OGB',
	                                 pre_transform=T.ToSparseTensor())
	data = dataset[0]
	data.y = data.y.view(-1)
	split_idx = dataset.get_idx_split()
	data.train_mask = index2mask(split_idx['train'], data.num_nodes)
	data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
	data.test_mask = index2mask(split_idx['test'], data.num_nodes)
	return data, dataset.num_features, dataset.num_classes


def get_yelp(root: str) -> Tuple[Data, int, int]:
	dataset = Yelp(f'{root}/YELP', pre_transform=T.ToSparseTensor())
	data = dataset[0]
	data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
	return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str) -> Tuple[Data, int, int]:
	dataset = Flickr(f'{root}/Flickr', pre_transform=T.ToSparseTensor())
	return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str) -> Tuple[Data, int, int]:
	dataset = Reddit2(f'{root}/Reddit2', pre_transform=T.ToSparseTensor())
	data = dataset[0]
	data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
	return data, dataset.num_features, dataset.num_classes


def get_ppi(root: str, split: str = 'train') -> Tuple[Data, int, int]:
	dataset = PPI(f'{root}/PPI', split=split, pre_transform=T.ToSparseTensor())
	data = Batch.from_data_list(dataset)
	data.batch = None
	data.ptr = None
	data[f'{split}_mask'] = torch.ones(data.num_nodes, dtype=torch.bool)
	return data, dataset.num_features, dataset.num_classes


def get_sbm(root: str, name: str) -> Tuple[Data, int, int]:
	dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train',
	                              pre_transform=T.ToSparseTensor())
	data = Batch.from_data_list(dataset)
	data.batch = None
	data.ptr = None
	return data, dataset.num_features, dataset.num_classes


def get_pm25(root):
	dataset = PM25(root)
	data = dataset[0]
	# return data, data.num_features, data.num_classes
	return data, data.num_features, 1


def get_data(root: str, name: str) -> Tuple[Data, int, int]:
	if name.lower() in ['cora', 'citeseer', 'pubmed']:
		return get_planetoid(root, name)
	elif name.lower() in ['coauthorcs', 'coauthorphysics']:
		return get_coauthor(root, name[8:])
	elif name.lower() in ['amazoncomputers', 'amazonphoto']:
		return get_amazon(root, name[6:])
	elif name.lower() == 'wikics':
		return get_wikics(root)
	elif name.lower() in ['cluster', 'pattern']:
		return get_sbm(root, name)
	elif name.lower() == 'reddit':
		return get_reddit(root)
	elif name.lower() == 'ppi':
		return get_ppi(root)
	elif name.lower() == 'flickr':
		return get_flickr(root)
	elif name.lower() == 'yelp':
		return get_yelp(root)
	elif name.lower() in ['ogbn-arxiv', 'arxiv']:
		return get_arxiv(root)
	elif name.lower() in ['ogbn-products', 'products']:
		return get_products(root)
	elif name.lower() == 'amazon':
		return get_amazon_products(root)
	elif name.lower() == 'papers100m':
		return get_papers100m(root)
	elif name.lower() == 'pm25':
		return get_pm25(root)
	else:
		raise NotImplementedError
