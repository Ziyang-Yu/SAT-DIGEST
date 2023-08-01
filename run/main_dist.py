import os.path
import time
import torch
import torch.distributed as dist
import argparse
import pandas as pd
from mini_batch import *
from collections import defaultdict
import threading
from pympler import asizeof
import random
import hydra
import sys

random.seed(30)

# interval = 1


def read_conf():
	initialize(config_path="conf", job_name="test_app")
	conf = compose(config_name="config")
	conf.model.params = conf.model.params[conf.dataset.name]
	params = conf.model.params
	print(OmegaConf.to_yaml(conf))
	return conf, params


def daemon_thread(req):
	req.wait()


def thread_to_send(data, dst, tag):
	dist.send(data, dst=dst, tag=tag)


def thread_to_receive(data, rank, index):
	dist.recv(data, src=rank, tag=index)
	

def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-r', '--rank', default=0, type=int, help='rank of current process')
	parser.add_argument('-i', '--servr_ip', default='127.0.0.1', help="servr_ip", type=str)
	# parser.add_argument('-b', '--backend', default='gloo', help='backend', type=str)
	# parser.add_argument('-k', '--topk', default='0.1', help='ratio of topk for historical embeddings', type=float)
	parser.add_argument('-n', '--num_parts', default='2', help="total number of works", type=int)
	parser.add_argument('-t', '--interval', default='10', help="interval to pull and push", type=int)
	args = parser.parse_args()
	conf, params = read_conf()
	rank = args.rank
	num_parts = args.num_parts
	interval = args.interval
	servr_ip = args.servr_ip
	params['num_parts'] = num_parts
	
	client = Minibatch(rank, conf, params)
	dist.init_process_group(backend='gloo', init_method='tcp://' + servr_ip + ':23456', rank=rank,
	                        world_size=client.batch_num)

	

	if conf.dataset.name == 'pm25':
		best_val_acc = np.Inf
		test_acc = np.Inf
	else:
		best_val_acc = 0
		test_acc = 0
	results = defaultdict(list)
	start_experiments = time.time()

	#auxiliary_model = LSTM_GCN(params['hidden_channels'], params['hidden_channels'], 2)
	for epoch in range(params.epochs):
		print("### Epoch ", epoch, " ###")
		epoch_start_time = time.time()
		all_train_loss = [None for _ in range(client.batch_num)]
		all_push_data = [None for _ in range(client.batch_num)]
		if rank == 0:
			for index, data in enumerate(all_push_data):
				tensor_size = (client.index_parameters[index]['batch_size'], client.model.hidden_channels)
				all_push_data[index] = [torch.zeros(tensor_size, dtype=torch.float, device=torch.device('cpu')) for _ in
				                        range(params.architecture.num_layers - 1)]
		
		# pull_data_receiver = torch.zeros((client.args[0][client.batch_size:].size()[0], client.model.hidden_channels), dtype=torch.float)
		# print(pull_data_receiver.element_size() * pull_data_receiver.nelement())
		# local_pull_data = [torch.zeros((client.args[0][client.batch_size:].size()[0], client.model.hidden_channels), dtype=torch.float)] * (params.architecture.num_layers - 1)
		start_time = time.time()
		# num_neighbour_to_selected = 1000
		if epoch % interval == 0:
			if rank == 0:
				'''
				Get needed embeddings for all mini-batch
				'''
				print("start to send pull data to clients")
				# randomly select n_ids for all client
				
				# all_subset_mini_batch_pull_data = dict()
				# for client_index in range(client.batch_num):
				# 	all_subset_mini_batch_pull_data[client_index] = random.sample(range(len(client.index_parameters[client_index]['n_id'][
				#                      client.index_parameters[client_index]['batch_size']:])), num_neighbour_to_selected)
				
				for client_index in range(client.batch_num):
					pull_time = time.time()
					
					mini_batch_pull_data = client.get_pull_data(client_index)
					
					print("pull time is ", time.time() - pull_time)
					if client_index == 0:
						local_pull_data = mini_batch_pull_data
					else:
						# print("****")
						# print(client_index)
						for index, pull_data in enumerate(mini_batch_pull_data):
							t = threading.Thread(target=thread_to_send, args=(pull_data, client_index, index,),
							                     daemon=True)
							t.start()
						# print(index)
						# print("send time ", time.time())
			
			else:
				local_pull_data = []
				for his_layer in range(params.architecture.num_layers - 1):
					init_time = time.time()
					# pull_data_receiver = torch.zeros((num_neighbour_to_selected, client.model.hidden_channels), dtype=torch.float)
					pull_data_receiver = torch.zeros(
						(client.args[0][client.batch_size:].size()[0], client.model.hidden_channels), dtype=torch.float)
					print("Init receiver takes ", time.time() - init_time)
					# req = dist.irecv(pull_data_receiver, src=0, tag=his_layer)
					# req.wait()
					dist.recv(pull_data_receiver, src=0, tag=his_layer)
					local_pull_data.append(pull_data_receiver)
			# print(local_pull_data)
		results['get_pull_time'].append(time.time() - start_time)
		print("get all pull takes ", time.time() - start_time)
		start_time = time.time()
		


		if epoch % interval == 0 and epoch != 0:
			loss, local_push_data = client.train(local_pull_data, False, True)
		elif epoch % interval == 0 and epoch == 0:
			loss, local_push_data = client.train(local_pull_data, False, False)
		else:
			loss, local_push_data = client.train(None, True, False)
		
		local_push_data = [i.to(torch.device('cpu')) for i in local_push_data]
		results['local_train_time'].append(time.time() - start_time)
		print("local train takes ", time.time() - start_time)
		start_time = time.time()
		local_model_state_dict = client.model.state_dict()
		for key in local_model_state_dict:
			local_model_state_dict[key] = local_model_state_dict[key].to(torch.device('cpu'))
			output = [torch.zeros_like(local_model_state_dict[key]) for _ in range(num_parts)]
			dist.gather(local_model_state_dict[key], output if rank == 0 else None, dst=0)
			local_model_state_dict[key] = torch.sum(torch.stack(output), dim=0)/num_parts
			dist.broadcast(local_model_state_dict[key], src=0)

			# local_model_state_dict[key] = local_model_state_dict[key].to(torch.device('cpu'))
			# dist.all_reduce(local_model_state_dict[key], op=dist.ReduceOp.SUM, async_op=False)
			# local_model_state_dict[key] = local_model_state_dict[key] / client.batch_num
		client.set_weights(local_model_state_dict)
		results['agg_time'].append(time.time() - start_time)
		print("aggregation takes ", time.time() - start_time)
		start_time = time.time()
		receive_threads = []
		if (epoch + 1) % interval == 0:
			for index, push_data in enumerate(local_push_data):
				if rank == 0:
					all_push_data[0] = local_push_data
					for client_index in range(1, client.batch_num):
						t = threading.Thread(target=thread_to_receive,
						                     args=(all_push_data[client_index][index], client_index, index,),
						                     daemon=True)
						receive_threads.append(t)
				else:
					dist.send(push_data, dst=0, tag=index)
			if rank == 0:
				for thread in receive_threads:
					thread.start()
				for thread in receive_threads:
					thread.join()
		
		results['get_push_time'].append(time.time() - start_time)
		print("get all push takes ", time.time() - start_time)
		start_time = time.time()
		
		if rank == 0:
			dist.gather_object(loss, all_train_loss, dst=0)
		else:
			dist.gather_object(loss, None, dst=0)
		results['get_loss_time'].append(time.time() - start_time)
		print("get all loss takes ", time.time() - start_time)
		
		'''
		if args.rank == 0:
			dist.gather_object(local_push_data, all_push_data, dst=0)
			dist.gather_object(loss, all_train_loss, dst=0)
		else:
			dist.gather_object(local_push_data, None, dst=0)
			dist.gather_object(loss, None, dst=0)
		'''
		
		if rank == 0:
			start_time = time.time()
			if (epoch + 1) % interval == 0:
				client.process_local_push(all_push_data)
			results['process_push_time'].append(time.time() - start_time)
			print("process all push takes ", time.time() - start_time)
			'''
			eval
			'''
			start_time = time.time()
			out = client.mini_test()
			if conf.dataset.name == 'pm25':
				# "acc is mse"
				train_acc = client.compute_mse_train(out)
				val_acc = client.compute_mse_val(out)
				tmp_test_acc = client.compute_mse_test(out)
				if val_acc < best_val_acc:
					best_val_acc = val_acc
					test_acc = tmp_test_acc
			else:
				train_acc = client.micro_f1_train(out)
				if conf.dataset.name != 'ppi':
					val_acc = client.micro_f1_val(out)
					tmp_test_acc = client.micro_f1_test(out)
				else:
					# We need to perform inference on a different graph as PPI is an
					# inductive dataset.
					val_acc = client.micro_f1_val_ppi()
					tmp_test_acc = client.micro_f1_test_ppi()
				if val_acc > best_val_acc:
					best_val_acc = val_acc
					test_acc = tmp_test_acc
			results['eval_time'].append(time.time() - start_time)
			print("eval takes ", time.time() - start_time)
			print(
				"Epoch: {}, Train: {}, Val: {}, Test: {}, Final: {}, Loss: {}, Epoch_time: {}".format(epoch, train_acc,
				                                                                                      val_acc,
				                                                                                      tmp_test_acc,
				                                                                                      test_acc,
				                                                                                      sum(all_train_loss) / len(
					                                                                                      all_train_loss),
				                                                                                      time.time() - epoch_start_time))
			results['Epoch'].append(epoch)
			results['Train'].append(train_acc)
			results['val'].append(val_acc)
			results['test'].append(tmp_test_acc)
			results['final'].append(test_acc)
			results['loss'].append(sum(all_train_loss) / len(all_train_loss))
			results['time'].append(time.time() - start_experiments)
	
	if rank == 0:
		print("Finish!!!!!!!!!")
		df = pd.DataFrame.from_dict(results)
		df.to_csv("results_" + str(num_parts) + "_gpus_interval_" + str(interval) + ".csv", index=False)
		print("Save results successfully!")


if __name__ == '__main__':
	main()
