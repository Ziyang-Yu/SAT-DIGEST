from main_dist import *
import os
from math import ceil
import subprocess
from multiprocessing import Process
import time
conf, params = read_conf()
# client_num = ceil(params.num_parts / params.batch_size)


def run_command(command):
	os.system(command)
	
	
for total_client in range(50, 51):
	for interval in [10]:
		os.system("sudo pkill -f main_dist")
		for i in range(total_client):
			command = 'python3 main_dist.py +rank=' + str(i) + ' +num_parts=' + str(total_client) + ' +interval=' + str(interval) \
						+ " hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled" \
			          +' > ' + str(i) + '.log  &4'
			p = Process(target=run_command, args=(command,))
			p.start()
			# print(command)
			# os.system(command)
			# time.sleep(1)
		while True:
			# print("results_" + str(total_client) + "_gpus_interval_" + str(interval) + ".csv")
			# print(os.path.exists("results_" + str(total_client) + "_gpus_interval_" + str(interval) + ".csv"))
			if os.path.exists("results_" + str(total_client) + "_gpus_interval_" + str(interval) + ".csv"):
				print("Finished!!!!!!!!!!!!!")
		# 		os.rename("results.csv", "results_" + str(total_client) + "_gpus_interval_" + str(interval) + ".csv")
				break