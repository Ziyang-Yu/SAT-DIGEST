total_client = 4
interval = 10
num_machine = 1
servr_ip = '127.0.0.1'
client_per_machine = total_client // num_machine
all_files = {}
for machine in range(1, num_machine+1):
	file_name = 'bash_' + str(machine)+'.sh'
	all_files[machine] = open(file_name, 'w')
	all_files[machine].write('#!/bin/bash\n')
	all_files[machine].write('export PYTHONUNBUFFERED=yes\n')

for i in range(1, total_client):
	all_files[i//client_per_machine + 1].write('python3 -u run/main_dist.py --rank ' + str(i) + ' --num_parts ' + str(total_client) + ' --interval ' + str(interval) \
	+ ' --servr_ip ' + servr_ip
	+ ' > ' + str(i) + '.log  &\n')

for machine in range(1, num_machine+1):
	all_files[machine].close()
	
	
