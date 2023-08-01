#!/bin/bash
export PYTHONUNBUFFERED=yes
python3 -u main_dist.py --rank 1 --num_parts 7 --interval 10 --servr_ip 127.0.0.1 > 1.log  &
python3 -u main_dist.py --rank 2 --num_parts 7 --interval 10 --servr_ip 127.0.0.1 > 2.log  &
python3 -u main_dist.py --rank 3 --num_parts 7 --interval 10 --servr_ip 127.0.0.1 > 3.log  &
python3 -u main_dist.py --rank 4 --num_parts 7 --interval 10 --servr_ip 127.0.0.1 > 4.log  &
python3 -u main_dist.py --rank 5 --num_parts 7 --interval 10 --servr_ip 127.0.0.1 > 5.log  &
python3 -u main_dist.py --rank 6 --num_parts 7 --interval 10 --servr_ip 127.0.0.1 > 6.log  &
