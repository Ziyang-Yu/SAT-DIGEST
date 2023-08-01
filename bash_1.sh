#!/bin/bash
export PYTHONUNBUFFERED=yes
python3 -u run/main_dist.py --rank 1 --num_parts 4 --interval 10 --servr_ip 127.0.0.1 > 1.log  &
python3 -u run/main_dist.py --rank 2 --num_parts 4 --interval 10 --servr_ip 127.0.0.1 > 2.log  &
python3 -u run/main_dist.py --rank 3 --num_parts 4 --interval 10 --servr_ip 127.0.0.1 > 3.log  &
