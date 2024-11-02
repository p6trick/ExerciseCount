#!/bin/bash

result=ver2_result
csv_ver=0


python video_test_dir.py \
--data_dir ./dataset \
--type patient \
--direction left \
--result $result \
--en_th 9.9 \
--ex_th 0.1 \
--max_dis 20 \
--mean_dis 10 \
--csv_ver $csv_ver

python video_test_dir.py \
--data_dir ./dataset \
--type patient \
--direction right \
--result $result \
--en_th 9.9 \
--ex_th 0.1 \
--max_dis 20 \
--mean_dis 10 \
--csv_ver $csv_ver 

python video_test_dir.py \
--data_dir ./dataset \
--type not_patient \
--direction left \
--result $result \
--en_th 9.9 \
--ex_th 0.1 \
--max_dis 20 \
--mean_dis 10 \
--csv_ver $csv_ver

python video_test_dir.py \
--data_dir ./dataset \
--type not_patient \
--direction right \
--result $result \
--en_th 9.9 \
--ex_th 0.1 \
--max_dis 20 \
--mean_dis 10 \
--csv_ver $csv_ver
