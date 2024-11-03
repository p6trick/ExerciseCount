#!/bin/bash

result=ver2_result
csv_ver=2
data_dir=./ver2_dataset


python video_test_dir.py \
--data_dir $data_dir \
--type patient \
--direction left \
--result $result \
--en_th 9.9 \
--ex_th 0.1 \
--max_dis 20 \
--mean_dis 10 \
--csv_ver $csv_ver

python video_test_dir.py \
--data_dir $data_dir \
--type patient \
--direction right \
--result $result \
--en_th 9.9 \
--ex_th 0.1 \
--max_dis 20 \
--mean_dis 10 \
--csv_ver $csv_ver 

python video_test_dir.py \
--data_dir $data_dir \
--type not_patient \
--direction left \
--result $result \
--en_th 9.9 \
--ex_th 0.1 \
--max_dis 20 \
--mean_dis 10 \
--csv_ver $csv_ver

python video_test_dir.py \
--data_dir $data_dir \
--type not_patient \
--direction right \
--result $result \
--en_th 9.9 \
--ex_th 0.1 \
--max_dis 20 \
--mean_dis 10 \
--csv_ver $csv_ver
