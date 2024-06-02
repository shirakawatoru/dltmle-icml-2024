#!/bin/bash

simple_data_list="simple-n1000-t10 simple-n1000-t20 simple-n1000-t30"
complex_data_list="complex-n1000-t10-p5-h5 complex-n1000-t20-p5-h5 complex-n1000-t30-p5-h5"
data_list=$simple_data_list" "$complex_data_list

for data_name in $data_list
do
    echo $data_name
    python main.py gen_data --data_name $data_name --seed 1234 --n_dataset 500
done
