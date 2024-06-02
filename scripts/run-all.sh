#!/bin/bash

n_sim=100
n_random_search=50

overwrite_option=""
# overwrite_option="--overwrite"

configuration_list="dltmle deepace"

simple_data_list="lay-cont-t10 simple-n1000-t10 simple-n1000-t20 simple-n1000-t30"
complex_data_list="complex-n1000-t10-p5-h10 complex-n1000-t20-p5-h20 complex-n1000-t30-p5-h30"
data_list=$simple_data_list" "$complex_data_list

for data_name in $data_list
do
    for configuration_name in $configuration_list
    do
        echo $data_name $configuration_name $overwrite_option
        python main.py tune --data_name $data_name --configuration_name $configuration_name --n_random_search $n_random_search $overwrite_option
        python main.py run --data_name $data_name --configuration_name $configuration_name --n_sim $n_sim $overwrite_option
    done
done

source scripts/savio/submit_sbatches.sh

for data_name in $data_list
do
    echo $data_name
    python main.py summarize --data_name $data_name
done

python main.py summarize --data_name lay-cont-t10 --exclude_from_plot 002-td-targeting