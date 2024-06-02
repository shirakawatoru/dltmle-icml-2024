'''generate shell scripts for analysis on savio'''

import os
import argparse
import itertools

import numpy as np

def main():
    data_names = [
        "way-n500",
        "way-n1000",
        "way-n2000",
        "way-n5000",
        "simple-n1000-t10",
        "simple-n1000-t20",
        "simple-n1000-t30",
        "complex-n1000-t10-p5-h10",
        "complex-n1000-t20-p5-h20",
        "complex-n1000-t30-p5-h30",
        "lay-cont-t10",
    ]

    names = [
        "way500",
        "way1k",
        "way2k",
        "way5k",
        "s10",
        "s20",
        "s30",
        "c10",
        "c20",
        "c30",
        "con-t10",
    ]

    model_names = [
        "glm",
        "sl",
    ]

    for name, data_name in zip(names, data_names):
        for model_name in model_names:
            gen_script(f"{name}_{model_name}", data_name, model_name)

def gen_script(name, data_name, model_name):

    src = f'''#!/bin/bash

Rscript src/r/run.R --seed 1234 --n_sim 100 --data_name {data_name} --model_name {model_name} --verbose
'''

    script_file_dir = os.path.join(os.path.dirname(__file__), 'batches')
    os.makedirs(script_file_dir, exist_ok=True)

    script_file_name = f'{name}.sh'
    script_file_path = os.path.join(script_file_dir, script_file_name)

    with open(script_file_path, mode='w') as f:
        f.write(src)

if __name__ == "__main__":
    main()
