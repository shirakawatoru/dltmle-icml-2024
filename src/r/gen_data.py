import numpy as np

import os
import sys
sys.path.append('./')

from src.utils import load_config
from src.data.loader import load_data

def gen_data(args):
    cfg = load_config(args['data_name'])

    rng = np.random.default_rng(args['seed']).spawn(args['b']+1)[args['b']]
    data, _ = load_data(rng, cfg)
    df_ltmle, df_abar0, df_abar1 = data.get_dfs_for_ltmle()

    os.makedirs(args['output_path'], exist_ok=True)

    df_ltmle.to_csv(os.path.join(args['output_path'], f'ltmle.csv'), index=False)
    df_abar0.to_csv(os.path.join(args['output_path'], f'abar0.csv'), index=False)
    df_abar1.to_csv(os.path.join(args['output_path'], f'abar1.csv'), index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-b', type=int)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args().__dict__

    gen_data(args)