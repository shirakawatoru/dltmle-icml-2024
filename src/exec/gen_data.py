import numpy as np
from tqdm import tqdm

import os
import sys
sys.path.append('./')

from ..data.loader import load_data
from ..utils import load_config

def gen_data(args):
    cfg = load_config(args['data_name'])
    n_dataset = args['n_dataset']
    rng = np.random.default_rng(args['seed']).spawn(n_dataset)

    for b in tqdm(range(n_dataset)):
        data, _ = load_data(rng[b], cfg)
        df_ltmle, df_abar0, df_abar1 = data.get_dfs_for_ltmle()

        path = os.path.join(args['artifact_path'], 'ltmle', 'data', args['data_name'])
        os.makedirs(path, exist_ok=True)
        base_name = f'b{b:03d}'

        df_ltmle.to_csv(os.path.join(path, f'{base_name}_ltmle.csv'), index=False)
        df_abar0.to_csv(os.path.join(path, f'{base_name}_abar0.csv'), index=False)
        df_abar1.to_csv(os.path.join(path, f'{base_name}_abar1.csv'), index=False)

    # print('cd artifact/synth')
    # print('tar cvzf synth-complex.tgz complex_datasets/')
