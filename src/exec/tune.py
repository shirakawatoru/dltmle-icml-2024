# nohup python jobs/tune/simple.py > log/hp-tuning-simple.out 2>&1 &

import numpy as np

import torch
import random

import itertools

from lightning.pytorch import Trainer
from lightning.pytorch.utilities.seed import isolate_rng

import os
import json
from datetime import datetime
import sys

sys.path.append('./')
from src.data.loader import load_data
from src.utils import seed_everything, load_config
from src.model.loader import load_model


def tune(args):
    """
    This function tunes the hyperparameters of a model.

    Parameters:
    args (dict): A dictionary containing the necessary parameters. The keys should include:
        - 'seed': The seed for the random number generator.
        - 'data_name': The name of the dataset to use.
        - 'configuration_name': The name of the configuration to use.
        - 'use_cpu': A boolean indicating whether to use the CPU instead of a GPU.
        - 'n_random_search': The number of random searches to perform.

    Returns:
    None
    """
    output_hparam_dir_path = os.path.join('results', 'hparams', args['data_name'], args['configuration_name'])
    output_hparam_file_path = os.path.join(output_hparam_dir_path, f'hparams.json')
    if os.path.exists(output_hparam_file_path) and not args['overwrite']:
        return

    rng = np.random.default_rng(args['seed'])
    cfg = load_config(args['data_name'])
    
    data, loader = load_data(rng, cfg)
    device = "gpu" if torch.cuda.is_available() and not args['use_cpu'] else "cpu"

    seed_everything(args['seed'])

    candidates = cfg['configurations'][args['configuration_name']]['hparams']
    model_name = cfg['configurations'][args['configuration_name']]['model_name']

    profiles = itertools.product(*candidates.values())
    profiles = random.sample(list(profiles), args['n_random_search'])

    def _profile_to_hparams(profile):
        return {k: v for k, v in zip(candidates, profile)}
    
    logger_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    hparams_a = [_profile_to_hparams(profile) for profile in profiles]
    metrics_a = [run_exp(args, model_name, data, loader, device, hparams, logger_name) for hparams in hparams_a]

    best_hparams = hparams_a[np.argmin(metrics_a)]

    os.makedirs(output_hparam_dir_path, exist_ok=True)

    with open(output_hparam_file_path, 'w') as f:
        json.dump(best_hparams, f, indent=4)
        

def run_exp(args, model_name, data, loader, device, hparams, logger_name):
    """
    This function runs a single experiment.

    Parameters:
    args (dict): A dictionary containing the necessary parameters. The keys should include:
        - 'artifact_path': The path where artifacts should be stored.
        - 'data_name': The name of the dataset to use.
        - 'configuration_name': The name of the configuration to use.
    model_name (str): The name of the model to use.
    data (DataLoader): The data for the experiment.
    loader (DataLoader): The data loader for the experiment.
    device (str): The device to use for computations ("gpu" or "cpu").
    hparams (dict): The hyperparameters for the model.
    logger_name (str): The name of the logger to use.

    Returns:
    Empirical loss of :math:`\\mathcal{L}^Q + \\mathcal{L}^G` on the validation set.
    """
    seed_everything(1234)

    with isolate_rng():
        model = load_model(model_name, hparams, data)
        
    seed_everything(1234)

    from lightning.pytorch.loggers import TensorBoardLogger

    save_dir = os.path.join(args["artifact_path"], "tune", args['data_name'], args['configuration_name'])
    os.makedirs(save_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir, name=logger_name, default_hp_metric=False)
    logger.log_hyperparams(hparams, metrics={'val/L':0, 'val/G':0, 'val/Q':0, 'val/GQ':0, 'val/Q_star':0, 'train/PnIC':0})

    with isolate_rng():
        trainer = Trainer(
            accelerator=device,
            gradient_clip_val=0.5,
            check_val_every_n_epoch=1,
            max_epochs=hparams['max_epochs'],
            logger=logger
        )

        trainer.fit(model, loader.load_train(), loader.load_val())
    
    return trainer.logged_metrics['val/GQ']
