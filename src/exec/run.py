
# nohup python src/experiment/eval.py > log/eval.out 2>&1 &

import numpy as np
import pandas as pd

import torch

import datetime
import json
import hashlib

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

import os
import sys
sys.path.append('./')

from src.data.loader import load_data
from src.model.loader import load_model
from src.utils import seed_everything, load_config, load_optimal_hparams, get_torch_device

_model_names_to_save = ['dltmle', 'dltmle2', 'deepace']

def run(args):
    cfg = load_config(args['data_name'])
    hparams = load_optimal_hparams(args['data_name'], args['configuration_name'])
    device = get_torch_device(args)

    exp_cfgs = cfg['experiments'][args['configuration_name']]

    for exp_cfg in exp_cfgs:
        results_dir = os.path.join('results', 'eval', args['data_name'], args['configuration_name'], exp_cfg['name'])
        result_csv_file = os.path.join(results_dir, 'result.csv')
        
        if os.path.exists(result_csv_file) and not args['overwrite']:
            print(f"Skipping {result_csv_file}")
            continue

        _hparams = hparams | exp_cfg['hparams']

        tb_log_dir = os.path.join(args['artifact_path'], 'eval', args['data_name'], args['configuration_name'], exp_cfg['name'])
        log_subdir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df = _run_experiments(device, args, cfg, _hparams, os.path.join(tb_log_dir, 'runs'), log_subdir_name)

        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(result_csv_file, index=False)

        with open(os.path.join(results_dir, "hparams.json"), "w") as f:
            json.dump(_hparams, f, indent=4)

        with open(os.path.join(results_dir, "config.json"), "w") as f:
            json.dump(exp_cfg, f, indent=4)

def _run_experiments(device, args, cfg, hparams, tb_log_dir, log_subdir_name):
    seed_everything(1234)
    rngs = np.random.default_rng(args['seed']).spawn(args['n_sim'])

    model_name = cfg['configurations'][args['configuration_name']]['model_name']

    rows = []

    for b, rng in enumerate(rngs):
        data, loader = load_data(rng, cfg)
        seed_train, seed_eval = rng.integers(0, 1000000, 2)
        rows.append(_run_single_experiment(seed_train, seed_eval, args, b, model_name, data, loader, device, hparams, tb_log_dir, log_subdir_name))

    df = pd.DataFrame(rows, columns=['est', 'se', 'PnIC', 'LQ', 'LG', 'Lstar', 'LQ_last', 'EIC', 'time'])
    return df

def _run_single_experiment(seed_train, seed_eval, args, b, model_name, data, loader, device, hparams, tb_log_dir, log_subdir_name):
    start_time = datetime.datetime.now()

    model, is_loaded = _load_model(args, args['seed'], b, model_name, hparams, args['data_name'], data)

    logger = TensorBoardLogger(tb_log_dir, name=log_subdir_name, default_hp_metric=False)
    logger.log_hyperparams(hparams, metrics={'test/L':0, 'test/G':0, 'test/Q':0, 'test/GQ':0, 'test/Q_star':0, 'test/PnIC':0})

    trainer = Trainer(
        accelerator=device,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=1,
        max_epochs=hparams['max_epochs'],
        logger=logger
    )

    if not is_loaded:
        print(f"Training {model_name} with {hparams}")

        seed_everything(seed_train)

        if "use_whole_sample" in hparams and hparams["use_whole_sample"]:
            train_loader = loader.load_whole(shuffle=True)
        else:
            train_loader = loader.load_train()

        trainer.fit(model, train_loader)
        _save_model(args, args['seed'], b, model_name, hparams, args['data_name'], model)

    if "use_whole_sample" in hparams and hparams["use_whole_sample"]:
        predict_loader = loader.load_whole(shuffle=False)
    else:
        predict_loader = loader.load_train_wo_shuffle()

    # move this targeting codes to model.LAY.dltmle
    if 'exact_tmle' in hparams and hparams['exact_tmle']:
        model.solve_canonical_gradient(trainer, predict_loader, data.tau)
    elif 'exact_tmle_common_eps' in hparams and hparams['exact_tmle_common_eps']:
        model.solve_canonical_gradient_common_eps(
            trainer, predict_loader, data.tau,
            stop_pnic_se_ratio=hparams['stop_pnic_se_ratio'] if 'stop_pnic_se_ratio' in hparams else False,
            max_delta_eps=hparams['max_delta_eps'] if 'max_delta_eps' in hparams else None
            )

    seed_everything(seed_eval)
    pred = trainer.predict(model, predict_loader)
    LQ = np.mean([x["loss"]["Q"] for x in pred])
    LG = np.mean([x["loss"]["G"] for x in pred])
    Lstar = np.mean([x["loss"]["Q_star"] for x in pred])
    LQ_last = np.mean([x["loss"]["Q_last"] for x in pred])
    
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time = elapsed_time.total_seconds()
    print(f"Elapsed time: {elapsed_time} seconds")

    est, se, IC, EIC = model.get_estimates_from_prediction(pred, predict_loader)

    return est, se, IC.mean(), LQ, LG, Lstar, LQ_last, EIC, elapsed_time

def _load_model(args, seed, b, model_name, hparams, data_name, data):
    '''Load model from file if exists. Otherwise, load from scratch.
    
    Returns:
        model: model
        is_loaded: True if model is loaded from file. False otherwise.
    '''
    if model_name in _model_names_to_save and not args["overwrite_model"]:
        _, key = _model_name(seed, b, model_name, hparams, data_name)

        model_dir_path = os.path.join(args['artifact_path'], 'models', data_name, model_name, key)
        model_file_path = os.path.join(model_dir_path, 'model.pt')

        if os.path.exists(model_file_path):
            print(f"Loading {model_file_path}")
            model = torch.load(model_file_path)
            return model, True
    
    return load_model(model_name, hparams, data), False

def _save_model(args, seed, b, model_name, hparams, data_name, model):  
    if model_name not in _model_names_to_save:
        return
    
    name, key = _model_name(seed, b, model_name, hparams, data_name)

    model_dir_path = os.path.join(args['artifact_path'], 'models', data_name, model_name, key)
    model_file_path = os.path.join(model_dir_path, 'model.pt')

    os.makedirs(model_dir_path, exist_ok=True)
    torch.save(model, model_file_path)

    with open(os.path.join(model_dir_path, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=4)

    with open(os.path.join(model_dir_path, "name.txt"), "w") as f:
        f.write(name)

def _model_name(seed, b, model_name, hparams, data_name):
    if model_name not in _model_names_to_save:
        return model_name
    
    hparams_keys = [
        'dim_static', 'dim_dynamic', 'tau', 'dim_emb', 'dim_emb_time', 'dim_emb_type', 
        'hidden_size', 'num_layers', 'nhead', 'dropout', 'learning_rate', 'alpha', 'beta',
        'survival_outcome'
        ]
    kv = {k: hparams[k] for k in hparams_keys if k in hparams}
    name = f'{model_name}_{data_name}_{seed}_{b}_' + '_'.join([f'{k}_{v}' for k, v in kv.items()])

    if "use_whole_sample" in hparams and hparams["use_whole_sample"]:
        name += "_whole_sample"

    h = hashlib.sha256()
    h.update(name.encode('utf-8'))
    return name, h.hexdigest()
