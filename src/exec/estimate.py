# nohup python main.py estimate --data_name circs-lacy-mort-medium --configuration_name dltmle > log/train-deepace.out 2>&1 &

import os
import json
from datetime import datetime
import hashlib

import torch
import numpy as np
import pandas as pd

from lightning.pytorch import loggers, Trainer

from ..data.loader import load_data, LongitudinalDataLoader
from ..model.loader import load_model
from ..utils import seed_everything, load_config, load_optimal_hparams, get_torch_device

_available_data_names = [
    "circs-lacy",
    "circs-lyca",
]

def estimate(args):
    cfg = load_config(args["data_name"])

    if cfg["data"]["name"] not in _available_data_names:
        raise ValueError(f"Invalid data_name: {args['data_name']}")

    hparams = load_optimal_hparams(args["data_name"], args["configuration_name"])
    device = get_torch_device(args)

    model_name = cfg["configurations"][args["configuration_name"]]["model_name"]

    dt = datetime.now().strftime("%Y%m%d_%H%M%S")

    seed_everything(args['seed'])
    rng = np.random.default_rng(args['seed'])

    data_orig, _ = load_data(rng, cfg)

    max_tau = args['max_tau'] if args['max_tau'] is not None else data_orig.tau
    max_epochs = args['max_epochs'] if args['max_epochs'] is not None else hparams["max_epochs"]

    taus = args['tau'] if args['tau'] is not None else range(1, max_tau + 1)

    for tau in taus:
        for a in [140, 120]:
            print(f"a = {a}, tau = {tau}")

            data = data_orig.get_sub_data(tau)

            if a == 140:
                data.a = data.a_140
            elif a == 120:
                data.a = data.a_120
            else:
                raise ValueError(f"Invalid a: {a}")

            loader = LongitudinalDataLoader(data)

            model, is_loaded = _load_model(args, args['seed'], model_name, hparams, args['data_name'], data, a, tau)

            writer = loggers.TensorBoardLogger(save_dir=os.path.join(args["artifact_path"], "estimate", dt))
            trainer = Trainer(
                accelerator=device,
                gradient_clip_val=0.5,
                check_val_every_n_epoch=1,
                max_epochs=max_epochs,
                logger=writer
                )

            if not is_loaded:
                seed_everything(args['seed'])
                trainer.fit(model, loader.load_whole(shuffle=True))
                _save_model(args, args['seed'], model_name, hparams, args['data_name'], model, a, tau)

            whole_loader = loader.load_whole(shuffle=False)

            model.solve_canonical_gradient_common_eps(
                trainer,
                whole_loader,
                tau,
                stop_pnic_se_ratio=True,
                )
            
            pred = trainer.predict(model, whole_loader)

            V_star = torch.cat([x["V_star"] for x in pred], axis=0).detach().numpy().squeeze()
            ic = torch.cat([x["IC"] for x in pred], axis=0).detach().numpy().squeeze()
            est = V_star[:, 0].mean()

            save_dir = os.path.join("results", "estimate", args["data_name"], args["configuration_name"], str(a), str(tau))
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, "est.json"), "w") as f:
                json.dump({"est": str(est)}, f, indent=4)

            pd.Series(ic).to_csv(os.path.join(save_dir, "ic.csv"), index=False, header=False)


def _load_model(args, seed, model_name, hparams, data_name, data, a, tau):
    '''Load model from file if exists. Otherwise, load from scratch.
    
    Returns:
        model: model
        is_loaded: True if model is loaded from file. False otherwise.
    '''
    if not args["overwrite_model"]:
        _, key = _model_name(seed, model_name, hparams, data_name, a, tau)

        model_dir_path = os.path.join(args['artifact_path'], 'models', 'estimate', data_name, model_name, key)
        model_file_path = os.path.join(model_dir_path, 'model.pt')

        if os.path.exists(model_file_path):
            print(f"Loading {model_file_path}")
            model = torch.load(model_file_path)
            return model, True
    
    return load_model(model_name, hparams, data), False

def _save_model(args, seed, model_name, hparams, data_name, model, a, tau):  
    name, key = _model_name(seed, model_name, hparams, data_name, a, tau)

    model_dir_path = os.path.join(args['artifact_path'], 'models', 'estimate', data_name, model_name, key)
    model_file_path = os.path.join(model_dir_path, 'model.pt')

    os.makedirs(model_dir_path, exist_ok=True)
    torch.save(model, model_file_path)

    with open(os.path.join(model_dir_path, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=4)

    with open(os.path.join(model_dir_path, "name.txt"), "w") as f:
        f.write(name)

def _model_name(seed, model_name, hparams, data_name, a, tau):
    hparams_keys = [
        'dim_static', 'dim_dynamic', 'tau', 'dim_emb', 'dim_emb_time', 'dim_emb_type', 
        'hidden_size', 'num_layers', 'nhead', 'dropout', 'learning_rate', 'alpha', 'beta'
        ]
    kv = {k: hparams[k] for k in hparams_keys if k in hparams}
    name = f'{model_name}_{data_name}_a_{a}_tau_{tau}' + '_'.join([f'{k}_{v}' for k, v in kv.items()])

    if "use_whole_sample" in hparams and hparams["use_whole_sample"]:
        name += "_whole_sample"

    h = hashlib.sha256()
    h.update(name.encode('utf-8'))
    return name, h.hexdigest()

