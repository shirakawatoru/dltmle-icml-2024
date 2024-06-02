import os
import json
from datetime import datetime
import hashlib

import torch
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from lightning.pytorch import loggers, Trainer

from ..data.loader import load_data, LongitudinalDataLoader
from ..model.loader import load_model
from ..utils import seed_everything, load_config, load_optimal_hparams, get_torch_device

def plot(args):
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.size': 14})

    cfg = load_config(args["data_name"])

    if cfg["data"]["name"] != "circs-lacy":
        raise NotImplementedError("Only circs-lacy data is supported for now.")
    
    data, _ = load_data(None, cfg)

    max_tau = args['tau']

    est_120 = np.zeros(max_tau + 1)
    est_140 = np.zeros(max_tau + 1)

    ic_120 = np.zeros((data.n, max_tau + 1))
    ic_140 = np.zeros((data.n, max_tau + 1))

    for a in [140, 120]:
        for tau in range(1, max_tau + 1):
            print(f"a = {a}, tau = {tau}")

            save_dir = os.path.join("results", "estimate", args["data_name"], args["configuration_name"], str(a), str(tau))
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, "est.json"), "r") as f:
                est = json.load(f)['est']

            ic = pd.read_csv(os.path.join(save_dir, "ic.csv"), header=None).squeeze().values

            if a == 120:
                est_120[tau] = est
                ic_120[:,tau] = ic
            else:
                est_140[tau] = est
                ic_140[:,tau] = ic

    plt.figure(figsize=(6, 3*2))

    plt.subplot(2,1,1)
    
    _plot_est(est_140, ic_140, data.n, "darkblue", "Standard")
    _plot_est(est_120, ic_120, data.n, "darkred", "Intensive")
    plt.legend()
    plt.grid()
    plt.ylabel("Cumulative Mortality")
    plt.xlabel("Years after baseline")

    plt.subplot(2,1,2)
    _plot_est(est_120 - est_140, ic_120 - ic_140, data.n, "darkblue", None)
    plt.grid()
    plt.ylabel("Averaget Treatment Effect")
    plt.xlabel("Years after baseline")

    plt.tight_layout()

    save_dir = os.path.join("results", "plot", args["data_name"], args["configuration_name"])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "cumulative_incidence.png"), dpi=300)

    taus = [10, 20, 30]
    ic = (ic_120 - ic_140)[:,taus]
    se = ((ic ** 2).mean(axis=0) / data.n) ** 0.5
    q = simultaneous_q(ic[:, se>0], data.n)
    ate = [(est_120[tau] - est_140[tau]) * 100 for tau in taus]
    se *= 100
    disp = ["{:.02f} ({:.02f} -- {:.02f})".format(ate[i], ate[i] - q * se[i], ate[i] + q * se[i]) for i in range(len(taus))]
    df_disp = pd.DataFrame({"tau": [f"$\\tau={tau}$" for tau in taus], "disp": disp}).T
    print(df_disp.to_latex(index=False, header=False))    


def _plot_est(est, ic, n, color, label):
    se = ((ic ** 2).mean(axis=0) / n) ** 0.5
    q = simultaneous_q(ic[:, se>0], n)

    plt.plot(est, color=color, label=label)
    plt.fill_between(np.arange(est.shape[0]), est - q * se, est + q * se, alpha=0.2, color=color)

def simultaneous_q(ic, n):
    Sigma = (ic.T @ ic) / n

    sd = np.sqrt(np.diag(Sigma))
    sd_outer = np.outer(sd, sd)
    rho = Sigma / sd_outer

    Z = np.random.multivariate_normal(np.zeros(rho.shape[0]), rho, size=100000)
    return np.quantile(np.max(np.abs(Z), axis=1), 0.95)

