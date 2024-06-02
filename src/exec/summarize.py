import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import os
import json
import sys

sys.path.append('./')
from src.utils import get_true_param, load_config

def summarize(args):
    cfg = load_config(args['data_name'])

    summary_path = os.path.join('results', 'summary', args['data_name'])
    os.makedirs(summary_path, exist_ok=True)

    ltmle_result_path = os.path.join('results', 'eval', args['data_name'], 'ltmle')

    phi_0 = get_true_param(args['data_name'])

    dfs = []
    exclude_from_plot = []
    labels = []

    for configuration_name in cfg['experiments']:
        eval_results_path = os.path.join('results', 'eval', args['data_name'], configuration_name)

        if not os.path.exists(eval_results_path):
            continue

        for dir in sorted(os.listdir(eval_results_path)):
            dir = os.path.join(eval_results_path, dir)

            result_file_path = os.path.join(dir, 'result.csv')
            config_file_path = os.path.join(dir, 'config.json')

            if os.path.exists(result_file_path):                
                with open(config_file_path, 'r') as f:
                    eval_cfg = json.load(f)

                exclude_from_plot.append(args['exclude_from_plot'] is not None and eval_cfg['name'] in args['exclude_from_plot'])
                dfs.append(pd.read_csv(result_file_path))
                labels.append(eval_cfg['label'])

    for model_name in ["glm", "sl"]:
        result_file_path = os.path.join(ltmle_result_path, f'{model_name}.csv')
        if os.path.exists(result_file_path):                
            df = pd.read_csv(result_file_path)
            df['LQ'] = np.nan
            df['LG'] = np.nan
            df['Lstar'] = np.nan
            df['LQ_last'] = np.nan
            df['EIC'] = np.nan

            exclude_from_plot.append(False)
            dfs.append(df)
            labels.append(f'ltmle ({model_name})')

    rows = []
    for label, df in zip(labels, dfs):
        est = df['est'].mean()
        time = df['time'].mean()
        bias = est - phi_0
        variance =  df['est'].var()
        se = df['se'].mean()
        coverage = _coverage(df['est'], df['se'], phi_0)
        oracle_coverage = _coverage(df['est'], np.sqrt(variance), phi_0)
        rows.append([label, len(df), est, phi_0, bias, variance, se, coverage, oracle_coverage, time])

    df_result = pd.DataFrame(rows, columns=['name', 'M', 'est', 'phi_0', 'bias', 'var', 'se', 'coverage', 'oracle_coverage', 'time'])
    df_result.to_csv(os.path.join(summary_path, 'summary.csv'), index=False)

    dfs = [df for df, ignore in zip(dfs, exclude_from_plot) if not ignore]
    labels = [label for label, ignore in zip(labels, exclude_from_plot) if not ignore]

    for df in dfs:
        if not 'LQ_last' in df:
            df['LQ_last'] = np.nan

    _plot(summary_path, dfs, labels, phi_0)

def _coverage(est, se, phi_0):
    lcb = est - 1.959963984540054 * se
    ucb = est + 1.959963984540054 * se
    return ((lcb <= phi_0) * (ucb >= phi_0)).mean()

def _plot(result_path, dfs, labels, phi_0):
    matplotlib.use('Agg')

    nrow, ncol, w, h = 5, 2, 8, 4
    # nrow, ncol, w, h = 3, 3, 5, 4
    # nrow, ncol, w, h = 5, 1, 12, 4
    # nrow, ncol, w, h = 1, 4, 3.8, 3

    plt.figure(figsize=(ncol * w, nrow * h))

    params = {
        'labels': labels,
        'meanline': True,
        'showmeans': True,
        'medianprops': {'linewidth': 0},
        'meanprops': {'linewidth': 2, 'linestyle':'solid', 'color':'black'},
    }

    xs = [np.random.normal(i+1, 0.04, len(df)) for i, df in enumerate(dfs)]

    plt.subplot(nrow, ncol, 1)
    plt.boxplot([df['est'] for df in dfs], **params)
    plt.plot([0.5, len(dfs)+0.5], [phi_0, phi_0], lw=1)
    for x, df in zip(xs, dfs):
        plt.scatter(x, df['est'], alpha=0.4)
    plt.title('Estimates')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.subplot(nrow, ncol, 2)
    plt.scatter(labels, [_coverage(df['est'], df['se'], phi_0) for df in dfs], label='Coverage')
    plt.scatter(labels, [_coverage(df['est'], df['est'].std(), phi_0) for df in dfs], label='Oracle Coverage')
    plt.title('Coverage')
    plt.xticks(rotation=60, ha='right')
    plt.grid()
    plt.legend()

    plt.subplot(nrow, ncol, 3)
    plt.boxplot([df['PnIC'] for df in dfs], **params)
    for x, df in zip(xs, dfs):
        plt.scatter(x, df['PnIC'], alpha=0.4)
    plt.title('$P_n\\phi(\\hat{P}^\\star_n)$')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.subplot(nrow, ncol, 4)
    plt.boxplot([df['se'] for df in dfs], **params)
    for x, df in zip(xs, dfs):
        plt.scatter(x, df['se'], alpha=0.4)
    plt.title('${\widehat{\mathrm{SE}}}_n=\\sqrt{n^{-1}P_n\\phi^2(\\hat{P}^\\star_n)}$')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.subplot(nrow, ncol, 5)
    plt.boxplot([(df['est'] - phi_0) / (1.959963984540054 * df['se']) for df in dfs], **params)
    for x, df in zip(xs, dfs):
        plt.scatter(x, (df['est'] - phi_0) / (1.959963984540054 * df['se']), alpha=0.4)
    plt.title('$(\hat{\psi}_n - \psi_0) / 1.96{\widehat{\mathrm{SE}}}_n$')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.subplot(nrow, ncol, 6)
    plt.boxplot([df['LQ'] for df in dfs], **params)
    for x, df in zip(xs, dfs):
        plt.scatter(x, df['LQ'], alpha=0.4)
    plt.title('$P_n\\mathcal{L}^Q$')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.subplot(nrow, ncol, 7)
    plt.boxplot([df['LQ_last'] for df in dfs], **params)
    for x, df in zip(xs, dfs):
        plt.scatter(x, df['LQ_last'], alpha=0.4)
    plt.title('$P_n\\mathcal{L}^Q_\\tau$')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.subplot(nrow, ncol, 8)
    plt.boxplot([df['LG'] for df in dfs], **params)
    for x, df in zip(xs, dfs):
        plt.scatter(x, df['LG'], alpha=0.4)
    plt.title('$P_n\\mathcal{L}^G$')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.subplot(nrow, ncol, 9)
    plt.boxplot([df['Lstar'] for df in dfs], **params)
    for x, df in zip(xs, dfs):
        plt.scatter(x, df['Lstar'], alpha=0.4)
    plt.title('$P_n\\mathcal{L}^\\star$')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.subplot(nrow, ncol, 10)
    plt.boxplot([df['EIC'] for df in dfs], **params)
    for x, df in zip(xs, dfs):
        plt.scatter(x, df['EIC'], alpha=0.4)
    plt.title('$P_nD^\star/\\sqrt{P_n(D^\star)^2}$')
    plt.xticks(rotation=60, ha='right')
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "summary.pdf"), dpi=300)
