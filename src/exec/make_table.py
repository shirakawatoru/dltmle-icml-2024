import os 
import re

import pandas as pd
import numpy as np

import json

def make_table(args):
    if args['table_name'] in ['simple', 'complex']:
        make_table_one_two(args)
    elif args['table_name'] == 'hparams':
        make_table_hparams(args)
    else:
        raise ValueError("table_name must be simple, complex or hparams")

def make_table_one_two(args):
    df_results = pd.DataFrame()

    models = [
        'ltmle (glm)',
        'ltmle (sl)',
        'deepace',
        'dltmle',
        'dltmle*',
        'dltmle‡'
        ]
    
    model_display_names = [
        'LTMLE (GLM)',
        'LTMLE (SL)',
        'DeepACE',
        'Deep LTMLE',
        'Deep LTMLE\\dagger',
        'Deep LTMLE\\star'
        ]

    model_index = {
        model: i for i, model in enumerate(models)
    }

    index_model_display_names = {
        i: model_display_name for i, model_display_name in enumerate(model_display_names)
    }

    if args['table_name'] == 'simple':
        data_names = ['simple-n1000-t10','simple-n1000-t20','simple-n1000-t30']
    elif args['table_name'] == 'complex':
        data_names = ['complex-n1000-t10-p5-h10','complex-n1000-t20-p5-h20','complex-n1000-t30-p5-h30']
    else:
        raise ValueError("table_name must be simple or complex")

    # mtd = pd.Series(['LTMLE(GLM)',
    #                  'LTMLE(SL)',
    #                  'DeepACE',
    #                  'DeepACE 0.05',
    #                  'DLTMLE^{\\text{\ding{61}}}(0)',
    #                  'DLTMLE^{\\text{\ding{61}}}\smallstar',
    #                  'DLTMLE^{\\text{\ding{61}}}\\filledstar\\filledstar'])

    
    for data_name in data_names:
        tau = int(re.search(r't(\d+)', data_name).group(1))

        summary_file_path = os.path.join('results/summary', data_name, 'summary.csv')
        df_result = pd.read_csv(summary_file_path)

        df_result = df_result[df_result['name'].isin(models)]

        df_result['tau'] = f"$\\tau = {tau}$"

        mse = df_result['bias']**2 + df_result['var']
        df_result['mse'] = mse

        df_result['sd'] = np.sqrt(df_result['var'])
        df_result['rmse'] = np.sqrt(df_result['mse'])

        # df_result['bias'] = [f"{x:.3f}" for x in df_result['bias']]
        # df_result['sd'] = [f"{x:.3f}" for x in df_result['sd']]
        # df_result['rmse'] = [f"{x:.3f}" for x in df_result['rmse']]
        # df_result['coverage'] = [f"{x:.3f}" for x in df_result['coverage']]

        df_result['model_index'] = [model_index[x] for x in df_result['name']]

        val_init = df_result.loc[df_result['model_index']==4,'time'].values
        df_result.loc[df_result['model_index'].isin([5,6]), 'time'] += val_init

        # if args['table_name'] == 'simple':
        #     df_result = df_result.loc[df_result['model_index'].isin([0,1,2,4,5,6])]
        # else:
        #     df_result = df_result.loc[df_result['model_index'].isin([0,1,2,4,5,6])]

        df_result = df_result[['model_index', 'tau', 'bias', 'sd', 'rmse', 'coverage', 'se']]
        df_result.columns = ['model_index', "tau", "Bias", "SD", "RMSE", "Coverage", "SE"]

        df_results = pd.concat([df_results, df_result], axis=0)

    df_table = df_results.pivot(index='model_index', columns='tau', values=['Bias', 'RMSE', 'Coverage', 'SE'])

    measure_formats = {
        'Bias': "{:.04f}",
        'SD': "{:.04f}",
        'RMSE': "{:.04f}",
        'Coverage': "{:.02f}",
        'SE': "{:.02f}",
    }

    values = df_table.values

    for j in range(len(df_table.columns)):
        df_table.iloc[:, j] = df_table.iloc[:, j].astype(str)

    for j in range(len(df_table.columns)):
        measure_name = df_table.columns[j][0]
        x_format = measure_formats[measure_name]

        def _perfomance(x):
            if measure_name == 'Coverage':
                return np.abs(x - 0.95)
            else:
                return np.abs(x)

        v_min = np.nanmin(_perfomance(values[:, j]))

        def _format(x):
            v = _perfomance(x)
            if np.isnan(x):
                sep = "|" if j in [2, 5, 8, 11] else ""
                return f"\\multicolumn{{1}}{{c{sep}}}{{---}}"
            elif v != v_min:
                return x_format.format(x)
            else:
                x = x_format.format(x)
                return x
                #return f"\\textbf{{{x}}}"

        df_table.iloc[:, j] = [_format(x) for x in values[:, j]]

    df_table['Model'] = [index_model_display_names[i] for i in df_table.index]
    df_table.set_index('Model', inplace=True)
    # df_table = df_table.reindex(labels=[index_model_display_names[i] for i in df_table.index])

    print(df_table)

    # Table path
    dir_path = "results/latex"
    os.makedirs(dir_path, exist_ok=True) 

    #Write .tex file
    if args['table_name'] == 'simple':
        f = open(os.path.join(dir_path, 'table1.tex'), "w+")
        cpt = "Results from simple synthetic data"
        label = "table1"
    else:
        f = open(os.path.join(dir_path, 'table2.tex'), "w+")
        cpt = "Results from complex synthetic data"
        label = "table2"

    s = df_table.style
    s.hide(names=True)
    #s.hide(axis=1, names=True)
    latex_s = s.to_latex(
        #buf=f,
               column_format="lrrrrrrcccccc",
               position="h",
               position_float="centering",
               hrules=True,
               label=label,
               caption= cpt,
               multicol_align="c")
    
    text = latex_s.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
    text = text.replace("\end{tabular}", "\end{tabular}}")
    text = text.replace("{table}", "{table*}")
    text = text.replace("DLTMLE &", "\hline DLTMLE &")
    text = re.sub("^tau", "Model", text, flags=re.MULTILINE)
    f.seek(0)
    f.write(text)
    f.close()


def make_table_hparams(args):
    settings = [
        ['simple-n1000-t10', 'dltmle'],
        ['simple-n1000-t20', 'dltmle'],
        ['simple-n1000-t30', 'dltmle'],
        ['simple-n1000-t10', 'deepace'],
        ['simple-n1000-t20', 'deepace'],
        ['simple-n1000-t30', 'deepace'],
        ['complex-n1000-t10-p5-h10', 'dltmle'],
        ['complex-n1000-t20-p5-h20', 'dltmle'],
        ['complex-n1000-t30-p5-h30', 'dltmle'],
        ['complex-n1000-t10-p5-h10', 'deepace'],
        ['complex-n1000-t20-p5-h20', 'deepace'],
        ['complex-n1000-t30-p5-h30', 'deepace'],
    ]
    
    def _read_hparams(data_name, model_name):
        hparams_file_path = os.path.join('results/hparams', data_name, model_name, 'hparams.json')
        with open(hparams_file_path, 'r') as f:
            hparams = json.load(f)
        
        params = {
            'dltmle' : ['dim_emb', 'dropout', 'hidden_size', 'num_layers', 'nhead', 'learning_rate', 'alpha', None, 'max_epochs'],
            'deepace': [None, 'dropout', 'hidden_size', 'num_layers', None, 'learning_rate', 'alpha', 'beta', 'max_epochs']
        }
        
        return [hparams[param] if param is not None else '---' for param in params[model_name]]
    
    results = [_read_hparams(*setting) for setting in settings]
    results.append([32, 0, 16, 8, 4, 5e-4, 0.1, 0.01, 100])

    print(results)
        
    results = list(map(list, zip(*results))) # transpose

    for r in results:
        print(' & '.join(map(str, r)))
