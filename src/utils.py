import math

import torch
import torch.nn as nn

import numpy as np

import os
import json

from scipy.special import expit, logit
from scipy.optimize import minimize

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class SinusoidalEncoder(nn.Module):
    def __init__(self, data=None, dim=None, max_period=10000, dim_scale=0.25):
        super().__init__()
        assert data is not None or dim is not None

        if dim is None:
            dim = len(data)**dim_scale

        half = dim // 2
        self.freqs = nn.Parameter(
            torch.exp(-math.log(max_period) * 
                torch.arange(start=0, end=half, dtype=torch.float) / half
            ), requires_grad=False)

        self.res = dim % 2

    def forward(self, timesteps):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.FloatTensor(timesteps)
        timesteps = timesteps.to(self.freqs.device)

        args = (timesteps[..., None] * self.freqs).view(timesteps.shape[0], -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return torch.cat([embedding, torch.zeros_like(embedding[:, :self.res])], dim=-1)

def get_lambda_rate(decay_iters, warmup_iters=100, decay_rate=0.1):
    def f(iteration):
        rate = 1.0 if iteration > warmup_iters else iteration / warmup_iters
        return rate * decay_rate ** (iteration // decay_iters)
    return f

def load_config(data_name):
    with open(os.path.join('config/scenarios', data_name) + '.json') as f:
        return json.load(f)

def load_optimal_hparams(data_name, configuration_name):
    path = os.path.join('results/hparams', data_name, configuration_name, 'hparams.json')
    with open(path) as f:
        return json.load(f)

def get_true_param(data_name):
    if data_name == 'simple-n1000-t10':
        return 0.102777399122715
    elif data_name == 'simple-n1000-t20':
        return 0.19789110124111176
    elif data_name == 'simple-n1000-t30':
        return 0.28251850605010986
    elif data_name == 'complex-n1000-t10-p5-h10':
        return 0.09062200039625168
    elif data_name == 'complex-n1000-t20-p5-h20':
        return 0.17119309306144714
    elif data_name == 'complex-n1000-t30-p5-h30':
        return 0.24475570023059845
    elif data_name == 'way-n500':
        return 0.0068246
    elif data_name == 'way-n1000':
        return 0.0068246
    elif data_name == 'way-n2000':
        return 0.0068246
    elif data_name == 'way-n5000':
        return 0.0068246
    elif data_name == 'lay-cont-t10':
        return 0.011241548
    
    else:
        raise ValueError('data_name not recognized: {}'.format(data_name))
    
def solve_one_dimensional_submodel(y_hat, y, H):
    '''fit univariate logistic regression: y ~ reg.predict(X) + eps with weight H
    
    Parameters
    ----------
    logit_y_hat: n dimensional vector
        logit of predicted probability of y = 1
    y: n dimensional vector
        labels for each sample
    H: n dimensional vector
        weight for logistic regression

    Returns
    -------
    eps: `float`
        translation which minimize the logistic loss    
    '''
    logit_y_hat = logit(y_hat)

    def _safelog(x, delta=1e-8):
        return np.log(np.clip(x, delta, np.inf))

    def _loss(eps):
        '''weighted logistic loss function (binary cross entropy loss)'''
        s = expit(logit_y_hat + eps)
        return -np.mean(H * (y * _safelog(s) + (1 - y) * _safelog(1 - s)))
    
    def _jac(eps):
        '''gradient of the loss function'''
        s = expit(logit_y_hat + eps)
        return -np.mean(H * (y - s))
    
    return minimize(_loss, 0, method='L-BFGS-B', jac=_jac, tol=1e-14).x[0]

def get_torch_device(args):
    if args['use_cpu']:
        return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    # elif torch.backends.mps.is_available():
    #     return 'mps'
    
    return 'cpu'