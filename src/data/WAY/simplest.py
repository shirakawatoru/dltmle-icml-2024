import numpy as np
import pandas as pd
from scipy.special import expit

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple

class SimpleWAYSyntheticData(Dataset):
    def __init__(self, rng:np.random.Generator, n:int, a_cf:Optional[int]=None, dtype=torch.float32):
        self.n = n
        self.tau = 1
        self.p = 1
        self.dtype = dtype

        W, L, A, Y, G, Q_last = _gen(rng, n, a_cf=a_cf)

        self.W = torch.tensor(W, dtype=dtype)
        self.L = torch.tensor(L, dtype=dtype)
        self.A = torch.tensor(A, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.G = torch.tensor(G, dtype=dtype)
        self.Q_tau = torch.tensor(Q_last, dtype=dtype)

        self.A_cf_0 = torch.zeros_like(self.A)
        self.A_cf_1 = torch.ones_like(self.A)

        self.dim_static = W.shape[1]
        self.dim_dynamic = self.p

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {
            "W": self.W[index],
            "L": self.L[index],
            "A": self.A[index],
            "Y": self.Y[index],
            "a": self.A_cf_1[index],
        }

    def get_dfs_for_ltmle(self):
        dic = {}

        for j in range(self.W.shape[1]):
            dic[f'W_{j:02d}'] = self.W[:,j]

        for t in range(self.tau):
            # for j in range(self.dim_dynamic):
            #     dic[f'L_{j:02d}_{t:02d}'] = self.L[:,t,j]

            dic[f'A_{t:02d}'] = self.A[:,t,0]
            dic[f'Y_{t:02d}'] = self.Y[:,t,0]

        df_ltmle = pd.DataFrame(dic)
        df_abar_0 = pd.DataFrame({f'A_{t:02d}': self.A_cf_0[:,t,0] for t in range(self.tau)})
        df_abar_1 = pd.DataFrame({f'A_{t:02d}': self.A_cf_1[:,t,0] for t in range(self.tau)})

        return df_ltmle, df_abar_0, df_abar_1

def compute_true_param(rng:np.random.Generator, n:int, a_cf:int) -> float:
    _, _, _, Y, _, _ = _gen(rng, n, a_cf)
    return Y.mean()


def _gen(rng:np.random.Generator, n:int, a_cf:Optional[int] = None):
    W = rng.normal(0, 1, n)
    L = np.zeros(n)
    G = expit(-0.5 * W + L) if a_cf is None else np.full(n, a_cf)
    A = rng.binomial(1, G)
    pY = expit(-3 + 0.2 * W + 0.2 * L - 2 * A)
    Y = rng.binomial(1, pY)

    return W[:,None], L[:,None,None], A[:,None,None], Y[:,None,None], G[:,None,None], pY[:,None,None]