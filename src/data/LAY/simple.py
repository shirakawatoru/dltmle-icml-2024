import numpy as np
import pandas as pd
from scipy.special import expit

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple

class SimpleSyntheticData(Dataset):
    def __init__(self, rng:np.random.Generator, n:int, tau:int, a_cf:Optional[int]=None, dtype=torch.float32):
        self.n = n
        self.tau = tau
        self.p = 1
        self.dtype = dtype

        W, L, A, Y, G, Q_last = _gen(rng, n, tau, a_cf=a_cf)

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
    
    def plot(self, plt):
        _plot(plt, self.L, self.A, self.Y)

    def get_dfs_for_ltmle(self):
        dic = {}

        for j in range(self.W.shape[1]):
            dic[f'W_{j:02d}'] = self.W[:,j]

        for t in range(self.tau):
            for j in range(self.dim_dynamic):
                dic[f'L_{j:02d}_{t:02d}'] = self.L[:,t,j]

            dic[f'A_{t:02d}'] = self.A[:,t,0]
            dic[f'Y_{t:02d}'] = self.Y[:,t,0]

        df_ltmle = pd.DataFrame(dic)
        df_abar_0 = pd.DataFrame({f'A_{t:02d}': self.A_cf_0[:,t,0] for t in range(self.tau)})
        df_abar_1 = pd.DataFrame({f'A_{t:02d}': self.A_cf_1[:,t,0] for t in range(self.tau)})

        return df_ltmle, df_abar_0, df_abar_1

def compute_true_param(rng:np.random.Generator, n:int, tau:int, a_cf:int) -> float:
    _, _, _, Y, _, _ = _gen(rng, n, tau, a_cf)
    return Y[:,-1].mean()


def _gen(rng:np.random.Generator, n:int, tau:int, a_cf:Optional[int] = None):
    W = rng.normal(0, 1, n)

    L = np.zeros((n, tau))
    A = np.zeros((n, tau))
    Y = np.zeros((n, tau))
    G = np.zeros((n, tau))
    pY = np.zeros((n, tau))

    t = 0
    L[:, t] = rng.normal(0.1 * W, 1)
    G[:, t] = expit(-0.5 * W + L[:, t]) if a_cf is None else a_cf
    A[:, t] = rng.binomial(1, G[:, t])
    pY[:, t] = expit(-3 + 0.2 * W + 0.2 * L[:, t] - 2 * A[:, t])
    Y[:, t] = rng.binomial(1, pY[:, t])

    for t in range(1, tau):
        L[:, t] = rng.normal(0.1 * W - 0.1 * A[:, t-1], 1)
        G[:, t] = expit(-0.5  + 0.3 * W + 0.3 * L[:, t] + 6.5 * A[:, t-1] - 2.6) if a_cf is None else a_cf
        A[:, t] = rng.binomial(1, G[:, t])
        pY[:, t] = expit(-4 + 0.2 * W + 0.2 * L[:, t] - 0.5 * A[:, t])
        Y[:, t] = rng.binomial(1, pY[:, t])

        Y[Y[:, t-1] == 1, t] = 1

    return W[:,None], L[:,:,None], A[:,:,None], Y[:,:,None], G[:,:,None], pY[:,-1,None]

def _plot(plt, L, A, Y):
    plt.figure(figsize=(6*3, 4))

    n, tau, p = L.shape

    T = np.arange(tau)

    plt.subplot(1, 3, 1)
    plt.title('L')
    plt.ylabel('t')
    plt.plot(T, L[:,:].mean(axis=0))
    plt.grid()

    print((A==1).all(axis=1).float().mean())
    print((A==0).all(axis=1).float().mean())

    plt.subplot(1, 3, 2)
    plt.title('A')
    plt.ylabel('t')
    plt.plot(T, A[:,:].mean(axis=0))
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.title('Y')
    plt.ylabel('t')
    plt.plot(T, Y.mean(axis=0))
    plt.grid()

    plt.tight_layout()

    plt.show()