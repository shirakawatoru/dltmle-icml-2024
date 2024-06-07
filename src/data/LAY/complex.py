import numpy as np
import pandas as pd
from scipy.special import expit

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple

class ComplexSyntheticData(Dataset):
    def __init__(self, rng_data:np.random.Generator, n:int, tau:int, p:int, lag:int, a_cf:Optional[int]=None, dtype=torch.float32):
        self.n = n
        self.tau = tau
        self.p = p
        self.lag = lag
        self.dtype = dtype

        W, L, A, Y, G = _gen(rng_data, n, tau, p, lag, a_cf=a_cf)

        self.W = torch.tensor(W, dtype=dtype)
        self.L = torch.tensor(L, dtype=dtype)
        self.A = torch.tensor(A, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.G = torch.tensor(G, dtype=dtype)

        self.A_cf_0 = torch.zeros_like(self.A)
        self.A_cf_1 = torch.ones_like(self.A)
        self.A_cf_1 = torch.tensor(expit(G) > 0.5, dtype=dtype)

        self.dim_static = W.shape[1]
        self.dim_dynamic = p

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

def compute_true_param(rng_data:np.random.Generator, n:int, tau:int, p:int, lag:int, a_cf:int) -> float:
    _, _, _, Y, _ = _gen(rng_data, n, tau, p, lag, a_cf)
    return Y[:,-1].mean()


def _gen(rng_data:np.random.Generator, n:int, tau:int, p:int, lag:int, a_cf:Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng_model = np.random.default_rng(1234)
    alpha = np.array([rng_model.normal(1 / (i + 2), 0.02) for i in reversed(range(lag))])
    beta = np.array([rng_model.normal(1 / (i + 2), 0.02) for i in reversed(range(lag))])
    gamma = rng_model.binomial(1, 0.5, size=lag) * 2 - 1

    L = rng_data.normal(0, 0.1, (n, tau + lag, p))
    A = rng_data.binomial(1, 0.5, (n, tau + lag))
    G = np.zeros((n, tau))

    Y = np.zeros((n, tau))
    
    e_L = rng_model.normal(0, 0.3, size=(n, tau, p))
    e_A1 = rng_model.normal(0, 0.2, size=(n, tau))
    e_A2 = rng_model.normal(0, 0.05, size=(n, tau))

    for t in range(tau):
        L[:, t+lag] = np.tanh((
            np.einsum('j,ijk->ijk', alpha, L[:, t:t+lag]) + 
            np.einsum('j,j,ijk->ijk', beta, gamma, 2 * A[:, t:t+lag, None] - 1).repeat(p, axis=-1)
            ).sum(axis=1) + e_L[:,t])
        
        G[:, t] =np.tan(L[:, t:t+lag].mean(axis=-1).prod(axis=1)) + e_A1[:, t]
        # pi = np.tan(L[:, t:t+lag].mean(axis=-1).prod(axis=1) + A[:,t:t+lag].mean(axis=1)) + \
        #     4 * L[:, t:t+lag].mean(axis=-1).prod(axis=1) + 6.5 * A[:,t:t+lag].mean(axis=1) - 3
        # pi = np.sinh(L[:, t:t+lag].mean(axis=-1).prod(axis=1) + 3.6 * (A[:,t:t+lag].mean(axis=1) - 0.5))
        A[:, t+lag] = (expit(G[:, t] + e_A2[:, t]) > 0.5).astype(int)

        if a_cf is not None: # counterfactual a
            A[:, t+lag] = a_cf

        lam = np.tan(L[:, t:t+lag].mean(axis=-1).prod(axis=1) - 0.7 * (A[:,t:t+lag].mean(axis=1) - 0.5)) - 4.5
        Y[:, t] = rng_data.binomial(1, expit(lam))

        if t > 0:
            Y[Y[:,t-1]==1, t] = 1

    W = np.concatenate([L[:,:lag], A[:,:lag,None]], axis=-1).reshape(n, -1)
    L = L[:, lag:] # shape (n, tau, p)
    A = A[:, lag:, None] # shape (n, tau, 1)
    Y = Y[:, :, None] # shape (n, tau, 1)
    G = G[:, :, None] # shape (n, tau, 1)

    return W, L, A, Y, G

def _plot(plt, L, A, Y):
    plt.figure(figsize=(6*3, 4))

    n, tau, p = L.shape

    T = np.arange(tau)

    plt.subplot(1, 3, 1)
    for j in range(p):
        # plt.plot(L[:,lag:,j].mean(axis=0))
        plt.title('L')
        plt.ylabel('t')
        plt.errorbar(T, L[:,:,j].mean(axis=0), np.sqrt(L[:,:,j].var(axis=0)), capsize=3)
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