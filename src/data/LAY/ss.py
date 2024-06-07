import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import os

class CIRCSDGP:
    def __init__(self, data_path):
        def _load(node):
            df =  pd.read_csv(os.path.join(data_path, f"{node}.csv.gz"))
            if node == "W":
                return df.iloc[:,1:] # remove id column
            else:
                return df.iloc[:,2:] # remove id and k columns

        # read Circs data
        self.df_W = _load("W")
        self.df_L = _load("L")
        self.df_V = _load("V")
        self.df_A = _load("A")
        self.df_C = _load("C")
        self.df_Y = _load("Y")
        self.df_A_cf_120 = _load("abar_120")
        self.df_A_cf_140 = _load("abar_140")

        self.n = self.df_W.shape[0]
        self.tau = self.df_L.shape[0] // self.n

        self.W = self.df_W.values
        _L = self.df_L.values.reshape(self.n, self.tau, -1)
        _V = self.df_V.values.reshape(self.n, self.tau, -1)
        self.L = np.concatenate([_L, _V], axis=2)
        self.A = self.df_A.values.reshape(self.n, self.tau, -1)
        self.C = self.df_C.values.reshape(self.n, self.tau, -1)
        self.Y = self.df_Y.values.reshape(self.n, self.tau, -1)

        A_cf_120 = self.df_A_cf_120.values.reshape(self.n, self.tau - 1, -1)
        A_cf_140 = self.df_A_cf_140.values.reshape(self.n, self.tau - 1, -1)

        self.A_cf_120 = np.zeros((self.n, self.tau, 1))
        self.A_cf_120[:,:-1] = A_cf_120
        self.A_cf_120[:,-1] = A_cf_120[:,-1]

        self.A_cf_140 = np.zeros((self.n, self.tau, 1))
        self.A_cf_140[:,:-1] = A_cf_140
        self.A_cf_140[:,-1] = A_cf_140[:,-1]

        self.p_Y = np.zeros((self.n, self.tau, 1))

        self.lambda_y = []

        for t in range(self.tau):
            idx = (self.Y[:,t-1,0] == 0) if t > 0 else np.full(self.n, True)
            idx &= (self.C[:,t,0] == 0)

            n = idx.sum()

            print("Fitting Regression for t=%d, n=%d" % (t, idx.sum()))

            X = np.concatenate([
                self.W[idx],
                self.L[idx, :t+1].reshape(n, -1),
                self.A[idx, :t+1].reshape(n, -1)
                ], axis=1)
            y = self.Y[idx, t, 0]


            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1234)
            clf = XGBClassifier(early_stopping_rounds=10)
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            clf = XGBClassifier(n_estimators=clf.best_iteration).fit(X, y)

            # Predict the probabilities
            prob_y = clf.predict_proba(X)[:, 1]

            print("mean observed Y  = %f" % y.mean())
            print("mean predicted Y = %f" % prob_y.mean())

            X = np.concatenate([
                self.W,
                self.L[:, :t+1].reshape(self.n, -1),
                self.A[:, :t+1].reshape(self.n, -1)
                ], axis=1)
            self.p_Y[:, t, 0] = clf.predict_proba(X)[:, 1]

            self.lambda_y.append(clf)

class SemiSyntheticData(Dataset):
    def __init__(self, rng_data:np.random.Generator, circs_dgp:CIRCSDGP, dtype=torch.float32):
        self.circs_dgp = circs_dgp
        self.dtype = dtype

        # remove Y=1 observations
        idx = (self.circs_dgp.Y==0).all(axis=1)[:,0]
        idx &= (self.circs_dgp.C==0).all(axis=1)[:,0]
        
        self.W_orig = self.circs_dgp.W[idx]
        self.L_orig = self.circs_dgp.L[idx]
        self.A_orig = self.circs_dgp.A[idx]
        self.A_cf_120_orig = self.circs_dgp.A_cf_120[idx]
        self.A_cf_140_orig = self.circs_dgp.A_cf_140[idx]
        self.p_Y_orig = self.circs_dgp.p_Y[idx]

        self.n = self.W_orig.shape[0]
        self.tau = self.L_orig.shape[1]
        self.dim_static = self.W_orig.shape[1]
        self.dim_dynamic = self.L_orig.shape[2]

        #generate semi synthetic data
        W, L, A, Y, A_cf_120, A_cf_140 = \
            _gen(rng_data, self.n, self.tau,
                self.W_orig, self.L_orig, self.A_orig, 
                self.A_cf_120_orig, self.A_cf_140_orig, self.p_Y_orig)
        
        self.W = torch.tensor(W, dtype=dtype)
        self.L = torch.tensor(L, dtype=dtype)
        self.A = torch.tensor(A, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.A_cf_120 = torch.tensor(A_cf_120, dtype=dtype)
        self.A_cf_140 = torch.tensor(A_cf_140, dtype=dtype)
    
    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.W[index], self.L[index], self.A[index], self.Y[index], self.A_cf_1[index]
    
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
        df_abar_0 = pd.DataFrame({f'A_{t:02d}': self.A_cf_120[:,t,0] for t in range(self.tau)})
        df_abar_1 = pd.DataFrame({f'A_{t:02d}': self.A_cf_140[:,t,0] for t in range(self.tau)})

        return df_ltmle, df_abar_0, df_abar_1

def compute_true_param(rng_data:np.random.Generator) -> float:
    _, _, _, Y = _gen(rng_data)
    return Y[:,-1].mean()

def _gen(
        rng_data:np.random.Generator, n, tau, 
        W_orig, L_orig, A_orig, A_cf_120_orig, A_cf_140_orig, p_Y_orig
        ):
    rng_model = np.random.default_rng(1234)

    sample_idx = rng_model.choice(n, n)

    W = W_orig[sample_idx]
    L = L_orig[sample_idx]
    A = A_orig[sample_idx]
    A_cf_120 = A_cf_120_orig[sample_idx]
    A_cf_140 = A_cf_140_orig[sample_idx]

    Y = np.zeros((n, tau))  

    for t in range(tau):
        Y[:, t] = rng_data.binomial(1, p_Y_orig[:, t, 0])
        print("t=%d, mean Y = %f" % (t, Y[:,t].mean()))

        if t > 0:
            Y[Y[:,t-1]==1, t] = 1

    Y = Y[:, :, None]

    return W, L, A, Y, A_cf_120, A_cf_140

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