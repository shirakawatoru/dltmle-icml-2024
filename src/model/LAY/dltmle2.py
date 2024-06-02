import torch
import torch.nn as nn

import numpy as np
from scipy.special import logit, expit

import lightning.pytorch as L

from ...utils import SinusoidalEncoder, solve_one_dimensional_submodel

class DeepLTMLE2(L.LightningModule):
    '''Class for deep ltmle that uses transformer as Q and G model for simultaneous quasi stochastic gradient descent with targeted loss

    .. math ::
        \\mathcal{L} = \\sum_{t=0}^\\tau \\mathcal{L}^Q_t + \\alpha \\mathcal{L}^G_t + \\beta \\mathcal{L}^\star_t 
    '''
    def __init__(self,
                 dim_static,
                 dim_dynamic,
                 tau,
                 dim_emb=16,
                 dim_emb_time=8,
                 dim_emb_type=8,
                 hidden_size=32,
                 num_layers=2,
                 nhead=8,
                 dropout=0.1,
                 learning_rate=1e-3,
                 alpha = 1,
                 beta = 1,
                 survival_outcome=True,
                 **kwargs):
        '''Initialization of DeepLTMLE with specified hyperparameters
        
        Parameters
        ----------
        dim_static: `integer`
            dimension of static covariates (baseline covariates)
        dim_dynamic: `integer`
            dimension of dynaic covariates (time-dependent covariates)
        tau: `integer`
            length of time horizon
        dim_emb: `integer`
            dimension of value embedding to transformer
        dim_emb_time: `integer`
            dimension of time embedding to transformer
        dim_emb_type: `integer`
            dimension of type embedding to transformer
        hidden_size: `integer`
            dimension of hidden state in transformer
        num_layers: `integer`
            number of layers in transformer
        nhead: `integer`
            number of attention heads in transformer
        dropout: `float`
            drop-out rate of transformer
        alpha: `float`
            weight of loss for propensity score
        beta: `float`
            weight of targeting loss
        '''
        super().__init__()

        self.tau = tau

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta

        self.dim_input_L = dim_static + dim_dynamic

        # embeddings
        self.emb_W = nn.Linear(dim_static, dim_emb)
        self.emb_L = nn.Linear(dim_dynamic, dim_emb)
        self.emb_A = nn.Linear(1,   dim_emb)
        self.emb_Y = nn.Linear(1,   dim_emb)

        # temporal embeddings
        self.emb_time = nn.Sequential(
            SinusoidalEncoder(dim=dim_emb_time),
            nn.Linear(dim_emb_time, dim_emb_time)
        )

        # type embeddings
        self.emb_type = nn.Parameter(torch.randn(4, dim_emb_type), requires_grad=True)

        # transformer encoder
        d_model = dim_emb + dim_emb_time + dim_emb_type
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

        self.logit_Q = nn.Linear(d_model, 1)
        self.G = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        self.survival_outcome  = survival_outcome

        self.eps = nn.Parameter(torch.zeros(tau, requires_grad=False))

    def _get_attention_mask(self, tau, device):
        if hasattr(self, '_attention_mask') and hasattr(self, '_attention_mask_tau'):
            if self._attention_mask_tau == tau:
                return self._attention_mask

        # input nodes = [W[0:1], L[0:tau], A[0:tau]]
        # DGP at t: L[t] > A[t]

        I = torch.triu(torch.full((tau, tau), True), diagonal=1) # attention < t
        J = torch.triu(torch.full((tau, tau), True), diagonal=0) # attention <= t

        # [0, 1, 1, 1]
        # [0, I, J, J]
        # [0, I, I, J]
        # [0, I, I, I]

        mask = torch.full((tau * 3 + 1, tau * 3 + 1), True)
        mask[:, 0] = False

        for i in range(3):
            for j in range(3):
                mask[(i*tau+1):((i+1)*tau+1), (j*tau+1):((j+1)*tau+1)] = I if i >= j else J
        
        self._attention_mask = mask.to(device)
        self._attention_mask_tau = tau

        return self._attention_mask

    def forward(self, W, L, A, Y, a):
        batch_size, tau = L.shape[0], L.shape[1]

        # embeddings
        # shape (batch_size, tau, dim_emb)
        z_W = self.emb_W(W[:,None,:])
        z_L = self.emb_L(L)
        z_A = self.emb_A(A)
        z_Y = self.emb_Y(Y)
        z_a = self.emb_A(a)

        # add time embeddings
        # shape (batch_size, tau, dim_emb+dim_emb_time)
        T_W = self.emb_time(torch.tensor([-1])).repeat(batch_size, 1, 1)
        T = self.emb_time(torch.arange(tau)).repeat(batch_size, 1, 1)

        # add type embeddings
        # shape (batch_size, tau, dim_emb+dim_emb_time+dim_emb_type)
        type_W = self.emb_type[0].repeat(batch_size, 1, 1)
        type_L = self.emb_type[1].repeat(batch_size, tau, 1)
        type_A = self.emb_type[2].repeat(batch_size, tau, 1)
        type_Y = self.emb_type[3].repeat(batch_size, tau, 1)

        z_W = torch.cat([z_W, T_W, type_W], axis=-1)
        z_L = torch.cat([z_L, T,   type_L], axis=-1)
        z_A = torch.cat([z_A, T,   type_A], axis=-1)
        z_Y = torch.cat([z_Y, T,   type_Y], axis=-1)
        z_a = torch.cat([z_a, T,   type_A], axis=-1)

        # transformer
        mask = self._get_attention_mask(tau, z_L.device)
        x = torch.cat([z_W, z_L, z_A, z_Y], axis=1) # shape: (batch_size, 3*tau, dim_emb+dim_emb_time+dim_emb_type)
        x = self.transformer(x, mask=mask)

        # V[t-1] > A[t-1] > (L[t] > Y[t] > C[t] > V[t] > A[t]) > L(t+1)

        # input:  W[0:1], L[0:tau], A[0:tau], Y[0:tau]
        # output: _[0:1], G[0:tau], Q[0:tau], _
        z_G, z_Q, _ = x[:,1:,:].reshape(batch_size, 3, tau, -1).transpose(0, 1)

        G = self.G(z_G)

        logit_Q_l = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_Q_l[:, 1:] = self.logit_Q(z_Q)

        eps = self.eps.view(1, tau, 1).repeat(batch_size, 1, 1)

        logit_Q_l_star = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_Q_l_star[:, 1:] = logit_Q_l[:, 1:] + eps

        Q_l = torch.sigmoid(logit_Q_l)
        Q_l_star = torch.sigmoid(logit_Q_l_star)

        # ----------------------------------------------
        # Counterfactual
        x = torch.cat([z_W, z_L, z_a, z_Y], axis=1) # shape: (batch_size, 3*tau, dim_emb+dim_emb_time+dim_emb_type)
        x = self.transformer(x, mask=mask)

        _, z_Q_a, _ = x[:,1:,:].reshape(batch_size, 3, tau, -1).transpose(0, 1)

        logit_Q_a = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_Q_a[:, :-1] = self.logit_Q(z_Q_a).detach() # block back propagation

        logit_Q_a_star = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_Q_a_star[:, :-1] = logit_Q_a[:, :-1] + eps.detach()

        Q_a = torch.sigmoid(logit_Q_a)
        Q_a_star = torch.sigmoid(logit_Q_a_star)

        # degeneration of Q
        Q_l, Q_l_star, Q_a, Q_a_star = self._set_deterministic_Q(Q_l, Q_l_star, Q_a, Q_a_star, Y)
 
        # IPW
        J = (A == a) / (G * A + (1 - G) * (1 - A)).detach()
        g = torch.ones(batch_size, tau + 1, 1, device=A.device)
        g[:, 1:] = J.cumprod(dim=1)
        g = torch.clip(g, 0, 100)

        # influence curve
        IC = (g * (Q_a_star - Q_l_star)).sum(dim=1)

        return {
            "Q_l": Q_l,
            "Q_l_star": Q_l_star,
            "Q_a": Q_a,
            "Q_a_star": Q_a_star,
            "G": G,
            "g": g,
            "IC": IC
        }
    
    def _set_deterministic_Q(self, Q_l, Q_l_star, Q_a, Q_a_star, Y):
        n, tau, _ = Y.shape

        R = torch.ones((n, tau + 1, 1), device=Y.device) # suvrival indicator
        R[:, 2:] = 1 - Y[:, :-1]

        T0 = torch.zeros((n, tau + 1, 1), device=Y.device) # indicator of t == 0
        T0[:, 0] = 1

        Q_a[:, -1] = Y[:, -1]
        if self.survival_outcome:
            Q_a[:, 1:-1] = torch.where(Y[:, :-1] == 1, 1, Q_a[:, 1:-1]) # Q^a_{t+1} = 1 if Y_t = 1 for t = 0, ..., tau-1    
        
        if self.survival_outcome:
            Q_l = torch.where(R == 1, Q_l, 1) # Q^l_{t+2} = 1 if Y_t = 1 for t = 0, ..., tau-1
        Q_l = torch.where(T0 == 0, Q_l, Q_a[:, 0].mean())

        Q_a_star[:, -1] = Y[:, -1]
        if self.survival_outcome:
            Q_a_star[:, 1:-1] = torch.where(Y[:, :-1] == 1, 1, Q_a_star[:, 1:-1]) # Q^a_{t+1} = 1 if Y_t = 1 for t = 0, ..., tau-1
        
        if self.survival_outcome:
            Q_l_star = torch.where(R == 1, Q_l_star, 1) # Q^l_{t+2} = 1 if Y_t = 1 for t = 0, ..., tau-1
        Q_l_star = torch.where(T0 == 0, Q_l_star, Q_a_star[:, 0].mean())

        return Q_l, Q_l_star, Q_a, Q_a_star

    def loss(self, S_hat, S):
        Q_l, Q_l_star, Q_a, Q_a_star, G, g, IC = S_hat.values()
        W, L, A, Y, a = S.values()

        H = nn.BCELoss(reduction='none')

        R = torch.ones_like(Y)
        if self.survival_outcome:
            R[:, 1:] = 1 - Y[:, :-1] # indicator of survival

        loss_Q = (R * H(Q_l[:, 1:], Q_a[:, 1:])).sum(dim=1).mean()
        loss_G = (R * H(G, A)).sum(dim=1).mean()
        loss_Q_star = (g[:, 1:] * R * H(Q_l_star[:, 1:], Q_a_star[:, 1:])).sum(dim=1).mean()

        loss_Q_last = (R[:,-1] * H(Q_l_star[:, -1], Q_a_star[:, -1])).mean()

        loss = loss_Q + self.alpha * loss_G + self.beta * loss_Q_star

        return {
            "L": loss,
            "Q": loss_Q,
            "G": loss_G,
            "GQ": loss_G + loss_Q,
            "Q_star": loss_Q_star,
            "Q_last": loss_Q_last,
            "PnIC": IC.mean(),
            "PnIC2": (IC ** 2).mean()
        }

    def training_step(self, batch, batch_idx):
        loss = self.loss(self(**batch), batch)

        for k, v in loss.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss["L"]

    def validation_step(self, batch, batch_idx):
        loss = self.loss(self(**batch), batch)

        for k, v in loss.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    
    def test_step(self, batch, batch_idx):
        loss = self.loss(self(**batch), batch)

        for k, v in loss.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss["L"]

    def predict_step(self, batch, batch_idx):
        x = self(**batch)
        x["loss"] = self.loss(x, batch)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def solve_canonical_gradient(self, trainer, loader, tau):
        eps = torch.zeros(tau+1)

        # survival indicator (needed for weight computation)
        r = np.ones((len(loader.dataset), tau))
        Y = np.concatenate([x["Y"] for x in loader], axis=0)[:,:,0]
        r[:, 1:] = 1 - Y[:,:-1]

        preds = trainer.predict(self, loader)
        y_hat = np.concatenate([x["Q_l"] for x in preds], axis=0)[:,:,0]
        y = np.concatenate([x["Q_a"] for x in preds], axis=0)[:,:,0]
        g = np.concatenate([x["g"] for x in preds], axis=0)[:,:,0]

        for t in reversed(range(tau)):
            eps[t] = solve_one_dimensional_submodel(
                y_hat[:,t+1], 
                expit(logit(y[:,t+1]) + float(eps[t+1])), 
                r[:,t] * g[:,t+1]
                )

            print(t + 1, eps[t])

        self.eps = torch.nn.Parameter(eps[:-1], requires_grad=False)
    
    def solve_canonical_gradient_common_eps(
            self,
            trainer, 
            loader, 
            tau, 
            max_iter=1000, 
            tol=1e-6, 
            stop_pnic_se_ratio=False,
            max_delta_eps=None,
            ):
        n = len(loader.dataset)
        r = np.ones((n, tau))
        Y = np.concatenate([x["Y"] for x in loader], axis=0)[:,:,0]
        r[:, 1:] = 1 - Y[:,:-1]

        preds = trainer.predict(self, loader)
        y_hat = np.concatenate([x["Q_l"] for x in preds], axis=0)[:,1:tau+1,0]
        y = np.concatenate([x["Q_a"] for x in preds], axis=0)[:,1:tau+1,0]
        g = np.concatenate([x["g"] for x in preds], axis=0)[:,1:tau+1,0]
        H = r * g

        eps = np.zeros(max_iter)

        for i in range(max_iter):
            _eps = solve_one_dimensional_submodel(y_hat.ravel(), y.ravel(), H.ravel())

            if max_delta_eps is not None:
                _eps = np.clip(_eps, -max_delta_eps, max_delta_eps)

            eps[i] = _eps

            y_hat = expit(logit(y_hat) + eps[i])
            y[:,:-1] = expit(logit(y[:,:-1]) + eps[i])

            if stop_pnic_se_ratio:
                ic = (g * (y - y_hat)).sum(axis=1)
                se = np.sqrt((ic ** 2).mean() / n)
                if np.abs(ic.mean() / se) < 1 / np.log(n):
                    break

            if np.abs(eps[i]) < tol:
                break

        print('eps: ', eps[:i+1])
        print('eps.sum = ', eps.sum())

        eps = torch.full((tau,), eps.sum())
        self.eps = torch.nn.Parameter(eps, requires_grad=False)

    def get_estimates_from_prediction(self, pred, loader, verbose=True):
        Q_l_star = torch.cat([x["Q_l_star"] for x in pred], axis=0).detach().numpy().squeeze()
        Q_a_star = torch.cat([x["Q_a_star"] for x in pred], axis=0).detach().numpy().squeeze()
        IC = torch.cat([x["IC"] for x in pred], axis=0).detach().numpy().squeeze()

        PnIC = IC.mean()
        PnIC2 = (IC ** 2).mean()
        EIC = np.abs(PnIC / PnIC2 ** 0.5)

        # self.W[index], self.L[index], self.A[index], self.Y[index], self.A_cf_1[index]
        Y = torch.cat([x["Y"] for x in loader], axis=0).detach().numpy()[:,:,0]
        R = np.ones((Y.shape[0], Y.shape[1] + 1))
        R[:, 2:] = 1 - Y[:, :-1]

        est = Q_a_star[:, 0].mean()
        se = np.sqrt((IC ** 2).mean() / IC.shape[0])

        if verbose:
            print("est: ", est)
            print("CI: ", est - 1.96 * se, est + 1.96 * se)
            print("se: ", se)

            # print("lambda = {}".format(model.lam))
            print("E_n[IC] = {}".format(PnIC))
            print("PnIC/√PnIC2 = {}".format(EIC))

            print("Q_l_star", (R * Q_l_star).sum(axis=0) / R.sum(axis=0))
            print("Q_a_star", (R * Q_a_star).sum(axis=0) / R.sum(axis=0))

        return est, se, PnIC, EIC
