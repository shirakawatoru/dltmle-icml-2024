import torch
import torch.nn as nn
import lightning.pytorch as L

import numpy as np
from scipy.optimize import minimize

class VariationalLSTM(nn.Module):
    '''
    A LSTM with variational dropouts [1].

    References
    ----------
    .. [1] Gal, Y., & Ghahramani, Z. (2016).
        A theoretically grounded application of dropout in recurrent neural networks.
        Advances in neural information processing systems (Vol. 29).
        https://proceedings.neurips.cc/paper_files/paper/2016/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf
    '''
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=1,
            dropout=0.0
            ):
        super().__init__()

        self.lstm_layers = nn.ModuleList(
            [nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)] + 
            [nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size) for _ in range(num_layers - 1)]
            )

        self.hidden_size = hidden_size
        self.dropout = dropout

    def forward(self, x, init_states=None):
        for lstm_cell in self.lstm_layers:

            # Customised LSTM-cell for variational LSTM dropout (Tensorflow-like implementation)
            if init_states is None:  # Encoder - init states are zeros
                hx = torch.zeros((x.shape[0], self.hidden_size)).type_as(x)
                cx = torch.zeros((x.shape[0], self.hidden_size)).type_as(x)
            else:  # Decoder init states are br of encoder
                hx, cx = init_states[0][0,:,:], init_states[1][0,:,:]

            # Variational dropout - sampled once per batch
            out_dropout = torch.bernoulli(hx.data.new(hx.data.size()).fill_(1 - self.dropout)) / (1 - self.dropout)
            h_dropout = torch.bernoulli(hx.data.new(hx.data.size()).fill_(1 - self.dropout)) / (1 - self.dropout)
            c_dropout = torch.bernoulli(cx.data.new(cx.data.size()).fill_(1 - self.dropout)) / (1 - self.dropout)

            output = []
            for t in range(x.shape[1]):
                hx, cx = lstm_cell(x[:, t, :], (hx, cx))
                if lstm_cell.training:
                    out = hx * out_dropout
                    hx, cx = hx * h_dropout, cx * c_dropout
                else:
                    out = hx
                output.append(out)

            x = torch.stack(output, dim=1)

        return x, (torch.unsqueeze(hx, 0), torch.unsqueeze(cx, 0))

class DeepACE(L.LightningModule):
    '''Class for Deep ACE [1].
    
    References
    ----------
    .. [1] Frauen, D., Hatt, T., Melnychuk, V., & Feuerriegel, S. (2023).
        Estimating average causal effects from patient trajectories.
        Proceedings of the AAAI Conference on Artificial Intelligence, 37(6), 7586–7594.
        https://doi.org/10.1609/aaai.v37i6.25921
    '''

    def __init__(
            self,
            dim_static,
            dim_dynamic,
            tau,
            hidden_size,
            num_layers,
            dropout,
            learning_rate,
            alpha,
            beta,
            **kwargs
            ):
        super().__init__()

        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        
        dim_lstml_input = dim_dynamic + 1 + 1 # L + Y + A

        self.lstm = VariationalLSTM(
            input_size=dim_lstml_input,
            hidden_size=hidden_size,
            num_layers=num_layers, 
            dropout=dropout
        )

        def _QHead():
            return nn.Sequential(
                nn.Linear(dim_static + 1 + hidden_size, hidden_size),
                nn.Dropout(p=dropout),
                nn.ELU(),
                nn.Linear(hidden_size, 1),
            )

        def _GHead():
            return nn.Sequential(
                nn.Linear(dim_static + hidden_size, hidden_size),
                nn.Dropout(p=dropout),
                nn.ELU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

        self.Q = nn.ModuleList([_QHead() for _ in range(tau)])
        self.G = nn.ModuleList([_GHead() for _ in range(tau)])
        self.eps = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, batch):
        W = batch["W"]
        L = batch["L"]
        A = batch["A"]
        Y = batch["Y"]
        a = batch["a"]

        n, tau, _ = A.shape

        W = W[:, None, :].repeat((1, tau, 1))

        L_Y  = torch.cat([torch.zeros_like(Y[:, :1]), Y[:, :-1]], dim=1)
        L = torch.cat([L, L_Y], dim=2)
        
        x = torch.cat([L, A], dim=2)
        x[:, 1:, -1] = x[:, :-1, -1]
        x[:, 0, -1] = 0
        x, _ = self.lstm(x)
        x = torch.cat([W, x], dim=2)

        x_cf = torch.cat([L, a], dim=2)
        x_cf[:, 1:, -1] = x_cf[:, :-1, -1]
        x_cf[:, 0, -1] = 0
        x_cf, _ = self.lstm(x_cf)
        x_cf = torch.cat([W, x_cf], dim=2)

        # ---- G part ----

        G = torch.empty((n, tau, 1), device=A.device)
        for t in range(tau):
            G[:, t] = self.G[t](x[:, t])

        g = (A == a) / (A * G + (1 - A) * (1 - G)).detach()
        g = g.cumprod(dim = 1)
        g = torch.clip(g, 0, 100)

        # ---- Q part ----
        Q = torch.zeros((n, tau, 1), device=A.device)
        V = torch.zeros((n, tau + 1, 1), device=A.device)

        x = torch.cat([A, x], dim=2)
        x_cf = torch.cat([a, x_cf], dim=2)

        for t in range(tau):
            Q[:, t] = self.Q[t](x[:,t])
            V[:, t] = self.Q[t](x_cf[:,t]).detach()

        V[:, -1] = Y[:, -1]

        # Clever covariate
        q = torch.zeros((n, tau + 1, 1), device=A.device, requires_grad=False)
        for t in reversed(range(tau)):
            q[:, t] = q[:, t+1] - g[:, t]

        Q_star = Q.detach() + q[:, :-1] * self.eps
        V_star = V.detach() + q * self.eps

        # Q, V, Q_star, V_star = self._deterministic_Q(Q, V, Q_star, V_star, Y)

        IC = (g * (V_star[:, 1:] - V_star[:, :-1])).sum(dim=1)

        return {
            "Q": Q,
            "Q_star": Q_star,
            "V": V,
            "V_star": V_star,
            "G": G,
            "g": g,
            "q": q,
            "IC": IC
        }

    def loss(self, S_hat, S):
        Q, Q_star, V, V_star, G, g, q, IC = S_hat.values()
        W, L, A, Y, A_cf = S.values()

        H = nn.MSELoss(reduction='none')
        K = nn.BCELoss(reduction='none')

        loss_Q = H(Q, V[:, 1:]).sum(dim=1).mean()
        loss_Q_star = H(V_star[:, :-1], V_star[:, 1:]).sum(dim=1).mean()
        loss_G = K(G, A).sum(dim=1).mean()

        loss_Q_last = K(torch.clip(V[:, -2], 0, 1), V[:, -1]).mean()

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
        loss = self.loss(self(batch), batch)

        for k, v in loss.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        return loss["L"]
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch)

        for k, v in loss.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def predict_step(self, batch, batch_idx):
        x = self(batch)
        x["loss"] = self.loss(x, batch)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def solve_canonical_gradient(self, trainer, loader, tau):
        n = len(loader.dataset)
        r = np.ones((n, tau))
        Y = np.concatenate([x["Y"] for x in loader], axis=0)[:,:,0]
        r[:, 1:] = 1 - Y[:,:-1]

        preds = trainer.predict(self, loader)
        V = np.concatenate([x["V"] for x in preds], axis=0)[:,:,0]
        q = np.concatenate([x["q"] for x in preds], axis=0)[:,:,0]

        eps = _solve_one_dimensional_submodel(
            (V[:, 1:] - V[:, :-1]).ravel(),
            (q[:, 1:] - q[:, :-1]).ravel(),
            )
        print('eps: ', eps)

        self.eps = torch.nn.Parameter(torch.tensor(eps), requires_grad=False)

    def get_estimates_from_prediction(self, pred, loader, verbose=True):
        Q_l_star = torch.cat([x["Q_star"] for x in pred], axis=0).detach().numpy().squeeze()
        Q_a_star = torch.cat([x["V_star"] for x in pred], axis=0).detach().numpy().squeeze()
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

            print("Q_l_star", (R[:,:-1] * Q_l_star).sum(axis=0) / R[:,:-1].sum(axis=0))
            print("Q_a_star", (R * Q_a_star).sum(axis=0) / R.sum(axis=0))

        return est, se, PnIC, EIC

def _solve_one_dimensional_submodel(dy, dq):
    def _loss(eps):
        '''weighted logistic loss function (binary cross entropy loss)'''
        return ((dy + dq * eps) ** 2).mean()
    
    def _jac(eps):
        '''gradient of the loss function'''
        return 2 * ((dy + dq * eps) * dq).mean()
    
    return minimize(_loss, 0, method='L-BFGS-B', jac=_jac, tol=1e-14).x[0]

