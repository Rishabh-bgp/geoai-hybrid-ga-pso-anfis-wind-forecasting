"""Layers 3-5: ANFIS (first-order TSK, Gaussian MFs) -> CNN-LSTM with MC dropout -> quantile head."""
import torch
import torch.nn as nn

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


class ANFIS(nn.Module):
    """Rule firing strengths w_r = prod_i exp(-(x_i-c_ri)^2 / 2 s_ri^2), normalised; output per rule
    is a linear consequent. Returns normalised firing strengths and rule outputs as features."""
    def __init__(self, n_in, n_rules):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_rules, n_in)); self.log_s = nn.Parameter(torch.zeros(n_rules, n_in))
        self.conseq = nn.Linear(n_in, n_rules)

    def forward(self, x):                                       # (..., n_in)
        d = (x.unsqueeze(-2) - self.c) / self.log_s.exp()
        w = torch.softmax(-0.5 * (d ** 2).sum(-1), -1)           # normalised firing strengths
        return torch.cat([w, w * self.conseq(x)], -1)


class GeoHybrid(nn.Module):
    def __init__(self, n_sites, n_static, horizons, filters=32, hidden=32, n_rules=8, dropout=0.1):
        super().__init__()
        self.anfis = ANFIS(n_sites, n_rules)
        c = 2 * n_rules + n_sites
        self.cnn = nn.Sequential(nn.Conv1d(c, filters, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(filters, 2 * filters, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(2 * filters, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.static = nn.Linear(n_static, hidden)
        self.out = nn.Linear(hidden, len(horizons) * len(QUANTILES)); self.H = len(horizons)

    def forward(self, x, static):                                # x (B,T,S), static (B,3)
        z = torch.cat([x, self.anfis(x)], -1)
        h = self.cnn(z.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        h = self.drop(h[:, -1] + torch.tanh(self.static(static)))
        q = self.out(h).view(-1, self.H, len(QUANTILES))
        return q[..., :1] + torch.cat([torch.zeros_like(q[..., :1]),
                                       torch.cumsum(nn.functional.softplus(q[..., 1:]), -1)], -1)  # monotone quantiles


def pinball(q, y):
    t = torch.tensor(QUANTILES).view(1, 1, -1)
    e = y.unsqueeze(-1) - q
    return torch.maximum(t * e, (t - 1) * e).mean()
