"""Stateful GRU regressor with explicit hidden state, exportable to ONNX for step-wise inference."""
from __future__ import annotations

import torch
import torch.nn as nn


class Highway(nn.Module):
    """h <- g * relu(Wt h + bt) + (1 - g) * h, g = sigmoid(Wg h + bg) (Srivastava et al. 2015); gate bias starts at -1."""

    def __init__(self, d):
        super().__init__()
        self.gate = nn.Linear(d, d); self.lin = nn.Linear(d, d)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, h):
        g = torch.sigmoid(self.gate(h))
        return g * torch.relu(self.lin(h)) + (1 - g) * h


class GRUModel(nn.Module):
    def __init__(self, n_features=112, hidden=128, layers=1, in_proj=0, dropout=0.0, x_clip=10.0, chrono=0.0,
                 head="linear", head_layers=1):
        super().__init__()
        self.head_kind, self.head_layers = head, head_layers
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.x_clip = x_clip
        self.hidden, self.layers = hidden, layers
        d_in = n_features
        self.proj = None
        if in_proj:
            self.proj = nn.Sequential(nn.Linear(n_features, in_proj), nn.GELU())
            d_in = in_proj
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        if head == "highway":  # LayerNorm -> k Highway blocks -> Linear
            self.head = nn.Sequential(nn.LayerNorm(hidden), *[Highway(hidden) for _ in range(head_layers)], nn.Linear(hidden, 2))
        else:
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        if chrono > 0:  # chrono initialisation (Tallec & Ollivier 2018): update-gate bias = log T, T log-uniform in [1, chrono]
            with torch.no_grad():
                for k in range(layers):
                    T = torch.exp(torch.rand(hidden) * torch.log(torch.tensor(float(chrono))))
                    getattr(self.gru, f"bias_ih_l{k}")[hidden:2 * hidden] = torch.log(T)
                    getattr(self.gru, f"bias_hh_l{k}")[hidden:2 * hidden] = 0.0

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean)); self.std.copy_(torch.as_tensor(std))

    def normalize(self, x):
        x = (x.float() - self.mean) / self.std
        x = torch.nan_to_num(x, nan=0.0, posinf=self.x_clip, neginf=-self.x_clip)
        return x.clamp(-self.x_clip, self.x_clip)

    def forward(self, x, h=None):
        """x: (B,T,F) raw features; h: (L,B,H) or None. Returns y (B,T,2), h."""
        x = self.normalize(x)
        if self.proj is not None:
            x = self.proj(x)
        out, h = self.gru(x, h)
        return self.head(out), h

    def init_state(self, batch, device):
        return torch.zeros(self.layers, batch, self.hidden, device=device)


class StepWrapper(nn.Module):
    """Single-step interface for ONNX: (x (1,F), h (L,1,H)) -> (y (2,), h)."""

    def __init__(self, m: GRUModel):
        super().__init__()
        self.m = m

    def forward(self, x, h):
        y, h = self.m(x.unsqueeze(0), h)
        return y[0, 0], h


def export_onnx(m: GRUModel, path: str, opset: int = 17):
    m = m.eval().cpu()
    w = StepWrapper(m)
    x = torch.zeros(1, m.mean.numel())
    h = m.init_state(1, "cpu")
    torch.onnx.export(w, (x, h), path, input_names=["x", "h"], output_names=["y", "h_out"],
                      opset_version=opset, dynamo=False)
    return path
