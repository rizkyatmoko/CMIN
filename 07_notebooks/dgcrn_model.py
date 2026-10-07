"""Compact DGCRN baseline (Li et al., 2023, ACM TKDD 17(1):9) under the CMIN data contract.

The same class is pasted verbatim into CMIN_dgcrn.ipynb by make_dgcrn_notebook.py, so the
local shape and parameter tests below exercise exactly the code that trains on Colab.

What is kept from DGCRN
  * a recurrent cell whose gates are graph convolutions (DGCRM), not dense layers;
  * a dynamic adjacency regenerated at EVERY time step by a hypernetwork that reads the
    current input and the previous hidden state, filtered through the predefined graph;
  * dynamic node embeddings formed as learned static embeddings modulated by the hypernetwork
    output, and an asymmetric adjacency ReLU(tanh(a (E1 E2' - E2 E1'))) with self-loops and
    degree normalisation;
  * propagation over both the predefined graph and the dynamic graph (forward and reverse).

What is simplified, as for every compact baseline in the protocol
  * encoder only, with a direct multi-horizon head instead of the seq2seq decoder, so no
    curriculum learning or scheduled sampling;
  * one propagation hop per graph instead of K-step diffusion;
  * no separate time-of-day embedding (the calendar features are already node inputs).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DGCRN(nn.Module):
    def __init__(self, Fn, d, H, N, A, emb=16, hyper=16, alpha=3.0):
        super().__init__()
        self.d, self.alpha = d, alpha
        self.register_buffer('A', A)
        self.E1 = nn.Parameter(torch.randn(N, emb) * 0.1)
        self.E2 = nn.Parameter(torch.randn(N, emb) * 0.1)
        c = Fn + d
        self.hyper1 = nn.Sequential(nn.Linear(2 * c, hyper), nn.Tanh(), nn.Linear(hyper, emb))
        self.hyper2 = nn.Sequential(nn.Linear(2 * c, hyper), nn.Tanh(), nn.Linear(hyper, emb))
        self.g_u = nn.Linear(4 * c, d)       # update gate
        self.g_r = nn.Linear(4 * c, d)       # reset gate
        self.g_c = nn.Linear(4 * c, d)       # candidate state
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, H))

    def dynamic_adjacency(self, xh):
        """xh: (B, N, Fn+d) -> row-normalised dynamic adjacency (B, N, N)."""
        z = torch.cat([xh, torch.einsum('ij,bjc->bic', self.A, xh)], -1)
        e1 = torch.tanh(self.alpha * self.E1.unsqueeze(0) * self.hyper1(z))
        e2 = torch.tanh(self.alpha * self.E2.unsqueeze(0) * self.hyper2(z))
        Ad = F.relu(torch.tanh(self.alpha * (e1 @ e2.transpose(1, 2) - e2 @ e1.transpose(1, 2))))
        Ad = Ad + torch.eye(Ad.shape[-1], device=Ad.device).unsqueeze(0)
        return Ad / Ad.sum(-1, keepdim=True)

    def gconv(self, z, Ad, lin):
        """One hop over the predefined graph and over the dynamic graph in both directions."""
        return lin(torch.cat([z,
                              torch.einsum('ij,bjc->bic', self.A, z),
                              Ad @ z,
                              Ad.transpose(1, 2) @ z], -1))

    def forward(self, xn, xg=None, *a):
        B, N, L_, _ = xn.shape
        h = xn.new_zeros(B, N, self.d)
        for t in range(L_):
            x = xn[:, :, t, :]
            xh = torch.cat([x, h], -1)
            Ad = self.dynamic_adjacency(xh)
            u = torch.sigmoid(self.gconv(xh, Ad, self.g_u))
            r = torch.sigmoid(self.gconv(xh, Ad, self.g_r))
            cand = torch.tanh(self.gconv(torch.cat([x, r * h], -1), Ad, self.g_c))
            h = u * h + (1.0 - u) * cand
        return self.head(h)


if __name__ == '__main__':
    torch.manual_seed(0)
    N, L, H = 120, 30, 30
    A = torch.rand(N, N); A.fill_diagonal_(0); A = A / A.sum(1, keepdim=True)
    for Fn in (12, 24, 40):
        for d in (64, 96, 128):
            m = _DGCRN(Fn, d, H, N, A)
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print('Fn=%-3d d=%-4d params=%d' % (Fn, d, n))
    m = _DGCRN(24, 64, H, N, A)
    x = torch.randn(4, N, L, 24)
    y = m(x)
    assert y.shape == (4, N, H), y.shape
    y.abs().mean().backward()
    assert all(p.grad is not None for p in m.parameters()), 'a parameter receives no gradient'
    Ad = m.dynamic_adjacency(torch.randn(4, N, 24 + 64))
    assert torch.allclose(Ad.sum(-1), torch.ones(4, N), atol=1e-5)
    print('forward/backward ok; every parameter receives a gradient; dynamic A row-stochastic')
