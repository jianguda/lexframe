"""The closure reading of a hidden state.

The state x_k(t) at layer k and position t runs through the model's own layers k..L-1 on its own column: norms, feed-forward blocks
and residual connections as the model computes them. Each attention write, the only place where other positions enter, is replaced
by its calibrated prediction from the column's own attention input plus an offset for the position, and an independent Gaussian
innovation with the calibrated residual covariance:

    a_l = typical_l(u) + offset_l[bucket(t)] + eps_l,    eps_l ~ N(0, Sigma_l).

The reading R_k is the mean next-token distribution over the innovations, integrated by randomised quasi-Monte Carlo: scrambled Sobol
points on the eigen-coordinates of every Sigma_l, the same points for every layer, position and text, so that a reading is a
deterministic function of the state and readings of different layers are paired. Position 0 sees only itself, so its attention is
computed exactly and draws no innovation.
"""

from __future__ import annotations

import math

import torch

from ._model import Model, block, logits
from ._transport import attend

NB = 16  # position buckets: 0..7 one each, then doubling widths (8..15, 16..31, ...)
SOBOL = 21201  # the largest dimension of torch's Sobol engine; further coordinates are pseudo-random


def bucket(t: int) -> int:
    return t if t < 8 else min(8 + int(math.log2(t / 8)), NB - 1)


def buckets(t: torch.Tensor) -> torch.Tensor:
    return torch.tensor([bucket(int(p)) for p in t.tolist()], device=t.device, dtype=torch.long)


def points(fac: torch.Tensor, M: int, seed: int = 0) -> torch.Tensor:
    """Standard-normal integration points [M, layer, hidden]: scrambled Sobol on the coordinates of the largest eigenvalues of all
    innovation covariances (fac[l] = V sqrt(lambda), so column j of layer l has eigenvalue |fac[l][:, j]|^2), the rest pseudo-random.
    """
    L, d, _ = fac.shape
    lam = torch.stack([(fac[l].float() ** 2).sum(0).cpu() for l in range(L)]).reshape(-1)
    order = torch.argsort(lam, descending=True)
    D = min(SOBOL, lam.numel())
    u = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed).draw(M, dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
    flat = torch.randn(M, lam.numel(), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    flat[:, order[:D]] = torch.special.ndtri(u)
    return flat.float().view(M, L, d)


def innovations(fac: torch.Tensor, xi: torch.Tensor, device) -> torch.Tensor:
    """The innovations [M, layer, hidden] at the points xi: eps_l = fac[l] xi_l, one layer at a time (fac may stay on the CPU)."""
    return torch.stack([xi[:, l].to(device) @ fac[l].to(device).float().T for l in range(fac.shape[0])], 1)


def read(m: Model, cal, X: dict, t: torch.Tensor, eps: torch.Tensor, tokens: torch.Tensor | None = None, center: bool = True,
         slab: int = 1 << 26):  # fmt: skip
    """Closure readings of the states X {layer k < L: [N, hidden]} at the positions t [N], integrated over the innovations eps
    [M, layer, hidden]: {k: log R_k [N, vocab]}, or with `tokens`
    log R_k of those tokens [N, token], centred by the mean over the vocabulary unless center=False. Differentiable when grad is
    enabled."""
    L, dev = m.n_layers, m.device
    M, N = eps.shape[0], t.shape[0]
    ks = sorted(X)
    rows = N * M
    rt = t.to(dev).repeat_interleave(M)
    rb = buckets(t.to(dev)).repeat_interleave(M)
    rm = torch.arange(M, device=dev).repeat(N)
    first = rt == 0
    H = None
    for l in range(ks[0], L):
        if l in X:
            h = X[l].repeat_interleave(M, 0)
            H = h if H is None else torch.cat([H, h])
        add = cal.off[l].to(dev)[rb] + eps[:, l][rm]
        typ = cal.typical(l)

        def fn(u, s, typ=typ, add=add, l=l):
            r = torch.arange(s, s + u.shape[0], device=dev) % rows
            a = typ(u, s).float() + add[r]
            own = first[r]
            if bool(own.any()):
                a = a.clone()
                a[own] = attend(m, l, u[own][:, None])[:, 0].float()
            return a.to(u.dtype)

        H = block(m, l, H, fn)
    out = {}
    V = m.lm_head.weight.shape[0]
    step = max(1, slab // (M * V))  # states per slice of the softmax
    for g, k in enumerate(ks):
        Hg = H[g * rows : (g + 1) * rows]
        parts = []
        for s in range(0, N, step):
            z = torch.log_softmax(logits(m, Hg[s * M : (s + step) * M]), -1).view(-1, M, V)
            lr = torch.logsumexp(z, 1) - math.log(M)
            parts.append(lr if tokens is None else lr[:, tokens] - (lr.mean(-1, keepdim=True) if center else 0))
        out[k] = torch.cat(parts)
    return out
