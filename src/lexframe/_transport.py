"""The transport of a state through the model's remaining layers, with the context visible or invisible.

With the context visible, the native transport carries states through the model's own remaining layers, each position attending to
the true states of its prefix and to its own key (or, jointly, to the carried states of its prefix: the model's own forward pass from
moved states).

With the context invisible, the expected transport reads a state by itself: the expectation of the native one over the context's law
given the state, approximated layer by layer. The state x_k(t) at layer k and position t runs through the model's own layers
k..L-1 on its own column: norms, feed-forward blocks and residual connections as the model computes them. Each attention write, the
only place where other positions enter, is replaced by its calibrated prediction from the column's own attention input plus an
offset for the position, and an independent Gaussian innovation with the calibrated residual covariance:

    a_l = typical_l(u) + offset_l[bucket(t)] + eps_l,    eps_l ~ N(0, Sigma_l).

The reading R_k is the mean next-token distribution over the innovations, integrated by randomised quasi-Monte Carlo: scrambled Sobol
points on the eigen-coordinates of every Sigma_l, the same points for every layer, position and text, so that a reading is a
deterministic function of the state and readings of different layers are paired. Position 0 sees only itself, so its attention is
computed exactly and draws no innovation.
"""

from __future__ import annotations

import math

import torch

from ._model import Model, block, eager, innorm, logits, own_attention


def _window(m, k):
    at = own_attention(m, k)
    if hasattr(at, "sliding_window"):
        return at.sliding_window
    cfg = getattr(m.hf.config, "text_config", None) or m.hf.config
    types = getattr(cfg, "layer_types", None)
    if types is not None:
        return cfg.sliding_window if types[k] == "sliding_attention" else None
    return getattr(cfg, "sliding_window", None) if getattr(cfg, "use_sliding_window", True) else None


def _additive(ok, dt):
    return torch.zeros(ok.shape, dtype=dt, device=ok.device).masked_fill(~ok, torch.finfo(dt).min)[None, None]


def mask(n, ref, w, dev, dt):
    """Additive mask over [reference; query] (or only the query): query t sees reference positions s < t within the
    window and its own key, and nothing else; the reference rows see a causal prefix so that no row is empty."""
    q = torch.arange(n, device=dev)
    if ref:
        s = torch.arange(n, device=dev)
        see = s[None] < q[:, None]
        if w:
            see &= q[:, None] - s[None] < w
        top = torch.cat([s[None] <= q[:, None], torch.zeros(n, n, dtype=torch.bool, device=dev)], 1)
        bot = torch.cat([see, torch.eye(n, dtype=torch.bool, device=dev)], 1)
        ok = torch.cat([top, bot])
    else:
        ok = torch.eye(n, dtype=torch.bool, device=dev)
    return _additive(ok, dt)


def causal(m, k, n, dev, dt):
    """Layer k's own causal mask (with its sliding window, if any) over n positions."""
    q = torch.arange(n, device=dev)
    ok = q[None] <= q[:, None]
    w = _window(m, k)
    if w:
        ok &= q[:, None] - q[None] < w
    return _additive(ok, dt)


SUB = 32  # queries per pass of attend()


def attend(m, k, uq, R=None):
    """Layer k's own attention for the queries uq [A, n, d] (attention inputs), each over the reference states R
    [A, n, d] of its prefix and its own key; with R None, over its own key alone."""
    at = own_attention(m, k)
    A, n, d = uq.shape
    pe = m.position_embeddings(k, n, twice=R is not None)
    am = mask(n, R is not None, _window(m, k), uq.device, uq.dtype)
    out = []
    with eager(m):
        for a in range(0, A, SUB):
            h = uq[a : a + SUB] if R is None else torch.cat([R[a : a + SUB].to(uq.dtype), uq[a : a + SUB]], 1)
            o = at(hidden_states=h, position_embeddings=pe, attention_mask=am)[0]
            out.append(o[:, -n:])
    return torch.cat(out)


def step(m, k, h, R=None):
    """Layer k of the native transport on the states h [b, T, d]: each position attends to the reference attention inputs R
    ([b, T, d], or [1, T, d] shared by every row) of its prefix and to its own key; with R None, to the carried states h of its
    prefix (the model's own layer)."""
    b, T, d = h.shape
    q = innorm(m, k)(h)
    a = attend(m, k, q, q if R is None else R.expand(b, -1, -1)).reshape(-1, d)
    return block(m, k, h.reshape(-1, d), lambda u, s: a[s : s + u.shape[0]].to(u.dtype)).view(b, T, d)


def native(m, xs, LS, X=None, joint=False):
    """The states of the layers LS < L (or the moved states X, one per layer of LS) carried to the output, [len(LS), B, T, d]:
    every layer as the model computes it, with each position attending to the true states of its prefix and to its own key (each
    position on its own column). joint=True carries all positions together, each attending to the carried states of
    its prefix: the model's own forward pass from the moved states. Unmoved states reproduce the model exactly either way."""
    L = int(m.n_layers)
    B, Tn, d = xs[0].shape
    rd = xs[0][:0].reshape(0, B, Tn, d)
    for k in range(LS[0], L):
        if k in LS:
            x0 = xs[k] if X is None else X[LS.index(k)]
            rd = torch.cat([rd, x0[None].to(xs[k].dtype)])
        n = rd.shape[0]
        R = None if joint else innorm(m, k)(xs[k])[None].expand(n, *xs[k].shape).flatten(0, 1)
        rd = step(m, k, rd.flatten(0, 1), R).view(n, B, Tn, d)
    return rd


def run(m, xs, LS, X=None, joint=False, chunk=8):
    """The final states of the native transport of the layers LS, [len(LS), B, T, d] (layer L is the output itself), `chunk` layers
    per pass; joint=True carries moved states X with each other, as the model's own forward pass does."""
    L = m.n_layers
    inner = [l for l in LS if l < L]
    out = []
    for c in range(0, len(inner), chunk):
        Xc = None if X is None else X[c : c + chunk]
        out.append(native(m, xs, inner[c : c + chunk], Xc, joint=joint))
    if L in LS:
        out.append((xs[L] if X is None else X[len(inner)])[None])
    return torch.cat(out)


# ---- the expected transport (context invisible)

NB = 16  # position buckets: 0..7 one each, then doubling widths (8..15, 16..31, ...)
SOBOL = 21201  # the largest dimension of torch's Sobol engine; further coordinates are pseudo-random


def bucket(t: int) -> int:
    return t if t < 8 else min(8 + int(math.log2(t / 8)), NB - 1)


def buckets(t: torch.Tensor) -> torch.Tensor:
    return torch.tensor([bucket(int(p)) for p in t.tolist()], device=t.device, dtype=torch.long)


SEED = 0  # of the integration points


def points(fac: torch.Tensor, M: int) -> torch.Tensor:
    """Standard-normal integration points [M, layer, hidden]: scrambled Sobol on the coordinates of the largest eigenvalues of all
    innovation covariances (fac[l] = V sqrt(lambda), so column j of layer l has eigenvalue |fac[l][:, j]|^2), the rest pseudo-random.
    """
    L, d, _ = fac.shape
    lam = torch.stack([(fac[l].float() ** 2).sum(0).cpu() for l in range(L)]).reshape(-1)
    order = torch.argsort(lam, descending=True)
    D = min(SOBOL, lam.numel())
    u = torch.quasirandom.SobolEngine(D, scramble=True, seed=SEED).draw(M).clamp(1e-7, 1 - 1e-7)
    flat = torch.randn(M, lam.numel(), generator=torch.Generator().manual_seed(SEED))
    flat[:, order[:D]] = torch.special.ndtri(u)
    return flat.view(M, L, d)


def innovations(fac: torch.Tensor, xi: torch.Tensor, device) -> torch.Tensor:
    """The innovations [M, layer, hidden] at the points xi: eps_l = fac[l] xi_l, one layer at a time (fac may stay on the CPU)."""
    return torch.stack([xi[:, l].to(device) @ fac[l].to(device).float().T for l in range(fac.shape[0])], 1)


SLAB = 1 << 26  # floats of one slice of the softmax over the vocabulary


def read_expected(m: Model, cal, X: dict, t: torch.Tensor, eps: torch.Tensor, tokens: torch.Tensor | None = None,
                  center: bool = True):  # fmt: skip
    """Readings with the context invisible (the expected transport) of the states X {layer k < L: [N, hidden]} at the positions t [N], integrated over the innovations eps
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
    step = max(1, SLAB // (M * V))  # states per slice of the softmax
    for g, k in enumerate(ks):
        Hg = H[g * rows : (g + 1) * rows]
        parts = []
        for s in range(0, N, step):
            z = torch.log_softmax(logits(m, Hg[s * M : (s + step) * M]), -1).view(-1, M, V)
            lr = torch.logsumexp(z, 1) - math.log(M)
            parts.append(lr if tokens is None else lr[:, tokens] - (lr.mean(-1, keepdim=True) if center else 0))
        out[k] = torch.cat(parts)
    return out
