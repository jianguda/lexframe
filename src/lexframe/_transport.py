"""The native transport: a state carried through the model's own remaining layers with the true context. Each position attends to
the true states of its prefix and to its own key (or, jointly, to the carried states of its prefix: the model's own forward pass
from moved states). The true states are read once per layer as keys and values (the Prefix), which every carried state attends to;
the positions are independent columns, so only the requested ones are carried.

The expected transport: a state carried through the remaining layers on its own column, every attention write replaced by its
expectation given the state (calibrated) plus a Gaussian innovation, averaged by quasi-Monte Carlo: the reading without context."""

from __future__ import annotations

import math

import torch
from transformers import DynamicCache

from ._model import Model, _tree, block, eager, innorm, logits, own_attention


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
PREFIX_FLOATS = 1 << 25  # floats of the copied prefix keys and values of one pass of attend_prefix()


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


class Prefix:
    """The context of the native transport: the true states xs [B, T, d] of every position as each layer's attention keys and values,
    computed once per layer (the model's own key and value projections, position embeddings and norms) and shared by every carried
    state that attends to them."""

    def __init__(self, m: Model, xs):
        self.m, self.xs, self.kv = m, xs, {}

    def __call__(self, k: int):
        if k not in self.kv:
            m, R = self.m, innorm(self.m, k)(self.xs[k])
            at = own_attention(m, k)
            c = DynamicCache()
            n = R.shape[1]
            with torch.no_grad(), eager(m):
                at(hidden_states=R, position_embeddings=m.position_embeddings(k, n), attention_mask=causal(m, k, n, R.device, R.dtype),
                   past_key_values=c)  # fmt: skip
            lay = c.layers[at.layer_idx]
            self.kv[k] = (lay.keys.detach(), lay.values.detach())
        return self.kv[k]


def attend_prefix(m, k, uq, pre: Prefix, P=None):
    """Layer k's own attention for the queries uq [A, n, d] (attention inputs) at the positions P (default all), each over the keys of
    its prefix in pre and its own key; row r belongs to text r % B of the prefix's B texts."""
    at = own_attention(m, k)
    A, n, _ = uq.shape
    K, V = pre(k)
    B, T = K.shape[0], pre.xs[k].shape[1]
    Pt = torch.arange(T, device=uq.device) if P is None else torch.as_tensor(P, device=uq.device)
    pe = _tree(lambda t: t.index_select(-2, Pt), m.position_embeddings(k, T))
    s = torch.arange(T, device=uq.device)
    see = s[None] < Pt[:, None]
    w = _window(m, k)
    if w:
        see &= Pt[:, None] - s[None] < w
    am = _additive(torch.cat([see, torch.eye(n, dtype=torch.bool, device=uq.device)], 1), uq.dtype)
    sub = max(1, PREFIX_FLOATS // (2 * K[0].numel() + 2 * V[0].numel()))
    out = []
    with eager(m):
        for a in range(0, A, sub):
            b = min(A, a + sub) - a
            if B == 1:
                Kc, Vc = K.expand(b, *K.shape[1:]), V.expand(b, *V.shape[1:])
            else:
                rows = torch.arange(a, a + b, device=uq.device) % B
                Kc, Vc = K[rows], V[rows]
            c = DynamicCache()
            c.update(Kc.to(uq.dtype), Vc.to(uq.dtype), at.layer_idx)
            out.append(at(hidden_states=uq[a : a + b], position_embeddings=pe, attention_mask=am, past_key_values=c)[0])
    return torch.cat(out)


def step(m, k, h, pre: Prefix | None = None, P=None):
    """Layer k of the native transport on the states h [b, n, d]: with pre, each of the positions P (default all) attends to the true
    states of its prefix and to its own key; without, every position attends to the carried states h of its prefix (the model's own
    layer, all positions)."""
    b, n, d = h.shape
    q = innorm(m, k)(h)
    a = (attend(m, k, q, q) if pre is None else attend_prefix(m, k, q, pre, P)).reshape(-1, d)
    return block(m, k, h.reshape(-1, d), lambda u, s: a[s : s + u.shape[0]].to(u.dtype)).view(b, n, d)


def native(m, xs, LS, X=None, joint=False, P=None, pre=None):
    """The states of the layers LS < L (or the moved states X, one per layer of LS) carried to the output, [len(LS), B, T, d]:
    every layer as the model computes it, with each position attending to the true states of its prefix and to its own key (each
    position on its own column), only the positions P if given ([len(LS), B, len(P), d]). joint=True carries all positions together,
    each attending to the carried states of its prefix: the model's own forward pass from the moved states. Unmoved states
    reproduce the model exactly either way."""
    if joint and P is not None:
        raise ValueError("the joint transport carries every position")
    L = int(m.n_layers)
    B, Tn, d = xs[0].shape
    pre = None if joint else (Prefix(m, xs) if pre is None else pre)
    cols = (lambda x: x) if P is None else (lambda x: x[:, P])
    n_ = Tn if P is None else len(P)
    rd = xs[0][:0, :, :n_].reshape(0, B, n_, d)
    for k in range(LS[0], L):
        if k in LS:
            x0 = xs[k] if X is None else X[LS.index(k)]
            rd = torch.cat([rd, cols(x0)[None].to(xs[k].dtype)])
        n = rd.shape[0]
        rd = step(m, k, rd.flatten(0, 1), pre, P).view(n, B, n_, d)
    return rd


def run(m, xs, LS, X=None, joint=False, chunk=8, P=None):
    """The final states of the native transport of the layers LS, [len(LS), B, T, d] (layer L is the output itself), `chunk` layers
    per pass, only the positions P if given; joint=True carries moved states X with each other, as the model's own forward pass does."""
    L = m.n_layers
    inner = [l for l in LS if l < L]
    pre = None if joint else Prefix(m, xs)
    out = []
    for c in range(0, len(inner), chunk):
        Xc = None if X is None else X[c : c + chunk]
        out.append(native(m, xs, inner[c : c + chunk], Xc, joint=joint, P=P, pre=pre))
    if L in LS:
        x = xs[L] if X is None else X[len(inner)]
        out.append((x if P is None else x[:, P])[None])
    return torch.cat(out)


# ---- the expected transport

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


def expected_write(m, cal, l, rb, rm, first, some, eps):
    """Layer l's attention write in the expected transport, for rows with position buckets rb, integration points rm and the
    position-0 mask first (some: whether any row is at position 0), the rows repeating with period len(rb): the typical write plus
    the offset and the innovation, or the exact attention at position 0."""
    dev = m.device
    rows = rb.shape[0]
    add = cal.off[l].to(dev)[rb] + eps[:, l][rm]
    typ = cal.typical(l)

    def fn(u, s):
        r = torch.arange(s, s + u.shape[0], device=dev) % rows
        a = typ(u, s).float() + add[r]
        if some:
            own = first[r]
            if bool(own.any()):
                a = a.clone()
                a[own] = attend(m, l, u[own][:, None])[:, 0].float()
        return a.to(u.dtype)

    return fn


def read_expected(m: Model, cal, X: dict, t: torch.Tensor, eps: torch.Tensor, tokens: torch.Tensor | None = None,
                  center: bool = True):  # fmt: skip
    """Readings with the context invisible (the expected transport) of the states X {layer k < L: [N, hidden]} at the positions t
    [N], integrated over the innovations eps [M, layer, hidden]: {k: log R_k [N, vocab]}, or with `tokens` log R_k of those tokens
    [N, token], centred by the mean over the vocabulary unless center=False. Differentiable when grad is enabled."""
    L, dev = m.n_layers, m.device
    M, N = eps.shape[0], t.shape[0]
    ks = sorted(X)
    rows = N * M
    rt = t.to(dev).repeat_interleave(M)
    rb = buckets(t.to(dev)).repeat_interleave(M)
    rm = torch.arange(M, device=dev).repeat(N)
    first = rt == 0
    some = bool(first.any())  # once here, not at every layer: each would wait for the device
    H = None
    for l in range(ks[0], L):
        if l in X:
            h = X[l].repeat_interleave(M, 0)
            H = h if H is None else torch.cat([H, h])
        H = block(m, l, H, expected_write(m, cal, l, rb, rm, first, some, eps))
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
