"""The native transport: states carried through the model's own remaining layers, each position attending to the true states of its
prefix and to its own key."""

import torch

from ._model import block, eager, innorm, own_attention


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


def _mask(n, ref, w, dev, dt):
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


def attend(m, k, uq, R=None, sub=32):
    """Layer k's own attention for the queries uq [A, n, d] (attention inputs), each over the reference states R
    [A, n, d] of its prefix and its own key; with R None, over its own key alone."""
    at = own_attention(m, k)
    A, n, d = uq.shape
    pe = m.position_embeddings(k, n, twice=R is not None)
    mask = _mask(n, R is not None, _window(m, k), uq.device, uq.dtype)
    out = []
    with eager(m):
        for a in range(0, A, sub):
            h = uq[a : a + sub] if R is None else torch.cat([R[a : a + sub].to(uq.dtype), uq[a : a + sub]], 1)
            o = at(hidden_states=h, position_embeddings=pe, attention_mask=mask)[0]
            out.append(o[:, -n:])
    return torch.cat(out)


def native(m, xs, LS, X=None):
    """The states of the layers LS < L (or the moved states X, one per layer of LS) carried to the output, [len(LS), B, T, d]:
    every layer as the model computes it, with each position attending to the true states of its prefix and to its own key.
    Unmoved states reproduce the model exactly."""
    L = int(m.n_layers)
    B, Tn, d = xs[0].shape
    rd = xs[0][:0].reshape(0, B, Tn, d)
    for k in range(LS[0], L):
        if k in LS:
            x0 = xs[k] if X is None else X[LS.index(k)]
            rd = torch.cat([rd, x0[None].to(xs[k].dtype)])
        n = rd.shape[0]
        norm = innorm(m, k)
        ref = norm(xs[k])[None].expand(n, *xs[k].shape).flatten(0, 1)
        a = attend(m, k, norm(rd.flatten(0, 1)), ref).reshape(-1, d)
        rd = block(m, k, rd.reshape(-1, d), lambda u, s, a=a: a[s : s + u.shape[0]].to(u.dtype)).view(n, B, Tn, d)
    return rd
