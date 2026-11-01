"""Ablation with the full context: what a layer wrote at some positions removed (the whole write, its attention part or its MLP
part), or the layer run again without what its attention read from one source position."""

from __future__ import annotations

import torch

from ._model import _second, block, eager, innorm, own_attention
from ._transport import attend, causal, mask


@torch.no_grad()
def sources(m, j, u, t, a):
    """Layer j's attention write at position t split by source position, (C [position, hidden], w [position]) with C summing to
    the write and w the head-averaged attention weights; None where the attention module does not allow the split."""
    at = own_attention(m, j)
    o = getattr(at, "o_proj", None)
    if not isinstance(o, torch.nn.Linear):
        return None
    n, dev, dt = t + 1, u.device, u.dtype
    h = u[None, :n]
    pe = m.position_embeddings(j, n)
    got = {}
    hook = o.register_forward_pre_hook(lambda _m, inp: got.__setitem__("ctx", inp[0]))
    try:
        with eager(m):
            r = at(hidden_states=h, position_embeddings=pe, attention_mask=causal(m, j, n, dev, dt))
            weights = r[1] if isinstance(r, tuple) and len(r) > 1 else None
            # each position attending only to itself: the hook then captures every source's value vectors
            at(hidden_states=h, position_embeddings=pe, attention_mask=mask(n, False, None, dev, dt))
    finally:
        hook.remove()
    if weights is None or "ctx" not in got:
        return None
    alpha = weights[0, :, t, :n].float()
    ctx = got["ctx"][0].float()
    heads = alpha.shape[0]
    if ctx.shape[-1] % heads:
        return None
    values = ctx.view(n, heads, -1)
    C = (alpha.T[:, :, None] * values).reshape(n, -1) @ o.weight.float().T
    w = alpha.mean(0)
    if o.bias is not None:
        C = C + w[:, None] * o.bias.float()
    if float((C.sum(0) - a).norm() / a.norm().clamp_min(1e-30)) > 0.05:
        return None
    return C, w


@torch.no_grad()
def _parts(m, h, k):
    """Layer k run as the model runs it on the states h [1, T, d] of a run: its attention input u [T, d], its attention output a
    [T, d], and the state between its two sublayers [1, T, d]."""
    u = innorm(m, k)(h)
    a = attend(m, k, u, u)
    mid = []
    hook = _second(m.layers[k]).register_forward_pre_hook(lambda _m, inp: mid.append(inp[0]))
    try:
        block(m, k, h[0], lambda q, s: a[0, s : s + q.shape[0]].to(q.dtype))
    finally:
        hook.remove()
    return u[0], a[0].float(), torch.cat(mid, 1) if len(mid) > 1 else mid[0]


@torch.no_grad()
def ablate(m, xs, P, layer, part, source):
    """The states of layer + 1 of the run xs (its states at every layer) after removing what the layer wrote at the positions P
    (part "all", "attention" or "mlp"), or, with a source position, after the layer runs again at each of them without what its
    attention read from that source, [1, T, d]."""
    L = m.n_layers
    if not 0 <= layer < L:
        raise ValueError(f"layer must lie in 1..{L}")
    if part not in ("all", "attention", "mlp"):
        raise ValueError("part is 'all', 'attention' or 'mlp'")
    if source is not None and part != "all":
        raise ValueError("a source is removed from the layer's attention: give part='all' (the default) with source")
    h = xs[layer + 1].clone()
    if source is None and part == "all":
        h[0, P] = xs[layer][0, P]
        return h
    u, a, mid = _parts(m, xs[layer], layer)
    if source is None:
        written = mid - xs[layer] if part == "attention" else xs[layer + 1] - mid
        h[0, P] = (h[0, P].float() - written[0, P].float()).to(h.dtype)
        return h
    for t in P:
        s = int(source) % (t + 1) if source < 0 else int(source)
        if not 0 <= s <= t:
            raise ValueError(f"source must be a position in 0..{t}")
        split = sources(m, layer, u, t, a[t])
        if split is None:
            raise ValueError("the attention of this model cannot be split by source position")
        # the layer runs again with the source's share taken out of its attention output, so that whatever the layer does to
        # that output before the residual (a norm, a scale) and its MLP respond as in the model
        at = (a[t] - split[0][s])[None]
        h[0, t] = block(m, layer, xs[layer][0, t][None], lambda q, s_: at.to(q.dtype))[0]
    return h
