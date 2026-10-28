"""The trail of one prediction: the effect of removing what each layer wrote, with context (attribution and ablation); and the split
of the label's log-probability across layers: the readings without context along depth, each layer's change (this context's departure
from the expected write) split by source position by integrated gradients through the expected transport."""

from __future__ import annotations

import numpy as np
import torch

from ._model import block, eager, forward, logprobs, own_attention
from ._transport import bucket, causal, mask, read_expected, run

STEPS = 8  # integrated-gradient points of the split by source


def _centred_logit(m, h, Wd):
    return (m.final_norm(h).float() * m.output_scale * Wd).sum(-1)


@torch.no_grad()
def attribute(f, ids, part, v):
    """The change of the label v's centred output logit at every position, [layer, position], when what the layer (or its part,
    "attention" or "mlp") wrote there is removed, with the context visible."""
    m = f._m
    hL, xs, mid = forward(m, ids, mid=True)
    L = m.n_layers
    Wd = (m.lm_head.weight[v].float() - f._head_mean()).expand(*ids.shape, -1)
    states = xs + [hL]
    written = {
        "all": lambda k: states[k + 1] - xs[k],
        "attention": lambda k: mid[k] - xs[k],
        "mlp": lambda k: states[k + 1] - mid[k],
    }[part]
    D = [written(k) for k in range(L)]
    out = torch.empty(L, *ids.shape, device=m.device)
    LS = list(range(1, L))
    X = [(xs[l].float() - D[l - 1].float()).to(xs[l].dtype) for l in LS]
    full = _centred_logit(m, hL, Wd)  # the unmoved states carry to the model's own output
    out[: L - 1] = full - _centred_logit(m, run(m, xs, LS, X, chunk=f._chunk), Wd)
    out[L - 1] = full - _centred_logit(m, (hL.float() - D[L - 1].float()).to(hL.dtype), Wd)
    return out[:, 0].cpu().numpy()


@torch.no_grad()
def split(f, ids, t, label):
    """The split of position t for the label id (None: the model's prediction): a dict with logp [L + 1], innovation, noise,
    check, residual [L] and sources [L, t + 1] (None where the attention cannot be split), the split integrated over STEPS points;
    residual is what the split leaves of each innovation."""
    f._ready()
    m, cal = f._m, f.calibration
    ids = ids[:, : t + 1]
    L, dev = m.n_layers, m.device
    u_in, a_out = {}, {}

    def cap(k, u, a):
        u_in[k], a_out[k] = u[0], a[0, t]

    hL, xs, _ = forward(m, ids, on_attention=cap)
    out = logprobs(m, hL[0, t])
    v = int(out.argmax()) if label is None else label
    xi, tt = f._points(), torch.tensor([t], device=dev)
    R = read_expected(m, cal, {k: xs[k][0, t][None] for k in range(L)}, tt, xi, torch.tensor([v], device=dev), center=False)
    logp = np.array([float(R[k][0, 0]) for k in range(L)] + [float(out[v])])
    innovation, noise, src, residual, check, split = _split_rows(m, cal, u_in, a_out, xs, t, xi, v, logp, STEPS)
    return {"logp": logp, "innovation": innovation, "noise": noise, "check": check,
            "sources": src if split else None, "residual": residual if split else None}  # fmt: skip


def _split_rows(m, cal, u_in, a_out, xs, t, xi, v, logp, steps):
    """Per layer of the split: innovation, noise, sources by position, residual, check, and whether the split was possible."""
    L, dev = m.n_layers, m.device
    innovation, noise, check, residual = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
    src = np.zeros((L, t + 1))
    split = True
    alphas = ((torch.arange(steps, device=dev) + 0.5) / steps)[:, None]
    for j in range(L):
        ahat = cal.typical(j)(u_in[j][t][None], 0).float()[0] + cal.off[j, bucket(t)].to(dev)
        a = a_out[j].float()
        A = torch.cat([ahat[None], a[None], ahat + alphas * (a - ahat)]).requires_grad_(True)
        with torch.enable_grad():
            g = _after(m, cal, j, xs[j][0, t], A, t, xi, v)
            grad = torch.autograd.grad(g[2:].sum(), A)[0][2:].mean(0)
        g = g.detach()
        check[j] = abs(float(g[1]) - logp[j + 1])
        innovation[j], noise[j] = logp[j + 1] - float(g[0]), float(g[0]) - logp[j]
        parts = sources(m, j, u_in[j], t, a) if split else None
        if parts is None:
            split = False
            continue
        C, w = parts
        src[j] = ((C - w[:, None] * ahat[None]) @ grad).cpu().numpy()
        residual[j] = innovation[j] - src[j].sum()
    return innovation, noise, src, residual, check, split


def _after(m, cal, j, x, A, t, xi, v):
    """log R_{j+1}(v) of the states that layer j makes from x [hidden] with the attention writes A [n, hidden]."""
    h = block(m, j, x[None].expand(A.shape[0], -1).contiguous(), lambda u, s: A[s : s + u.shape[0]].to(u.dtype))
    if j + 1 == m.n_layers:
        return logprobs(m, h)[:, v]
    tt = torch.full((A.shape[0],), t, device=m.device)
    return read_expected(m, cal, {j + 1: h}, tt, xi, torch.tensor([v], device=m.device), center=False)[j + 1][:, 0]


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


def ablate(m, ids, t, layer, part, source):
    """The states after removing what the layer wrote at position t (part "all", "attention" or "mlp"), or, with a source position,
    after the layer runs again without what its attention read from that source: (xs, h) with xs the model's own states of every
    layer and h the ablated states of layer + 1, [1, T, d]."""
    L = m.n_layers
    if not 0 <= layer < L:
        raise ValueError(f"layer must lie in 0..{L - 1}")
    if part not in ("all", "attention", "mlp"):
        raise ValueError("part is 'all', 'attention' or 'mlp'")
    u_in, a_out = {}, {}

    def cap(k, u, a):
        u_in[k], a_out[k] = u[0], a[0, t]

    hL, xs, mid = forward(m, ids, on_attention=cap, mid=True)
    xs = xs + [hL]
    h = xs[layer + 1].clone()
    if source is None:
        written = {"all": xs[layer + 1] - xs[layer], "attention": mid[layer] - xs[layer], "mlp": xs[layer + 1] - mid[layer]}[part]
        h[0, t] = (h[0, t].float() - written[0, t].float()).to(h.dtype)
    else:
        if part != "all":
            raise ValueError("a source is removed from the layer's attention: give part='all' (the default) with source")
        s = int(source) % (t + 1) if source < 0 else int(source)
        if not 0 <= s <= t:
            raise ValueError(f"source must be a position in 0..{t}")
        parts = sources(m, layer, u_in[layer], t, a_out[layer].float())
        if parts is None:
            raise ValueError("the attention of this model cannot be split by source position")
        # the layer runs again with the source's share taken out of its attention output, so that whatever the layer does to
        # that output before the residual (a norm, a scale) and its MLP respond as in the model
        a = (a_out[layer].float() - parts[0][s])[None]
        h[0, t] = block(m, layer, xs[layer][0, t][None], lambda u, s_: a.to(u.dtype))[0]
    return xs, h
