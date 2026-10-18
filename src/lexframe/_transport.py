"""Transport of order n: a state passes the model's own blocks with n rounds of attention between positions."""

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
    return torch.zeros(ok.shape, dtype=dt, device=dev).masked_fill(~ok, torch.finfo(dt).min)[None, None]


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


def _step(m, cal, k, h, fi):
    base = cal.typical(k)

    def fn(u, s):
        a = base(u, s)
        sel = fi[(fi >= s) & (fi < s + u.shape[0])] - s
        if sel.numel():
            own = attend(m, k, u[sel][:, None])[:, 0]
            a.index_copy_(0, sel, own.to(a.dtype))
        return a

    return block(m, k, h, fn)


def _streams(m, cal, xs, LS, st):
    L = int(m.n_layers)
    LS = list(LS)
    B, Tn, d = xs[0].shape
    dev = xs[0].device
    ch = xs[0][:0].reshape(0, B, Tn, d)
    rd = {nm: ch for nm in st}
    nA = 0
    for k in range(LS[0], L):
        norm = innorm(m, k)
        if k in LS:
            ch = torch.cat([ch, xs[k][None]])
            nA += 1
            for nm, s in st.items():
                x0 = s["x"][LS.index(k)] if "x" in s else xs[k]
                rd[nm] = torch.cat([rd[nm], x0[None].to(xs[k].dtype)])
        refs = {}

        def pick(over, a):
            if over not in refs:
                src = xs[k][None].expand(nA, *xs[k].shape) if over == "true" else ch if over is None else rd[over]
                refs[over] = norm(src.flatten(0, 1))
            return refs[over][: a * B]

        att = {
            nm: attend(m, k, norm(rd[nm].flatten(0, 1)), pick(s.get("over"), rd[nm].shape[0])) for nm, s in st.items()
        }
        refs.clear()
        fi = torch.arange(0, nA * B * Tn, Tn, device=dev)
        ch = _step(m, cal, k, ch.reshape(-1, d), fi).view(nA, B, Tn, d)
        for nm in att:
            h = rd[nm]
            a = att[nm].reshape(-1, d)
            rd[nm] = block(m, k, h.reshape(-1, d), lambda u, s, a=a: a[s : s + u.shape[0]].to(u.dtype)).view(h.shape)
        del att
    return {**rd, "none": ch}


def name(o):
    return "none" if o == 0 else "layer" + (f"o{o}" if o > 1 else "")


def chain(o):
    return {name(i): {"over": None if i == 1 else name(i - 1)} for i in range(1, o + 1)}
