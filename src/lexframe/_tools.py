"""Research tools, not part of the public API: the first-order audit of a weight step, and the ceiling of fixed lenses."""

from __future__ import annotations

import contextlib
import re

import torch
from torch.autograd import forward_ad as fwAD

from . import _transport as TR
from ._model import forward, head, label_id


@contextlib.contextmanager
def _tangent(model, delta):
    """Within a forward-AD dual level: the model with every parameter named in delta (name -> tangent) made a dual tensor with that
    tangent, tied copies included, restored on exit."""
    params = dict(model.named_parameters(remove_duplicate=False))
    tangents = {id(params[k]): v for k, v in delta.items()}
    slots = []
    try:
        for mod in model.modules():
            for attr, p in list(mod._parameters.items()):
                if p is not None and id(p) in tangents:
                    slots.append((mod, attr, p))
                    t = tangents[id(p)].to(p.device, p.dtype)
                    mod._parameters[attr] = fwAD.make_dual(p.detach(), t)
        yield
    finally:
        for mod, attr, p in slots:
            mod._parameters[attr] = p


@torch.no_grad()
def step_audit(frame, text, delta, target="next"):
    """The first-order effect of a weight step `delta` (name -> tensor) on the log-probability of the target at every position of
    `text`, exactly, by forward-mode differentiation, per group of parameters (each layer, and all of them together, "joint"):

        gain           the mean first-order change of log p(target) along the step;
        gain_own       the same with every other position's states held at the unstepped run (the step still acts on everything at
                       this position, the keys and values it reads from the held states included);
        context_share  1 - gain_own / gain: the share that comes through the other positions' changed states;
        curvature      the mean Fisher energy of the change of the logits, Var_p(dz).

    target: "next" (the next token; the last position is left out), "model" (the model's own top label) or a label."""
    frame._ready(calibrated=False)
    m, hf = frame._m, frame._m.hf
    ids = frame._one(text)
    T = ids.shape[1]
    params = dict(hf.named_parameters())
    bad = [k for k in delta if k not in params]
    if bad:
        raise ValueError(f"not parameters of the model: {bad[:3]}")
    hL, xs, _ = forward(m, ids)
    z0 = head(m, m.final_norm(hL[0]).float())
    p = torch.softmax(z0, -1)
    if target == "next":
        y, valid = torch.cat([ids[0, 1:], ids[0, -1:]]), torch.arange(T, device=m.device) < T - 1
    elif target == "model":
        y, valid = z0.argmax(-1), torch.ones(T, dtype=torch.bool, device=m.device)
    else:
        y, valid = torch.full((T,), label_id(m, target), device=m.device), torch.ones(T, dtype=torch.bool, device=m.device)
    groups = {}
    for k in delta:
        mt = re.search(r"layers\.(\d+)\.", k)
        groups.setdefault(int(mt.group(1)) if mt else k, []).append(k)
    groups = {"joint": list(delta), **groups}

    def dz(keys, own):
        """The first-order change of every position's logits along the step restricted to keys, [T, V]."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with fwAD.dual_level(), _tangent(hf, {k: delta[k] for k in keys}), sdpa_kernel(SDPBackend.MATH):
            h = TR.native(m, xs, [0])[0, 0] if own else forward(m, ids)[0][0]
            t = fwAD.unpack_dual(head(m, m.final_norm(h).float())).tangent
        return torch.zeros_like(z0) if t is None else t.float()

    res = {}
    for name, keys in groups.items():
        row = {}
        for own in (False, True):
            d = dz(keys, own)
            mu = (p * d).sum(-1, keepdim=True)
            G = d.gather(1, y[:, None])[:, 0] - mu[:, 0]  # the change of log p_y
            E = (p * (d - mu) ** 2).sum(-1)
            row[own] = (G[valid], E[valid])
        Gf, Ef = row[False]
        Go, _ = row[True]
        res[name] = {
            "gain": float(Gf.mean()),
            "gain_own": float(Go.mean()),
            "context_share": float(1 - Go.sum() / Gf.sum()) if float(Gf.sum()) != 0 else float("nan"),
            "curvature": float(Ef.mean()),
            "keys": len(keys),
        }
    return res


def bound(frame, texts, token, layers=None):
    """The best mean cosine that any lens with one direction per token and layer can reach with the model's own gradient of the
    token's centred output logit, per layer, over every position of the texts: the square root of the mean inner product of the
    unit gradients of different texts (at least two)."""
    LS = list(range(frame.num_layers)) if layers is None else layers
    sums, counts = [], []
    for text in texts:
        g = frame.state(text)[LS].gradient([token])[..., 0, :]  # [layer, position, hidden]
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        sums.append(g.sum(1).float())
        counts.append(g.shape[1])
    if len(sums) < 2:
        raise ValueError("the bound needs at least two texts")
    S = torch.stack(sums)
    n = torch.tensor(counts, dtype=torch.float32)
    kappa = ((S.sum(0) ** 2).sum(-1) - (S**2).sum((0, -1))) / (n.sum() ** 2 - (n**2).sum())
    return kappa.clamp_min(0).sqrt().cpu().numpy()
