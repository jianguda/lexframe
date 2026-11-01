"""The semantic bases of a state x of layer k: one vector b_v for every label of the vocabulary, the polar basis of the coefficient
function

    c(h) = log R_k(x) + z(h) - z(x)    (centred over the vocabulary),

with R_k the reading (what the state carries: State.logits) and z the model's own output log-probabilities, the context fixed. At
the state, the coefficients are the reading; a move changes them exactly as it changes the model's output; their gradients are the
model's own (State.gradient). At the output layer the reading is the output, and c is z.

The polar basis of c at x: the state is the sum of the bases weighted by its coefficients, its projection on a basis is that
coefficient, and a move along a basis changes that coefficient by the step, to first order: with u = x / |x|, g_v = grad c_v,
rho_v = <g_v, u> and p_v = g_v - rho_v u (the readout is almost wholly tangential: the layers read the state's direction),

    b_v = r_v u + beta_v p_v + c_v lam,    r_v = c_v |x| / |c|^2:

the radial parts add up to the state (sum_v c_v r_v = |x|) and give the projections (<x, b_v> |c|^2 / |x|^2 = c_v); the tangential
parts are the shortest, over the whole vocabulary together, that give every basis a unit gain (<g_v, b_v> = 1) and cancel in the
composition (sum_v c_v t_v = 0):

    (|c|^2 I - sum_v c_v^2 p^_v p^_v^T) lam = -sum_v c_v a_v p_v / |p_v|^2,    beta_v = (a_v - c_v <p_v, lam>) / |p_v|^2,

a_v = 1 - r_v rho_v, a system of the hidden size that the wide cone of the vocabulary's covectors keeps well conditioned. Each
label's basis is its own: it does not depend on the other labels asked for. The gradients of the whole vocabulary come from the
Jacobian of the final norm's output with respect to the state (one backward pass per hidden dimension, for several positions
together) and the head.
"""

from __future__ import annotations

import torch

from . import _state as S

GFLOATS = 1 << 29  # floats of the vocabulary's gradients held at once (positions x vocabulary x hidden): positions go together
VROWS = 1 << 14  # head rows per block of the vocabulary's gradients


def _vocabulary(f, xs, l, P, moves=None):
    """At the states of layer l and positions P: the states h [position, hidden], their coefficients c (the centred log readings)
    [position, vocabulary] and the coefficients' gradients g (the model's own) [position, vocabulary, hidden], from the column
    Jacobians (one backward pass per hidden dimension for all the positions together) and the head."""
    m = f._m
    W = m.lm_head.weight
    V, s = W.shape[0], float(m.output_scale)
    with torch.no_grad():
        c, _ = S.expected(f, xs, [l], P, None)
    J, _ = S.jacobians(f, xs, l, P, moves)  # [position, out, in]
    g = torch.empty(len(P), V, J.shape[-1], device=J.device)
    with torch.no_grad():
        mean = f._head_mean()
        for j in range(len(P)):
            for v0 in range(0, V, VROWS):
                g[j, v0 : v0 + VROWS] = (s * (W[v0 : v0 + VROWS].float() - mean)) @ J[j]
    return xs[l][0, P].detach().float(), c[0].float(), g


def bases(f, xs, LS, P, vs, moves=None):
    """The semantic bases of the labels vs (None: the whole vocabulary) at the layers LS and positions P of the states xs,
    [layer, position, label, hidden]."""
    m = f._m
    V, d = m.lm_head.weight.shape
    group = max(1, min(GFLOATS // (V * d), S.JFLOATS // (d * d)))
    out = []
    for l in LS:
        dl = []
        for p0 in range(0, len(P), group):
            hs, cs, gs = _vocabulary(f, xs, l, list(P[p0 : p0 + group]), moves)
            for h, c, g in zip(hs, cs, gs):
                h, c = h.to(g.device), c.to(g.device)
                hn = h.norm()
                u = h / hn
                c2 = (c * c).sum()
                rho = g @ u
                p = g - rho[:, None] * u
                n = (p * p).sum(-1)
                r = c * hn / c2
                a = 1 - r * rho
                A = c2 * (torch.eye(len(u), device=u.device) + torch.outer(u, u)) - (p * (c * c / n)[:, None]).T @ p
                rhs = -((c * a / n)[:, None] * p).sum(0)
                lam = torch.linalg.solve(A, rhs)
                lam = lam + torch.linalg.solve(A, rhs - A @ lam)  # one refinement: the sums over the vocabulary lose float32 digits
                sel = slice(None) if vs is None else vs
                beta = (a[sel] - c[sel] * (p[sel] @ lam)) / n[sel]
                dl.append(r[sel, None] * u + beta[:, None] * p[sel] + c[sel, None] * lam)
        out.append(torch.stack(dl))
    return torch.stack(out)
