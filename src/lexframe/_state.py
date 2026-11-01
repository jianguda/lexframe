"""The operations of a State: the reading (what each state carries by itself: the expected transport, calibrated); the model's
own gradients and column Jacobians (the context fixed), from which the gradients and the semantic bases (_basis) come; moves carried
forward through the model; and steering."""

from __future__ import annotations

import warnings

import torch

from ._model import final, head, label_id, logprobs, times_head
from ._model import logits as logits_of
from ._transport import Prefix, read_expected, run, step


# ---- the reading

def expected(f, xs, LS, P, vs=None, grad=False):
    """What the states of the layers LS carry by themselves at the positions P: the centred log readings without context
    [layer, position, label] for the label ids vs (None: the whole vocabulary), and with grad=True their gradients [layer,
    position, label, hidden]. At the output layer, the model's own centred log-probabilities and gradients."""
    m, L = f._m, f._m.n_layers
    cal = f.calibration
    xi = f._points() if any(l < L for l in LS) else None
    V = m.lm_head.weight.shape[0]
    samples = f._settings["samples"]
    width = V if vs is None else len(vs)
    # a gradient keeps the probabilities of every draw: about 2 GB of them per chunk of positions
    n = max(1, (f._rows // samples) if not grad else (1 << 29) // (samples * V * 4))
    Pt = torch.as_tensor(P, device=m.device)
    grads = []
    inner = [l for l in LS if l < L]
    if not grad and len(inner) > 1:  # the readings alone: every layer below the output in one pass of the expected transport
        nb = max(1, f._rows // (samples * len(inner)))
        parts = {l: [] for l in inner}
        for s in range(0, len(Pt), nb):
            rows = Pt[s : s + nb]
            r = read_expected(m, cal, {l: xs[l][0][rows] for l in inner}, rows, xi, vs)
            for l in inner:
                c = r[l]
                parts[l].append((c - c.mean(-1, keepdim=True) if vs is None else c).detach().float())
        done = {l: torch.cat(parts[l]) for l in inner}
        LS = [l for l in LS if l >= L]
    else:
        done = {}
    for l in LS:
        h = xs[l][0]
        cs, gs = [], []
        for s in range(0, len(Pt), n):
            rows = Pt[s : s + n]
            x = h[rows].detach().float().requires_grad_(grad)
            with torch.set_grad_enabled(grad):
                if l < L:
                    c = read_expected(m, cal, {l: x.to(h.dtype)}, rows, xi, vs)[l]
                    if vs is None:
                        c = c - c.mean(-1, keepdim=True)
                else:
                    lp = logprobs(m, x.to(h.dtype))
                    c = (lp if vs is None else lp[:, vs]) - lp.mean(-1, keepdim=True)
                if grad:
                    gs.append(torch.stack([torch.autograd.grad(c[:, j].sum(), x, retain_graph=j < width - 1)[0]
                                           for j in range(width)], 1))  # fmt: skip
            cs.append(c.detach().float())
        done[l] = torch.cat(cs)
        if grad:
            grads.append(torch.cat(gs).float())
    return torch.stack([done[l] for l in sorted(done)]), (torch.stack(grads) if grad else None)


# ---- the model's gradients

JFLOATS = 1 << 28  # floats of the column Jacobians held at once (positions x hidden x hidden)
OUT_ROWS = 1 << 24  # floats of the cotangent rows of one batched backward pass to a layer


def _columns(f, xs, l, P, moves=None):
    """The map of the states of layer l at the positions P (each on its own column, attending to the states of its prefix in the
    run xs, the run's moves {layer: [1, T, hidden]} added where it passes them) to the final norm's output there: h [1, position,
    hidden] -> [1, position, hidden]; the norm alone at the output."""
    m = f._m
    L = int(m.n_layers)
    pre = Prefix(m, xs) if l < L else None
    moves = moves or {}

    def to_output(h):
        for k in range(l, L):
            h = step(m, k, h, pre, P)
            if k + 1 in moves:  # a move is a fixed change: it shifts the column and passes its gradient unchanged
                h = h + moves[k + 1][:, P].to(h.device, h.dtype)
        return final(m, h)

    return to_output


def _pull(f, xs, l, P, rows, moves=None):
    """The cotangents rows(z) [row, position, out] at the final norm's output, given there the logits z [position, vocabulary],
    pulled back to the states of layer l at the positions P: [row, position, hidden], and z."""
    m = f._m
    to_output = _columns(f, xs, l, P, moves)
    X = xs[l][:, P].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        nu = to_output(X)
        with torch.no_grad():
            z = head(m, nu[0])
        E = rows(z)
        chunk = max(1, OUT_ROWS // (len(P) * E.shape[-1]))
        out = torch.cat([torch.autograd.grad(nu, X, E[a : a + chunk, None].to(nu.dtype), retain_graph=True, is_grads_batched=True)[0][:, 0]
                         for a in range(0, E.shape[0], chunk)])  # fmt: skip
    return out.float(), z


def gradients(f, xs, LS, P, vs, moves=None):
    """The model's own gradients of the labels' centred log-probabilities at the output with respect to the states of the layers LS
    at the positions P, the context fixed: [layer, position, label, hidden]."""
    m = f._m
    W = m.lm_head.weight
    s = float(m.output_scale)

    def rows(z):  # d (z_v - mean z) / d nu = s (w_v - mean w)
        return (s * (W[vs].float() - f._head_mean()))[:, None].expand(-1, z.shape[0], -1)  # [label, position, out]

    return torch.stack([_pull(f, xs, l, P, rows, moves)[0].permute(1, 0, 2) for l in LS])


def jacobians(f, xs, l, P, moves=None):
    """The Jacobians of the final norm's output with respect to the states of layer l at the positions P, the context fixed:
    [position, out, in], and the output logits there [position, vocabulary]."""
    d = xs[0].shape[-1]
    J, z = _pull(f, xs, l, P, lambda z: torch.eye(d, device=z.device)[:, None].expand(-1, len(P), -1), moves)
    return J.permute(1, 0, 2), z


# ---- moves

@torch.no_grad()
def propagate(m, xs, deltas, moves=None):
    """The states of every layer of the run xs after the new moves deltas {layer: [1, T, hidden]}, each added to its layer's states
    and carried forward through the model's own layers (every position attending to the moved states of its prefix), in increasing
    order of layer, the run's earlier moves (moves, in the same form) added again above the lowest new one: [L + 1] states, the
    same as xs below the lowest moved layer."""
    L = m.n_layers
    l0 = min(deltas)
    every = combine(moves, deltas)

    def add(h, l, d):
        return (h.float() + d[l].to(h.device).float()).to(h.dtype) if l in d else h

    out = list(xs[:l0])
    h = add(xs[l0], l0, deltas)  # xs[l0] holds the earlier moves at l0 already
    out.append(h)
    for k in range(l0, L):
        h = add(step(m, k, h), k + 1, every)
        out.append(h)
    return out


def combine(moves, deltas):
    """The moves of a run after new ones: the changes at each layer added up."""
    out = dict(moves or {})
    for l, d in deltas.items():
        d = d.float()
        out[l] = out[l] + d.to(out[l].device) if l in out else d
    return out


# ---- steering

EPS = torch.finfo(torch.float32).eps  # the working precision: the smallest curvature, decrease and step it tells apart
TOLERANCE = 0.02  # the relative error of the labels' changes at which steering stops, and D's relative predicted decrease
ZETA = 0.8  # the normal step's share of the radius (Byrd-Omojokun)
VROWS = 1 << 14  # head rows per block of the Fisher metric
LOW, HIGH = 0.25, 0.75  # the ratios of actual to predicted decrease that shrink and widen the radius (Nocedal-Wright, alg. 4.1)


def steer(state, weights, tol):
    """Move the selected states of one layer so that each label's coordinate (its centred output log-probability, as the bases'
    coefficients change) changes by its weight, at the selected positions (the other positions stay), while the output's
    distribution over the other labels changes least, and carry the move forward through the model. Each selected state x is
    rotated at its norm to solve

        min D(x) = KL(q0 || q)   subject to   dc_S = w,   |x| = |x0|,

    q the output's distribution over the other labels (renormalised) and q0 that before the move: every label weighed by its
    probability. D is -q0^T z + log sum exp z of the others' logits z (up to a constant), so its gradient is q - q0 and its curvature
    the Fisher metric of q (diag q - q q^T), exactly. The norm is kept because a state's norm weighs it against every later write:
    scaling it is a channel the model's states do not use. It is solved by a trust-region SQP method (Byrd-Omojokun's composite
    step, judged change first): with J the map of the state to the final norm's output (the context fixed; the bases' Jacobian), K
    the Fisher metric in that space (one pass over the head), M = J^T K J, g the gradient of D and R the labels' gradients (as
    State.gradient), all orthogonal to the state, each step within the radius D is
        n = R^+ rem, cut to 0.8 D:                         the shortest rotation towards the change still to make (the normal step);
        t with R t = 0, |t| <= (D^2 - |n|^2)^1/2:          the least of (g + M n)^T t + t^T M t / 2 there (the tangential step),
                                                           solved exactly in the eigenbasis of M.
    The trust region keeps the problem well posed where the metric is flat (moves that change only improbable labels). The
    positions are moved from left to right: a position's move changes no earlier position's output, so each is solved exactly
    given the moves before it. A step is measured on the moved run's own output, the change first: while the change is not made
    within the tolerance, a step is taken where it brings the change still to make down; once it is, where the change stays made
    and D comes down. The radius follows the ratio of the actual to the predicted decrease of what is judged (quartered below 1/4,
    doubled above 3/4 at the boundary), from the length of the shortest rotation up to the state's norm; J and M are taken again
    where a step with an older J disappoints. It stops where the change is made within the tolerance and D's predicted decrease is
    within the tolerance of D, or where the radius falls below the working precision; it warns where a change is not reached."""
    f = state._frame
    m = f._m
    if len(state._layers) != 1:
        raise ValueError("steer one layer at a time; steer the moved State again to move another layer")
    l = state._layers[0]
    labels = list(weights)
    vs = torch.tensor([label_id(m, t) for t in labels], device=m.device)
    w = torch.tensor([float(weights[t]) for t in labels], dtype=torch.float32)
    xs = state._xs
    X0 = xs[l][0].float()
    X = X0.clone()
    P = list(state._positions)
    err = torch.zeros(len(P))
    for i in sorted(range(len(P)), key=lambda i: P[i]):  # left to right: a position's move changes no earlier position's output,
        X[P[i]], err[i] = _steer_one(f, xs, state._moves, X, l, P[i], vs, w, tol)  # so each solves exactly given those before it
    if float(err.max()) > tol:
        warnings.warn(f"steer reached the change within the tolerance {tol} at {int((err <= tol).sum())} of {len(P)} positions; "
                      f"the largest relative error is {float(err.max()):.3g}: a change beyond the reach of a rotation stops short")
    new = {l: X[None] - xs[l].float()}
    return state._run(propagate(m, xs, new, state._moves), combine(state._moves, new))


def _fisher(f, z, others):
    """The Fisher metric of the others' distribution q in the space of the final norm's output, K [d, d], at the logits z
    [vocabulary]: s^2 (W^T diag(q) W - a a^T), a = W^T q. One pass over the head."""
    m = f._m
    W = m.lm_head.weight
    V, s = W.shape[0], float(m.output_scale)
    q = torch.where(others, z, torch.full_like(z, -torch.inf)).softmax(-1)
    K = torch.zeros(W.shape[1], W.shape[1], device=z.device)
    a = torch.zeros(W.shape[1], device=z.device)
    for b in range(0, V, VROWS):
        Wb = W[b : b + VROWS].float()
        wb = q[b : b + VROWS]
        K += (Wb * wb[:, None]).T @ Wb
        a += wb @ Wb
    return s * s * (K - torch.outer(a, a))


def _tangential(lam, gv, Cv, radius):
    """The trust-region step of the model g^T t + t^T M t / 2 with C t = 0 and |t| <= radius, exactly, in the eigenbasis of M
    (eigenvalues lam, gv = V^T g, Cv = C V): t(mu) = -(M + mu I)^-1 (g + C^T kappa) with kappa making C t = 0 and the least mu >= 0
    that fits, found by bisection of log mu to the working precision. Returns V^T t."""
    if radius <= 0 or float(gv.norm()) == 0:
        return torch.zeros_like(gv)

    def step(mu):
        wt = 1 / (lam + mu)
        CW = Cv * wt
        kappa = torch.linalg.lstsq(CW @ Cv.T, -(CW @ gv)[:, None]).solution[:, 0]
        return -wt * (gv + Cv.T @ kappa)

    lo = EPS * lam.max().clamp_min(torch.finfo(lam.dtype).tiny)
    t = step(lo)
    if float(t.norm()) <= radius:
        return t
    a_, b_ = lo.log(), (gv.norm() / radius).clamp_min(lo).log()  # |t(mu)| <= |g| / mu
    for _ in range(int(torch.ceil(torch.log2((b_ - a_).abs().clamp_min(EPS) / EPS)).item())):
        mid = (a_ + b_) / 2
        if float(step(mid.exp()).norm()) > radius:
            a_ = mid
        else:
            b_ = mid
    return step(b_.exp())


def _steer_one(f, xs, moves, X, l, t, vs, w, tol):
    """Rotate the state of layer l at position t in the run xs (with the moves {layer: change}) with the states X of layer l [T,
    hidden] (the earlier positions already moved) so that the labels vs change by w from their values in the run xs, the others'
    distribution changing least: the moved state and the relative error."""
    m = f._m
    L = int(m.n_layers)
    s = float(m.output_scale)
    W = m.lm_head.weight
    X0 = xs[l][0].float()
    V = W.shape[0]
    others = torch.ones(V, dtype=torch.bool, device=W.device)
    others[vs] = False

    def run_with(Xl):  # the run with the states Xl of layer l, carried forward
        return xs if bool((Xl == X0).all()) else propagate(m, xs, {l: (Xl - X0)[None]}, moves)

    def logits(run_):  # the output logits at t
        with torch.no_grad():
            return logits_of(m, run_[L][0, t][None])[0]

    def dist(z):  # the others' log-distribution, 0 at the labels
        lq = torch.where(others, z, torch.full_like(z, -torch.inf)).log_softmax(-1)
        return torch.where(others, lq, torch.zeros_like(lq))

    z0 = logits(xs)
    lab0 = z0[vs] - z0.mean()
    lq0 = dist(z0)
    q0 = lq0.exp() * others
    wn = w.norm().clamp_min(1e-30)

    def state_of(z):  # D = KL(q0 || q), q - q0 (D's gradient in the others' logits), the change still to make, the relative error
        lq = dist(z)
        rem = (w.to(z.device) - ((z[vs] - z.mean()) - lab0)).cpu()
        return float((q0 * (lq0 - lq)).sum()), lq.exp() * others - q0, rem, float(rem.norm() / wn)

    x = X.clone()
    r = float(x[t].norm())
    run_ = run_with(x)
    z = logits(run_)
    D, e, rem, err = state_of(z)
    radius, lin, fresh = None, None, False
    while True:
        if lin is None:  # J and M at the moved state: the bases' Jacobian and the others' Fisher metric
            J, _ = jacobians(f, run_, l, [t], moves)
            u = x[t] / r
            J = J[0] - (J[0] @ u)[:, None] * u  # [out, in], orthogonal to the state: a rotation
            Mx = J.T @ _fisher(f, z, others) @ J
            lam, Vm = torch.linalg.eigh((Mx + Mx.T) / 2)
            lam = lam.clamp_min(0)
            lin, fresh = True, True
        # one batched backward pass at the moved state: the labels' gradients and D's gradient
        def rows(zz, e=e):
            lab_rows = s * (W[vs].float() - f._head_mean())  # the labels' centred logits
            d_row = s * times_head(m, e)  # sum over the others of (q - q0)_v times their logits' covectors
            return torch.cat([lab_rows, d_row[None]])[:, None]

        G, _ = _pull(f, run_, l, [t], rows, moves)
        G = G[:, 0]
        u = x[t] / r
        G = G - (G @ u)[:, None] * u
        k = len(vs)
        R, g = G[:k], G[k]
        rj = rem.to(G.device)
        n_ = R.T @ torch.linalg.lstsq(R @ R.T, rj[:, None]).solution[:, 0]  # the shortest rotation towards the change
        if radius is None:
            radius = min(float(n_.norm()), r)  # the scale of the problem: the shortest rotation's length
        if float(n_.norm()) > ZETA * radius:
            n_ = n_ * (ZETA * radius / float(n_.norm()))
        nv, gv = Vm.T @ n_, Vm.T @ g
        tv = _tangential(lam, gv + lam * nv, torch.cat([u[None], R]) @ Vm, max(radius * radius - float(n_ @ n_), 0.0) ** 0.5)
        dv = nv + tv
        d = Vm @ dv
        pred_D = float(-(gv @ dv) - 0.5 * (lam * dv * dv).sum())  # D's predicted decrease, to second order
        pred_c = float(rj.norm() - (rj - R @ d).norm())
        if err <= tol and pred_D <= tol * D:
            break  # the change made, and the others as still as the model can tell
        if radius <= EPS * r:
            break  # the steps fell below the working precision
        y = x.clone()
        step_ = x[t] + d
        y[t] = r * step_ / step_.norm()
        run1 = run_with(y)
        z1 = logits(run1)
        D1, e1, rem1, err1 = state_of(z1)
        if err > tol:  # the change first: the step is judged by the change still to make
            ared, pred = float(rem.norm() - rem1.norm()), pred_c
        else:  # then the others: by D, the change kept within the tolerance
            ared, pred = (D - D1 if err1 <= tol else -float("inf")), pred_D
        ratio = ared / pred if pred > 0 else 0.0
        if ratio < LOW:
            if fresh:
                radius /= 4
            else:
                lin = None  # the model disappoints with an older J: take J and M again before shrinking
        elif ratio > HIGH and float(d.norm()) >= radius * (1 - EPS**0.5):
            radius = min(2 * radius, r)
        if ared > 0 and pred > 0:  # taken
            x, run_, z, D, e, rem, err = y, run1, z1, D1, e1, rem1, err1
            fresh = False
    return x[t], err
