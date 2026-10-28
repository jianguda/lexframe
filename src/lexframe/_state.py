"""The operations of a State, in the frame's terms: the decoding, with the context visible (the native transport) or invisible (the
expected transport); the coframe, the labels' gradients in either state; the frame vectors as applied moves (the damped least-squares
solution of J s = y, by LSMR on Jacobian-vector products of the native transport, without forming J); the labels' coordinates of a
displacement, from the coframe and the output-side pairing; and steering."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.autograd import forward_ad as fwAD
from torch.utils.checkpoint import checkpoint

from . import _transport as TR
from ._model import innorm, kl, label_id, logprobs
from ._transport import read_expected, run


# ---- the decoding

def reading(f, xs, LS, X, P, k, context):
    """Top-k ids and probabilities [B, layer, position, k] and the divergence from the output [B, layer, position] of the (moved)
    states X at the layers LS and positions P: with the context visible, the model's output from them through the native transport;
    invisible, through the expected transport."""
    if context:
        with torch.no_grad():
            H = run(f._m, xs, LS, X, joint=X is not None, chunk=f._chunk)  # moved states: the model's output after the move
            ref = logprobs(f._m, xs[-1])
            lps = [logprobs(f._m, h) for h in H]
        top = [lp.topk(k, -1) for lp in lps]
        out = [
            torch.stack([t.indices for t in top], 1),
            torch.stack([t.values.exp() for t in top], 1),
            torch.stack([kl(ref, lp) for lp in lps], 1),
        ]
        return out if P is None else [o[:, :, P] for o in out]
    # the context invisible: the expected transport
    f._ready()
    m, L = f._m, f._m.n_layers
    B, T, _ = xs[0].shape
    Pt = list(range(T)) if P is None else list(P)
    state = lambda i, l: (xs[l] if X is None else X[i])[:, Pt].reshape(B * len(Pt), -1)
    ref = logprobs(m, xs[L][:, Pt]).reshape(B * len(Pt), -1)
    t = torch.tensor(Pt, device=m.device).repeat(B)
    N = t.shape[0]
    inner = [(i, l) for i, l in enumerate(LS) if l < L]
    res = {}
    with torch.no_grad():
        if inner:
            xi = f._points()
            n = max(1, f._rows // (f._settings["samples"] * len(inner)))
            for s in range(0, N, n):
                lr = read_expected(m, f.calibration, {l: state(i, l)[s : s + n] for i, l in inner}, t[s : s + n], xi)
                for l, r in lr.items():
                    top = r.topk(k, -1)
                    res.setdefault(l, []).append((top.indices, top.values.exp(), kl(ref[s : s + n], r)))
        if L in LS:
            r = logprobs(m, state(LS.index(L), L))
            top = r.topk(k, -1)
            res[L] = [(top.indices, top.values.exp(), kl(ref, r))]
    out = [torch.stack([torch.cat([p[j] for p in res[l]]) for l in LS]) for j in range(3)]
    return [o.view(len(LS), B, len(Pt), *o.shape[2:]).transpose(0, 1) for o in out]


# ---- the coframe

def coframe(f, xs, LS, Wc, X=None) -> torch.Tensor:
    """The coframe with the context visible: the native gradients of the centred logits Wc at the (moved) states of the layers LS,
    [layer, position, label, hidden]."""
    out = []
    with torch.enable_grad():
        for c in range(0, len(LS), f._chunk):
            lv = [(xs[l] if X is None else X[l]).clone().requires_grad_(True) for l in LS[c : c + f._chunk]]
            y = f._centred_logits(run(f._m, xs, LS[c : c + f._chunk], lv, chunk=f._chunk), Wc)
            n = Wc.shape[0]
            g = [torch.stack(torch.autograd.grad(y[..., i].sum(), lv, retain_graph=i < n - 1)) for i in range(n)]
            out.append(torch.stack(g).float()[:, :, 0])
    return torch.cat(out, 1).permute(1, 2, 0, 3)



def _expected(f, xs, LS, X, vs, P, grad):
    f._ready()
    m, L = f._m, f._m.n_layers
    T = xs[0].shape[1]
    Pt = torch.tensor(list(range(T)) if P is None else list(P), device=m.device)
    xi = f._points() if any(l < L for l in LS) else None
    V = m.lm_head.weight.shape[0]
    samples = f._settings["samples"]
    # a gradient keeps the probabilities of every draw: about 2 GB of them per chunk of positions
    n = max(1, (f._rows // samples) if not grad else (1 << 29) // (samples * V * 4))
    coefs, grads = [], []
    for l in LS:
        h = (xs[l] if X is None else X[l])[0]
        cs, gs = [], []
        for s in range(0, len(Pt), n):
            rows = Pt[s : s + n]
            x = h[rows].detach().float().requires_grad_(grad)
            with torch.set_grad_enabled(grad):
                if l < L:
                    c = read_expected(m, f.calibration, {l: x.to(h.dtype)}, rows, xi, vs)[l]
                else:
                    lp = logprobs(m, x.to(h.dtype))
                    c = lp[:, vs] - lp.mean(-1, keepdim=True)
                if grad:
                    r = len(vs)
                    gs.append(torch.stack([torch.autograd.grad(c[:, j].sum(), x, retain_graph=j < r - 1)[0] for j in range(r)], 1))
            cs.append(c.detach())
        coefs.append(torch.cat(cs))
        if grad:
            grads.append(torch.cat(gs))
    return torch.stack(coefs), (torch.stack(grads) if grads else None)



def expected_logits(f, xs, LS, X, vs, P=None) -> torch.Tensor:
    """With the context invisible: the centred log readings of the labels vs [layer, position, label] at the positions P (all by
    default)."""
    return _expected(f, xs, LS, X, vs, P, False)[0]


def expected_coframe(f, xs, LS, X, vs, P=None) -> torch.Tensor:
    """The coframe with the context invisible: the gradients of expected_logits, [layer, position, label, hidden]."""
    return _expected(f, xs, LS, X, vs, P, True)[1]


# ---- the frame vectors

def references(m, xs, l):
    """The true attention inputs of every position at layers l..L-1, which the transported states attend to."""
    return [innorm(m, k)(xs[k]) for k in range(l, int(m.n_layers))]


def stream(m, l, refs, X, ckpt=False):
    """The native transport of the states X [b, T, d] of layer l to the output (the final-norm output), each position attending to
    the references refs (from references()) of its prefix and to its own key; ckpt recomputes each layer in the backward pass instead
    of keeping its activations."""
    h = X
    for k in range(l, int(m.n_layers)):
        h = checkpoint(TR.step, m, k, h, refs[k - l], use_reentrant=False) if ckpt else TR.step(m, k, h, refs[k - l])
    return m.final_norm(h).float()


FORWARD_AD = None  # forward-mode differentiation: None decides by device (Apple MPS has none for linear layers)


def forward_ad(device) -> bool:
    return torch.device(device).type != "mps" if FORWARD_AD is None else FORWARD_AD


def jvp(m, l, kv, X0, V):
    """The tangents of the final-norm output along V [n, T, d] at X0.
    Without forward mode, J V is the derivative of the vector-Jacobian product J^T eta . V in eta (two reverse passes)."""
    dt = X0.dtype
    if not forward_ad(X0.device):
        X = X0.expand(V.shape[0], -1, -1).clone().requires_grad_(True)
        with torch.enable_grad():
            out = stream(m, l, kv, X)
            eta = torch.zeros_like(out, requires_grad=True)
            g = torch.autograd.grad(out, X, eta, create_graph=True)[0]
            return torch.autograd.grad(g, eta, V.to(g.dtype))[0].float()
    with torch.no_grad(), fwAD.dual_level():
        out = stream(m, l, kv, fwAD.make_dual(X0.expand(V.shape[0], -1, -1).contiguous(), V.to(dt)))
        return fwAD.unpack_dual(out).tangent.float()


def vjp(m, l, kv, X0, U):
    X = X0.expand(U.shape[0], -1, -1).clone().requires_grad_(True)
    with torch.enable_grad():
        return torch.autograd.grad(stream(m, l, kv, X, ckpt=True), X, U)[0].float()


def _ortho(a, b):
    r = torch.sqrt(a * a + b * b)
    z = r == 0
    return (
        torch.where(z, torch.ones_like(a), a / r.clamp_min(1e-30)),
        torch.where(z, torch.zeros_like(a), b / r.clamp_min(1e-30)),
        r,
    )


def lsmr(Av, Atu, b, x0, damp, steps, tol=0.0):
    """LSMR (Fong & Saunders 2011) for min |A x - b|^2 + damp^2 |x|^2, batched over rows (and positions): at most `steps` iterations;
    with tol > 0 a row stops once its damped normal-equation residual |A^T (b - A x) - damp^2 x| (LSMR's |zetabar|) falls below
    tol |A^T b|."""
    nrm = lambda u: u.norm(dim=-1, keepdim=True)
    A = lambda v: (Av(v), damp * v)
    At = lambda u: Atu(u[0]) + damp * u[1]
    u = b - Av(x0), -damp * x0
    beta = torch.sqrt(nrm(u[0]) ** 2 + nrm(u[1]) ** 2)
    u = tuple(p / beta.clamp_min(1e-30) for p in u)
    v = At(u)
    alpha = nrm(v)
    v = v / alpha.clamp_min(1e-30)
    zetabar, alphabar = alpha * beta, alpha
    one, zero = torch.ones_like(alpha), torch.zeros_like(alpha)
    rho, rhobar, cbar, sbar, zeta = one, one, one, zero, zero
    h, hbar, dx = v.clone(), torch.zeros_like(v), torch.zeros_like(v)
    solved = alpha * beta == 0
    start = zetabar.abs()
    for _ in range(0 if bool(solved.all()) else steps):
        Avv = A(v)
        u = Avv[0] - alpha * u[0], Avv[1] - alpha * u[1]
        beta = torch.sqrt(nrm(u[0]) ** 2 + nrm(u[1]) ** 2)
        u = tuple(p / beta.clamp_min(1e-30) for p in u)
        v = At(u) - beta * v
        alpha = nrm(v)
        v = v / alpha.clamp_min(1e-30)
        rhoold, rhobarold = rho, rhobar
        c, s, rho = _ortho(alphabar, beta)
        thetanew, alphabar = s * alpha, c * alpha
        thetabar = sbar * rho
        cbar, sbar, rhobar = _ortho(cbar * rho, thetanew)
        zeta, zetabar = cbar * zetabar, -sbar * zetabar
        hbar = h - thetabar * rho / (rhoold * rhobarold).clamp_min(1e-30) * hbar
        step = zeta / (rho * rhobar).clamp_min(1e-30) * hbar
        h = v - thetanew / rho.clamp_min(1e-30) * h
        dx = dx + torch.where(solved, torch.zeros_like(step), step)
        if tol > 0:
            solved = solved | (zetabar.abs() <= tol * start)
            if bool(solved.all()):
                break
    return x0 + dx


CAP = 64  # most rows of one LSMR batch


def auto_batch(m, probe, n):
    dev = m.device
    if dev.type != "cuda":
        return min(n, 8)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    probe(2)
    per = max(1, (torch.cuda.max_memory_allocated(dev) - base) // 2)
    # what this process may still take: memory free on the device, within its own cap, both net of what it holds
    limit = torch.cuda.get_device_properties(dev).total_memory * torch.cuda.get_per_process_memory_fraction(dev)
    free = min(torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev), limit) - base
    return max(1, min(n, CAP, int(0.8 * max(free, 0) / per)))


def solve(m, xs, LS, Y, damping, steps, tol, after=None, X0=None):
    """The damped least-squares solutions of J s = y at the layers LS for the targets Y, by LSMR from zero: at most `steps`
    iterations to the relative tolerance `tol`, with the relative damping `damping`; X0 {layer: [1, T, d]} moves the states they are
    taken at (the true prefix states xs stay the context). Without damping, these are the directions of _tools.staged()."""
    args = (m, xs, LS, Y, damping, steps, after, tol, X0)
    if m.device.type == "cuda":
        with torch.cuda.device(m.device):
            return _solve(*args)
    return _solve(*args)


def _solve(m, xs, LS, Y, damping, steps, after, tol, X0=None):
    nr = Y.shape[0]
    LS = sorted(LS, reverse=True)
    X0s = {l: (xs[l][:1] if X0 is None else X0[l].to(xs[l].dtype)) for l in LS}
    K0 = references(m, xs, min(LS))
    kvof = lambda l: K0[l - min(LS) :]

    def run(rows, l, x0, kv):
        X0 = X0s[l]
        y = Y[rows].float()
        Av = lambda v: jvp(m, l, kv, X0, v)
        Atu = lambda u: vjp(m, l, kv, X0, u)
        damp = math.sqrt(damping) * Atu(y).norm(dim=-1, keepdim=True) / y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        return lsmr(Av, Atu, y, x0, damp, steps, tol)

    def probe(n):
        l = LS[-1]
        v = Y[:1].float().expand(n, -1, -1)
        jvp(m, l, kvof(l), X0s[l], v)
        vjp(m, l, kvof(l), X0s[l], v)

    b = auto_batch(m, probe, nr)
    r0 = 0
    while r0 < nr:
        rows = slice(r0, min(nr, r0 + b))
        try:
            for l in LS:
                kv = kvof(l)
                x = run(rows, l, torch.zeros_like(Y[rows].float()), kv)
                if after is not None:
                    after(l, X0s[l], kv, x, rows)
            r0 = rows.stop
        except torch.cuda.OutOfMemoryError:
            if b == 1:
                raise
            b = max(1, b // 2)
            torch.cuda.empty_cache()


def directions(f, xs, LS, vs, Wc, X0=None) -> torch.Tensor:
    """The frame vectors of the labels vs at the states of the layers LS, as applied moves, [layer, position, label, hidden], each
    scaled to raise its label's centred logit (Wc: the centred head rows of vs) by one: the converged LSMR solution with the frame f's
    damping, the patch that keeps a move within the transport's linear range where it is ill conditioned; at the model's own
    states or at the moved states X0 {layer: [1, T, d]}."""
    m, cfg = f._m, f._solver
    T = xs[0].shape[1]
    Y = (m.lm_head.weight[vs].float() @ f._gram_inverse())[:, None].expand(-1, T, -1).contiguous()
    S = {}

    def save(l, X, kv, x, rows):
        gain = m.output_scale * (jvp(m, l, kv, X, x) * Wc[rows][:, None]).sum(-1)
        S[(l, rows.start)] = x / gain[..., None]

    solve(m, xs, LS, Y, cfg["damping"], cfg["steps"], cfg["tol"], save, X0)
    return torch.stack([torch.cat([S[k] for k in sorted(S) if k[0] == l]) for l in LS]).transpose(1, 2).detach()



# ---- the output side: the reachable tangent and the labels' pairing

CHUNK = 2**24  # floats of one hidden-state tensor of a forward-mode pass (tangents x positions x hidden)


def working_dtype(X):
    """The working precision of the directions: the states' own, at least float32."""
    return torch.promote_types(X.dtype, torch.float32)


def column_jacobian(f, X0, P, chunk):
    """The Jacobian of a column-wise map f at X0 [1, T, d] for the positions P, [len(P), d_out, d_in]: columns by forward mode (tangent
    e_i at every position at once, the columns are independent), or rows by reverse mode where the device has no forward mode;
    `chunk` directions per pass."""
    T, d = X0.shape[1], X0.shape[2]
    J = None
    for s in range(0, d, chunk):
        e = min(d, s + chunk)
        E = torch.zeros(e - s, T, d, device=X0.device, dtype=X0.dtype)
        E[torch.arange(e - s), :, torch.arange(s, e)] = 1
        if forward_ad(X0.device):
            with torch.no_grad(), fwAD.dual_level():
                out = fwAD.unpack_dual(f(fwAD.make_dual(X0.expand(e - s, -1, -1).contiguous(), E))).tangent.to(working_dtype(X0))
            out = out[:, P].permute(1, 2, 0)  # [P, d_out, chunk of inputs]
            J = torch.empty(len(P), out.shape[1], d, device=X0.device, dtype=out.dtype) if J is None else J
            J[:, :, s:e] = out
        else:
            X = X0.expand(e - s, -1, -1).clone().requires_grad_(True)
            with torch.enable_grad():
                y = f(X)
                out = torch.autograd.grad(y, X, E.to(y.dtype))[0].to(working_dtype(X0))
            out = out[:, P].permute(1, 0, 2)  # [P, chunk of outputs, d_in]
            J = torch.empty(len(P), d, d, device=X0.device, dtype=out.dtype) if J is None else J
            J[:, s:e, :] = out
    return J



def n_gauge(Ji):
    """The number of gauge directions of the final norm's Jacobian Ji [d, d]: one (the radial direction, N(a x) = N(x) up to the
    norm's epsilon), two where the norm also removes the mean (a layer norm)."""
    d = Ji.shape[-1]
    one = torch.full((d,), d**-0.5, dtype=Ji.dtype, device=Ji.device)
    smax = float(torch.linalg.svdvals(Ji)[0])
    return 2 if float((Ji @ one).norm()) < 1e-3 * smax else 1



def tangent(m, xs, t, chunk=None):
    """The final norm's Jacobian at position t and its split of the output space: (JN [d, d], U [d, d-g] an orthonormal basis of the
    reachable output tangent (the range of JN), normal [d, g] the output directions no move of the state can make (the left null
    space: for an RMS norm the ellipsoid normal h / g^2, for a layer norm also the normal of the mean), Vh and S of the SVD of JN, g).
    The g gauge directions are gauge, not physics, whatever their epsilon-sized singular values."""
    L, (T, d) = int(m.n_layers), xs[0].shape[1:]
    chunk = max(8, min(d, CHUNK // (T * d))) if chunk is None else chunk
    wk = working_dtype(xs[L])
    JN = column_jacobian(lambda h: m.final_norm(h).to(wk), xs[L][:1].to(wk), [t], chunk)[0]
    g = n_gauge(JN)
    Usvd, Ssvd, Vh = torch.linalg.svd(JN)
    return JN, Usvd[:, : d - g], Usvd[:, d - g :], Vh, Ssvd, g


def output_pairing(f, xs, P, vs) -> torch.Tensor:
    """The labels' pairing at the output end, [position, label, label]: u's centred-logit change along v's frame vector, each frame
    vector scaled to raise its own label's logit by one. It is the same at every layer, so it needs only the head and the landing point
    (the final norm's reachable tangent at each position)."""
    m = f._m
    Wc = m.lm_head.weight[vs].float() - f._head_mean()
    Y = m.lm_head.weight[vs].float() @ f._gram_inverse()
    G = []
    for p in P:
        U = tangent(m, xs, p)[1].float()
        g = m.output_scale * (Wc @ U) @ (U.T @ Y.T)
        G.append(g / g.diagonal()[None])
    return torch.stack(G)


# ---- steering

JACOBI = 2  # rounds of the joint correction on the model's own forward pass
MAX_STEPS = 24  # steps of the path where the corrected step still misses


def steer(state, weights, tol):
    """Move the selected states so that each label's centred logit changes by its weight, at the selected positions of each selected
    layer; the other positions stay. The move is a combination of the labels' frame vectors (as applied moves) solved with the inverse
    of their pairing with the coframe, so that the cross-talk cancels to first order; its second-order term is removed by a corrected
    step D <- D_1 - B^T G^-1 c_2(D), c_2 = (c(x+D) + c(x-D) - 2 c(x)) / 2. The result is checked on the model's own forward pass with every
    moved position carried together and corrected there (Jacobi rounds); where it still misses `tol` (relative error), the move is
    discarded and a path from the model's own state is followed instead, re-solving at each step within the linear radius. Returns the
    moved State, with `error` and `move` per selected position and, in `_info`, the steps, the regime and |c_2|/|w|."""
    if state._X is not None:
        raise ValueError("steering starts from the model's own states, not from moved ones")
    f = state._frame
    labels = list(weights)
    vs = torch.tensor([label_id(f._m, t) for t in labels], device=f._m.device)
    Wc = f._m.lm_head.weight[vs].float() - f._head_mean()
    xs = state._xs
    T = xs[0].shape[1]
    sel = torch.zeros(T, dtype=torch.bool, device=f._m.device)
    sel[state._positions] = True
    w = torch.tensor([float(weights[t]) for t in labels], dtype=torch.float32, device=Wc.device)
    X, info = {}, {}
    for l in state._layers:
        Xl, info[l] = _steer_layer(f, xs, l, xs[l][0].float(), vs, Wc, w, tol, sel)
        X[l] = Xl[None].to(xs[l].dtype)
    moved = state._with(X)
    P = state._positions
    moved.error = state._shape(torch.stack([info[l]["error"][P] for l in state._layers]).numpy())
    moved.move = state._shape(torch.stack([info[l]["move"][P] for l in state._layers]).numpy())
    moved._info = info
    return moved


def _steer_layer(f, xs, l, X0, vs, Wc, w, tol, sel):
    """Move the states X0 [T, d] of layer l so that the labels' centred logits change by w at the positions sel; the other positions
    stay where they are."""
    m = f._m
    dt = xs[l].dtype

    # the logits after a move come from the model's own forward pass with every position moved (each position then attends to the
    # moved states of its prefix), so the reported error is the one the model shows; the frame stays that of each position's column
    def logits_at(X):  # [T, n]
        return f._centred_logits(TR.run(m, xs, [l], [X[None].to(dt)], joint=True), Wc)[0, 0]

    def zfull(X):  # centred full logits [T, V]
        from ._model import logits

        z = logits(m, TR.run(m, xs, [l], [X[None].to(dt)], joint=True)[0, 0])
        return z - z.mean(-1, keepdim=True)

    def frame_at(X):  # coframe R and frame vectors B [T, n, d] at the (moved) states X
        B = directions(f, xs, [l], vs, Wc, X0={l: X[None]})[0]
        Rd = coframe(f, xs, [l], Wc, {l: X[None].to(dt)})[0]  # [T, n, d]
        return Rd, B

    def step(Rd, B, rem):
        G = Rd @ B.mT
        return (torch.linalg.solve(G, rem)[..., None] * B).sum(-2) * keep, G

    def corrected(X, D, B, G):
        c0 = logits_at(X)
        D1 = D.clone()
        c2 = None
        for _ in range(2):
            c2 = 0.5 * (logits_at(X + D1) + logits_at(X - D1) - 2 * c0)
            D1 = D - (torch.linalg.solve(G, c2)[..., None] * B).sum(-2) * keep
        return D1, c2

    c_start = logits_at(X0)
    wn = w.norm().clamp_min(1e-30)
    T = X0.shape[0]
    keep = sel[:, None].to(X0.dtype)
    steps = torch.zeros(T, dtype=torch.long)
    regime = np.array(["oneshot"] * T, dtype=object)
    target = w.expand(T, -1) * keep
    # the corrected one-shot step
    Rd, B = frame_at(X0)
    D, G = step(Rd, B, target)
    D, c2 = corrected(X0, D, B, G)
    c2ratio = (c2.norm(dim=-1) / wn).cpu()
    X = X0 + D
    steps += 1
    # Jacobi rounds: the positions move together, so each one's prefix has moved too; the residual of the model's own forward pass is
    # solved again on the same frame (one forward per round)
    cX = None  # logits_at(X) where it is already known
    for _ in range(JACOBI):
        cX = logits_at(X)
        rem = target - (cX - c_start)
        todo = (rem.norm(dim=-1) / wn > tol) & sel
        if not bool(todo.any()):
            break
        D2, _ = step(Rd, B, rem)
        X = torch.where(todo[:, None], X + D2, X)
        steps += todo.cpu().long()
        cX = None
    err = (((logits_at(X) if cX is None else cX) - c_start) - target).norm(dim=-1) / wn * sel
    # where it still misses, the step is discarded (its overshoot would stay in the move) and a path from the model's own state is
    # followed, each step within the linear radius
    todo = (err > tol) & sel
    if bool(todo.any()):
        X = torch.where(todo[:, None], X0, X)
        for _ in range(MAX_STEPS):
            rem = target - (logits_at(X) - c_start)
            e = rem.norm(dim=-1) / wn
            todo = todo & (e > tol) & ((X - X0).norm(dim=-1) <= 5 * X0.norm(dim=-1))
            if not bool(todo.any()):
                break
            Rd, B = frame_at(X)
            D, G = step(Rd, B, rem)
            # the linear radius r = 2|J u| / |D^2 z[u, u]| along the move (central differences)
            u = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            eps = 0.1 * D.norm(dim=-1, keepdim=True)
            zp, z0, zm = zfull(X + eps * u), zfull(X), zfull(X - eps * u)
            d1 = (zp - zm) / (2 * eps)
            d2 = (zp - 2 * z0 + zm) / eps**2
            r = 2 * d1.norm(dim=-1) / d2.norm(dim=-1).clamp_min(1e-12)
            D = D * torch.clamp(r / D.norm(dim=-1).clamp_min(1e-30), max=1.0)[:, None]
            D, _ = corrected(X, D, B, G)
            X = torch.where(todo[:, None], X + D, X)
            steps += todo.cpu().long()
            regime[todo.cpu().numpy()] = "path"
        err = ((logits_at(X) - c_start) - target).norm(dim=-1) / wn * sel
    info = {
        "error": err.cpu(),
        "steps": steps,
        "move": ((X - X0).norm(dim=-1) / X0.norm(dim=-1)).cpu(),
        "regime": regime,
        "c2_ratio": c2ratio,
    }
    return X, info
