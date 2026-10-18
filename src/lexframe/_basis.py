"""The lexical basis at a layer, s = J^+ y, solved by LSMR on Jacobian-vector products without forming J."""

import math

import torch
from torch.autograd import forward_ad as fwAD
from torch.utils.checkpoint import checkpoint

from . import _transport as TR
from ._model import block, innorm


def references(m, cal, xs, l, order):
    L = int(m.n_layers)
    B, Tn, d = xs[0].shape
    fi = torch.arange(0, B * Tn, Tn, device=xs[0].device)
    if order is None:
        return [innorm(m, k)(xs[k]) for k in range(l, L)]
    low = [xs[l]] * order
    out = []
    with torch.no_grad():
        for k in range(l, L):
            norm = innorm(m, k)
            refs = [norm(h) for h in low]
            out.append(refs[order - 1])
            att = [TR.attend(m, k, refs[j], refs[j - 1]) for j in range(1, order)]
            new = [TR._step(m, cal, k, low[0].reshape(-1, d), fi).view(B, Tn, d)]
            for j in range(1, order):
                a = att[j - 1].reshape(-1, d)
                new.append(
                    block(m, k, low[j].reshape(-1, d), lambda u, s, a=a: a[s : s + u.shape[0]].to(u.dtype)).view(
                        B, Tn, d
                    )
                )
            low = new
    return out


def stream(m, l, refs, X, ckpt=False):
    L = int(m.n_layers)
    b, Tn, d = X.shape

    def step(h, k):
        R = refs[k - l]
        R = R.expand(b, *R.shape[1:]) if R.shape[0] != b else R
        a = TR.attend(m, k, innorm(m, k)(h), R).reshape(-1, d)
        return block(m, k, h.reshape(-1, d), lambda u, s: a[s : s + u.shape[0]].to(u.dtype)).view(b, Tn, d)

    h = X
    for k in range(l, L):
        h = checkpoint(step, h, k, use_reentrant=False) if ckpt else step(h, k)
    return m.final_norm(h).float()


def jvp(m, l, kv, X0, V):
    dt = X0.dtype
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


def lsmr(Av, Atu, b, x0, damp, steps):
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
    return x0 + dx


def auto_batch(m, probe, n, cap=64):
    if m.device.type != "cuda":
        return min(n, 8)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    probe(2)
    per = max(1, (torch.cuda.max_memory_allocated() - base) // 2)
    free = torch.cuda.get_device_properties(m.device).total_memory - base
    return max(1, min(n, cap, int(0.8 * free / per)))


def basis(m, cal, xs, LS, Y, order=None, ridge=0.001, steps=30, after=None, warm=False):
    nr = Y.shape[0]
    LS = sorted(LS, reverse=True)
    X0s = {l: xs[l][:1] for l in LS}
    K0 = references(m, cal, xs, min(LS), None) if order is None else None
    kvof = lambda l: K0[l - min(LS) :] if order is None else references(m, cal, xs, l, order)

    def run(rows, l, x0, kv):
        X0 = X0s[l]
        y = Y[rows].float()
        Av = lambda v: jvp(m, l, kv, X0, v)
        Atu = lambda u: vjp(m, l, kv, X0, u)
        damp = math.sqrt(ridge) * Atu(y).norm(dim=-1, keepdim=True) / y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        return lsmr(Av, Atu, y, x0, damp, steps)

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
            x = torch.zeros_like(Y[rows].float())
            for l in LS:
                kv = kvof(l)
                x = run(rows, l, x if warm else torch.zeros_like(x), kv)
                if after is not None:
                    after(l, X0s[l], kv, x, rows)
            r0 = rows.stop
        except torch.cuda.OutOfMemoryError:
            if b == 1:
                raise
            b = max(1, b // 2)
            torch.cuda.empty_cache()
