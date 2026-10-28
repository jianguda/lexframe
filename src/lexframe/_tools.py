"""Research tools, not part of the public API: the staged QR chart of a column and the undamped frame vectors it gives in factored form,
with the pairing self-test; the audit of a weight step; and the bound of fixed lenses."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import torch

from . import _transport as TR
from ._model import forward, label_id
from ._model import innorm
from ._state import CHUNK, column_jacobian, tangent, working_dtype


# ---- the staged chart

def one_layer(m, xs, k):
    """Layer k as a map of the states of every column, each attending to the true states xs of its prefix and to its own key."""
    R = innorm(m, k)(xs[k])
    return lambda h: TR.step(m, k, h, R)



class Staged:
    """The chart of one column (one position) as a staged QR transport (the Benettin / Shimada-Nagashima method of Lyapunov spectra),
    never forming the end-to-end product J_k = N' A_{L-1} ... A_k:

        J_L^T U = Q_L R_L,   A_l^T Q_{l+1} = Q_l R_l (l = L-1 .. k),   so  U^T J_k = R_L^T ... R_k^T Q_k^T,

    with U [d, d-g] an orthonormal basis of the reachable output tangent (the range of the final norm's Jacobian; g gauge directions),
    A_l the column Jacobian of block l and R_l upper triangular. Q_k [d, d-g] is the chart's orientation at layer k (an orthonormal basis
    of its row space, the radial null excluded by construction); the scalars D_k = prod_{l >= k} diag R_l carry its conditioning
    (log|D_k| is the finite-time Lyapunov spectrum of the column), and R_k ... R_L = D_k Ut_k with Ut_k unit upper triangular (its
    condition number says how well the scalars carry the conditioning). Solutions are formed by one triangular solve per block, so the
    small singular values are never lost to the rounding of a product or to squaring (J^T J).

    Attributes: layers (the layers kept), U [d, d-g], normal [d, g], Q {k: [d, d-g]}, R {l: [d-g, d-g]}, top (the layer L where
    the stages start), dtype."""

    def __init__(self, U, normal, Q, R, top):
        self.U, self.normal, self.Q, self.R, self.top = U, normal, Q, R, top
        self.layers = sorted(Q)
        self.dtype = U.dtype
        self._ut = {}

    def graded(self, k):
        """(D_k [d-g], Ut_k [d-g, d-g]) with R_k ... R_L = diag(D_k) Ut_k, Ut_k unit upper triangular, by the recursion
        Ut_k = (D_{k+1}^-1 U_k D_{k+1}) Ut_{k+1} (U_k = diag(R_k)^-1 R_k): the product is never formed unscaled. D_k is returned as its
        log magnitude and sign: (logabs, sign)."""
        if k in self._ut:
            return self._ut[k]
        la, sg, Ut = None, None, None
        for l in range(self.top, k - 1, -1):
            R = self.R[l]
            dg = R.diagonal()
            Ul = R / dg[:, None]
            if Ut is None:
                Ut, la, sg = Ul, dg.abs().log(), dg.sign()
            else:
                ratio = torch.exp(la[None, :] - la[:, None]) * (sg[None, :] * sg[:, None])  # d_j / d_i of the layer above
                Ut = (Ul * ratio) @ Ut
                la, sg = la + dg.abs().log(), sg * dg.sign()
            self._ut[l] = (la, sg, Ut)
        return self._ut[k]

    def solve(self, y, k, tol=None):
        """The minimum-norm move at layer k whose output effect is the orthogonal projection of y [d, n] (output tangent targets) onto
        the reachable tangent: returns (z [d-g, n] Oseledets coordinates, s = Q_k z [d, n], reach [n]). tol None: exact (triangular solves
        through every block, no truncation); tol > 0: graded truncation, the coordinates with |D_k,i| < tol max|D_k| set to zero, reach =
        the share of the target's energy the truncated move delivers."""
        yp = self.U.T @ y.to(self.dtype)
        if tol is None:
            z = yp
            for l in range(self.top, k - 1, -1):
                z = torch.linalg.solve_triangular(self.R[l].T, z, upper=False)
            reach = torch.ones(y.shape[1], dtype=self.dtype, device=y.device)
        else:
            # U^T J_k = Ut^T D Q_k^T: the live directions (scale >= tol max) map onto the columns C = Ut^T[:, live] (well conditioned when
            # Ut is); the truncated move is the least-squares fit of the target on them, i.e. the orthogonal projection of the target onto
            # the output directions the layer delivers
            la, sg, Ut = self.graded(k)
            live = la >= la.max() + math.log(tol)
            C = Ut.T[:, live]
            Qc, Rc = torch.linalg.qr(C)
            u = torch.zeros_like(yp)
            u[live] = torch.linalg.solve_triangular(Rc, Qc.T @ yp, upper=True)
            z = u * (sg * torch.exp(-la))[:, None]
            eff = Qc @ (Qc.T @ yp)
            reach = (eff * eff).sum(0) / (yp * yp).sum(0).clamp_min(1e-30)
        return z, self.Q[k] @ z, reach

    def apply(self, s, k):
        """The output effect U^T J_k s [d-g, n] of moves s [d, n] at layer k, by the staged factors (Q_k^T, then R_k^T ... R_L^T)."""
        w = self.Q[k].T @ s.to(self.dtype)
        for l in range(k, self.top + 1):
            w = self.R[l].T @ w
        return w

    def apply_z(self, z, k):
        """The output effect [d-g, n] of a move given in the Oseledets coordinates z of layer k (no hidden-space vector formed)."""
        w = z.to(self.dtype)
        for l in range(k, self.top + 1):
            w = self.R[l].T @ w
        return w


def staged(m, xs, LS, position=-1, store=None):
    """The staged QR chart (see Staged) of one column at the layers LS, each layer attending to the true states xs of its prefix, in the
    states' working precision, starting from the left singular vectors of the final norm's Jacobian. store: the
    device that keeps the factors (one [d, d] triangle per block; default the model's device; "cpu" frees the card for long columns of
    large models)."""
    L = int(m.n_layers)
    T, d = xs[0].shape[1], xs[0].shape[2]
    t = position % T
    chunk = max(8, min(d, CHUNK // (T * d)))
    wk = working_dtype(xs[L])
    # the top stage is the SVD of the final norm's Jacobian with its g gauge directions removed exactly (input: the radial direction, and
    # the constant for a layer norm; output: the normal): J_L^T U = V S on the reachable tangent
    JN, U, normal, Vh, Ssvd, g = tangent(m, xs, t, chunk)
    Q, R = Vh[: d - g].T, torch.diag(Ssvd[: d - g])
    keep = (lambda a: a) if store is None else (lambda a: a.to(store))
    Qs, Rs = {}, {L: keep(R)}
    if L in LS:
        Qs[L] = keep(Q)
    for k in range(L - 1, min(LS) - 1, -1):
        A = column_jacobian(one_layer(m, xs, k), xs[k][:1], [t], chunk)[0].to(wk)
        Q, R = torch.linalg.qr(A.T @ Q)
        del A
        Rs[k] = keep(R)
        if k in LS:
            Qs[k] = keep(Q)
    return Staged(keep(U), keep(normal), Qs, Rs, L)


# ---- the chart of one column


@dataclass
class Chart:
    """The chart of one column (one position of one text) at the read layers as a staged QR transport (see Staged): it is never
    formed as an end-to-end product, so its small singular values are not lost to the rounding of the product or to squaring.

    layers; position; lyapunov {layer: [d-g]}: log10 of the chart's graded scales |D_k| relative to the largest, sorted descending
    (the finite-time Lyapunov spectrum of the column, the conditioning carried by scalars); graded_cond {layer}: the condition number of
    the unit triangular coupling Ut_k (how well the scalars carry the conditioning: small means they do); temperature: rms of the last
    state x_L (the final norm's scale; logits scale as 1/temperature); direction: x_L / |x_L| (the base point on the norm sphere);
    staged: the factors, for solves (Staged.solve / apply / apply_z)."""

    layers: list
    position: int
    lyapunov: dict
    graded_cond: dict
    temperature: float
    direction: torch.Tensor
    staged: object
    states: dict = field(default=None, repr=False)
    _frame: object = field(default=None, repr=False)

    def frame(self, tokens, layer, tol=None):
        """The undamped frame vectors of the tokens at one layer from the staged chart, with their pairing self-test.

        tol None: exact (no truncation; the conditioning stays in the scalars); tol > 0: graded truncation of the directions whose scale
        is below tol times the largest. Returns a dict: z [n, d-g] (Oseledets coordinates), vectors [n, d] (hidden-space vectors, scaled
        to raise each token's centred logit by one: their live part is lost in storage where |vector| exceeds 1/eps of it), reach [n],
        realised / ideal [n, n] (cross-pairings b_u . s(v), realised through the factors without forming the vectors, ideal computed at
        the output), deviation (max |realised - ideal|), deviation_vectors (the same through the stored vectors), norm [n] (|vector| / |x|).
        """
        m = self._frame._m
        S = self.staged
        Wc, Y = self._targets(tokens)  # centred head rows [n, d], output dual vectors [d, n]
        z, s, reach = S.solve(Y, layer, tol)
        eff = S.U @ S.apply_z(z, layer)  # [d, n]
        effv = S.U @ S.apply(s, layer)
        ideal = S.U @ (S.U.T @ Y)
        pair = lambda e: (Wc @ e) / (Wc * e.T).sum(-1)[None]
        realised, ideal_p, viav = pair(eff), pair(ideal), pair(effv)
        gain = m.output_scale * (Wc * eff.T).sum(-1)
        x = self.states[layer]
        return {
            "z": (z / gain[None]).T,
            "vectors": (s / gain[None]).T,
            "reach": reach,
            "realised": realised,
            "ideal": ideal_p,
            "deviation": float((realised - ideal_p).abs().max()),
            "deviation_vectors": float((viav - ideal_p).abs().max()),
            "norm": (s / gain[None]).norm(dim=0) / x.norm().to(S.dtype),
        }

    def _targets(self, tokens):
        f, m, S = self._frame, self._frame._m, self.staged
        vs = torch.tensor([label_id(f._m, t) for t in tokens], device=m.device)
        Wc = (m.lm_head.weight[vs].float() - f._head_mean()).to(S.dtype)
        Y = (m.lm_head.weight[vs].float() @ f._gram_inverse()).to(S.dtype).T
        return Wc, Y

    def factored(self, tokens, layer):
        """The undamped frame vectors of the tokens at one layer in factored form, s_v = Q_k diag(sign_k exp(-log_scale_k)) profile_v:
        no number in it is large. Returns a dict: profile [n, d-g] (the token's part, Ut_k^-T U^T y_v, finite and O(1), scaled so that
        the token's centred logit rises by one), log_scale [d-g] (minus the log of the chart's graded scales |D_k|, shared by every token:
        the length of the frame vector along each Oseledets direction), sign [d-g], and the orthonormal frame Q [d, d-g]. The hidden-space unit
        direction of s_v is dominated by the most contracted directions; the profile is what tells the tokens apart."""
        S = self.staged
        Wc, Y = self._targets(tokens)
        la, sg, Ut = S.graded(layer)
        u = torch.linalg.solve_triangular(Ut.T, S.U.T @ Y, upper=False, unitriangular=True)  # [d-g, n]
        eff = S.U @ (Ut.T @ u)  # the output effect of the frame vector: the orthogonal projection of y
        gain = self._frame._m.output_scale * (Wc * eff.T).sum(-1)
        return {"profile": (u / gain[None]).T, "log_scale": -la, "sign": sg, "Q": S.Q[layer]}

    def decompose(self, delta, tokens, layer):
        """A displacement delta [d] of this column's state at one layer split on the tokens' frame, in factored form:
        c = G^-1 (R delta) with R the tokens' coframe and G = R B^T its pairing with the frame vectors, which equals its
        output-side value (layer-invariant); so c is finite and exact for any layer. Returns c [n], the residual's profile [d-g]
        (delta's graded coordinates D_k Q_k^T delta minus sum_v c_v profile_v: the coframe sees none of it), and check, the relative
        error of the synthesis: R (sum_v c_v s_v), formed through the factors, against R delta; and the pairing G [n, n]."""
        S = self.staged
        Wc, Y = self._targets(tokens)
        F = self.factored(tokens, layer)
        la, sg, Ut = S.graded(layer)
        sc = self._frame._m.output_scale
        d = delta.to(S.dtype).reshape(-1, 1)
        w = S.apply(d, layer)  # U^T J_k delta [d-g, 1]
        Rd = sc * (Wc @ (S.U @ w))[:, 0]  # the first-order change of the tokens' centred logits
        P = F["profile"].T  # [d-g, n]
        G = sc * Wc @ (S.U @ (Ut.T @ P))  # R B^T through the factors: [n, n]
        c = torch.linalg.solve(G, Rd)
        syn = sc * Wc @ (S.U @ (Ut.T @ (P @ c)))
        prof = (sg * torch.exp(la))[:, None] * (S.Q[layer].T @ d)  # delta in graded coordinates (finite: delta is an ordinary vector)
        return {"c": c, "residual_profile": prof[:, 0] - P @ c, "check": float((syn - Rd).norm() / Rd.norm().clamp_min(1e-30)),
                "pairing": G}


def _chart(frame, xs, LS, t, store=None):
    m = frame._m
    S = staged(m, xs, LS, t, store=store)
    lyap, cond = {}, {}
    for k in LS:
        la, _, Ut = S.graded(k)
        v = (la - la.max()) / math.log(10)
        lyap[k] = torch.sort(v, descending=True).values.cpu()
        cond[k] = float(torch.linalg.cond(Ut.float()))
    xL = xs[m.n_layers][0, t].float()
    states = {k: xs[k][0, t] for k in LS}
    return Chart(LS, t, lyap, cond, float(xL.pow(2).mean().sqrt()), (xL / xL.norm()).cpu(), S, states, frame)


def chart(frame, text, layers=None, position=-1, store=None):
    """The staged chart (see Chart) of one position of `text` at `layers` (default all)."""
    frame._ready(calibrated=False)
    m = frame._m
    ids = frame._one(text)
    hL, xs, _ = forward(m, ids)
    xs = xs + [hL]
    L = m.n_layers
    LS = list(range(L + 1)) if layers is None else sorted({int(l) for l in ([layers] if isinstance(layers, int) else layers)})
    return _chart(frame, xs, LS, position % ids.shape[1], store)


# ---- the undamped frame and its pairing self-test


@dataclass
class Pairing:
    """The cross-pairings of the labels' frame vectors with their coframe at every read layer, against their value at the output.

    realised [layer, position, label, label]: b_u . s_k(v) (one on the diagonal); ideal [layer, position, label, label]: the same
    quantity computed at the output from the last hidden state alone (it is the same at every layer: the realised value must equal it
    wherever the layer reaches the labels); deviation [layer, position]: max |realised - ideal| (the self-test)."""

    layers: list
    realised: torch.Tensor
    ideal: torch.Tensor
    deviation: torch.Tensor

    def __repr__(self) -> str:
        return f"Pairing(layers={self.layers}, max deviation={float(self.deviation.max()):.2e})"


def _own_states(state, what):
    if state._X is not None:
        raise ValueError(f"{what} is taken at the model's own states, not at moved ones")
    f = state._frame
    f._ready(calibrated=False)
    return f, state._xs, state._layers


def _exact(state, labels, what):
    """Chart.frame of the labels at every selected layer and position: [layer][position] dicts."""
    f, xs, LS = _own_states(state, what)
    charts = [_chart(f, xs, LS, p) for p in state._positions]
    return [[c.frame(labels, l) for c in charts] for l in LS]


def pairing(state, labels):
    """The self-test of the undamped frame: the cross-pairings of the labels' frame vectors with their coframe at the State's layers,
    against their value computed at the output (identical at every layer wherever the layer reaches the labels)."""
    B = _exact(state, labels, "the pairing")
    st = lambda key: torch.stack([torch.stack([b[key] for b in row]) for row in B]).float().cpu()
    realised, ideal = st("realised"), st("ideal")
    return Pairing(state._layers, realised, ideal, (realised - ideal).abs().amax((-1, -2)))


def exact_frame(state, labels):
    """The labels' undamped frame vectors (the minimum-norm lifts of the output targets, from the staged chart), [..., label, hidden]
    at the State's selection, each raising its label's centred logit by one; see pairing()."""
    B = _exact(state, labels, "the exact frame")
    D = torch.stack([torch.stack([b["vectors"] for b in row]) for row in B]).float().to(state._xs[0].device)
    return state._shape(D)


# ---- the audit of a weight step


@contextlib.contextmanager
def swapped(model, weights):
    """The model with the given parameters (name -> tensor) in place of its own, restored on exit."""
    if not weights:
        yield
        return
    params = dict(model.named_parameters(remove_duplicate=False))
    old = {}
    try:
        for k, v in weights.items():
            old[k] = params[k].data.clone()
            params[k].data.copy_(v.to(params[k].device, params[k].dtype))
        yield
    finally:
        for k, v in old.items():
            params[k].data.copy_(v)



def tail_mass(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - p_y from logits z [..., V] for labels y [...], as exp(logsumexp of the other logits - logsumexp of all), without forming 1 - p_y
    (which rounds to 0 for confident labels)."""
    oth = z.float().scatter(-1, y[..., None], -float("inf"))
    return torch.exp(torch.logsumexp(oth, -1) - torch.logsumexp(z.float(), -1))


@torch.no_grad()
def step_audit(frame, text, delta, target="next"):
    """The first-order effect of a weight step `delta` (name -> tensor) on the log-probability of the target at every position of
    `text`, per parameter group: gain, curvature, the share that comes through the context, and related ratios."""
    import re

    from ._model import logits

    frame._ready(calibrated=False)
    m, hf = frame._m, frame._m.hf
    ids = frame._one(text)
    T = ids.shape[1]
    params = dict(hf.named_parameters())
    bad = [k for k in delta if k not in params]
    if bad:
        raise ValueError(f"not parameters of the model: {bad[:3]}")
    hL, xs, _ = forward(m, ids)
    z0 = logits(m, hL[0])
    p = torch.softmax(z0, -1)
    if target == "next":
        y, valid = torch.cat([ids[0, 1:], ids[0, -1:]]), torch.arange(T, device=m.device) < T - 1
    elif target == "model":
        y, valid = z0.argmax(-1), torch.ones(T, dtype=torch.bool, device=m.device)
    else:
        y, valid = torch.full((T,), label_id(frame._m, target), device=m.device), torch.ones(T, dtype=torch.bool, device=m.device)
    groups = {}  # by layer
    for k in delta:
        mt = re.search(r"layers\.(\d+)\.", k)
        groups.setdefault(int(mt.group(1)) if mt else k, []).append(k)
    groups = {"joint": list(delta), **groups}
    py = p.gather(1, y[:, None])[:, 0]
    tail = tail_mass(z0, y)  # 1 - p_y without rounding to 0 at confident labels

    def dz(keys, mode):
        """The first-order change of every position's logits along the step restricted to keys: the full model (context recomputed)
        or the own column (every other position's states frozen at the base), by central differences."""
        nrm = math.sqrt(sum(float(delta[k].float().pow(2).sum()) for k in keys))
        wn = math.sqrt(sum(float(params[k].float().pow(2).sum()) for k in keys))
        e = 1e-3 * wn / max(nrm, 1e-30)
        out = []
        for sgn in (1, -1):
            w = {k: params[k].data + sgn * e * delta[k].to(params[k].device, params[k].dtype) for k in keys}
            with swapped(hf, w):
                if mode == "full":
                    out.append(logits(m, forward(m, ids)[0][0]))
                else:
                    out.append(logits(m, TR.native(m, xs, [0])[0, 0]))
        return (out[0] - out[1]) / (2 * e)

    res = {}
    for name, keys in groups.items():
        row = {}
        D = {mode: dz(keys, mode) for mode in ("full", "own")}
        for mode, d in D.items():
            dd = d.float()
            mu = (p * dd).sum(-1, keepdim=True)
            G = (dd.gather(1, y[:, None])[:, 0] - mu[:, 0]).float()  # dz . (e_y - p): the first-order gain of log p_y
            E = (p * (dd - mu) ** 2).sum(-1)  # dz^T F dz, two-pass (the one-pass form cancels at confident sites)
            row[mode] = (G[valid], E[valid])
        Gf, Ef = row["full"]
        Go, Eo = row["own"]
        Eu = Gf**2 * py[valid] / tail[valid].clamp_min(1e-30)
        per = (Gf**2 / (2 * Ef.clamp_min(1e-30))).mean()
        res[name] = {
            "gain": float(Gf.mean()),
            "gain_own": float(Go.mean()),
            "context_share": float(1 - Go.sum() / Gf.sum()) if float(Gf.sum()) != 0 else float("nan"),
            "curvature": float(Ef.mean()),
            "curvature_ratio": float(Ef.sum() / Eo.sum().clamp_min(1e-30)),
            "useful_fraction": float(Eu[Gf > 0].sum() / Ef.sum().clamp_min(1e-30)),
            "interference": float((Gf.sum() ** 2 / (2 * Ef.sum().clamp_min(1e-30))) / (len(Gf) * per.clamp_min(1e-30))),
            "keys": len(keys),
        }
    return res


# ---- the bound of fixed lenses


def bound(frame, texts, token, layers=None):
    """The best mean cosine that any lens, with one direction per token and layer, can reach with the coframe of `token` (context
    visible), per layer: sqrt of the coframe's agreement across at least two texts."""
    sums, counts = [], []
    for text in texts:
        LS = list(range(frame.num_layers)) if layers is None else layers
        b = frame.state(text)[LS].gradient([token])[..., 0, :]
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        sums.append(b.sum(-2).float())
        counts.append(b.shape[-2])
    if len(sums) < 2:
        raise ValueError("the bound needs at least two texts")
    S = torch.stack(sums)
    n = torch.tensor(counts, dtype=torch.float32)
    kappa = ((S.sum(0) ** 2).sum(-1) - (S**2).sum((0, -1))) / (n.sum() ** 2 - (n**2).sum())
    return kappa.clamp_min(0).sqrt().cpu().numpy()
