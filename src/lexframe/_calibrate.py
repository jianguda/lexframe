"""The calibration of a frame, from one pass of the model over unlabelled text and a second for the residuals.

Per layer: the typical attention output, a ridge regression of the attention output on its input, with the ridge chosen by the
MacKay evidence fixed point (started from two-fold cross-validation); and the closure statistics of its residuals, the mean residual
per position bucket and the posterior-predictive residual covariance (the pooled covariance times n / (n - gamma), with gamma the
effective number of parameters). Position 0 attends only to itself and is left out of the residuals."""

from __future__ import annotations

from typing import Iterable

import torch

from ._closure import NB, bucket
from ._model import Model, forward

RHO = [10 ** (e / 8) for e in range(-56, 25)]


class _Moments:

    def __init__(self, m: Model, batches: Iterable[torch.Tensor], folds: int = 2, work=torch.float32, progress=False):
        L, d = m.n_layers, m.hidden_dim
        self.L, self.d, self.K, self.dev = L, d, folds, m.device
        self.n = 0
        acc = torch.float64 if 2 * folds * L * d * d * 8 < 4000000000.0 else torch.float32
        z = lambda *s: torch.zeros(*s, dtype=acc, device=m.device)
        self.C = [[[z(d, d), z(d, d)] for _ in range(folds)] for _ in range(L)]
        self.cnt = [0] * folds
        at = [0]
        self.mu = torch.zeros(L, folds, d, dtype=torch.float64, device=m.device)
        self.ma = torch.zeros_like(self.mu)
        self.tr = torch.zeros(L, folds, dtype=torch.float64)

        def cap(k, u, a):
            g = (at[0] + torch.arange(u.shape[0], device=u.device)) % folds
            for f in range(folds):
                if not bool((g == f).any()):
                    continue
                n2, mu2, ma2, C2, B2, t2 = _moments(
                    u[g == f].reshape(-1, d).to(work), a[g == f].reshape(-1, d).to(work)
                )
                n1 = self.cnt[f]
                w = n1 * n2 / (n1 + n2)
                du, da = mu2 - self.mu[k, f], ma2 - self.ma[k, f]
                Cf = self.C[k][f]
                Cf[0] += (C2 + w * torch.outer(du, du)).to(acc)
                Cf[1] += (B2 + w * torch.outer(du, da)).to(acc)
                self.tr[k, f] += t2 + w * float(da @ da)
                self.mu[k, f] += du * (n2 / (n1 + n2))
                self.ma[k, f] += da * (n2 / (n1 + n2))
                if k == L - 1:
                    self.cnt[f] = n1 + n2

        from tqdm.auto import tqdm

        for ids in tqdm(batches, desc="Calibration", disable=not progress, leave=False):
            ids = m.ids(ids)
            forward(m, ids, on_attention=cap)
            at[0] += ids.shape[0]
            self.n += ids.numel()
        if at[0] < folds:
            raise ValueError(f"calibration needs at least {folds} sequences")

    def moments(self, k):
        out = [
            (
                self.cnt[f],
                self.mu[k, f],
                self.ma[k, f],
                self.C[k][f][0].double(),
                self.C[k][f][1].double(),
                float(self.tr[k, f]),
            )
            for f in range(self.K)
        ]
        self.C[k] = None
        return out


def _moments(U, A):
    n = U.shape[0]
    mu, ma = U.mean(0), A.mean(0)
    Uc = U - mu
    Ac = A - ma
    return n, mu.double(), ma.double(), (Uc.T @ Uc).double(), (Uc.T @ Ac).double(), float((Ac.double() ** 2).sum())


def _merge(ms):
    n = sum((m[0] for m in ms))
    mu = sum((m[0] * m[1] for m in ms)) / n
    ma = sum((m[0] * m[2] for m in ms)) / n
    C = sum((m[3] + m[0] * torch.outer(m[1] - mu, m[1] - mu) for m in ms))
    B = sum((m[4] + m[0] * torch.outer(m[1] - mu, m[2] - ma) for m in ms))
    return n, mu, ma, C, B, sum((m[5] + m[0] * float((m[2] - ma) @ (m[2] - ma)) for m in ms))


class Calibration:
    """Per layer k: the typical attention output (ubar, abar, the eigenvectors Q and eigenvalues lam of the input covariance, the
    projected cross-covariance QA and the ridge rho), and the closure statistics (the offsets off [bucket, hidden] and a square root
    fac of the innovation covariance, fac fac^T = Sigma)."""

    FIELDS = ("ubar", "abar", "Q", "lam", "QA", "rho", "off", "fac", "count")

    def __init__(self, ubar, abar, Q, lam, QA, rho, off, fac, count, stats, model_id):
        self.ubar, self.abar, self.Q, self.lam, self.QA, self.rho = ubar, abar, Q, lam, QA, rho
        self.off, self.fac, self.count = off, fac, count
        self.stats, self.model_id = stats, model_id

    def __repr__(self) -> str:
        return f"Calibration(model={self.model_id['architecture']}, layers={self.model_id['layers']})"

    def typical(self, k: int):
        ub, ab = self.ubar[k], self.abar[k]
        dev = ub.device
        Q, QA, lam, rid = self.Q[k].to(dev), self.QA[k].to(dev), self.lam[k], self.rho[k]

        def fn(u, s):
            uc = u.float() - ub
            return (ab.expand_as(uc) + uc @ Q / (lam[None] + rid) @ QA).to(u.dtype)

        return fn

    def to(self, device):
        big = lambda t: t if t.device.type == "cpu" else t.to(device)  # large tables stay where calibrate() put them
        f = {k: getattr(self, k) for k in self.FIELDS}
        f = {k: (big(v) if k in ("Q", "QA", "fac") else v if k == "count" else v.to(device)) for k, v in f.items()}
        return Calibration(**f, stats=self.stats, model_id=self.model_id)

    def save(self, path: str) -> None:
        torch.save(
            {
                "format": 2,
                "model_id": self.model_id,
                "stats": self.stats,
                **{k: getattr(self, k).cpu() for k in self.FIELDS},
            },
            path,
        )

    @staticmethod
    def load(path: str, model: Model | None = None) -> Calibration:
        z = torch.load(path, map_location="cpu", weights_only=True)
        if z.get("format") != 2:
            raise ValueError(f"{path}: not a lexframe calibration of format 2 (lexframe 1.3 or later)")
        c = Calibration(**{k: z[k] for k in Calibration.FIELDS}, stats=z["stats"], model_id=z["model_id"])
        if model is not None:
            if c.model_id != model_id(model):
                raise ValueError(f"{path} was calibrated for {c.model_id}, not for {model_id(model)}")
            c = c.to(model.device)
        return c


def model_id(m: Model) -> dict:
    w = lambda p: float(sum((t.double().sum() for t in p.parameters())))
    return {
        "architecture": type(m.hf).__name__,
        "layers": m.n_layers,
        "hidden": m.hidden_dim,
        "checksum": round(w(m.layers[0]) + w(m.layers[-1]), 3),
    }


def chunks(m: Model, texts: Iterable[str], length: int = 128, batch: int = 8, limit: int | None = None):
    if m.tokenizer is None:
        raise ValueError("this model has no tokenizer: pass token ids to calibrate")
    eos = m.tokenizer.eos_token_id
    buf, out, n = [], [], 0
    for t in texts:
        if not t or not t.strip():
            continue
        buf += m.tokenizer.encode(t, add_special_tokens=False) + ([eos] if eos is not None else [])
        while len(buf) >= length:
            out.append(buf[:length])
            buf = buf[length:]
            n += 1
            if len(out) == batch:
                yield torch.tensor(out)
                out = []
            if limit is not None and n >= limit:
                if out:
                    yield torch.tensor(out)
                return
    if out:
        yield torch.tensor(out)


def spread(m: Model, texts: list, size: int, length: int = 128, window: int = 400) -> list[torch.Tensor]:
    """`size` blocks of `length` tokens, one starting at each of `size` evenly spaced texts (documents joined by EOS, as in
    chunks()), in batches of 8; contiguous chunks when there are fewer texts than blocks."""
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < size:
        return list(chunks(m, texts, length=length, limit=size))
    out = []
    for j in range(size):
        s = j * len(texts) // size
        out += [b for b in chunks(m, texts[s : s + window], length=length, batch=1, limit=1)]
    out = torch.cat(out)
    return list(out.split(8))


def evidence(lam, q, tr, n, rho, iterations=500):
    """The MacKay evidence fixed point for the ridge of one layer's regression, in its eigen-coordinates: lam the eigenvalues of
    the input scatter, q the squared projected cross-scatter, tr the trace of the output scatter, n the rows, rho the start.
    Returns the ridge and gamma, the effective number of parameters."""
    rss = lambda r: tr - float((q * (2 / (lam + r) - lam / (lam + r) ** 2)).sum())
    r = rho
    for _ in range(iterations):
        g = float((lam / (lam + r)).sum())
        new = g * rss(r) / (float((q / (lam + r) ** 2).sum()) * (n - g))
        done = abs(new / r - 1) < 1e-10
        r = new
        if done:
            break
    return r, float((lam / (lam + r)).sum())


@torch.no_grad()
def residuals(m: Model, cal: Calibration, batches, gamma, store) -> tuple:
    """The closure statistics of the typical attention's residuals: offsets [L, bucket, d], factors [L, d, d], counts [bucket]."""
    L, d, dev = m.n_layers, m.hidden_dim, m.device
    acc = torch.float64 if L * d * d * 8 < 4e9 else torch.float32
    S = torch.zeros(L, d, d, dtype=acc, device=dev)
    s = torch.zeros(L, NB, d, dtype=torch.float64, device=dev)
    c = torch.zeros(NB, dtype=torch.float64, device=dev)
    typ = [cal.typical(k) for k in range(L)]

    def cap(k, u, a):
        B, T, _ = u.shape
        r = (a.reshape(-1, d).float() - typ[k](u.reshape(-1, d), 0).float()).view(B, T, d)[:, 1:]
        b = torch.tensor([bucket(t) for t in range(1, T)], device=dev)
        S[k] += torch.einsum("btd,bte->de", r.to(acc), r.to(acc))
        s[k].index_add_(0, b, r.double().sum(0))
        if k == 0:
            c.index_add_(0, b, torch.full((T - 1,), float(B), dtype=torch.float64, device=dev))

    for ids in batches:
        forward(m, m.ids(ids), on_attention=cap)
    off = s / c.clamp_min(1)[None, :, None]
    for b in range(2, NB):  # buckets beyond the calibration length take the last one seen
        if c[b] == 0:
            off[:, b] = off[:, b - 1]
    fac = torch.empty(L, d, d, device=store)
    n = float(c.sum())
    for k in range(L):
        C = (S[k].double() - (off[k].T * c) @ off[k]) / n * (n / (n - gamma[k]))
        lam, V = torch.linalg.eigh(C)
        fac[k] = (V * lam.clamp_min(0).sqrt()).float().to(store)
    return off.float(), fac, c.cpu()


@torch.no_grad()
def calibrate(m: Model, batches: Iterable, folds: int = 2, work=torch.float32, progress: bool = False) -> Calibration:
    batches = list(batches)
    bank = _Moments(m, batches, folds, work, progress)
    L, d, dev, K = bank.L, bank.d, bank.dev, bank.K
    f64 = torch.float64
    store = "cpu" if L * d * d * 8 > 1500000000.0 else dev
    ubar = torch.empty(L, d, device=dev)
    abar = torch.empty(L, d, device=dev)
    stats = []
    Q = torch.empty(L, d, d, device=store)
    QA = torch.empty(L, d, d, device=store)
    lam = torch.empty(L, d, device=dev)
    rho = torch.empty(L, device=dev)
    grid = torch.tensor(RHO, dtype=f64, device=dev)
    gamma = []
    for k in range(L):
        mo = bank.moments(k)
        n, mu, ma, Cuu, Cua, tr = _merge(mo)
        ubar[k], abar[k] = mu.float(), ma.float()
        rg = grid * (Cuu.diagonal().sum() / d)
        err = torch.zeros_like(rg)
        tts = 0.0
        for g in range(K):
            _, mug, mag, Cg, Bg, _ = _merge([mo[i] for i in range(K) if i != g])
            nf, _, _, Cf, Bf, tf = mo[g]
            eu, ea = mo[g][1] - mug, mo[g][2] - mag
            Duu = Cf + nf * torch.outer(eu, eu)
            Dua = Bf + nf * torch.outer(eu, ea)
            tt = tf + nf * float(ea @ ea)
            tts += tt
            ev, Qg = torch.linalg.eigh(Cg.to(work))
            ev = ev.double().clamp_min(0)
            QAg = Qg.T @ Bg.to(work)
            Dr = Qg.T @ Duu.to(work) @ Qg
            p = (Qg.T @ Dua.to(work) * QAg).sum(1).double()
            KS = (QAg @ QAg.T * Dr).double()
            W = 1.0 / (ev[None] + rg[:, None])
            err += tt - 2 * (W @ p) + (W @ KS * W).sum(1)
        j = int(err.argmin())
        ev, Qk = torch.linalg.eigh(Cuu.to(work))
        ev = ev.double().clamp_min(0)
        QAk = Qk.T @ Cua.to(work)
        rcv = float(rg[j])
        rk, g = evidence(ev, (QAk.double() ** 2).sum(1), tr, n, rcv)
        gamma.append(g)
        stats.append(
            {
                "rows": n,
                "ridge_cv": rcv,
                "ridge_evidence": rk,
                "gamma": g,
                "r2_heldout_cv": 1 - float(err[j]) / tts,
            }
        )
        Q[k], QA[k], lam[k], rho[k] = Qk.float().to(store), QAk.float().to(store), ev.float(), rk
    cal = Calibration(ubar, abar, Q, lam, QA, rho, None, None, None, stats, model_id(m))
    cal.off, cal.fac, cal.count = residuals(m, cal, batches, gamma, store)
    for k in range(L):
        stats[k]["residual_count"] = float(cal.count.sum())
    return cal
