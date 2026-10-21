"""The typical attention output of every block: a ridge regression of the attention output on its input."""

from __future__ import annotations

from typing import Iterable

import torch

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

    def __init__(self, ubar, abar, Q, lam, QA, rho, stats, model_id):
        self.ubar, self.abar, self.Q, self.lam, self.QA, self.rho, self.stats, self.model_id = (
            ubar,
            abar,
            Q,
            lam,
            QA,
            rho,
            stats,
            model_id,
        )

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
        mv = lambda t: t.to(device)
        return Calibration(
            mv(self.ubar),
            mv(self.abar),
            self.Q if self.Q.device.type == "cpu" else mv(self.Q),
            mv(self.lam),
            self.QA if self.QA.device.type == "cpu" else mv(self.QA),
            mv(self.rho),
            self.stats,
            self.model_id,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "format": 1,
                "model_id": self.model_id,
                "stats": self.stats,
                **{k: getattr(self, k).cpu() for k in ("ubar", "abar", "Q", "lam", "QA", "rho")},
            },
            path,
        )

    @staticmethod
    def load(path: str, model: Model | None = None) -> Calibration:
        z = torch.load(path, map_location="cpu", weights_only=True)
        if z.get("format") != 1:
            raise ValueError(f"{path}: not a lexframe calibration")
        c = Calibration(z["ubar"], z["abar"], z["Q"], z["lam"], z["QA"], z["rho"], z["stats"], z["model_id"])
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


@torch.no_grad()
def calibrate(m: Model, batches: Iterable, folds: int = 2, work=torch.float32, progress: bool = False) -> Calibration:
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
    for k in range(L):
        mo = bank.moments(k)
        n, mu, ma, Cuu, Cua, _ = _merge(mo)
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
        rk = float(rg[j])
        stats.append(
            {
                "rows": n,
                "ridge_relative": RHO[j],
                "dof": float((ev / (ev + rk)).sum()),
                "r2_heldout": 1 - float(err[j]) / tts,
            }
        )
        Q[k], QA[k], lam[k], rho[k] = Qk.float().to(store), QAk.float().to(store), ev.float(), rk
    return Calibration(ubar, abar, Q, lam, QA, rho, stats, model_id(m))
