"""LexFrame: read the hidden state of every layer in the model's own vocabulary (the closure reading), account for what the context
adds after each layer (the ledger), and measure, attribute and steer the state in the native frame."""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from . import _transport as TR
from ._basis import basis as _solve_basis
from ._basis import jvp as _jvp
from ._calibrate import Calibration, calibrate, chunks, spread
from ._closure import bucket, innovations, points, read
from ._model import Model, block, eager, forward, kl, logprobs, own_attention, wrap

Text = str | Sequence[str] | torch.Tensor
Token = int | str
Layers = int | Iterable[int] | None
FRAMES = ("closure", "native")
CHECK_TEXT = "The quick brown fox jumps over the lazy dog."


def _single(text) -> bool:
    return isinstance(text, str) or (isinstance(text, torch.Tensor) and text.dim() == 1)


def _kind(frame: str) -> str:
    if frame not in FRAMES:
        raise ValueError(f"frame is 'closure' or 'native', not {frame!r}")
    return frame


class Carrier:
    """The hidden states of one or more texts at one or more layers (layer num_layers is the output), with what the frame reads in
    them.

    vector: the states, [..., position, hidden]; tokens, ids, probs [..., position, k] and kl [..., position]: the reading, as NumPy
    arrays, by default the closure reading (`reading` says which). first_layer() finds where a token first tops it. basis(),
    readout() and coef() give the lexical basis, the lexical readout and the coefficients of given tokens (ids or single-token
    strings such as " Paris") at the states of one text, in the native frame or, for readout() and coef(), the closure; gap() compares
    the two readouts. Adding or subtracting a vector moves the states, and steer() moves them along the basis; a moved Carrier is
    read again with frame.view(). A Carrier of chosen positions only (positions) holds its reading and nothing more.
    """

    def __init__(
        self, frame, vector, layers, ids, context, flags, moved=False, reading=None, positions=None, kind="closure"
    ):
        self.vector, self.layers, self.moved = vector, layers, moved
        self._frame, self._ids, self._context, self._flags = frame, ids, context, flags
        self.ids, self.probs, self.tokens, self.kl = reading or (None, None, None, None)
        self.positions = positions
        self.reading = kind if reading else None

    def _whole(self, what: str) -> None:
        if self.positions is not None:
            raise ValueError(f"{what} needs every position: view the text without positions=")

    def basis(self, tokens: Sequence[Token]) -> torch.Tensor:
        """The lexical basis of the tokens in the native frame, [..., position, token, hidden]: the move of the state that the
        remaining layers, with the true context, turn into raising that token alone, scaled to raise its logit by one. Unmoved states
        only."""
        self._whole("the basis")
        return self._frame._directions(self, tokens, "basis", "native")

    def readout(self, tokens: Sequence[Token], frame: str = "native") -> torch.Tensor:
        """The lexical readout of the tokens, [..., position, token, hidden]: in the native frame the gradient of each token's
        centred output logit, so that its inner product with a displacement of the state is the logit change it predicts; in the
        closure the gradient of the token's centred log reading."""
        self._whole("the readout")
        return self._frame._directions(self, tokens, "readout", _kind(frame))

    def coef(self, tokens: Sequence[Token], frame: str = "native") -> torch.Tensor:
        """The coefficients of the state on the tokens, [..., position, token]: their centred output logits when the state is
        carried with its true context (native), or their centred log reading (closure)."""
        self._whole("the coefficient")
        return self._frame._directions(self, tokens, "coef", _kind(frame))

    def gap(self, tokens: Sequence[Token]) -> torch.Tensor:
        """The frame gap of the tokens, [..., position, token]: one minus the cosine of the native readout with the closure readout.
        It measures how much the context will still reinterpret the state."""
        n, c = self.readout(tokens), self.readout(tokens, "closure")
        return 1 - torch.nn.functional.cosine_similarity(n, c.to(n), dim=-1)

    def steer(self, weights: Mapping[Token, float]) -> Carrier:
        """The states moved along the lexical basis of each token by its weight, in logits, at every position, e.g.
        {" Paris": -4, " London": 4}. Unmoved states of one text only."""
        tokens = list(weights)
        b = self.basis(tokens)
        w = torch.tensor([float(weights[t]) for t in tokens], dtype=b.dtype, device=b.device)
        return self + (b * w[:, None]).sum(-2)

    def first_layer(self, token: Token) -> np.ndarray:
        """The first of the read layers at which the token tops the reading, [..., position]; -1 where it never does."""
        if self.ids is None:
            raise ValueError("a moved Carrier has no reading; read it with frame.view() first")
        top = self.ids[..., 0] == self._frame.token_id(token)
        top = top[..., None, :] if isinstance(self.layers, int) else top
        layers = np.array(self._layer_list())
        return np.where(top.any(-2), layers[top.argmax(-2)], -1)

    def _layer_list(self) -> list[int]:
        return [self.layers] if isinstance(self.layers, int) else list(self.layers)

    def _canonical(self) -> torch.Tensor:
        v = self.vector
        single, one = self._flags
        v = v[None] if one else v
        return v[None] if single else v

    def _moved(self, other, sign) -> Carrier:
        self._whole("moving a state")
        d = other.vector if isinstance(other, Carrier) else torch.as_tensor(other)
        v = self.vector + sign * d.to(self.vector)
        return Carrier(self._frame, v, self.layers, self._ids, self._context, self._flags, True)

    def __add__(self, other):
        return self._moved(other, 1)

    def __radd__(self, other):
        return self._moved(other, 1)

    def __sub__(self, other):
        return self._moved(other, -1)

    def __repr__(self) -> str:
        extra = "" if self.positions is None else f", positions={self.positions}"
        return f"Carrier(layers={self.layers}, shape={tuple(self.vector.shape)}, moved={self.moved}, reading={self.reading}{extra})"


class Ledger:
    """What the context adds to the reading of one position after each layer, for one target token, in log-probability units.

    logp [layer]: log R_k(target) for k = 0..num_layers, the closure reading of every layer (the last is the model's output), and prob
    its exponent. step [layer]: log R_{j+1} - log R_j, the change that layer j makes, so log p_out - log R_k = step[k:].sum() exactly.
    Each step is the innovation of layer j's attention write (the true write against its closure prediction) plus a noise term (the
    closure's own spread at layer j): step = innovation + noise. sources [layer, position]: the innovation split by the position that
    the attention read it from (integrated gradients along the straight path of the write), with residual = innovation - its sum
    (None where the attention cannot be split). check: |the reading rebuilt from the true write - logp[j + 1]|, which must vanish.
    Layer 0 is the embedding, for diagnosis; the reading starts at layer 1."""

    def __init__(self, token, target, position, logp, innovation, noise, sources, residual, check):
        self.token, self.target, self.position = token, target, position
        self.logp, self.prob, self.step = logp, np.exp(logp), np.diff(logp)
        self.innovation, self.noise, self.sources, self.residual, self.check = (
            innovation,
            noise,
            sources,
            residual,
            check,
        )

    def __repr__(self) -> str:
        return f"Ledger(token={self.token!r}, position={self.position}, layers={len(self.step)}, sources={self.sources is not None})"


class LexFrame:
    """A lexical reference frame of a Hugging Face causal language model. It reads the hidden state of layer k at position t by the
    closure: the state runs through the model's own remaining layers on its own column, with every attention write replaced by its
    calibrated prediction plus a Gaussian innovation, averaged by quasi-Monte Carlo over `samples` points. Directions, attribution
    and steering use the native frame: the model's own remaining layers with the true context. Call init() once before use.
    """

    def __init__(self, model, tokenizer=None, samples: int = 64, **removed):
        if "order" in removed:
            raise TypeError(
                "order was removed in lexframe 1.3: view() gives the closure reading, which needs no order, and "
                "view(..., frame='native') the exact transport that order='inf' gave"
            )
        if removed:
            raise TypeError(f"unexpected arguments {sorted(removed)}")
        self._m = model if isinstance(model, Model) else wrap(model, tokenizer)
        self.calibration: Calibration | None = None
        self.identities: dict | None = None
        self._chunk = 8 if self._m.hidden_dim < 4000 else 6
        self._rows = 1 << 15  # rows of one closure pass
        self.solver = {"steps": 30, "ridge": 1e-3, "warm": False}
        self.samples = _samples(samples)
        self._xi = None

    @property
    def num_layers(self) -> int:
        """The number of layers; layer num_layers is the output."""
        return self._m.n_layers

    def __repr__(self) -> str:
        return f"LexFrame(model={type(self._m.hf).__name__}, layers={self._m.n_layers}, samples={self.samples})"

    def config(
        self,
        steps: int | None = None,
        ridge: float | None = None,
        warm: bool | None = None,
        samples: int | None = None,
    ) -> LexFrame:
        """Settings of the iterative LSMR solver behind basis(), kept in `frame.solver`: `steps` per layer (30), the relative `ridge`
        that damps directions the remaining layers barely pass on (1e-3), and `warm` to start each layer from the solution of the
        layer above (False); and the number of quasi-Monte Carlo `samples` of the closure reading (64)."""
        new = {k: v for k, v in dict(steps=steps, ridge=ridge, warm=warm).items() if v is not None}
        if "steps" in new and (not isinstance(new["steps"], int) or new["steps"] < 1):
            raise ValueError("steps must be a positive integer")
        if "ridge" in new and new["ridge"] < 0:
            raise ValueError("ridge must not be negative")
        if samples is not None:
            self.samples, self._xi = _samples(samples), None
        self.solver.update(new)
        return self

    def init(
        self,
        texts: Iterable,
        size: int = 1024,
        spread: bool = True,
        cache: bool = True,
        cache_dir: str | None = None,
        tol: float = 0.05,
        progress: bool = True,
    ) -> LexFrame:
        """Calibrate on unlabelled `texts`: `size` sequences of 128 tokens, one starting at each of `size` evenly spaced texts
        (spread=True, when there are enough texts) or cut contiguously from the start; then check the exact identities. With
        cache=True the calibration is stored under a fingerprint of the model's weights and configuration and the calibration
        tokens, and loaded again when all of them match."""
        batches = _batches(self._m, texts, size, spread)
        path = os.path.join(_folder(cache_dir), _fingerprint(self._m, batches) + ".pt") if cache else None
        if path and os.path.exists(path):
            cal = Calibration.load(path, self._m)
        else:
            cal = calibrate(self._m, batches, progress=progress)
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cal.save(path)
        self.calibration, self._xi = cal.to(self._m.device), None
        self.identities = self._check(tol)
        if not self.identities["ok"]:
            warnings.warn(f"the frame does not reproduce the model within {tol} nats: {self.identities}")
        return self

    @torch.no_grad()
    def view(
        self,
        x: Text | Carrier,
        k: int = 1,
        layers: Layers = None,
        positions: int | Iterable[int] | None = None,
        frame: str = "closure",
    ) -> Carrier | list[Carrier]:
        """The hidden states of one or more texts at every layer (or the given ones) with their reading, or the reading of a (moved)
        Carrier. frame="closure" (default) is the closure reading, what each state carries by itself; frame="native" carries the
        states to the output with their true context, which for unmoved states is the model's own output and for moved ones the
        output after the move. positions reads only the given positions of a text, which is cut after the last of them. Texts of
        different token lengths are read one at a time, and a list of Carriers is returned."""
        kind = _kind(frame)
        if positions is not None and not _is_int(positions):
            positions = list(positions)
        texts = self._ragged(x)
        if texts is not None:
            return [self.view(t, k, layers, positions, frame) for t in texts]
        self._ready()
        P = None
        if isinstance(x, Carrier):
            if positions is not None:
                raise ValueError("a Carrier is read as it is: positions apply to texts")
            if x.positions is not None:
                raise ValueError("a Carrier of chosen positions holds its reading only: view the text again")
            if x._frame._m.hf is not self._m.hf:
                raise ValueError("this Carrier comes from another model")
            if layers is not None:
                raise ValueError("a Carrier is read at its own layers")
            ids, xs, LS, flags = x._ids, x._context, x._layer_list(), x._flags
            V = x._canonical()
            X = [V[:, i].to(xs[l].dtype) for i, l in enumerate(LS)] if x.moved else None
        else:
            ids = self._m.ids(x)
            if positions is not None:
                P = chosen(positions, ids.shape[1])
                ids = ids[:, : P[-1] + 1]
            hL, xs, _ = forward(self._m, ids)
            xs = xs + [hL]
            LS = self._layers(layers)
            flags = (_single(x), isinstance(layers, int))
            V = torch.stack([xs[l] for l in LS], 1).float()
            X = None
        out = self._closure_reading(xs, LS, X, P, k) if kind == "closure" else self._native_reading(xs, LS, X, P, k)
        out = [o.cpu().numpy() for o in out]
        if P is not None:
            V = V[:, :, P]
        if flags[0]:
            out, V = [o[0] for o in out], V[0]
        if flags[1]:
            out = [out[0][..., 0, :, :], out[1][..., 0, :, :], out[2][..., 0, :]]
            V = V[..., 0, :, :]
        vector = x.vector if isinstance(x, Carrier) else V
        tok = self._m.tokenizer
        tokens = None if tok is None else np.vectorize(lambda i: tok.decode([int(i)]), otypes=[object])(out[0])
        moved = x.moved if isinstance(x, Carrier) else False
        reading = (out[0], out[1], tokens, out[2])
        return Carrier(self, vector, LS[0] if flags[1] else LS, ids, xs, flags, moved, reading, P, kind)

    @torch.no_grad()
    def attr(
        self, text: Text, sublayer: str | None = None, target: Token | None = None
    ) -> np.ndarray | list[np.ndarray]:
        """How much each layer (or its `sublayer`, "attn" or "mlp") contributed to the logit of the token `target` (default: the
        model's own prediction at every position), [layer, position]: the change of the target's centred output logit when what the
        layer wrote at a position is removed there, in the native frame."""
        if sublayer not in (None, "attn", "mlp"):
            raise ValueError("sublayer is None (the whole layer), 'attn' or 'mlp'")
        texts = self._ragged(text)
        if texts is not None:
            return [self.attr(t, sublayer, target) for t in texts]
        self._ready()
        ids = self._m.ids(text)
        hL, xs, mid = forward(self._m, ids, mid=True)
        L = self._m.n_layers
        W = self._m.lm_head.weight.float()
        top = (
            (self._m.final_norm(hL).float() @ W.T).argmax(-1)
            if target is None
            else torch.full_like(ids, self.token_id(target))
        )
        Wd = W[top] - W.mean(0)
        states = xs + [hL]
        written = {
            None: lambda k: states[k + 1] - xs[k],
            "attn": lambda k: mid[k] - xs[k],
            "mlp": lambda k: states[k + 1] - mid[k],
        }[sublayer]
        D = [written(k) for k in range(L)]
        out = torch.empty(L, *ids.shape, device=self._m.device)
        LS = list(range(1, L))
        X = [(xs[l].float() - D[l - 1].float()).to(xs[l].dtype) for l in LS]
        out[: L - 1] = self._coeff_rows(self._run(xs, LS), Wd) - self._coeff_rows(self._run(xs, LS, X), Wd)
        out[L - 1] = self._coeff_rows(hL, Wd) - self._coeff_rows((hL.float() - D[L - 1].float()).to(hL.dtype), Wd)
        out = out.transpose(0, 1).cpu().numpy()
        return out[0] if _single(text) else out

    def ledger(self, text: Text, target: Token | None = None, position: int = -1, steps: int = 8) -> Ledger:
        """The ledger of one position of one text (the last by default) for the token `target` (default: the model's prediction):
        how the closure reading of the target changes after each layer, split into the innovation of the layer's attention write
        and a noise term, and the innovation split by source position with `steps` points of integrated gradients."""
        self._ready()
        m, cal = self._m, self.calibration
        ids = self._one(text)
        t = chosen(position, ids.shape[1])[0]
        if t == 0:
            raise ValueError(
                "position 0 attends only to itself: its reading is exact at every layer and it has no ledger"
            )
        ids = ids[:, : t + 1]
        L, dev = m.n_layers, m.device
        u_in, a_out = {}, {}

        def cap(k, u, a):
            u_in[k], a_out[k] = u[0], a[0, t]

        hL, xs, _ = forward(m, ids, on_attention=cap)
        out = logprobs(m, hL[0, t])
        v = int(out.argmax()) if target is None else self.token_id(target)
        xi, tt = self._points(), torch.tensor([t], device=dev)
        with torch.no_grad():
            R = read(
                m, cal, {k: xs[k][0, t][None] for k in range(L)}, tt, xi, torch.tensor([v], device=dev), center=False
            )
        logp = np.array([float(R[k][0, 0]) for k in range(L)] + [float(out[v])])
        innovation, noise, check, residual = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
        sources = np.zeros((L, t + 1))
        split = True
        alphas = ((torch.arange(steps, device=dev) + 0.5) / steps)[:, None]
        for j in range(L):
            ahat = cal.typical(j)(u_in[j][t][None], 0).float()[0] + cal.off[j, bucket(t)].to(dev)
            a = a_out[j].float()
            A = torch.cat([ahat[None], a[None], ahat + alphas * (a - ahat)]).requires_grad_(True)
            with torch.enable_grad():
                g = self._after(j, xs[j][0, t], A, t, xi, v)
                grad = torch.autograd.grad(g[2:].sum(), A)[0][2:].mean(0)
            g = g.detach()
            check[j] = abs(float(g[1]) - logp[j + 1])
            innovation[j], noise[j] = logp[j + 1] - float(g[0]), float(g[0]) - logp[j]
            parts = self._sources(j, u_in[j], t, a) if split else None
            if parts is None:
                split = False
                continue
            C, w = parts
            sources[j] = ((C - w[:, None] * ahat[None]) @ grad).cpu().numpy()
            residual[j] = innovation[j] - sources[j].sum()
        token = m.tokenizer.decode([v]) if m.tokenizer is not None else None
        return Ledger(
            token, v, t, logp, innovation, noise, sources if split else None, residual if split else None, check
        )

    def bound(self, texts: Iterable[Text], token: Token, layers: Layers = None) -> np.ndarray:
        """The best mean cosine that any lens, with one direction per token and layer, can reach with the model's own readout of
        `token`, per layer: sqrt of the native readout's agreement across at least two texts."""
        sums, counts = [], []
        for text in texts:
            LS = list(range(self.num_layers)) if layers is None else layers
            b = self._delta(self._one(text), [token], LS, "readout")[0][..., 0, :]
            b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            sums.append(b.sum(-2).double())
            counts.append(b.shape[-2])
        if len(sums) < 2:
            raise ValueError("the bound needs at least two texts")
        S = torch.stack(sums)
        n = torch.tensor(counts, dtype=torch.float64)
        kappa = ((S.sum(0) ** 2).sum(-1) - (S**2).sum((0, -1))) / (n.sum() ** 2 - (n**2).sum())
        return kappa.clamp_min(0).sqrt().cpu().numpy()

    def token_id(self, token: Token) -> int:
        """The id of a token, given as an id or as a string that the tokenizer encodes to exactly one token."""
        return token_id(self._m, token)

    # ---- the closure

    def _points(self) -> torch.Tensor:
        """The innovations at the integration points, [samples, layer, hidden], computed once per calibration."""
        if self._xi is None or self._xi.shape[0] != self.samples:
            fac = self.calibration.fac
            self._xi = innovations(fac, points(fac, self.samples), self._m.device)
        return self._xi

    def _closure_reading(self, xs, LS, X, P, k):
        """Top-k ids and probabilities [B, layer, position, k] and the divergence from the output [B, layer, position]."""
        m, L = self._m, self._m.n_layers
        B, T, _ = xs[0].shape
        Pt = list(range(T)) if P is None else list(P)
        state = lambda i, l: (xs[l] if X is None else X[i])[:, Pt].reshape(B * len(Pt), -1)
        ref = logprobs(m, xs[L][:, Pt]).reshape(B * len(Pt), -1)
        t = torch.tensor(Pt, device=m.device).repeat(B)
        N = t.shape[0]
        inner = [(i, l) for i, l in enumerate(LS) if l < L]
        res = {}
        if inner:
            xi = self._points()
            n = max(1, self._rows // (self.samples * len(inner)))
            for s in range(0, N, n):
                lr = read(m, self.calibration, {l: state(i, l)[s : s + n] for i, l in inner}, t[s : s + n], xi)
                for l, r in lr.items():
                    top = r.topk(k, -1)
                    res.setdefault(l, []).append((top.indices, top.values.exp(), kl(ref[s : s + n], r)))
        if L in LS:
            r = logprobs(m, state(len(LS) - 1, L))
            top = r.topk(k, -1)
            res[L] = [(top.indices, top.values.exp(), kl(ref, r))]
        out = [torch.stack([torch.cat([p[j] for p in res[l]]) for l in LS]) for j in range(3)]
        return [o.view(len(LS), B, len(Pt), *o.shape[2:]).transpose(0, 1) for o in out]

    def _native_reading(self, xs, LS, X, P, k):
        H = self._run(xs, LS, X)
        ref = logprobs(self._m, xs[-1])
        lps = [logprobs(self._m, h) for h in H]
        top = [lp.topk(k, -1) for lp in lps]
        out = [
            torch.stack([t.indices for t in top], 1),
            torch.stack([t.values.exp() for t in top], 1),
            torch.stack([kl(ref, lp) for lp in lps], 1),
        ]
        return out if P is None else [o[:, :, P] for o in out]

    def _closure_directions(self, xs, LS, X, vs, kind):
        """Centred log readings of the tokens vs [layer, position, token] and, for kind 'readout', their gradients."""
        m, L = self._m, self._m.n_layers
        T = xs[0].shape[1]
        t = torch.arange(T, device=m.device)
        xi = self._points() if any(l < L for l in LS) else None
        V = m.lm_head.weight.shape[0]
        # a gradient keeps the probabilities of every draw: about 2 GB of them per chunk of positions
        n = max(1, (self._rows // self.samples) if kind != "readout" else (1 << 29) // (self.samples * V * 4))
        coefs, grads = [], []
        for i, l in enumerate(LS):
            h = (xs[l] if X is None else X[l])[0]
            cs, gs = [], []
            for s in range(0, T, n):
                x = h[s : s + n].detach().float().requires_grad_(kind == "readout")
                with torch.set_grad_enabled(kind == "readout"):
                    if l < L:
                        f = read(m, self.calibration, {l: x.to(h.dtype)}, t[s : s + n], xi, vs)[l]
                    else:
                        lp = logprobs(m, x.to(h.dtype))
                        f = lp[:, vs] - lp.mean(-1, keepdim=True)
                    if kind == "readout":
                        r = len(vs)
                        gs.append(
                            torch.stack(
                                [torch.autograd.grad(f[:, j].sum(), x, retain_graph=j < r - 1)[0] for j in range(r)], 1
                            )
                        )
                cs.append(f.detach())
            coefs.append(torch.cat(cs))
            if kind == "readout":
                grads.append(torch.cat(gs))
        return torch.stack(coefs), (torch.stack(grads) if grads else None)

    def _after(self, j, x, A, t, xi, v):
        """log R_{j+1}(v) of the states that layer j makes from x [hidden] with the attention writes A [n, hidden]."""
        m = self._m
        h = block(m, j, x[None].expand(A.shape[0], -1).contiguous(), lambda u, s: A[s : s + u.shape[0]].to(u.dtype))
        if j + 1 == m.n_layers:
            return logprobs(m, h)[:, v]
        tt = torch.full((A.shape[0],), t, device=m.device)
        return read(m, self.calibration, {j + 1: h}, tt, xi, torch.tensor([v], device=m.device), center=False)[j + 1][
            :, 0
        ]

    @torch.no_grad()
    def _sources(self, j, u, t, a):
        """Layer j's attention write at position t split by source position, (C [position, hidden], w [position]) with C summing to
        the write and w the head-averaged attention weights; None where the attention module does not allow the split.
        """
        m = self._m
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
                r = at(hidden_states=h, position_embeddings=pe, attention_mask=TR.causal(m, j, n, dev, dt))
                weights = r[1] if isinstance(r, tuple) and len(r) > 1 else None
                at(hidden_states=h, position_embeddings=pe, attention_mask=TR._mask(n, False, None, dev, dt))
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

    # ---- the native frame

    def _directions(self, x: Carrier, tokens, kind, frame) -> torch.Tensor:
        self._ready()
        if not x._flags[0]:
            raise ValueError("directions are taken at the states of one text")
        if kind == "basis" and x.moved:
            raise ValueError("the basis is taken at the model's own states, not at moved ones")
        X = None
        if x.moved:
            V = x._canonical()
            X = {l: V[:, i].to(x._context[l].dtype) for i, l in enumerate(x._layer_list())}
        if frame == "closure":
            vs = torch.tensor([self.token_id(t) for t in tokens], device=self._m.device)
            LS = x._layer_list()
            coef, vectors = self._closure_directions(x._context, LS, X, vs, kind)
            one = isinstance(x.layers, int)
            out = coef if kind == "coef" else vectors
            return out[0] if one else out
        vectors, coef = self._delta(x._ids, tokens, x.layers, kind, X)
        return coef if kind == "coef" else vectors

    def _delta(self, ids, tokens, layers, kind, X=None):
        self._ready()
        hL, xs, _ = forward(self._m, ids)
        xs = xs + [hL]
        T = ids.shape[1]
        one = isinstance(layers, int)
        LS = self._layers(layers)
        vs = torch.tensor([self.token_id(t) for t in tokens], device=self._m.device)
        W = self._m.lm_head.weight.float()
        Wc = W[vs] - W.mean(0)
        Xl = None if X is None else [X[l] for l in LS]
        with torch.no_grad():
            coef = self._coeff(self._run(xs, LS, Xl), Wc)[:, 0]
        vectors = None
        if kind == "basis":
            Y = (W[vs] @ self._gram_inverse())[:, None].expand(-1, T, -1).contiguous()
            S = {}
            scale = self._m.output_scale

            def save(l, X0, kv, x, rows):
                gain = scale * (_jvp(self._m, l, kv, X0, x) * Wc[rows][:, None]).sum(-1)
                S[(l, rows.start)] = x / gain[..., None]

            cfg = self.solver
            _solve_basis(self._m, xs, LS, Y, cfg["ridge"], cfg["steps"], save, cfg["warm"])
            vectors = torch.stack([torch.cat([S[k] for k in sorted(S) if k[0] == l]) for l in LS]).transpose(1, 2)
        elif kind == "readout":
            vectors = self._readout(xs, LS, Wc, X)
        if one:
            coef = coef[0]
            vectors = None if vectors is None else vectors[0]
        return (None if vectors is None else vectors.detach()), coef.detach()

    def _readout(self, xs, LS, Wc, X=None) -> torch.Tensor:
        out = []
        with torch.enable_grad():
            for c in range(0, len(LS), self._chunk):
                lv = [(xs[l] if X is None else X[l]).clone().requires_grad_(True) for l in LS[c : c + self._chunk]]
                f = self._coeff(self._run(xs, LS[c : c + self._chunk], lv), Wc)
                n = Wc.shape[0]
                g = [torch.stack(torch.autograd.grad(f[..., i].sum(), lv, retain_graph=i < n - 1)) for i in range(n)]
                out.append(torch.stack(g).float()[:, :, 0])
        return torch.cat(out, 1).permute(1, 2, 0, 3)

    def _run(self, xs, LS, X=None) -> torch.Tensor:
        """The final states of the native transport of the layers LS, [len(LS), B, T, d] (layer L is the output itself)."""
        L = self._m.n_layers
        inner = [l for l in LS if l < L]
        out = []
        for c in range(0, len(inner), self._chunk):
            Xc = None if X is None else X[c : c + self._chunk]
            out.append(TR.native(self._m, xs, inner[c : c + self._chunk], Xc))
        if L in LS:
            out.append((xs[L] if X is None else X[len(inner)])[None])
        return torch.cat(out)

    @torch.no_grad()
    def _check(self, tol: float) -> dict:
        m = self._m
        ids = m.ids(CHECK_TEXT if m.tokenizer is not None else torch.arange(8) % m.lm_head.weight.shape[0])
        hL, xs, _ = forward(m, ids)
        ref = logprobs(m, hL)
        L = m.n_layers
        native = max(kl(ref, logprobs(m, h)).max() for h in self._run(xs, list(range(L))))
        first = read(
            m,
            self.calibration,
            {l: xs[l][0, :1] for l in range(L)},
            torch.zeros(1, dtype=torch.long, device=m.device),
            self._points(),
        )
        first = max(kl(ref[0, :1], r).max() for r in first.values())
        result = {"native_all_layers": float(native), "closure_first_position": float(first)}
        result["ok"] = max(result.values()) < tol
        return result

    def _ready(self) -> None:
        if self.calibration is None:
            raise RuntimeError("call init(texts) first")
        if self.calibration.ubar.device != self._m.device:
            self.calibration, self._xi = self.calibration.to(self._m.device), None
            self.__dict__.pop("_gi", None)

    def _one(self, text: Text) -> torch.Tensor:
        ids = self._m.ids(text)
        if ids.shape[0] != 1:
            raise ValueError("this method takes one text at a time")
        return ids

    def _ragged(self, x) -> list | None:
        return ragged(self._m, x)

    def _layers(self, layers: Layers) -> list[int]:
        L = self._m.n_layers
        LS = (
            list(range(L + 1))
            if layers is None
            else sorted({int(l) for l in ([layers] if isinstance(layers, int) else layers)})
        )
        if not LS or LS[0] < 0 or LS[-1] > L:
            raise ValueError(f"layers must lie in 0..{L} (layer {L} is the output)")
        return LS

    def _coeff(self, h, W):
        return self._m.final_norm(h).float() * self._m.output_scale @ W.T

    def _coeff_rows(self, h, Wd):
        return (self._m.final_norm(h).float() * self._m.output_scale * Wd).sum(-1)

    def _gram_inverse(self) -> torch.Tensor:
        if getattr(self, "_gi", None) is None:
            W = self._m.lm_head.weight.double()
            lam, U = torch.linalg.eigh(W.T @ W)
            self._gi = (U / lam.clamp_min(lam.max() * 1e-10) @ U.T).float()
        return self._gi


def _samples(n) -> int:
    if not isinstance(n, int) or n < 1:
        raise ValueError("samples must be a positive integer")
    return n


def token_id(m: Model, token: Token) -> int:
    """The id of a token, given as an id or as a string that the tokenizer encodes to exactly one token."""
    if not isinstance(token, str):
        return int(token)
    if m.tokenizer is None:
        raise ValueError(f"a token given as the string {token!r} needs the tokenizer")
    ids = m.tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        pieces = [m.tokenizer.decode([i]) for i in ids]
        raise ValueError(f"{token!r} is {len(ids)} tokens, {pieces}; give a single token or its id")
    return ids[0]


def ragged(m: Model, x) -> list | None:
    """The texts of a batch whose token lengths differ, or None."""
    if not isinstance(x, (list, tuple)) or len(x) < 2 or not all(isinstance(t, str) for t in x):
        return None
    if m.tokenizer is None:
        return None
    lengths = {len(m.tokenizer.encode(t, add_special_tokens=False)) for t in x}
    return list(x) if len(lengths) > 1 else None


def chosen(positions, T: int) -> list[int]:
    """Positions given as one index or several, possibly negative, as sorted indices into a text of T tokens."""
    P = [positions] if _is_int(positions) else list(positions)
    if not P or any(not -T <= int(p) < T for p in P):
        raise ValueError(f"positions must lie in -{T}..{T - 1} for a text of {T} tokens")
    return sorted({int(p) % T for p in P})


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) or (isinstance(x, torch.Tensor) and x.dim() == 0)


def _folder(cache_dir: str | None) -> str:
    return cache_dir or os.environ.get("LEXFRAME_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "lexframe")


def _batches(m: Model, texts, size: int, even: bool = True) -> list[torch.Tensor]:
    texts = list(texts)
    if texts and isinstance(texts[0], torch.Tensor):
        return texts
    return spread(m, texts, size) if even else list(chunks(m, texts, length=128, limit=size))


def _fingerprint(m: Model, batches: list[torch.Tensor], shapes: bool = False) -> str:
    h = hashlib.sha256(b"lexframe-calibration-2")
    h.update(m.hf.config.to_json_string(use_diff=False).encode())
    with torch.no_grad():
        for name, p in m.hf.named_parameters():
            t = p.detach().double()
            h.update(f"{name}|{tuple(p.shape)}|{p.dtype}|{float(t.sum()):.15e}|{float(t.square().sum()):.15e}".encode())
    for b in batches:
        if shapes:
            h.update(str(tuple(b.shape)).encode())
        h.update(b.cpu().long().numpy().tobytes())
    return h.hexdigest()[:32]
