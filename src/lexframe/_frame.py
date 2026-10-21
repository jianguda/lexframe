"""LexFrame: read, attribute and steer the hidden states of every layer in the model's own vocabulary."""

from __future__ import annotations

import hashlib
import math
import os
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from . import _transport as TR
from ._basis import basis as _solve_basis
from ._basis import jvp as _jvp
from ._calibrate import Calibration, calibrate, chunks
from ._model import Model, forward, kl, logprobs, wrap

Text = str | Sequence[str] | torch.Tensor
Token = int | str
Layers = int | Iterable[int] | None
NATIVE = math.inf
CHECK_TEXT = "The quick brown fox jumps over the lazy dog."


def _single(text) -> bool:
    return isinstance(text, str) or (isinstance(text, torch.Tensor) and text.dim() == 1)


def _order(order) -> float:
    if order in ("inf", math.inf):
        return NATIVE
    if isinstance(order, int) and order >= 0:
        return order
    raise ValueError(f"order must be a non-negative integer or 'inf', not {order!r}")


class Carrier:
    """The hidden states of one or more texts at one or more layers (layer num_layers is the output), as a frame
    carries and reads them.

    vector: the states, [..., position, hidden]; tokens, ids, probs [..., position, k] and kl [..., position]: the
    frame's reading, as NumPy arrays. first_layer() finds where a token first tops the reading. basis(), readout() and
    coef() give the lexical basis, the lexical readout and the coefficients of given tokens (ids or single-token
    strings such as " Paris") at the states of one text. Adding or subtracting a vector moves the states, and steer()
    moves them along the basis; a moved Carrier is read again with frame.view(), which in the native frame gives the
    model's own output. A Carrier of chosen positions only (positions) holds its reading and nothing more.
    """

    def __init__(self, frame, vector, layers, ids, context, flags, moved=False, reading=None, positions=None):
        self.vector, self.layers, self.moved = vector, layers, moved
        self._frame, self._ids, self._context, self._flags = frame, ids, context, flags
        self.ids, self.probs, self.tokens, self.kl = reading or (None, None, None, None)
        self.positions = positions

    def _whole(self, what: str) -> None:
        if self.positions is not None:
            raise ValueError(f"{what} needs every position: view the text without positions=")

    def basis(self, tokens: Sequence[Token]) -> torch.Tensor:
        """The lexical basis of the tokens, [..., position, token, hidden]: the move of the state that the remaining
        layers turn into raising that token alone, scaled to raise its logit by one. Unmoved states only."""
        self._whole("the basis")
        return self._frame._directions(self, tokens, "basis")

    def readout(self, tokens: Sequence[Token]) -> torch.Tensor:
        """The lexical readout of the tokens, [..., position, token, hidden]: the gradient of each token's logit, so
        its inner product with a displacement of the state is the logit change it predicts."""
        self._whole("the readout")
        return self._frame._directions(self, tokens, "readout")

    def coef(self, tokens: Sequence[Token]) -> torch.Tensor:
        """The coefficients of the state on the tokens, [..., position, token]: their centred logits after the
        transport."""
        self._whole("the coefficient")
        return self._frame._directions(self, tokens, "coef")

    def steer(self, weights: Mapping[Token, float]) -> Carrier:
        """The states moved along the lexical basis of each token by its weight, in logits, at every position, e.g.
        {" Paris": -4, " London": 4}. Unmoved states of one text only."""
        tokens = list(weights)
        b = self.basis(tokens)
        w = torch.tensor([float(weights[t]) for t in tokens], dtype=b.dtype, device=b.device)
        return self + (b * w[:, None]).sum(-2)

    def first_layer(self, token: Token) -> np.ndarray:
        """The first of the read layers at which the token tops the frame's reading, [..., position]; -1 where it
        never does."""
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
        return f"Carrier(layers={self.layers}, shape={tuple(self.vector.shape)}, moved={self.moved}{extra})"


class LexFrame:
    """A lexical reference frame of a Hugging Face causal language model. It reads a hidden state at layer l through
    the model's own layers l..L-1 with `order` rounds of attention between positions (default 1; "inf" keeps all of
    them) and a calibrated typical attention output for the rest. Call init() once before use."""

    def __init__(self, model, tokenizer=None, order: int | str = 1):
        self._m = model if isinstance(model, Model) else wrap(model, tokenizer)
        self.order = _order(order)
        self.calibration: Calibration | None = None
        self.identities: dict | None = None
        self._chunk = 8 if self._m.hidden_dim < 4000 else 6
        self.solver = {"steps": 30, "ridge": 1e-3, "warm": False}

    @property
    def num_layers(self) -> int:
        """The number of layers; layer num_layers is the output."""
        return self._m.n_layers

    def __repr__(self) -> str:
        order = "inf" if self.order == NATIVE else self.order
        return f"LexFrame(model={type(self._m.hf).__name__}, layers={self._m.n_layers}, order={order})"

    def config(
        self,
        steps: int | None = None,
        ridge: float | None = None,
        warm: bool | None = None,
    ) -> LexFrame:
        """Settings of the iterative LSMR solver behind basis(), kept in `frame.solver`: `steps` per layer (30), the
        relative `ridge` that damps directions the remaining layers barely pass on (1e-3), and `warm` to start each
        layer from the solution of the layer above (False)."""
        new = {k: v for k, v in dict(steps=steps, ridge=ridge, warm=warm).items() if v is not None}
        if "steps" in new and (not isinstance(new["steps"], int) or new["steps"] < 1):
            raise ValueError("steps must be a positive integer")
        if "ridge" in new and new["ridge"] < 0:
            raise ValueError("ridge must not be negative")
        self.solver.update(new)
        return self

    def init(
        self,
        texts: Iterable,
        size: int = 256,
        cache: bool = True,
        cache_dir: str | None = None,
        tol: float = 0.05,
        progress: bool = True,
    ) -> LexFrame:
        """Calibrate on unlabelled `texts` (`size` sequences of 128 tokens are used) and check the exact identities.
        With cache=True the calibration is stored under a fingerprint of the model's weights and configuration, the
        calibration tokens and the settings, and loaded again when all of them match."""
        batches = _batches(self._m, texts, size)
        path = os.path.join(_folder(cache_dir), _fingerprint(self._m, batches) + ".pt") if cache else None
        if path and os.path.exists(path):
            cal = Calibration.load(path, self._m)
        else:
            cal = calibrate(self._m, batches, progress=progress)
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cal.save(path)
        self.calibration = cal.to(self._m.device)
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
    ) -> Carrier | list[Carrier]:
        """The hidden states of one or more texts at every layer (or the given ones) with what the frame reads in
        them, or the reading of a (moved) Carrier. In the native frame, a moved Carrier reads as the model's output.
        positions reads only the given positions of a text, which is cut after the last of them. Texts of different
        token lengths are read one at a time, and a list of Carriers is returned."""
        if positions is not None and not _is_int(positions):
            positions = list(positions)
        ragged = self._ragged(x)
        if ragged is not None:
            return [self.view(t, k, layers, positions) for t in ragged]
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
                P = self._positions(positions, ids.shape[1])
                ids = ids[:, : P[-1] + 1]
            hL, xs, _ = forward(self._m, ids)
            xs = xs + [hL]
            LS = self._layers(layers)
            flags = (_single(x), isinstance(layers, int))
            V = torch.stack([xs[l] for l in LS], 1).float()
            X = None
        H = self._run(xs, LS, X)
        ref = logprobs(self._m, xs[-1])
        top = [logprobs(self._m, h).topk(k, -1) for h in H]
        out = [
            torch.stack([t.indices for t in top], 1),
            torch.stack([t.values.exp() for t in top], 1),
            torch.stack([kl(ref, logprobs(self._m, h)) for h in H], 1),
        ]
        out = [o.cpu().numpy() for o in out]
        if P is not None:
            out = [out[0][:, :, P], out[1][:, :, P], out[2][:, :, P]]
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
        layers_ = LS[0] if flags[1] else LS
        reading = (out[0], out[1], tokens, out[2])
        return Carrier(self, vector, layers_, ids, xs, flags, moved, reading, P)

    @torch.no_grad()
    def attr(
        self, text: Text, sublayer: str | None = None, target: Token | None = None
    ) -> np.ndarray | list[np.ndarray]:
        """How much each layer (or its `sublayer`, "attn" or "mlp") contributed to the logit of the token
        `target` (default: the model's own prediction at every position), [layer, position]."""
        if sublayer not in (None, "attn", "mlp"):
            raise ValueError("sublayer is None (the whole layer), 'attn' or 'mlp'")
        ragged = self._ragged(text)
        if ragged is not None:
            return [self.attr(t, sublayer, target) for t in ragged]
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

    def bound(self, texts: Iterable[Text], token: Token, layers: Layers = None) -> np.ndarray:
        """The best mean cosine that any lens, with one direction per token and layer, can reach with the model's own
        readout of `token`, per layer: sqrt of the readout's agreement across at least two texts. It is a property of
        the model (the native frame), whatever the order of this frame."""
        native = self._native()
        sums, counts = [], []
        for text in texts:
            LS = list(range(self.num_layers)) if layers is None else layers
            b = native._delta(native._one(text), [token], LS, "readout")[0][..., 0, :]
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

    def _directions(self, x: Carrier, tokens, kind) -> torch.Tensor:
        self._ready()
        if not x._flags[0]:
            raise ValueError("directions are taken at the states of one text")
        if kind == "basis" and self.order == 0:
            raise ValueError("the basis needs order >= 1 or 'inf'")
        if kind == "basis" and x.moved:
            raise ValueError("the basis is taken at the model's own states, not at moved ones")
        X = None
        if x.moved:
            V = x._canonical()
            X = {l: V[:, i].to(x._context[l].dtype) for i, l in enumerate(x._layer_list())}
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
            order = None if self.order == NATIVE else int(self.order)
            scale = self._m.output_scale

            def save(l, X0, kv, x, rows):
                gain = scale * (_jvp(self._m, l, kv, X0, x) * Wc[rows][:, None]).sum(-1)
                S[(l, rows.start)] = x / gain[..., None]

            cfg = self.solver
            _solve_basis(self._m, self.calibration, xs, LS, Y, order, cfg["ridge"], cfg["steps"], save, cfg["warm"])
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

    @torch.no_grad()
    def _check(self, tol: float) -> dict:
        ids = self._m.ids(
            CHECK_TEXT if self._m.tokenizer is not None else torch.arange(8) % self._m.lm_head.weight.shape[0]
        )
        hL, xs, _ = forward(self._m, ids)
        ref = logprobs(self._m, hL)
        L = self._m.n_layers
        result = {
            "orderinf_all_layers": float(
                max(kl(ref, logprobs(self._m, h)).max() for h in self._native()._run(xs, list(range(L))))
            )
        }
        for o in (1, 2):
            f = LexFrame(self._m, order=o)
            f.calibration = self.calibration
            result[f"order{o}_last_layer"] = float(kl(ref, logprobs(self._m, f._run(xs, [L - 1])[0])).max())
        result["ok"] = max(result.values()) < tol
        return result

    def _native(self) -> LexFrame:
        f = LexFrame(self._m, order="inf")
        f.calibration = self.calibration
        return f

    def _ready(self) -> None:
        if self.calibration is None:
            raise RuntimeError("call init(texts) first")
        if self.calibration.ubar.device != self._m.device:
            self.calibration = self.calibration.to(self._m.device)
            self.__dict__.pop("_gi", None)

    def _one(self, text: Text) -> torch.Tensor:
        ids = self._m.ids(text)
        if ids.shape[0] != 1:
            raise ValueError("this method takes one text at a time")
        return ids

    def _ragged(self, x) -> list | None:
        return ragged(self._m, x)

    @staticmethod
    def _positions(positions, T: int) -> list[int]:
        return chosen(positions, T)

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

    def _spec(self, X=None) -> dict:
        if self.order == NATIVE:
            return {"model": {"over": "true", **({} if X is None else {"x": X})}}
        if self.order == 0:
            return {}
        st = TR.chain(self.order)
        if X is not None:
            st[TR.name(self.order)] = {**st[TR.name(self.order)], "x": X}
        return st

    def _run(self, xs, LS, X=None) -> torch.Tensor:
        L = self._m.n_layers
        key = "none" if self.order == 0 else "model" if self.order == NATIVE else TR.name(self.order)
        inner = [l for l in LS if l < L]
        out = []
        for c in range(0, len(inner), self._chunk):
            Xc = None if X is None else X[c : c + self._chunk]
            if self.order == 0 and Xc is not None:
                xs = list(xs)
                for i, l in enumerate(inner[c : c + self._chunk]):
                    xs[l] = Xc[i]
                Xc = None
            out.append(TR._streams(self._m, self.calibration, xs, inner[c : c + self._chunk], self._spec(Xc))[key])
        if L in LS:
            out.append((xs[L] if X is None else X[len(inner)])[None])
        return torch.cat(out)

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


def _batches(m: Model, texts, size: int) -> list[torch.Tensor]:
    texts = list(texts)
    if texts and isinstance(texts[0], torch.Tensor):
        return texts
    return list(chunks(m, texts, length=128, limit=size))


def _fingerprint(m: Model, batches: list[torch.Tensor], shapes: bool = False) -> str:
    h = hashlib.sha256(b"lexframe-calibration-1")
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
