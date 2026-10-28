"""LexFrame: the hidden states of a text decoded in the model's own vocabulary, with the directions that measure and move them (a State),
and the trail of one prediction across layers and positions (a Trail)."""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from . import _state as S
from . import _trail as TL
from ._state import directions
from ._calibrate import Calibration, calibrate, spread
from ._model import Model, eigh, forward, kl, label_id, logprobs, wrap
from ._transport import innovations, points, read_expected, run

Text = str | torch.Tensor
Label = int | str
CHECK_TEXT = "The quick brown fox jumps over the lazy dog."
CHECK_TOL = 0.05  # nats within which the frame must reproduce the model (a warning otherwise)
SOLVER = {"steps": 50, "damping": 0.1, "tol": 1e-3}  # direction(): LSMR to convergence, relative damping 0.1
_UNSET = object()  # an argument of config() not given
_DEFAULT = {"samples": 64, "tolerance": 0.02, "size": 1024, "cache": "", "progress": True}  # cache "": the default folder


def select(items: list, key) -> tuple[list, bool]:
    """Index a list like a NumPy axis: an int (which drops the axis), a slice, or a list of ints. Returns (items, dropped)."""
    if isinstance(key, (int, np.integer)) or (isinstance(key, torch.Tensor) and key.dim() == 0):
        return [items[int(key)]], True
    if isinstance(key, slice):
        return items[key], False
    return [items[int(i)] for i in key], False


class Decoding:
    """What the selected states decode to, in the model's vocabulary. Returned by State.decode().

    top(k): the k most probable labels and their probabilities, (labels, probs), each [..., k] (labels as text).
    logits(labels): the labels' centred logits, [..., label]; without context, the centred log-probabilities that the state alone gives.
    kl: the divergence from the model's own output, in nats, [...].
    The leading axes are the State's: [layer, position], an axis dropped where it was indexed by an int."""

    def __init__(self, state: State, context: bool):
        self._s, self._context, self._top = state, context, None

    def top(self, k: int = 5):
        if self._top is None or self._top[0] < k:
            s = self._s
            f = s._frame
            X = s._moved_list()
            f._ready(calibrated=not self._context)
            out = S.reading(f, s._xs, s._layers, X, s._positions, k, self._context)
            ids, probs, div = (o[0].cpu().numpy() for o in out)
            tok = f._m.tokenizer
            labels = ids if tok is None else np.vectorize(lambda i: tok.decode([int(i)]), otypes=[object])(ids)
            self._top = (k, labels, probs, div)
        _, labels, probs, _ = self._top
        return self._s._shape(labels[..., :k]), self._s._shape(probs[..., :k])

    def logits(self, labels: Sequence[Label]) -> torch.Tensor:
        s = self._s
        f = s._frame
        vs = torch.tensor([label_id(f._m, t) for t in labels], device=f._m.device)
        if self._context:
            f._ready(calibrated=False)
            Wc = f._m.lm_head.weight[vs].float() - f._head_mean()
            X = s._moved_list()
            with torch.no_grad():
                c = f._centred_logits(run(f._m, s._xs, s._layers, X, joint=X is not None, chunk=f._chunk), Wc)[:, 0][:, s._positions]
        else:
            c = S.expected_logits(f, s._xs, s._layers, s._moved_dict(), vs, s._positions)
        return s._shape(c.detach())

    @property
    def kl(self) -> np.ndarray:
        if self._top is None:
            self.top(1)
        return self._s._shape(self._top[3])

    def __repr__(self) -> str:
        return f"Decoding(context={self._context}, layers={self._s._layers}, positions={len(self._s._positions)})"


class State:
    """The hidden states of a text, at every layer and position, with what they decode to and the directions that measure and move
    them. Returned by LexFrame.state(); index it like an array, s[layer] or s[layer, position] (ints, slices or lists; layer
    num_layers is the output).

    vector: the selected states, [layer, position, hidden], an axis dropped where it was indexed by an int.
    decode(context=True): what they decode to (a Decoding). With context, the states are carried through the remaining layers with the
    true context: what the model outputs from them (for a moved state, after the move). Without, what each state carries by itself.
    gradient(labels, context=True): the gradient of the labels' centred logits with respect to the states, [..., label, hidden]; its
    inner product with a change of a state is the logit change, to first order.
    direction(labels): for each label, a move that raises its logit by one and leaves the others' to first order, [..., label,
    hidden].
    coordinates(vector, labels): a change of the states written label by label, [..., label]: how far to move along each label's
    direction to make the same first-order logit changes.
    steer({label: change}): the states moved so that each label's logit changes by `change` (error and move report how well and how
    far); state + vector and state - vector move them freely. Moved states decode, and give gradients, like any other."""

    def __init__(self, frame, ids, xs, layers, positions, squeeze=(False, False), X=None):
        self._frame, self._ids, self._xs = frame, ids, xs
        self._layers, self._positions, self._squeeze, self._X = layers, positions, squeeze, X

    def __getitem__(self, key) -> State:
        keys = key if isinstance(key, tuple) else (key,)
        axes = [a for a in (0, 1) if not self._squeeze[a]]  # the axes still there: layer, position
        if len(keys) > len(axes):
            raise IndexError(f"too many indices: the State has {len(axes)} axes left")
        sel = [self._layers, self._positions]
        squeeze = list(self._squeeze)
        for a, k in zip(axes, keys):
            sel[a], squeeze[a] = select(sel[a], k)
        if not sel[0] or not sel[1]:
            raise IndexError("the selection is empty")
        if any(b <= a for a, b in zip(sel[0], sel[0][1:])):
            raise IndexError("select layers in increasing order, each once")
        return State(self._frame, self._ids, self._xs, sel[0], sel[1], tuple(squeeze), self._X)

    @property
    def layers(self) -> list[int]:
        return list(self._layers)

    @property
    def positions(self) -> list[int]:
        return list(self._positions)

    @property
    def vector(self) -> torch.Tensor:
        V = torch.stack([self._full(l)[0, self._positions] for l in self._layers]).float()
        return self._shape(V)

    def decode(self, context: bool = True) -> Decoding:
        return Decoding(self, context)

    def gradient(self, labels: Sequence[Label], context: bool = True) -> torch.Tensor:
        f = self._frame
        f._ready(calibrated=not context)
        vs = torch.tensor([label_id(f._m, t) for t in labels], device=f._m.device)
        if context:
            Wc = f._m.lm_head.weight[vs].float() - f._head_mean()
            g = S.coframe(f, self._xs, self._layers, Wc, self._moved_dict())[:, self._positions]
        else:
            g = S.expected_coframe(f, self._xs, self._layers, self._moved_dict(), vs, self._positions)
        return self._shape(g.detach())

    def direction(self, labels: Sequence[Label]) -> torch.Tensor:
        if self._X is not None:
            raise ValueError("directions are taken at the model's own states, not at moved ones")
        f = self._frame
        f._ready(calibrated=False)
        vs = torch.tensor([label_id(f._m, t) for t in labels], device=f._m.device)
        Wc = f._m.lm_head.weight[vs].float() - f._head_mean()
        return self._shape(directions(f, self._xs, self._layers, vs, Wc)[:, self._positions])

    def coordinates(self, vector, labels: Sequence[Label]) -> torch.Tensor:
        if self._X is not None:
            raise ValueError("coordinates are taken at the model's own states, not at moved ones")
        f = self._frame
        f._ready(calibrated=False)
        vs = torch.tensor([label_id(f._m, t) for t in labels], device=f._m.device)
        R = self.gradient(labels)  # [..., label, hidden]
        d = torch.as_tensor(vector.vector if isinstance(vector, State) else vector).to(R)
        Rd = (R @ d[..., None])[..., 0]
        G = S.output_pairing(f, self._xs, self._positions, vs)  # [position, label, label], the same at every layer
        if self._squeeze[1]:
            G = G[0]
        return torch.linalg.solve(G.to(Rd), Rd[..., None])[..., 0]

    def steer(self, changes: Mapping[Label, float]) -> State:
        self._frame._ready(calibrated=False)
        return S.steer(self, changes, self._frame._settings["tolerance"])

    def __add__(self, other):
        return self._moved_by(other, 1)

    def __radd__(self, other):
        return self._moved_by(other, 1)

    def __sub__(self, other):
        return self._moved_by(other, -1)

    def __repr__(self) -> str:
        return f"State(layers={self._layers}, positions={self._positions}, moved={self._X is not None})"

    # ---- internals

    def _full(self, l) -> torch.Tensor:
        return self._xs[l] if self._X is None or l not in self._X else self._X[l]

    def _shape(self, a):
        """[layer, position, ...] with the axes indexed by an int dropped."""
        if self._squeeze[1]:
            a = a[:, 0]
        if self._squeeze[0]:
            a = a[0]
        return a

    def _moved_dict(self):
        return None if self._X is None else {l: self._full(l) for l in self._layers}

    def _moved_list(self):
        return None if self._X is None else [self._full(l) for l in self._layers]

    def _with(self, X) -> State:
        merged = dict(self._X or {})
        merged.update(X)
        return State(self._frame, self._ids, self._xs, self._layers, self._positions, self._squeeze, merged)

    def _moved_by(self, other, sign) -> State:
        d = other.vector if isinstance(other, State) else torch.as_tensor(other)
        d = d.to(self._xs[0].device).float()
        shape = (len(self._layers), len(self._positions), self._xs[0].shape[-1])
        if self._squeeze[1]:
            d = d.unsqueeze(-2) if d.dim() >= 1 else d
        if self._squeeze[0]:
            d = d.unsqueeze(0) if d.dim() >= 2 else d
        d = d.expand(shape)
        X = {}
        for i, l in enumerate(self._layers):
            h = self._full(l).clone()
            h[0, self._positions] = (h[0, self._positions].float() + sign * d[i]).to(h.dtype)
            X[l] = h
        return self._with(X)


class Trail:
    """The trail of one prediction: how the model's output at one position comes about, for one label (by default the prediction
    itself). Returned by LexFrame.trail().

    label, position: the label (as text) and the position.
    logprobs [layer + 1]: the label's log-probability decoded from the state of every layer by itself; the last is the model's output.
    attribute() [layer, 3]: the change of the label's centred logit when what each layer wrote at this position is removed: the whole
    write, its attention part and its MLP part.
    upstream() [layer, position]: what each layer's attention brought in from each earlier position (None where the attention cannot
    be split by source); each row adds up to what the context added at that layer.
    downstream() [layer, later position]: what each later position's own prediction took from this position.
    ablate(layer, part="all", source=None): the State after removing what the layer wrote here (part "attention" or "mlp" for one
    sublayer), or, with a source position, the State after the layer runs again without what its attention read from that position;
    decode it with context to see the model's output after the removal."""

    def __init__(self, frame, ids, position, label_id, label, logprobs):
        self._frame, self._ids = frame, ids
        self.position, self._v, self.label, self.logprobs = position, label_id, label, logprobs
        self._up = None

    def attribute(self) -> np.ndarray:
        f, t = self._frame, self.position
        pre = self._ids[:, : t + 1]
        return np.stack([TL.attribute(f, pre, part, self._v)[:, t] for part in ("all", "attention", "mlp")], 1)

    def upstream(self) -> np.ndarray | None:
        if self._up is None:
            self._up = TL.split(self._frame, self._ids, self.position, self._v)
        return self._up["sources"]

    def downstream(self) -> np.ndarray | None:
        f, t, T = self._frame, self.position, self._ids.shape[1]
        rows = []
        for tp in range(t + 1, T):
            sources = TL.split(f, self._ids, tp, None)["sources"]
            if sources is None:
                return None
            rows.append(sources[:, t])
        return np.stack(rows, 1) if rows else np.zeros((f.num_layers, 0))

    def ablate(self, layer: int, part: str = "all", source: int | None = None) -> State:
        f = self._frame
        xs, h = TL.ablate(f._m, self._ids, self.position, layer, part, source)
        return State(f, self._ids, xs, [layer + 1], list(range(self._ids.shape[1])), (True, False), {layer + 1: h})

    def __repr__(self) -> str:
        return f"Trail(label={self.label!r}, position={self.position}, layers={len(self.logprobs) - 1})"


class LexFrame:
    """A frame of a Hugging Face causal language model: every hidden state read in the model's own vocabulary, by carrying it to the
    output. With context, a state runs through the model's remaining layers with the true context. Without, it runs through them on
    its own column, every attention write replaced by its expectation given the state plus a Gaussian innovation, averaged by
    quasi-Monte Carlo.

    config(...) sets the precision and the cost; init(texts) calibrates once; state(text) gives the states of a text, and trail(text)
    the trail of one prediction."""

    def __init__(self, model, tokenizer=None):
        self._m = model if isinstance(model, Model) else wrap(model, tokenizer)
        self.calibration: Calibration | None = None
        self._identities: dict | None = None
        self._chunk = 8 if self._m.hidden_dim < 4000 else 6
        self._rows = 1 << 15  # rows of one pass of the expected transport
        self._solver = dict(SOLVER)
        self._settings = dict(_DEFAULT)
        self._xi = None

    @property
    def num_layers(self) -> int:
        """The number of layers; layer num_layers is the output."""
        return self._m.n_layers

    def __repr__(self) -> str:
        return f"LexFrame(model={type(self._m.hf).__name__}, layers={self._m.n_layers}, settings={self._settings})"

    def config(self, samples: int | None = None, tolerance: float | None = None, size: int | None = None, cache=_UNSET,
               progress: bool | None = None) -> LexFrame:  # fmt: skip
        """The frame with new settings. samples: integration points of the decoding without context (64; more is more accurate, at a
        proportional cost). tolerance: the relative error at which steering stops (0.02). size: calibration sequences of 128 tokens
        (1024). cache: the folder of calibrations (default ~/.cache/lexframe or $LEXFRAME_CACHE), or None for no cache. progress:
        show the calibration's progress (True)."""
        if samples is not None:
            if not isinstance(samples, int) or samples < 1:
                raise ValueError("samples must be a positive integer")
            self._settings["samples"], self._xi = samples, None
        if tolerance is not None:
            if not 0 < tolerance < 1:
                raise ValueError("tolerance must lie in (0, 1)")
            self._settings["tolerance"] = float(tolerance)
        if size is not None:
            if not isinstance(size, int) or size < 2:
                raise ValueError("size must be an integer of at least 2")
            self._settings["size"] = size
        if cache is not _UNSET:
            self._settings["cache"] = cache
        if progress is not None:
            self._settings["progress"] = bool(progress)
        return self

    def init(self, texts: Iterable) -> LexFrame:
        """Calibrate on unlabelled `texts`: `size` sequences of 128 tokens, one from each of `size` evenly spaced texts (or batches of
        token ids as tensors, used as given); then check that the frame reproduces the model. The calibration is cached under a
        fingerprint of the model and the calibration tokens."""
        st = self._settings
        texts = list(texts)
        batches = texts if texts and isinstance(texts[0], torch.Tensor) else spread(self._m, texts, st["size"])
        folder = st["cache"]
        if folder == "":
            folder = os.environ.get("LEXFRAME_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "lexframe")
        path = None if folder is None else os.path.join(folder, fingerprint(self._m, batches) + ".pt")
        cal = None
        if path and os.path.exists(path):
            try:
                cal = Calibration.load(path, self._m)
            except ValueError:  # a calibration made by another version is made again
                cal = None
        if cal is None:
            cal = calibrate(self._m, batches, progress=st["progress"])
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cal.save(path)
        self.calibration, self._xi = cal.to(self._m.device), None
        self._stamp = _stamp(self._m)
        self._identities = self._check(CHECK_TOL)
        if not self._identities["ok"]:
            warnings.warn(f"the frame does not reproduce the model within {CHECK_TOL} nats: {self._identities}")
        return self

    @torch.no_grad()
    def state(self, text: Text) -> State:
        """The hidden states of one text at every layer and position; index the State to select layers and positions."""
        ids = self._one(text)
        hL, xs, _ = forward(self._m, ids)
        L = self._m.n_layers
        return State(self, ids, xs + [hL], list(range(L + 1)), list(range(ids.shape[1])))

    @torch.no_grad()
    def trail(self, text: Text, position: int = -1, label: Label | None = None) -> Trail:
        """The trail of the prediction at one position of one text (the last by default), for `label` (default: the model's
        prediction there)."""
        self._ready()
        ids = self._one(text)
        T = ids.shape[1]
        if not -T <= position < T:
            raise ValueError(f"position must lie in -{T}..{T - 1} for a text of {T} tokens")
        t = position % T
        if t == 0:
            raise ValueError("position 0 attends only to itself: its decoding is exact at every layer and it has no trail")
        m = self._m
        L = m.n_layers
        hL, xs, _ = forward(m, ids[:, : t + 1])
        out = logprobs(m, hL[0, t])
        v = int(out.argmax()) if label is None else label_id(self._m, label)
        tt = torch.tensor([t], device=m.device)
        R = read_expected(m, self.calibration, {k: xs[k][0, t][None] for k in range(L)}, tt, self._points(), torch.tensor([v], device=m.device),
                 center=False)  # fmt: skip
        lp = np.array([float(R[k][0, 0]) for k in range(L)] + [float(out[v])])
        name = m.tokenizer.decode([v]) if m.tokenizer is not None else v
        return Trail(self, ids, t, v, name, lp)

    # ---- checks and caches

    @torch.no_grad()
    def _check(self, tol: float) -> dict:
        m = self._m
        ids = m.ids(CHECK_TEXT if m.tokenizer is not None else torch.arange(8) % m.lm_head.weight.shape[0])
        hL, xs, _ = forward(m, ids)
        ref = logprobs(m, hL)
        L = m.n_layers
        native = max(kl(ref, logprobs(m, h)).max() for h in run(m, xs, list(range(L)), chunk=self._chunk))
        first = read_expected(m, self.calibration, {l: xs[l][0, :1] for l in range(L)}, torch.zeros(1, dtype=torch.long, device=m.device),
                     self._points())  # fmt: skip
        first = max(kl(ref[0, :1], r).max() for r in first.values())
        result = {"visible_all_layers": float(native), "invisible_first_position": float(first)}
        result["ok"] = max(result.values()) < tol
        return result

    def _ready(self, calibrated: bool = True) -> None:
        """Refresh the caches if the weights changed; with calibrated=True, also require init()."""
        stamp = _stamp(self._m)
        if getattr(self, "_stamp", None) is not None and stamp[2] != self._stamp[2]:
            # the model moved to another device: its tensors are new, its weights are not
            self._m.pe.clear()
            self.__dict__.pop("_gi", None)
            self.__dict__.pop("_wbar", None)
            self._stamp = None
        if self.calibration is None:
            if calibrated:
                raise RuntimeError("call init(texts) first: decoding without context and trails need the calibration")
            if getattr(self, "_stamp", None) is not None and stamp[1] != self._stamp[1]:
                self.__dict__.pop("_gi", None)
                self.__dict__.pop("_wbar", None)
            self._stamp = stamp
            return
        old = getattr(self, "_stamp", None)
        if old is None:
            self._stamp = stamp
        elif stamp != old:
            # the cached factors were measured on other weights: the output-side ones (head mean, head Gram) are recomputed, the
            # expected transport's calibration needs init() again, and the native transport is recomputed on every call
            if stamp[1] != old[1]:
                self.__dict__.pop("_gi", None)
                self.__dict__.pop("_wbar", None)
            if stamp[0] != old[0]:
                warnings.warn("the model's weights changed since init(): decoding without context uses the old calibration; call init() again")
            self._stamp = (old[0], stamp[1], stamp[2])
        if self.calibration.ubar.device != self._m.device:
            self.calibration, self._xi = self.calibration.to(self._m.device), None
            self.__dict__.pop("_gi", None)
            self.__dict__.pop("_wbar", None)

    def _one(self, text: Text) -> torch.Tensor:
        ids = self._m.ids(text)
        if ids.shape[0] != 1:
            raise ValueError("one text at a time")
        return ids

    def _centred_logits(self, h, W):
        return self._m.final_norm(h).float() * self._m.output_scale @ W.T

    def _points(self) -> torch.Tensor:
        """The innovations at the integration points, [samples, layer, hidden], computed once per calibration."""
        n = self._settings["samples"]
        if self._xi is None or self._xi.shape[0] != n:
            fac = self.calibration.fac
            self._xi = innovations(fac, points(fac, n), self._m.device)
        return self._xi

    def _gram_inverse(self) -> torch.Tensor:
        if getattr(self, "_gi", None) is None:
            W, n = self._m.lm_head.weight, 1 << 14  # in row blocks: no float32 copy of the head
            G = sum(W[i : i + n].float().T @ W[i : i + n].float() for i in range(0, W.shape[0], n))
            lam, U = eigh(G)
            self._gi = U / lam.clamp_min(lam.max() * 1e-6) @ U.T  # pseudo-inverse at float32 resolution
        return self._gi

    def _head_mean(self) -> torch.Tensor:
        """The mean output-embedding row, in row blocks (no float32 copy of the head)."""
        if getattr(self, "_wbar", None) is None:
            W, n = self._m.lm_head.weight, 1 << 14
            self._wbar = sum(W[i : i + n].float().sum(0) for i in range(0, W.shape[0], n)) / W.shape[0]
        return self._wbar


def _stamp(m: Model) -> tuple:
    """A cheap version stamp of the weights: (all parameters, the output head and final norm, the device). In-place updates
    (optimiser steps, edits under no_grad) advance a tensor's version counter; reassigning a tensor changes its storage."""

    def key(ps):
        return tuple((p.data_ptr(), p._version) for p in ps)

    hf = m.hf if hasattr(m, "hf") else None
    params = list(hf.parameters()) if isinstance(hf, torch.nn.Module) else []
    out = [m.lm_head.weight] + [p for p in getattr(m.final_norm, "parameters", lambda: [])()]
    return key(params), key(out), m.device


FINGERPRINT_SAMPLES = 4096  # evenly spaced values of each parameter that the fingerprint hashes


def fingerprint(m: Model, batches: list[torch.Tensor]) -> str:
    """The cache key of a calibration: the model's configuration, each parameter's name, shape, dtype and an evenly spaced sample of
    its values (exact bytes), and the calibration tokens."""
    from . import __version__

    h = hashlib.sha256(f"lexframe-calibration-{__version__}".encode())
    h.update(m.hf.config.to_json_string(use_diff=False).encode())
    with torch.no_grad():
        for name, p in m.hf.named_parameters():
            flat = p.detach().reshape(-1)
            sample = flat[:: max(1, flat.numel() // FINGERPRINT_SAMPLES)][:FINGERPRINT_SAMPLES]
            h.update(f"{name}|{tuple(p.shape)}|{p.dtype}|".encode())
            h.update(sample.float().cpu().numpy().tobytes())
    for b in batches:
        h.update(b.cpu().long().numpy().tobytes())
    return h.hexdigest()[:32]
