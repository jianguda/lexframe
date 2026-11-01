"""LexFrame: the hidden states of a text read in the model's own vocabulary, with their semantic bases, and the moves of the model's
run (a State)."""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from . import _basis
from . import _state as S
from . import _ablate as AB
from ._calibrate import Calibration, calibrate, spread
from ._model import Model, forward, kl, label_id, logprobs, wrap
from ._transport import innovations, points, read_expected, run

Text = str | torch.Tensor
Label = int | str
_UNSET = object()  # an argument of config() not given
CHECK_TEXT = "The quick brown fox jumps over the lazy dog."
CHECK_TOL = 0.05  # nats within which the frame must reproduce the model (a warning otherwise)
_DEFAULT = {"samples": 64, "size": 1024, "length": 256, "cache": "", "progress": True}  # cache "": default


def select(items: list, key) -> tuple[list, bool]:
    """Index a list like a NumPy axis: an int (which drops the axis), a slice, or a list of ints. Returns (items, dropped)."""
    if isinstance(key, (int, np.integer)) or (isinstance(key, torch.Tensor) and key.dim() == 0):
        return [items[int(key)]], True
    if isinstance(key, slice):
        return items[key], False
    return [items[int(i)] for i in key], False


class State:
    """The hidden states of a text, at every layer and position: what they carry, their semantic bases, and the moves that change
    them. Returned by LexFrame.state(); index it like an array, s[layer] or s[layer, position] (ints, slices or lists; layer
    num_layers is the output).

    vector: the selected states, [layer, position, hidden], an axis dropped where it was indexed by an int.

    logits(labels=None): what the states carry, centred logits [..., label] (default: the whole vocabulary), calibrated by
    init(); at the output layer, the model's own.
    gradient(labels): the model's own gradients of the labels' centred output logits with respect to the states, [..., label,
    hidden], the context fixed (no calibration).
    basis(labels): the labels' semantic bases at the states, [..., label, hidden] (labels None: the whole vocabulary, vocabulary x
    hidden floats per state), the polar basis of the coefficients c(h) = log R(x) + z(h) - z(x): the state is the sum of the bases
    of the whole vocabulary weighted by its logits, and a move along a label's basis changes the model's output logit of that
    label by the step, to first order. Each label's basis is its own, whatever other labels are asked for.

    Moves (each returns the State of the moved run, at every layer and position, whose output layer is the model's output after
    the move):
    steer({label: change}): the selected states of one layer rotated at their norms so that each label's centred output logit
    changes by `change`, the output's distribution over the other labels changing least (a warning where a change is beyond reach).
    ablate(part="all", source=None): the run without what the previous layer wrote at the selected positions (part "attention" or
    "mlp" for one sublayer), or, with a source position, with that layer run again without what its attention read from there.
    state + vector, state - vector: the selected states moved freely.
    A moved run is the model with its moves added (each kept as the change it made to its layer's states): every member works on
    it at every layer, and a further move, at any layer, keeps the earlier ones."""

    def __init__(self, frame, ids, xs, layers, positions, squeeze=(False, False), moves=None):
        self._frame, self._ids, self._xs = frame, ids, xs
        self._layers, self._positions, self._squeeze = layers, positions, squeeze
        self._moves = moves or {}  # {layer: [1, T, hidden]}: the changes the moves made, added again whenever the run is carried

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
        return State(self._frame, self._ids, self._xs, sel[0], sel[1], tuple(squeeze), self._moves)

    @property
    def layers(self) -> list[int]:
        return list(self._layers)

    @property
    def positions(self) -> list[int]:
        return list(self._positions)

    @property
    def vector(self) -> torch.Tensor:
        V = torch.stack([self._xs[l][0, self._positions] for l in self._layers]).float()
        return self._shape(V)

    def logits(self, labels: Sequence[Label] | None = None) -> torch.Tensor:
        f = self._frame
        f._ready(calibrated=min(self._layers) < f.num_layers)
        vs = None if labels is None else torch.tensor([label_id(f._m, t) for t in labels], device=f._m.device)
        with torch.no_grad():
            z, _ = S.expected(f, self._xs, self._layers, self._positions, vs)
        return self._shape(z)

    def gradient(self, labels: Sequence[Label]) -> torch.Tensor:
        f = self._frame
        f._ready(calibrated=False)
        vs = torch.tensor([label_id(f._m, t) for t in labels], device=f._m.device)
        return self._shape(S.gradients(f, self._xs, self._layers, self._positions, vs, self._moves))

    def basis(self, labels: Sequence[Label] | None) -> torch.Tensor:
        f = self._frame
        f._ready(calibrated=min(self._layers) < f.num_layers)
        vs = None if labels is None else torch.tensor([label_id(f._m, t) for t in labels], device=f._m.device)
        return self._shape(_basis.bases(f, self._xs, self._layers, self._positions, vs, self._moves))

    def steer(self, changes: Mapping[Label, float]) -> State:
        self._frame._ready(calibrated=False)
        return S.steer(self, changes, S.TOLERANCE)

    def ablate(self, part: str = "all", source: int | None = None) -> State:
        if len(self._layers) != 1:
            raise ValueError("ablate one layer at a time")
        k = self._layers[0]
        if k < 1:
            raise ValueError("layer 0 is the embedding: nothing was written before it")
        f = self._frame
        h = AB.ablate(f._m, self._xs, self._positions, k - 1, part, source)
        return self._moved({k: h.float() - self._xs[k].float()})

    def __add__(self, other):
        return self._moved_by(other, 1)

    def __radd__(self, other):
        return self._moved_by(other, 1)

    def __sub__(self, other):
        return self._moved_by(other, -1)

    def __repr__(self) -> str:
        return f"State(layers={self._layers}, positions={self._positions}, moved={bool(self._moves)})"

    # ---- internals

    def _shape(self, a):
        """[layer, position, ...] with the axes indexed by an int dropped."""
        if self._squeeze[1]:
            a = a[:, 0]
        if self._squeeze[0]:
            a = a[0]
        return a

    def _run(self, xs, moves) -> State:
        """The State of a moved run, at every layer and position."""
        L = self._frame.num_layers
        return State(self._frame, self._ids, xs, list(range(L + 1)), list(range(self._ids.shape[1])), (False, False), moves)

    def _moved(self, deltas) -> State:
        """The run after the new moves deltas {layer: [1, T, hidden]}, its earlier moves kept."""
        xs = S.propagate(self._frame._m, self._xs, deltas, self._moves)
        return self._run(xs, S.combine(self._moves, deltas))

    def _moved_by(self, other, sign) -> State:
        d = other.vector if isinstance(other, State) else torch.as_tensor(other)
        d = d.to(self._xs[0].device).float()
        shape = (len(self._layers), len(self._positions), self._xs[0].shape[-1])
        if self._squeeze[1]:
            d = d.unsqueeze(-2) if d.dim() >= 1 else d
        if self._squeeze[0]:
            d = d.unsqueeze(0) if d.dim() >= 2 else d
        d = d.expand(shape)
        deltas = {}
        for i, l in enumerate(self._layers):
            D = torch.zeros(1, *self._xs[l].shape[1:], device=d.device)
            D[0, self._positions] = sign * d[i]
            deltas[l] = D
        return self._moved(deltas)


class LexFrame:
    """A frame of a Hugging Face causal language model. State.logits carries a hidden state through the remaining layers on its
    own column, every attention write replaced by its expectation given the state plus a Gaussian innovation, averaged by
    quasi-Monte Carlo (calibrated once by init()); State.gradient and the moves (steer, ablate, + and -) run the model's own
    layers with the full context; State.basis is the geometric carrier of both.

    config(...) sets the precision and the cost; init(texts) calibrates once; state(text) gives the states of a text."""

    def __init__(self, model, tokenizer=None):
        self._m = model if isinstance(model, Model) else wrap(model, tokenizer)
        self.calibration: Calibration | None = None
        self._identities: dict | None = None
        self._chunk = 8 if self._m.hidden_dim < 4000 else 6
        self._rows = 1 << 15  # rows of one pass of the expected transport
        self._settings = dict(_DEFAULT)
        self._xi = None

    @property
    def num_layers(self) -> int:
        """The number of layers; layer num_layers is the output."""
        return self._m.n_layers

    def __repr__(self) -> str:
        return f"LexFrame(model={type(self._m.hf).__name__}, layers={self._m.n_layers}, settings={self._settings})"

    def config(self, samples: int | None = None, size: int | None = None, length: int | None = None, cache=_UNSET,
               progress: bool | None = None) -> LexFrame:  # fmt: skip
        """The frame with new settings. samples: integration points of logits (64; more is more accurate, at a proportional
        cost). size and length: the calibration's sequences and their
        tokens (1024 of 256: positions beyond the length read with the statistics of the last positions seen). cache: the folder of
        calibrations (default ~/.cache/lexframe or $LEXFRAME_CACHE), or None for no cache. progress: show the calibration's
        progress (True)."""
        if samples is not None:
            if not isinstance(samples, int) or samples < 1:
                raise ValueError("samples must be a positive integer")
            self._settings["samples"], self._xi = samples, None
        if size is not None:
            if not isinstance(size, int) or size < 2:
                raise ValueError("size must be an integer of at least 2")
            self._settings["size"] = size
        if length is not None:
            if not isinstance(length, int) or length < 2:
                raise ValueError("length must be an integer of at least 2")
            self._settings["length"] = length
        if cache is not _UNSET:
            self._settings["cache"] = cache
        if progress is not None:
            self._settings["progress"] = bool(progress)
        return self

    def _cache_folder(self):
        folder = self._settings["cache"]
        if folder == "":
            folder = os.environ.get("LEXFRAME_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "lexframe")
        return folder

    def init(self, texts: Iterable) -> LexFrame:
        """Calibrate on unlabelled `texts`: `size` sequences of `length` tokens, one from each of `size` evenly spaced texts (or batches of
        token ids as tensors, used as given); then check that the frame reproduces the model. The calibration is cached under a
        fingerprint of the model and the calibration tokens."""
        st = self._settings
        texts = _as_list(texts)
        batches = texts if texts and isinstance(texts[0], torch.Tensor) else spread(self._m, texts, st["size"], st["length"])
        folder = self._cache_folder()
        path = None if folder is None else os.path.join(folder, fingerprint(self._m, batches) + ".pt")
        cal = None
        if path and os.path.exists(path):
            try:
                cal = Calibration.load(path, self._m)
            except (ValueError, KeyError):  # a calibration made by another version is made again
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
        """Refresh the caches if the weights or the device changed; with calibrated=True, also require init()."""
        stamp = _stamp(self._m)
        old = getattr(self, "_stamp", None)
        if old is not None:
            if stamp[2] != old[2]:  # the model moved to another device: its tensors are new, its weights are not
                self._m.pe.clear()
            if stamp[1] != old[1] or stamp[2] != old[2]:  # the head or the final norm changed: the head mean is recomputed
                self.__dict__.pop("_wbar", None)
                self._m.out.clear()
            if self.calibration is not None and stamp[0] != old[0]:
                warnings.warn("the model's weights changed since init(): logits use the old calibration; call init() again")
                stamp = (old[0], stamp[1], stamp[2])  # warn again until init() runs on the new weights
        self._stamp = stamp
        if self.calibration is None:
            if calibrated:
                raise RuntimeError("call init(texts) first: below the output, what a state carries (its logits, gradient and "
                                   "bases) needs the calibration")  # fmt: skip
            return
        if self.calibration.ubar.device != self._m.device:
            self.calibration, self._xi = self.calibration.to(self._m.device), None

    def _one(self, text: Text) -> torch.Tensor:
        ids = self._m.ids(text)
        if ids.shape[0] != 1:
            raise ValueError("one text at a time")
        return ids

    def _points(self) -> torch.Tensor:
        """The innovations at the integration points, [samples, layer, hidden], computed once per calibration."""
        n = self._settings["samples"]
        if self._xi is None or self._xi.shape[0] != n:
            fac = self.calibration.fac
            self._xi = innovations(fac, points(fac, n), self._m.device)
        return self._xi

    def _head_mean(self) -> torch.Tensor:
        """The mean output-embedding row, in row blocks (no float32 copy of the head)."""
        if getattr(self, "_wbar", None) is None:
            W, n = self._m.lm_head.weight, 1 << 14
            self._wbar = sum(W[i : i + n].float().sum(0) for i in range(0, W.shape[0], n)) / W.shape[0]
        return self._wbar


def _as_list(texts) -> list:
    """The texts as a list; a sequence that reads a slice at once (a datasets column) in one read rather than item by item."""
    if not isinstance(texts, (list, tuple)) and hasattr(texts, "__len__") and hasattr(texts, "__getitem__"):
        try:
            out = texts[:]
        except (TypeError, KeyError, IndexError, ValueError):
            out = None
        if isinstance(out, (list, tuple)):
            return list(out)
    return list(texts)


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


def fingerprint(m: Model, batches: list[torch.Tensor], params=None) -> str:
    """A cache key: the model's configuration, each parameter's name, shape, dtype and an evenly spaced sample of its values (exact
    bytes), and the tokens of the batches; params: only these named parameters (default all)."""
    from . import __version__

    h = hashlib.sha256(f"lexframe-{__version__}".encode())
    h.update(m.hf.config.to_json_string(use_diff=False).encode())
    with torch.no_grad():
        for name, p in m.hf.named_parameters() if params is None else params:
            flat = p.detach().reshape(-1)
            sample = flat[:: max(1, flat.numel() // FINGERPRINT_SAMPLES)][:FINGERPRINT_SAMPLES]
            h.update(f"{name}|{tuple(p.shape)}|{p.dtype}|".encode())
            h.update(sample.float().cpu().numpy().tobytes())
    for b in batches:
        h.update(b.cpu().long().numpy().tobytes())
    return h.hexdigest()[:32]
