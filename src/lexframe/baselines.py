"""Baselines to compare a frame with: the logit lens, and the Jacobian lens of Gurnee et al. (2026) as its reference
implementation (github.com/anthropics/jacobian-lens) defines it. They are kept apart from the frame:

    from lexframe.baselines import JacobianLens, LogitLens

    LogitLens(model, tokenizer).state(text)[layer].logits()
    JacobianLens(model, tokenizer).fit(texts).state(text)[layer].logits()

A lens reads every hidden state of a layer with one map: the logit lens with the output head, the Jacobian lens with
lm_head(final_norm(J_l x_l)) for the input x_l of block l. J_l is the gradient of the last block's output, summed over
the target positions, with respect to x_l at a source position, averaged over the source positions and then over the
fitting sequences. Positions before SKIP and the last position are left out, and a sequence starts with the
tokenizer's BOS token when it has one. J_0 is the gradient with respect to the embedding, the input of block 0.
state() reads a text as given, without BOS, like a frame. A lens state gives centred logits and gradients, indexed like a frame's
State, but it has no bases and cannot steer."""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, Sequence

import torch

from ._frame import Label, Text, fingerprint, select
from ._model import Model, final, forward, head, label_id, wrap

SKIP = 16
LENGTH = 128  # tokens per fitting sequence
DIM_BATCH = 8  # output directions per backward pass of the fit


def _sequences(m: Model, texts: Iterable[str], limit: int = 100) -> list[torch.Tensor]:
    """Sequences of LENGTH tokens cut from the texts, each starting with BOS when the tokenizer has one."""
    from ._calibrate import chunks

    bos = getattr(m.tokenizer, "bos_token_id", None)
    body = LENGTH - (bos is not None)
    out = [s[None] for b in chunks(m, texts, length=body, batch=1, limit=limit) for s in b]
    return [torch.cat([torch.tensor([[bos]]), s], 1) if bos is not None else s for s in out]


def _jacobian(m: Model, ids: torch.Tensor, valid: torch.Tensor, dim_batch: int) -> torch.Tensor:
    """J_l of one sequence for l = 0..L-1: [L, hidden, hidden]."""
    L, d = m.n_layers, m.hidden_dim
    src, out, hooks = {}, [], []
    for k in range(L):

        def pre(_m, args, kwargs, k=k):
            x = args[0] if args else kwargs["hidden_states"]
            if k == 0:
                x = x.detach().requires_grad_(True)
                args, kwargs = ((x, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": x})
            src[k] = x
            return args, kwargs

        hooks.append(m.layers[k].register_forward_pre_hook(pre, with_kwargs=True))
    hooks.append(m.layers[-1].register_forward_hook(lambda _m, _i, o: out.append(o[0] if isinstance(o, tuple) else o)))
    rows = torch.zeros(L, d, d)
    try:
        with torch.enable_grad():
            m.backbone(input_ids=ids.expand(dim_batch, -1).to(m.device), use_cache=False, return_dict=True)
            target, sources = out[0], [src[k] for k in range(L)]
            b = torch.arange(dim_batch, device=target.device)[:, None]
            pos = valid.to(target.device)[None]
            for start in range(0, d, dim_batch):
                n = min(dim_batch, d - start)
                cot = torch.zeros_like(target)
                cot[b[:n], pos, start + b[:n]] = 1.0
                grads = torch.autograd.grad(target, sources, cot, retain_graph=start + dim_batch < d)
                for i, g in enumerate(grads):
                    rows[i, start : start + n] = g[:n, valid.to(g.device)].float().mean(1).cpu()
    finally:
        for h in hooks:
            h.remove()
    return rows


def _fit_jacobian(m: Model, seqs: list[torch.Tensor]) -> torch.Tensor:
    """The Jacobian lens J_l for l = 0..L-1, [L, hidden, hidden]."""
    from tqdm.auto import tqdm

    L, d = m.n_layers, m.hidden_dim
    total, n = torch.zeros(L, d, d), 0
    for ids in tqdm(seqs, desc="Jacobian lens", leave=False):
        valid = torch.arange(SKIP, ids.shape[1] - 1)
        if valid.numel() == 0:
            continue
        total += _jacobian(m, ids, valid, DIM_BATCH)
        n += 1
    if n == 0:
        raise ValueError(f"the Jacobian lens needs sequences longer than {SKIP + 1} tokens")
    return total / n


class LensState:
    """The hidden states of a text as a lens reads them, indexed like a frame's State: s[layer] or s[layer, position]. vector,
    layers and positions as for a State; logits(labels=None) the lens's centred logits,
    gradient(labels) the gradient of the labels' centred logits with respect to the states."""

    def __init__(self, lens, ids, xs, layers, positions, squeeze=(False, False)):
        self._lens, self._ids, self._xs = lens, ids, xs
        self._layers, self._positions, self._squeeze = layers, positions, squeeze

    def __getitem__(self, key) -> LensState:
        keys = key if isinstance(key, tuple) else (key,)
        axes = [a for a in (0, 1) if not self._squeeze[a]]
        if len(keys) > len(axes):
            raise IndexError(f"too many indices: the state has {len(axes)} axes left")
        sel, squeeze = [self._layers, self._positions], list(self._squeeze)
        for a, k in zip(axes, keys):
            sel[a], squeeze[a] = select(sel[a], k)
        return LensState(self._lens, self._ids, self._xs, sel[0], sel[1], tuple(squeeze))

    @property
    def layers(self) -> list[int]:
        return list(self._layers)

    @property
    def positions(self) -> list[int]:
        return list(self._positions)

    @property
    def lens(self) -> str:
        return self._lens.name

    @property
    def vector(self) -> torch.Tensor:
        return self._shape(torch.stack([self._xs[l][0, self._positions] for l in self._layers]).float())

    def logits(self, labels: Sequence[Label] | None = None) -> torch.Tensor:
        return self._lens._directions(self, labels, "logits")

    def gradient(self, labels: Sequence[Label]) -> torch.Tensor:
        return self._lens._directions(self, labels, "gradient")

    def _shape(self, a):
        if self._squeeze[1]:
            a = a[:, 0]
        if self._squeeze[0]:
            a = a[0]
        return a

    def __repr__(self) -> str:
        return f"LensState(lens={self.lens!r}, layers={self._layers}, positions={self._positions})"


class _Lens:
    name = ""

    def __init__(self, model, tokenizer=None):
        self._m = model if isinstance(model, Model) else wrap(model, tokenizer)

    @property
    def num_layers(self) -> int:
        """The number of layers; layer num_layers is the output."""
        return self._m.n_layers

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={type(self._m.hf).__name__}, layers={self._m.n_layers})"

    def _map(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """The state of `layer` as the lens hands it to the output head."""
        return h

    def _ready(self) -> None:
        pass

    @torch.no_grad()
    def state(self, text: Text) -> LensState:
        """The hidden states of one text at every layer and position, as the lens reads them."""
        self._ready()
        ids = self._m.ids(text)
        if ids.shape[0] != 1:
            raise ValueError("one text at a time")
        hL, xs, _ = forward(self._m, ids)
        L = self._m.n_layers
        return LensState(self, ids, xs + [hL], list(range(L + 1)), list(range(ids.shape[1])))

    def _directions(self, s: LensState, labels, kind) -> torch.Tensor:
        W = self._m.lm_head.weight  # no float32 copy of the head
        if labels is None:  # the whole vocabulary, through the head itself
            vs, Wc = torch.arange(W.shape[0], device=self._m.device), None
        else:
            vs = torch.tensor([label_id(self._m, t) for t in labels], device=self._m.device)
            Wc = W[vs].float() - W.mean(0, dtype=torch.float32)
        out = []
        for l in s._layers:
            h = s._xs[l][0, s._positions].to(self._m.device).float().clone().requires_grad_(kind == "gradient")
            with torch.enable_grad():
                x = final(self._m, self._map(l, h.to(s._xs[l].dtype)))  # the readout in float32, as a frame's
                if Wc is None:
                    z = head(self._m, x)
                    z = z - z.mean(-1, keepdim=True)
                else:
                    z = (x * self._m.output_scale) @ Wc.T
                if kind == "logits":
                    out.append(z.detach())
                    continue
                n = len(vs)
                grads = [torch.autograd.grad(z[..., j].sum(), h, retain_graph=j < n - 1)[0] for j in range(n)]
                out.append(torch.stack(grads, -2).float())
        return s._shape(torch.stack(out))


class LogitLens(_Lens):
    """The logit lens: every hidden state is read with the model's own output head."""

    name = "logit"


class JacobianLens(_Lens):
    """The Jacobian lens of Gurnee et al. (2026), fitted with fit() as its reference implementation does."""

    name = "jacobian"

    def __init__(self, model, tokenizer=None):
        super().__init__(model, tokenizer)
        self.jacobian: torch.Tensor | None = None

    def fit(self, texts: Iterable, size: int = 100) -> JacobianLens:
        """Fit J_l on `size` sequences of 128 tokens of unlabelled `texts`, or on given token ids (one sequence per
        row, used as given). The lens is kept in half precision, as the reference keeps it, and cached on disk in the
        frame's cache folder ($LEXFRAME_CACHE or ~/.cache/lexframe)."""
        texts = list(texts)
        if texts and isinstance(texts[0], torch.Tensor):
            seqs = [row[None].long() for t in texts for row in t.reshape(-1, t.shape[-1])][:size]
        else:
            seqs = _sequences(self._m, texts, limit=size)
        folder = os.environ.get("LEXFRAME_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "lexframe")
        shapes = hashlib.sha256("|".join(str(tuple(s_.shape)) for s_ in seqs).encode()).hexdigest()[:8]
        path = os.path.join(folder, "jlens0-" + fingerprint(self._m, seqs) + shapes + ".pt")  # jlens0: with J_0
        if os.path.exists(path):
            J = torch.load(path, map_location="cpu", weights_only=True)
        else:
            J = _fit_jacobian(self._m, seqs).half()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(J, path)
        self.jacobian = J
        return self

    def _ready(self) -> None:
        if self.jacobian is None:
            raise RuntimeError("call fit(texts) first")

    def _map(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        if layer == self._m.n_layers:
            return h
        return (h.float() @ self.jacobian[layer].to(h.device).float().T).to(h.dtype)
