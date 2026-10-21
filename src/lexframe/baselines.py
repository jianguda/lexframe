"""Baselines to compare a frame with: the logit lens, and the Jacobian lens of Gurnee et al. (2026) as its reference
implementation (github.com/anthropics/jacobian-lens) defines it. They are kept apart from the frame:

    from lexframe.baselines import JacobianLens, LogitLens

    LogitLens(model, tokenizer).view(text).tokens
    JacobianLens(model, tokenizer).fit(texts).view(text).tokens

A lens reads every hidden state of a layer with one map: the logit lens with the output head, the Jacobian lens with
lm_head(final_norm(J_l x_l)) for the input x_l of block l. J_l is the gradient of the last block's output, summed over
the target positions, with respect to x_l at a source position, averaged over the source positions and then over the
fitting sequences. Positions before SKIP and the last position are left out, and a sequence starts with the
tokenizer's BOS token when it has one. The embedding x_0 has no Jacobian lens, so its reading holds -1 and NaN.
view() reads a text as given, without BOS, like a frame. A lens gives coefficients (its centred logits) and readouts
(their gradients with respect to the state), but no basis, and it cannot steer."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import torch

from ._frame import Layers, Text, Token, _fingerprint, _folder, _is_int, _single, chosen, ragged, token_id
from ._model import Model, forward, kl, logprobs, wrap

SKIP = 16


def sequences(m: Model, texts: Iterable[str], length: int = 128, limit: int = 100) -> list[torch.Tensor]:
    """Sequences of `length` tokens cut from the texts, each starting with BOS when the tokenizer has one."""
    from ._calibrate import chunks

    bos = getattr(m.tokenizer, "bos_token_id", None)
    body = length - (bos is not None)
    out = [s[None] for b in chunks(m, texts, length=body, batch=1, limit=limit) for s in b]
    return [torch.cat([torch.tensor([[bos]]), s], 1) if bos is not None else s for s in out]


def _jacobian(m: Model, ids: torch.Tensor, valid: torch.Tensor, dim_batch: int) -> torch.Tensor:
    """J_l of one sequence for l = 1..L-1: [L-1, hidden, hidden]."""
    L, d = m.n_layers, m.hidden_dim
    src, out, hooks = {}, [], []
    for k in range(1, L):

        def pre(_m, args, kwargs, k=k):
            x = args[0] if args else kwargs["hidden_states"]
            if k == 1:
                x = x.detach().requires_grad_(True)
                args, kwargs = ((x, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": x})
            src[k] = x
            return args, kwargs

        hooks.append(m.layers[k].register_forward_pre_hook(pre, with_kwargs=True))
    hooks.append(m.layers[-1].register_forward_hook(lambda _m, _i, o: out.append(o[0] if isinstance(o, tuple) else o)))
    rows = torch.zeros(L - 1, d, d)
    try:
        with torch.enable_grad():
            m.backbone(input_ids=ids.expand(dim_batch, -1).to(m.device), use_cache=False, return_dict=True)
            target, sources = out[0], [src[k] for k in range(1, L)]
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


def fit_jacobian(m: Model, seqs: list[torch.Tensor], dim_batch: int = 8, progress: bool = True) -> torch.Tensor:
    """The Jacobian lens J_l for l = 0..L-1, [L, hidden, hidden], with J_0 undefined (NaN)."""
    from tqdm.auto import tqdm

    L, d = m.n_layers, m.hidden_dim
    total, n = torch.zeros(L - 1, d, d), 0
    for ids in tqdm(seqs, desc="Jacobian lens", disable=not progress, leave=False):
        valid = torch.arange(SKIP, ids.shape[1] - 1)
        if valid.numel() == 0:
            continue
        total += _jacobian(m, ids, valid, dim_batch)
        n += 1
    if n == 0:
        raise ValueError(f"the Jacobian lens needs sequences longer than {SKIP + 1} tokens")
    return torch.cat([torch.full((1, d, d), float("nan")), total / n])


class Reading:
    """What a lens reads in the hidden states of one or more texts at one or more layers (layer num_layers is the
    output). vector: the states, [..., position, hidden]; tokens, ids, probs [..., position, k] and kl
    [..., position]: the reading, as NumPy arrays, with -1 and NaN at a layer that the lens does not read.
    first_layer(), coef() and readout() work as for a frame's Carrier."""

    def __init__(self, lens, vector, layers, flags, reading, positions=None):
        self.vector, self.layers, self.positions = vector, layers, positions
        self._lens, self._flags = lens, flags
        self.ids, self.probs, self.tokens, self.kl = reading

    @property
    def lens(self) -> str:
        return self._lens.name

    def first_layer(self, token: Token) -> np.ndarray:
        """The first of the read layers at which the token tops the reading, [..., position]; -1 where it never
        does."""
        top = self.ids[..., 0] == self._lens.token_id(token)
        top = top[..., None, :] if isinstance(self.layers, int) else top
        layers = np.array(self._layer_list())
        return np.where(top.any(-2), layers[top.argmax(-2)], -1)

    def coef(self, tokens: Sequence[Token]) -> torch.Tensor:
        """The centred logits of the tokens as the lens reads the states, [..., position, token]."""
        return self._lens._directions(self, tokens, "coef")

    def readout(self, tokens: Sequence[Token]) -> torch.Tensor:
        """The gradient of each token's centred logit with respect to the state, [..., position, token, hidden]."""
        return self._lens._directions(self, tokens, "readout")

    def _layer_list(self) -> list[int]:
        return [self.layers] if isinstance(self.layers, int) else list(self.layers)

    def __repr__(self) -> str:
        extra = "" if self.positions is None else f", positions={self.positions}"
        return f"Reading(lens={self.lens!r}, layers={self.layers}, shape={tuple(self.vector.shape)}{extra})"


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

    def token_id(self, token: Token) -> int:
        """The id of a token, given as an id or as a string that the tokenizer encodes to exactly one token."""
        return token_id(self._m, token)

    def _map(self, layer: int, h: torch.Tensor) -> torch.Tensor | None:
        """The state of `layer` as the lens hands it to the output head, or None where it does not read the layer."""
        return h

    def _ready(self) -> None:
        pass

    @torch.no_grad()
    def view(
        self, x: Text, k: int = 1, layers: Layers = None, positions: int | Iterable[int] | None = None
    ) -> Reading | list[Reading]:
        """What the lens reads in the hidden states of one or more texts at every layer (or the given ones), at every
        position (or the given ones, after which the text is cut). Texts of different token lengths are read one at
        a time, and a list is returned."""
        self._ready()
        if positions is not None and not _is_int(positions):
            positions = list(positions)
        texts = ragged(self._m, x)
        if texts is not None:
            return [self.view(t, k, layers, positions) for t in texts]
        ids = self._m.ids(x)
        P = None
        if positions is not None:
            P = chosen(positions, ids.shape[1])
            ids = ids[:, : P[-1] + 1]
        hL, xs, _ = forward(self._m, ids)
        xs = xs + [hL]
        L = self._m.n_layers
        LS = (
            list(range(L + 1))
            if layers is None
            else sorted({int(l) for l in ([layers] if isinstance(layers, int) else layers)})
        )
        if not LS or LS[0] < 0 or LS[-1] > L:
            raise ValueError(f"layers must lie in 0..{L} (layer {L} is the output)")
        flags = (_single(x), isinstance(layers, int))
        ref = logprobs(self._m, hL)
        ids_, probs, kls = [], [], []
        for l in LS:
            h = self._map(l, xs[l])
            if h is None:
                ids_.append(torch.full((*hL.shape[:-1], k), -1, dtype=torch.long, device=hL.device))
                probs.append(torch.full((*hL.shape[:-1], k), float("nan"), device=hL.device))
                kls.append(torch.full(hL.shape[:-1], float("nan"), device=hL.device))
                continue
            lp = logprobs(self._m, h)
            top = lp.topk(k, -1)
            ids_.append(top.indices), probs.append(top.values.exp()), kls.append(kl(ref, lp))
        out = [torch.stack(o, 1).cpu().numpy() for o in (ids_, probs, kls)]
        V = torch.stack([xs[l] for l in LS], 1).float()
        if P is not None:
            out, V = [o[:, :, P] for o in out], V[:, :, P]
        if flags[0]:
            out, V = [o[0] for o in out], V[0]
        if flags[1]:
            out, V = [out[0][..., 0, :, :], out[1][..., 0, :, :], out[2][..., 0, :]], V[..., 0, :, :]
        tok = self._m.tokenizer
        word = lambda i: tok.decode([int(i)]) if i >= 0 else ""
        tokens = None if tok is None else np.vectorize(word, otypes=[object])(out[0])
        return Reading(self, V, LS[0] if flags[1] else LS, flags, (out[0], out[1], tokens, out[2]), P)

    def _directions(self, r: Reading, tokens, kind) -> torch.Tensor:
        if not r._flags[0]:
            raise ValueError("coefficients and readouts are taken at the states of one text")
        vs = torch.tensor([self.token_id(t) for t in tokens], device=self._m.device)
        W = self._m.lm_head.weight.float()
        Wc = W[vs] - W.mean(0)
        V = r.vector[None] if r._flags[1] else r.vector
        out = []
        for i, l in enumerate(r._layer_list()):
            h = V[i].to(self._m.device).clone().requires_grad_(kind == "readout")
            with torch.enable_grad():
                g = self._map(l, h)
                if g is None:
                    shape = (*h.shape[:-1], len(vs)) + ((h.shape[-1],) if kind == "readout" else ())
                    out.append(torch.full(shape, float("nan"), device=h.device))
                    continue
                z = self._m.final_norm(g.to(self._m.lm_head.weight.dtype)).float() * self._m.output_scale @ Wc.T
                if kind == "coef":
                    out.append(z.detach())
                    continue
                n = len(vs)
                grads = [torch.autograd.grad(z[..., j].sum(), h, retain_graph=j < n - 1)[0] for j in range(n)]
                out.append(torch.stack(grads, -2).float())
        out = torch.stack(out)
        return out[0] if r._flags[1] else out


class LogitLens(_Lens):
    """The logit lens: every hidden state is read with the model's own output head."""

    name = "logit"


class JacobianLens(_Lens):
    """The Jacobian lens of Gurnee et al. (2026), fitted with fit() as its reference implementation does."""

    name = "jacobian"

    def __init__(self, model, tokenizer=None):
        super().__init__(model, tokenizer)
        self.jacobian: torch.Tensor | None = None

    def fit(
        self,
        texts: Iterable,
        size: int = 100,
        dim_batch: int = 8,
        cache: bool = True,
        cache_dir: str | None = None,
        progress: bool = True,
    ) -> JacobianLens:
        """Fit J_l on `size` sequences of 128 tokens of unlabelled `texts`, or on given token ids (one sequence per
        row, used as given). The lens is kept in half precision, as the reference keeps it, and cached on disk like
        a frame's calibration."""
        texts = list(texts)
        if texts and isinstance(texts[0], torch.Tensor):
            seqs = [row[None].long() for t in texts for row in t.reshape(-1, t.shape[-1])][:size]
        else:
            seqs = sequences(self._m, texts, limit=size)
        path = (
            os.path.join(_folder(cache_dir), "jlens-" + _fingerprint(self._m, seqs, shapes=True) + ".pt")
            if cache
            else None
        )
        if path and os.path.exists(path):
            J = torch.load(path, map_location="cpu", weights_only=True)
        else:
            J = fit_jacobian(self._m, seqs, dim_batch=dim_batch, progress=progress).half()
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(J, path)
        self.jacobian = J
        return self

    def _ready(self) -> None:
        if self.jacobian is None:
            raise RuntimeError("call fit(texts) first")

    def _map(self, layer: int, h: torch.Tensor) -> torch.Tensor | None:
        if layer == 0:
            return None
        if layer == self._m.n_layers:
            return h
        return (h.float() @ self.jacobian[layer].to(h.device).float().T).to(h.dtype)
