"""The model as Hugging Face loads it, with the entry points that a frame needs."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class Model:
    hf: Any
    tokenizer: Any
    backbone: Any
    layers: Any
    final_norm: Any
    lm_head: Any
    output_scale: float
    softcap: float | None = None
    pe: dict = field(default_factory=dict)  # per layer, by (length, twice); see position_embeddings()

    def __repr__(self) -> str:
        return (
            f"Model({type(self.hf).__name__}, layers={self.n_layers}, hidden={self.hidden_dim}, device={self.device})"
        )

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_dim(self) -> int:
        cfg = getattr(self.hf.config, "text_config", None) or self.hf.config
        return int(cfg.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.hf.parameters()).device

    def ids(self, text_or_ids) -> torch.Tensor:
        if isinstance(text_or_ids, torch.Tensor):
            ids = text_or_ids
        else:
            if self.tokenizer is None:
                raise ValueError("this model has no tokenizer: pass token ids")
            texts = [text_or_ids] if isinstance(text_or_ids, str) else list(text_or_ids)
            enc = [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]
            if len({len(e) for e in enc}) != 1:
                raise ValueError("texts of different token lengths: pass them one at a time")
            ids = torch.tensor(enc, dtype=torch.long)
        ids = ids.long()
        return (ids[None] if ids.dim() == 1 else ids).to(self.device)

    def position_embeddings(self, k: int, n: int, twice: bool = False):
        """The position embeddings that layer k receives for positions 0..n-1, or for 0..n-1 twice in a row (a prefix of
        reference states followed by the queries at the same positions). They are produced by the model itself, from
        a forward pass with these position ids, so their form is whatever the architecture uses."""
        key = (n, twice)
        if key not in self.pe:
            pos = torch.arange(n, device=self.device)
            pos = torch.cat([pos, pos]) if twice else pos
            self.pe[key] = _capture_position_embeddings(self, pos)
        return self.pe[key].get(k)


def _tree(fn, x):
    """Apply fn to every tensor of a (nested) tuple or list."""
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, (tuple, list)):
        return type(x)(_tree(fn, v) for v in x)
    return x


class _Captured(Exception):
    pass


@torch.no_grad()
def _capture_position_embeddings(m: Model, pos: torch.Tensor) -> dict:
    """Layer k's position_embeddings argument for a sequence with position ids pos, from the model's own forward pass;
    the pass stops once the last layer has received its argument."""
    got, hooks = {}, []
    for k, layer in enumerate(m.layers):

        def pre(_m, args, kwargs, k=k):
            pe = kwargs.get("position_embeddings")
            if pe is not None:
                got[k] = _tree(lambda p: p[:1].detach(), pe)
            if k == len(m.layers) - 1:
                raise _Captured

        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    held = [(layer, layer.self_attn) for layer in m.layers]
    attns = [at.own[0] if isinstance(at, _Stub) else at for _, at in held]
    for (layer, _), at in zip(held, attns):  # the model's own attention modules, which compute nothing here
        layer.self_attn = at
        at.forward = _no_attention
    ids = torch.zeros(1, pos.numel(), dtype=torch.long, device=m.device)
    try:
        m.backbone(input_ids=ids, position_ids=pos[None], use_cache=False, return_dict=True)
    except _Captured:
        pass
    finally:
        for h in hooks:
            h.remove()
        for (layer, at), own in zip(held, attns):
            del own.forward
            layer.self_attn = at
    return got


def _no_attention(hidden_states=None, *args, **kwargs):
    return torch.zeros_like(hidden_states), None


def wrap(hf, tokenizer=None) -> Model:
    body = hf.get_decoder() if hasattr(hf, "get_decoder") else hf.model
    layers = getattr(body, "layers", None)
    if layers is None or not all(hasattr(layer, "self_attn") for layer in layers) or not hasattr(body, "norm"):
        raise NotImplementedError(
            f"{type(hf).__name__} is not a decoder whose layers have `self_attn` and a final `norm`"
        )
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    cfg = getattr(hf.config, "text_config", None) or hf.config
    scale = getattr(cfg, "logits_scaling", None) or 1.0
    return Model(
        hf,
        tokenizer,
        body,
        layers,
        body.norm,
        hf.get_output_embeddings(),
        1.0 / float(scale),
        getattr(cfg, "final_logit_softcapping", None),
    )


@contextlib.contextmanager
def eager(m: Model):
    """Run attention and mixture-of-experts modules with their plain implementations, which accept any additive mask
    and support forward-mode differentiation."""
    cfgs = [c for c in (m.hf.config, getattr(m.hf.config, "text_config", None)) if c is not None]
    keys = [k for k in ("_attn_implementation", "_experts_implementation") if all(hasattr(c, k) for c in cfgs)]
    old = [[getattr(c, k) for k in keys] for c in cfgs]
    try:
        for c in cfgs:
            for k in keys:
                if getattr(c, k) is not None:
                    setattr(c, k, "eager")
        yield
    finally:
        for c, o in zip(cfgs, old):
            for k, v in zip(keys, o):
                setattr(c, k, v)


def _second(layer):
    for nm in ("pre_feedforward_layernorm", "post_attention_layernorm"):
        if nm == "post_attention_layernorm" and hasattr(layer, "post_feedforward_layernorm"):
            return layer.mlp
        if hasattr(layer, nm):
            return getattr(layer, nm)
    return layer.mlp


@torch.no_grad()
def forward(
    m: Model, ids: torch.Tensor, on_attention: Callable | None = None, mid: bool = False
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    xs, md, fin, hooks = [], [], [], []
    for k, layer in enumerate(m.layers):

        def pre(_m, args, kwargs, k=k):
            xs.append(args[0] if args else kwargs["hidden_states"])

        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        if mid:
            hooks.append(_second(layer).register_forward_pre_hook(lambda _m, inp: md.append(inp[0])))
        if on_attention is not None:

            def post(_m, args, kwargs, out, k=k):
                on_attention(
                    k, kwargs.get("hidden_states", args[0] if args else None), out[0] if isinstance(out, tuple) else out
                )

            hooks.append(layer.self_attn.register_forward_hook(post, with_kwargs=True))
    hooks.append(m.layers[-1].register_forward_hook(lambda _m, _i, o: fin.append(o[0] if isinstance(o, tuple) else o)))
    try:
        m.backbone(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        for h in hooks:
            h.remove()
    return fin[0], xs, md


class _Stub(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.start = 0

    def forward(self, hidden_states=None, *args, **kwargs):
        return self.fn(hidden_states[0], self.start)[None], None


def own_attention(m: Model, k: int):
    at = m.layers[k].self_attn
    return at.own[0] if isinstance(at, _Stub) else at


def innorm(m: Model, k: int):
    return getattr(m.layers[k], "input_layernorm", None) or (lambda x: x)


def block(m: Model, k: int, x: torch.Tensor, read: Callable, chunk: int = 8192) -> torch.Tensor:
    layer = m.layers[k]
    own = layer.self_attn
    out = []
    layer.self_attn = stub = _Stub(read)
    stub.own = [own]
    try:
        with eager(m):
            for s in range(0, x.shape[0], chunk):
                stub.start = s
                y = layer(x[None, s : s + chunk])
                out.append((y[0] if isinstance(y, tuple) else y)[0])
    finally:
        layer.self_attn = own
    return out[0] if len(out) == 1 else torch.cat(out)


def logits(m: Model, h: torch.Tensor) -> torch.Tensor:
    z = (m.lm_head(m.final_norm(h)) * m.output_scale).float()
    return z if not m.softcap else torch.tanh(z / m.softcap) * m.softcap


def logprobs(m: Model, h: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits(m, h), -1)


def kl(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    return (lp.exp() * (lp - lq)).sum(-1)
