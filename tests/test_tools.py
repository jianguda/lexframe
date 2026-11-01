"""The research tools (lexframe._tools; not part of the public API)."""

import torch
from conftest import L, TEXT, hf_model, tokenizer

import lexframe as lf

from lexframe import _tools as TL
from lexframe._model import forward, logprobs

PARIS = tokenizer().convert_tokens_to_ids("paris")


def test_step_audit(kind):
    """The gain is the derivative of the mean log-probability of the next tokens along the step; the last layer's write reaches no
    other position, so none of its gain comes through the context. No calibration is needed."""
    frame = lf.LexFrame(hf_model(kind), tokenizer())  # its own model: the check below moves the weights
    P = dict(frame._m.hf.named_parameters())
    last = [n for n in P if f"layers.{L - 1}." in n and "down_proj" in n]
    first = [n for n in P if "layers.0." in n and "o_proj" in n]
    g = torch.Generator().manual_seed(2)
    delta = {n: 1e-2 * P[n].detach().std() * torch.randn(P[n].shape, generator=g) for n in last + first}
    a = TL.step_audit(frame, TEXT, delta)
    assert set(a) == {"joint", 0, L - 1}
    assert abs(a[L - 1]["context_share"]) < 1e-5 and abs(a[0]["context_share"]) > 1e-3
    ids = frame._m.ids(TEXT)
    y = ids[0, 1:]

    def ll(sgn, e=1e-1):
        old = {n: P[n].data.clone() for n in delta}
        try:
            for n, d in delta.items():
                P[n].data += sgn * e * d
            return logprobs(frame._m, forward(frame._m, ids)[0][0])[:-1].gather(1, y[:, None]).mean()
        finally:
            for n, v in old.items():
                P[n].data.copy_(v)

    fd = float((ll(1) - ll(-1)) / 2e-1)
    assert abs(fd - a["joint"]["gain"]) < 2e-2 * abs(fd) + 1e-6


def test_bound(frame):
    """The square root of the mean cosine of the unit gradients of different texts, over every position."""
    gen = torch.Generator().manual_seed(4)
    texts = [torch.randint(3, 20, (8,), generator=gen) for _ in range(3)]
    u = [frame.state(t)[1].gradient([PARIS])[:, 0] for t in texts]
    u = [x / x.norm(dim=-1, keepdim=True) for x in u]
    cos = [float((a @ b.T).mean()) for i, a in enumerate(u) for j, b in enumerate(u) if i != j]
    bound = TL.bound(frame, texts, PARIS, layers=[1])
    assert bound.shape == (1,) and abs(bound[0] - max(sum(cos) / len(cos), 0) ** 0.5) < 1e-5
