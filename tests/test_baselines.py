import os

import numpy as np
import pytest
import torch

from conftest import L, TEXT, TEXTS, kl_of, tokenizer, top_of
from lexframe import baselines as B

T = len(TEXT.split())
TOK = tokenizer()
PARIS, LONDON = (TOK.convert_tokens_to_ids(w) for w in ("paris", "london"))


def test_logit_lens(frame):
    from lexframe._model import forward, logprobs

    lens = B.LogitLens(frame._m)
    s = lens.state(TEXT)
    labels, probs = top_of(s, 3)
    hL, xs, _ = forward(frame._m, frame._m.ids(TEXT))
    want = logprobs(frame._m, xs[2][0]).topk(3, -1)
    assert s.lens == "logit" and labels.shape == (L + 1, T, 3)
    assert np.array_equal(labels[2], np.vectorize(lambda i: TOK.decode([int(i)]))(want.indices.cpu().numpy()))
    assert np.allclose(probs[2], want.values.exp().cpu().numpy(), atol=1e-6)
    assert np.allclose(kl_of(s)[L], 0, atol=1e-5) and np.array_equal(labels[L], top_of(frame.state(TEXT)[L], 3)[0])
    out = frame.state(TEXT)[L]
    assert torch.allclose(lens.state(TEXT)[L].logits([PARIS, LONDON]), out.logits([PARIS, LONDON]), atol=1e-5)
    assert torch.allclose(lens.state(TEXT)[L].gradient([PARIS]), out.gradient([PARIS]), atol=1e-5)
    r = s.gradient([PARIS, LONDON])
    assert r.shape == (L + 1, T, 2, frame._m.hidden_dim) and torch.isfinite(r).all()
    assert not hasattr(s, "basis") and not hasattr(s, "steer")


def test_jacobian_lens_matches_its_definition(frame, tmp_path):
    """J_l is the gradient of the last block's output, summed over the target positions, with respect to x_l at a
    source position, averaged over the source positions; here against finite differences on one short sequence."""
    m = frame._m
    ids = torch.tensor([[1] + [TOK.convert_tokens_to_ids(w) for w in TEXTS[0].split()[:19]]])
    valid = torch.arange(B.SKIP, ids.shape[1] - 1)
    J = B._jacobian(m, ids, valid, dim_batch=8)
    assert torch.allclose(J, B._jacobian(m, ids, valid, dim_batch=3), atol=1e-5)

    def last(layer, pos=None, delta=None):
        out = []

        def pre(_m, args, kwargs):
            if delta is None:
                return None
            x = (args[0] if args else kwargs["hidden_states"]).clone()
            x[0, pos] += delta
            return ((x, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": x})

        hk = m.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)
        ho = m.layers[-1].register_forward_hook(lambda _m, _i, o: out.append(o[0] if isinstance(o, tuple) else o))
        try:
            with torch.no_grad():
                m.backbone(input_ids=ids, use_cache=False)
        finally:
            hk.remove(), ho.remove()
        return out[0][0, valid].double().sum(0)

    for layer in (0, L - 1):
        eps = 1e-3
        for j in (0, 7):
            e = torch.zeros(m.hidden_dim)
            e[j] = eps
            col = torch.stack([(last(layer, int(s), e) - last(layer, int(s), -e)) / (2 * eps) for s in valid]).mean(0)
            assert torch.allclose(J[layer][:, j].double(), col, atol=2e-2, rtol=2e-2)


def test_jacobian_lens(frame, tmp_path, monkeypatch):
    m = frame._m
    monkeypatch.setenv("LEXFRAME_CACHE", str(tmp_path))
    with pytest.raises(RuntimeError, match="fit"):
        B.JacobianLens(m).state(TEXT)
    f = B.JacobianLens(m).fit(TEXTS, size=4)
    assert f.jacobian.shape == (L, m.hidden_dim, m.hidden_dim) and f.jacobian.dtype == torch.float16
    assert torch.isfinite(f.jacobian).all()  # J_0 included: the embedding is read too
    s = f.state(TEXT)
    labels, probs = top_of(s, 2)
    kl = kl_of(s)
    assert s.lens == "jacobian" and np.isfinite(probs).all() and np.isfinite(kl).all() and np.allclose(kl[L], 0, atol=1e-5)
    assert torch.isfinite(s.gradient([PARIS])).all()
    again = B.JacobianLens(m).fit(TEXTS, size=4)
    assert len(os.listdir(tmp_path)) == 1 and torch.equal(again.jacobian, f.jacobian)
    assert np.array_equal(top_of(again.state(TEXT), 2)[0], labels)

    seqs = [s_[0] for s_ in B._sequences(m, TEXTS, limit=2)]
    one = B.JacobianLens(m).fit(seqs)
    joined = B.JacobianLens(m).fit([torch.cat(seqs)])
    assert len(os.listdir(tmp_path)) == 3 and not torch.equal(one.jacobian, joined.jacobian)
    monkeypatch.setenv("LEXFRAME_CACHE", str(tmp_path / "rows"))  # a fresh cache: the rows are fitted, not loaded
    rows = B.JacobianLens(m).fit([torch.stack(seqs)])
    assert len(os.listdir(tmp_path / "rows")) == 1 and torch.equal(rows.jacobian, one.jacobian)


def test_lens_positions(frame):
    lens = B.LogitLens(frame._m)
    s = lens.state(TEXT)
    last = s[:, -1]
    whole = top_of(s, 2)[0]
    assert last.positions == [T - 1] and np.array_equal(top_of(last, 2)[0], whole[:, -1])
    kept = s[:, [-1]]
    assert kept.logits([PARIS]).shape == (L + 1, 1, 1) and kept.gradient([PARIS]).shape[-1] == frame._m.hidden_dim
    with pytest.raises(ValueError, match="one text"):
        lens.state([TEXT, TEXT])
    one = s[2]
    assert top_of(one, 1)[0].shape == (T, 1) and one.logits([PARIS]).shape == (T, 1)
    assert one.layers == [2] and torch.equal(one.vector, s.vector[2])
