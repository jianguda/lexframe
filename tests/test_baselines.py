import os

import numpy as np
import pytest
import torch

from conftest import L, TEXT, TEXTS, tokenizer
from lexframe import baselines as B

T = len(TEXT.split())
TOK = tokenizer()
PARIS, LONDON = (TOK.convert_tokens_to_ids(w) for w in ("paris", "london"))


def test_logit_lens(frame):
    from lexframe._model import forward, logprobs

    lens = B.LogitLens(frame._m)
    s = lens.state(TEXT)
    labels, probs = s.decode().top(3)
    hL, xs, _ = forward(frame._m, frame._m.ids(TEXT))
    want = logprobs(frame._m, xs[2][0]).topk(3, -1)
    assert s.lens == "logit" and labels.shape == (L + 1, T, 3)
    assert np.array_equal(labels[2], np.vectorize(lambda i: TOK.decode([int(i)]))(want.indices.cpu().numpy()))
    assert np.allclose(probs[2], want.values.exp().cpu().numpy(), atol=1e-6)
    assert np.allclose(s.decode().kl[L], 0, atol=1e-5) and np.array_equal(labels[L], frame.state(TEXT)[L].decode().top(3)[0])
    out = frame.state(TEXT)[L]
    assert torch.allclose(lens.state(TEXT)[L].decode().logits([PARIS, LONDON]), out.decode().logits([PARIS, LONDON]))
    assert torch.allclose(lens.state(TEXT)[L].gradient([PARIS]), out.gradient([PARIS]), atol=1e-5)
    r = s.gradient([PARIS, LONDON])
    assert r.shape == (L + 1, T, 2, frame._m.hidden_dim) and torch.isfinite(r).all()
    assert not hasattr(s, "direction") and not hasattr(s, "steer")


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

    for layer in (1, L - 1):
        base, eps = last(layer), 1e-3
        for j in (0, 7):
            e = torch.zeros(m.hidden_dim)
            e[j] = eps
            col = torch.stack([(last(layer, int(s), e) - base) / eps for s in valid]).mean(0)
            assert torch.allclose(J[layer - 1][:, j].double(), col, atol=2e-2, rtol=2e-2)


def test_jacobian_lens(frame, tmp_path, monkeypatch):
    m = frame._m
    monkeypatch.setenv("LEXFRAME_CACHE", str(tmp_path))
    with pytest.raises(RuntimeError, match="fit"):
        B.JacobianLens(m).state(TEXT)
    f = B.JacobianLens(m).fit(TEXTS, size=4)
    assert f.jacobian.shape == (L, m.hidden_dim, m.hidden_dim) and f.jacobian.dtype == torch.float16
    assert torch.isnan(f.jacobian[0]).all()
    s = f.state(TEXT)
    labels, probs = s.decode().top(2)
    kl = s.decode().kl
    assert s.lens == "jacobian" and (labels[0] == "").all() and np.isnan(probs[0]).all()
    assert np.isnan(kl[0]).all() and np.isfinite(kl[1:]).all() and np.allclose(kl[L], 0, atol=1e-5)
    r = s.gradient([PARIS])
    assert torch.isnan(r[0]).all() and torch.isfinite(r[1:]).all()
    assert torch.isnan(s.decode().logits([PARIS])[0]).all()
    again = B.JacobianLens(m).fit(TEXTS, size=4)
    assert len(os.listdir(tmp_path)) == 1 and torch.equal(again.jacobian[1:], f.jacobian[1:])
    assert np.array_equal(again.state(TEXT).decode().top(2)[0], labels)

    seqs = [s_[0] for s_ in B._sequences(m, TEXTS, limit=2)]
    one = B.JacobianLens(m).fit(seqs)
    joined = B.JacobianLens(m).fit([torch.cat(seqs)])
    assert len(os.listdir(tmp_path)) == 3 and not torch.equal(one.jacobian[1:], joined.jacobian[1:])
    monkeypatch.setenv("LEXFRAME_CACHE", str(tmp_path / "rows"))  # a fresh cache: the rows are fitted, not loaded
    rows = B.JacobianLens(m).fit([torch.stack(seqs)])
    assert len(os.listdir(tmp_path / "rows")) == 1 and torch.equal(rows.jacobian[1:], one.jacobian[1:])


def test_lens_positions(frame):
    lens = B.LogitLens(frame._m)
    s = lens.state(TEXT)
    last = s[:, -1]
    whole = s.decode().top(2)[0]
    assert last.positions == [T - 1] and np.array_equal(last.decode().top(2)[0], whole[:, -1])
    kept = s[:, [-1]]
    assert kept.decode().logits([PARIS]).shape == (L + 1, 1, 1) and kept.gradient([PARIS]).shape[-1] == frame._m.hidden_dim
    with pytest.raises(ValueError, match="one text"):
        lens.state([TEXT, TEXT])
    one = s[2]
    assert one.decode().top(1)[0].shape == (T, 1) and one.decode().logits([PARIS]).shape == (T, 1)
    assert one.layers == [2] and torch.equal(one.vector, s.vector[2])
