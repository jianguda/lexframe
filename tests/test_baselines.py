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
    v = lens.view(TEXT, k=3)
    hL, xs, _ = forward(frame._m, frame._m.ids(TEXT))
    want = logprobs(frame._m, xs[2][0]).topk(3, -1).indices.cpu().numpy()
    assert v.lens == "logit" and np.array_equal(v.ids[2], want)
    assert np.allclose(v.kl[L], 0, atol=1e-5) and np.array_equal(v.ids[L], frame.view(TEXT, k=3).ids[L])
    out = frame.view(TEXT, layers=L)
    assert torch.allclose(lens.view(TEXT, layers=L).coef([PARIS, LONDON]), out.coef([PARIS, LONDON]))
    assert torch.allclose(lens.view(TEXT, layers=L).readout([PARIS]), out.readout([PARIS]), atol=1e-5)
    r = v.readout([PARIS, LONDON])
    assert r.shape == (L + 1, T, 2, frame._m.hidden_dim) and torch.isfinite(r).all()
    assert not hasattr(v, "basis") and not hasattr(v, "steer")
    assert v.first_layer(int(v.ids[L, -1, 0]))[-1] <= L


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


def test_jacobian_lens(frame, tmp_path):
    m = frame._m
    with pytest.raises(RuntimeError, match="fit"):
        B.JacobianLens(m).view(TEXT)
    f = B.JacobianLens(m).fit(TEXTS, size=4, cache_dir=str(tmp_path))
    assert f.jacobian.shape == (L, m.hidden_dim, m.hidden_dim) and f.jacobian.dtype == torch.float16
    assert torch.isnan(f.jacobian[0]).all()
    v = f.view(TEXT, k=2)
    assert v.lens == "jacobian" and (v.ids[0] == -1).all() and np.isnan(v.kl[0]).all() and np.isfinite(v.kl[1:]).all()
    assert np.allclose(v.kl[L], 0, atol=1e-5) and (v.tokens[0] == "").all()
    r = v.readout([PARIS])
    assert torch.isnan(r[0]).all() and torch.isfinite(r[1:]).all()
    again = B.JacobianLens(m).fit(TEXTS, size=4, cache_dir=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1 and torch.equal(again.jacobian[1:], f.jacobian[1:])
    assert np.array_equal(again.view(TEXT, k=2).ids, v.ids)

    seqs = [s[0] for s in B.sequences(m, TEXTS, limit=2)]
    one = B.JacobianLens(m).fit(seqs, cache_dir=str(tmp_path), progress=False)
    joined = B.JacobianLens(m).fit([torch.cat(seqs)], cache_dir=str(tmp_path), progress=False)
    assert len(os.listdir(tmp_path)) == 3 and not torch.equal(one.jacobian[1:], joined.jacobian[1:])
    rows = B.JacobianLens(m).fit([torch.stack(seqs)], cache=False, progress=False)
    assert torch.equal(rows.jacobian[1:], one.jacobian[1:])


def test_lens_positions_and_batches(frame):
    lens = B.LogitLens(frame._m)
    last = lens.view(TEXT, k=2, positions=-1)
    whole = lens.view(TEXT, k=2)
    assert last.positions == [T - 1] and np.array_equal(last.ids[:, 0], whole.ids[:, -1])
    assert last.coef([PARIS]).shape == (L + 1, 1, 1) and last.readout([PARIS]).shape[-1] == frame._m.hidden_dim
    vs = lens.view([TEXT, "the city of"])
    assert isinstance(vs, list) and [v.ids.shape[1] for v in vs] == [T, 3]
    assert lens.view([TEXT, TEXT]).kl.shape == (2, L + 1, T)
    with pytest.raises(ValueError, match="one text"):
        lens.view([TEXT, TEXT]).coef([PARIS])
    one = lens.view(TEXT, layers=2)
    assert one.ids.shape == (T, 1) and one.coef([PARIS]).shape == (T, 1)
