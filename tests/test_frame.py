import os

import numpy as np
import pytest
import torch

import lexframe as lf
from conftest import FAMILIES, L, TEXT, TEXTS, V, hf_model, tokenizer, with_order

T = len(TEXT.split())
TOK = tokenizer()
PARIS, LONDON, CITY = (TOK.convert_tokens_to_ids(w) for w in ("paris", "london", "city"))


def test_init_checks_the_identities(frame):
    assert frame.identities["ok"] and frame.identities["orderinf_all_layers"] < 1e-4


def test_init_caches_by_fingerprint(kind, tmp_path):
    model = hf_model(kind)
    a = lf.LexFrame(model, tokenizer()).init(TEXTS, cache_dir=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1
    b = lf.LexFrame(model, tokenizer()).init(TEXTS, cache_dir=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1 and torch.equal(a.calibration.Q, b.calibration.Q)
    lf.LexFrame(model, tokenizer()).init(TEXTS[:12], cache_dir=str(tmp_path))
    lf.LexFrame(hf_model(kind, seed=5), tokenizer()).init(TEXTS, cache_dir=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 3


@pytest.mark.parametrize("family", [f for f in FAMILIES if f not in ("smollm3", "olmo3")])
def test_families(family):
    f = lf.LexFrame(hf_model(family), tokenizer()).init(TEXTS, cache=False)
    assert f.identities["ok"] and max(v for k, v in f.identities.items() if k != "ok") < 1e-4
    s = f.view(TEXT, layers=2)
    r, b = s.readout([PARIS]), s.basis([PARIS])
    assert torch.allclose(torch.diagonal(r @ b.mT, dim1=-2, dim2=-1), torch.ones(T, 1), atol=1e-3)


@pytest.mark.parametrize("order", [0, 1, 2, "inf"])
def test_view(frame, order):
    f = with_order(frame, order)
    s = f.view(TEXT, k=3)
    assert isinstance(s.ids, np.ndarray) and s.ids.shape == (L + 1, T, 3) and s.probs.shape == (L + 1, T, 3)
    assert s.tokens.shape == (L + 1, T, 3) and isinstance(s.tokens[0, 0, 0], str)
    assert s.kl.shape == (L + 1, T) and np.all((s.probs > 0) & (s.probs <= 1)) and s.kl[L].max() < 1e-5
    assert s.vector.shape == (L + 1, T, frame._m.hidden_dim) and s.layers == list(range(L + 1)) and not s.moved
    one = f.view(TEXT, k=3, layers=2)
    assert one.ids.shape == (T, 3) and one.kl.shape == (T,) and one.vector.shape == (T, frame._m.hidden_dim)
    assert np.array_equal(one.ids, s.ids[2]) and torch.equal(one.vector, s.vector[2])
    batch = f.view([TEXT, TEXT], k=3)
    assert batch.ids.shape == (2, L + 1, T, 3) and np.array_equal(batch.ids[0], s.ids)
    again = f.view(one, k=3)
    assert np.array_equal(again.ids, one.ids) and np.allclose(again.kl, one.kl, atol=1e-6)


@pytest.mark.parametrize("order", [2, 3])
def test_orders(frame, order):
    f = with_order(frame, order)
    kl = f.view(TEXT).kl
    assert kl[L - order :].max() < 1e-5 and kl[: L - order].max() > 1e-6


def _logit(f, s, token):
    W = f._m.lm_head.weight.float()
    Wc = W[[token]] - W.mean(0)
    xs = s._context
    return lambda X: f._coeff(f._run(xs, [s.layers], [X[None].to(xs[s.layers].dtype)]), Wc)[0, 0, :, 0]


def test_basis_has_logit_units(frame):
    f = with_order(frame, "inf")
    s = f.config(steps=200).view(TEXT, layers=1)
    d = s.basis([PARIS, CITY])
    assert d.shape == (T, 2, frame._m.hidden_dim) and s.coef([PARIS, CITY]).shape == (T, 2)
    c, eps = _logit(f, s, PARIS), 1e-3
    gain = (c(s.vector + eps * d[:, 0]) - c(s.vector - eps * d[:, 0])) / (2 * eps)
    assert torch.allclose(gain, torch.ones_like(gain), atol=0.01)


def test_readout_measures_displacements(frame):
    f = with_order(frame, 1)
    s = f.view(TEXT, layers=1)
    r = s.readout([PARIS, CITY])
    delta = torch.randn(T, frame._m.hidden_dim, generator=torch.Generator().manual_seed(3))
    c, eps = _logit(f, s, PARIS), 1e-3
    measured = (c(s.vector + eps * delta) - c(s.vector - eps * delta)) / (2 * eps)
    predicted = (r @ delta[..., None])[:, 0, 0]
    assert (measured - predicted).norm() / predicted.norm() < 1e-2
    moved = s + 0.5 * delta
    assert torch.allclose(r @ (moved - s).vector[..., None], 0.5 * (r @ delta[..., None]), atol=1e-5)


def test_output_layer(frame):
    s = with_order(frame, 1).view(TEXT, k=V, layers=L)
    c = s.coef([PARIS, CITY])
    logp = np.log(np.take_along_axis(s.probs, np.argsort(s.ids, -1), -1))
    assert s.kl.max() < 1e-5 and np.allclose(c[:, 0] - c[:, 1], logp[:, PARIS] - logp[:, CITY], atol=1e-4)
    d, eps = s.basis([PARIS])[:, 0], 1e-3
    gain = ((s + eps * d).coef([PARIS]) - (s - eps * d).coef([PARIS]))[:, 0] / (2 * eps)
    assert torch.allclose(gain, torch.ones_like(gain), atol=0.01) and frame.view(s + d, k=1).ids.shape == (T, 1)


def test_directions_at_every_layer(frame):
    s = frame.view(TEXT)
    b, r, c = s.basis([PARIS, LONDON]), s.readout([PARIS, LONDON]), s.coef([PARIS, LONDON])
    assert b.shape == r.shape == (L + 1, T, 2, frame._m.hidden_dim) and c.shape == (L + 1, T, 2)
    one = frame.view(TEXT, layers=2)
    assert torch.allclose(one.coef([PARIS, LONDON]), c[2]) and torch.allclose(
        one.readout([PARIS, LONDON]), r[2], atol=1e-6
    )


def test_steering(frame):
    f = with_order(frame, "inf")
    s = f.view(TEXT, k=V, layers=2)
    b = s.basis([PARIS, LONDON])
    moved = s + 2 * b[:, 0] - 2 * b[:, 1]
    assert moved.moved and not s.moved and moved.ids is None
    out = f.view(moved, k=V)
    logp = lambda v: np.log(np.take_along_axis(v.probs, np.argsort(v.ids, -1), -1))
    gain = (logp(out)[:, PARIS] - logp(out)[:, LONDON]) - (logp(s)[:, PARIS] - logp(s)[:, LONDON])
    assert out.moved and out.ids.shape == (T, V) and gain.mean() > 0
    c = moved.coef([PARIS, LONDON])
    assert np.allclose(c[:, 0] - c[:, 1], logp(out)[:, PARIS] - logp(out)[:, LONDON], atol=1e-4)
    assert moved.readout([PARIS]).shape == (T, 1, frame._m.hidden_dim)
    assert with_order(frame, 1).view(moved, k=1).ids.shape == (T, 1)


def test_steer(frame):
    s = frame.view(TEXT, layers=2)
    b = s.basis([PARIS, LONDON])
    moved = s.steer({"paris": 2, LONDON: -2})
    # steer solves the basis again, and an iterative solve repeats only up to rounding
    d, want = (moved - s).vector, 2 * b[:, 0] - 2 * b[:, 1]
    assert moved.moved and (d - want).norm() < 1e-2 * want.norm()
    with pytest.raises(ValueError, match="moved"):
        moved.steer({"paris": 1})


def test_first_layer(frame):
    v = frame.view(TEXT)
    token = int(v.ids[-1, -1, 0])
    first = v.first_layer(token)
    top = v.ids[..., 0] == token
    assert first.shape == (T,) and first[-1] == int(np.argmax(top[:, -1])) and v.first_layer("paris").min() >= -1
    assert (first[~top.any(0)] == -1).all()
    assert frame.view(TEXT, layers=L).first_layer(token)[-1] == L
    assert frame.view([TEXT, TEXT]).first_layer(token).shape == (2, T)
    with pytest.raises(ValueError, match="no reading"):
        (v + 0).first_layer(token)


def test_tokens_as_strings(frame):
    s = frame.view(TEXT, layers=2)
    assert torch.equal(s.coef(["paris", "london"]), s.coef([PARIS, LONDON]))
    assert np.array_equal(frame.attr(TEXT, target="paris"), frame.attr(TEXT, target=PARIS))
    assert frame.token_id("paris") == frame.token_id(PARIS) == PARIS
    with pytest.raises(ValueError, match="2 tokens"):
        s.coef(["paris london"])


@pytest.mark.parametrize("sublayer", [None, "attn", "mlp"])
def test_attr(frame, sublayer):
    a = frame.attr(TEXT, sublayer)
    assert isinstance(a, np.ndarray) and a.shape == (L, T) and np.isfinite(a).all()
    assert frame.attr([TEXT, TEXT], sublayer, target=PARIS).shape == (2, L, T)


def test_bound(frame):
    g = torch.Generator().manual_seed(4)
    texts = [torch.randint(3, 20, (T,), generator=g) for _ in range(3)]
    bound = frame.bound(texts, PARIS, layers=[1])
    native = with_order(frame, "inf")
    u = [native.view(t, layers=1).readout([PARIS])[:, 0] for t in texts]
    u = [x / x.norm(dim=-1, keepdim=True) for x in u]
    cos = [float((a @ b.T).mean()) for i, a in enumerate(u) for j, b in enumerate(u) if i != j]
    assert frame.bound(texts, PARIS).shape == (L,)
    assert bound.shape == (1,) and abs(bound[0] - max(sum(cos) / len(cos), 0) ** 0.5) < 1e-5


def test_refusals(frame, kind):
    with pytest.raises(RuntimeError):
        lf.LexFrame(hf_model(kind), tokenizer()).view(TEXT)
    with pytest.raises(ValueError):
        with_order(frame, 0).view(TEXT, layers=1).basis([PARIS])
    with pytest.raises(ValueError):
        frame.view([TEXT, TEXT], layers=1).basis([PARIS])
    s = frame.view(TEXT, layers=1)
    with pytest.raises(ValueError):
        (s + s.readout([PARIS])[:, 0]).basis([PARIS])
    with pytest.raises(ValueError):
        frame.view(s, layers=2)
    with pytest.raises(ValueError):
        frame.attr(TEXT, "block")
    with pytest.raises(ValueError):
        lf.LexFrame(hf_model(kind, seed=5), tokenizer()).init(TEXTS, cache=False).view(s)
    assert with_order(frame, "inf").view(s).layers == 1
    with pytest.raises(ValueError):
        lf.LexFrame(hf_model(kind), tokenizer(), order=-1)


def test_config_the_solver(frame):
    f = with_order(frame, "inf")
    assert f.solver == {"steps": 30, "ridge": 1e-3, "warm": False}
    short = f.config(steps=2).view(TEXT, layers=1).basis([PARIS])
    long_ = f.config(steps=60, warm=True).view(TEXT, layers=1).basis([PARIS])
    assert f.solver["steps"] == 60 and f.solver["warm"] and not torch.allclose(short, long_)
    for bad in ({"steps": 0}, {"steps": 2.5}, {"ridge": -1.0}):
        with pytest.raises(ValueError):
            f.config(**bad)
