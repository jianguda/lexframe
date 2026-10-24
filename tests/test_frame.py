import os

import numpy as np
import pytest
import torch

import lexframe as lf
from conftest import FAMILIES, L, TEXT, TEXTS, V, hf_model, tokenizer

T = len(TEXT.split())
TOK = tokenizer()
PARIS, LONDON, CITY = (TOK.convert_tokens_to_ids(w) for w in ("paris", "london", "city"))


def test_init_checks_the_identities(frame):
    i = frame.identities
    assert i["ok"] and i["native_all_layers"] < 1e-4 and i["closure_first_position"] < 1e-4


def test_calibration(frame):
    cal = frame.calibration
    d = frame._m.hidden_dim
    assert cal.off.shape == (L, 16, d) and cal.fac.shape == (L, d, d) and float(cal.count[0]) == 0
    for s in cal.stats:
        assert s["ridge_evidence"] > 0 and 0 < s["gamma"] < d
    # the innovation covariance is the posterior-predictive one: fac fac^T is symmetric positive semi-definite
    S = cal.fac[1].float() @ cal.fac[1].float().T
    assert torch.allclose(S, S.T, atol=1e-4) and torch.linalg.eigvalsh(S.double()).min() > -1e-6


def test_init_caches_by_fingerprint(kind, tmp_path):
    model = hf_model(kind)
    a = lf.LexFrame(model, tokenizer()).init(TEXTS, cache_dir=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1
    b = lf.LexFrame(model, tokenizer()).init(TEXTS, cache_dir=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1 and torch.equal(a.calibration.Q, b.calibration.Q)
    assert torch.equal(a.calibration.fac, b.calibration.fac)
    lf.LexFrame(model, tokenizer()).init(TEXTS[:12], cache_dir=str(tmp_path))
    lf.LexFrame(hf_model(kind, seed=5), tokenizer()).init(TEXTS, cache_dir=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 3


def test_spread_blocks(frame):
    from lexframe._calibrate import chunks, spread

    m = frame._m
    texts = [f"{w} " * 200 for w in ("the", "city", "tower", "river")]
    b = torch.cat(spread(m, texts, 4))
    assert b.shape == (4, 128) and len({int(r[0]) for r in b}) == 4  # one block from each text
    few = torch.cat(spread(m, texts, 8))
    assert torch.equal(few, torch.cat(list(chunks(m, texts, length=128, limit=8))))


@pytest.mark.parametrize("family", [f for f in FAMILIES if f not in ("smollm3", "olmo3")])
def test_families(family):
    f = lf.LexFrame(hf_model(family), tokenizer()).init(TEXTS, cache=False)
    assert f.identities["ok"] and max(v for k, v in f.identities.items() if k != "ok") < 1e-4
    s = f.view(TEXT, layers=2)
    r, b = s.readout([PARIS]), s.basis([PARIS])
    assert torch.allclose(torch.diagonal(r @ b.mT, dim1=-2, dim2=-1), torch.ones(T, 1), atol=1e-3)
    led = f.ledger(TEXT, target=PARIS)
    assert led.check.max() < 1e-4 and abs(led.step.sum() - (led.logp[-1] - led.logp[0])) < 1e-5


@pytest.mark.parametrize("frame_", ["closure", "native"])
def test_view(frame, frame_):
    s = frame.view(TEXT, k=3, frame=frame_)
    assert s.reading == frame_ and isinstance(s.ids, np.ndarray) and s.ids.shape == (L + 1, T, 3)
    assert s.probs.shape == (L + 1, T, 3) and s.tokens.shape == (L + 1, T, 3) and isinstance(s.tokens[0, 0, 0], str)
    assert s.kl.shape == (L + 1, T) and np.all((s.probs > 0) & (s.probs <= 1)) and s.kl[L].max() < 1e-5
    assert s.vector.shape == (L + 1, T, frame._m.hidden_dim) and s.layers == list(range(L + 1)) and not s.moved
    one = frame.view(TEXT, k=3, layers=2, frame=frame_)
    assert one.ids.shape == (T, 3) and one.kl.shape == (T,) and one.vector.shape == (T, frame._m.hidden_dim)
    assert np.array_equal(one.ids, s.ids[2]) and torch.equal(one.vector, s.vector[2])
    batch = frame.view([TEXT, TEXT], k=3, frame=frame_)
    assert batch.ids.shape == (2, L + 1, T, 3) and np.array_equal(batch.ids[0], s.ids)
    again = frame.view(one, k=3, frame=frame_)
    assert np.array_equal(again.ids, one.ids) and np.allclose(again.kl, one.kl, atol=1e-6)
    with pytest.raises(ValueError, match="closure"):
        frame.view(TEXT, frame="order1")


def test_native_reading_is_the_output(frame):
    s = frame.view(TEXT, k=V, frame="native")
    assert s.kl.max() < 1e-5 and (s.ids[:, :, 0] == s.ids[L, :, 0]).all()


def test_closure_reading(frame):
    v = frame.view(TEXT, k=V)
    assert v.kl[L].max() < 1e-5 and v.kl[:, 0].max() < 1e-5  # the output, and position 0, which sees only itself
    assert v.kl[1:L, 1:].min() > 1e-6  # elsewhere the reading is not the output: no tautology
    assert np.allclose(v.probs.sum(-1), 1, atol=1e-4)
    assert np.array_equal(frame.view(TEXT, k=V).probs, v.probs)  # deterministic: the same points every time


def test_closure_reads_the_state_alone(frame):
    """The swap test: a state carried into another text reads exactly as in its own text."""
    other = "the river is located near the city of"
    a, b = frame.view(TEXT, layers=2), frame.view(other, layers=2)
    swapped = frame.view(a + (b.vector - a.vector))
    assert np.allclose(swapped.probs, b.probs, atol=1e-5) and np.array_equal(swapped.ids, b.ids)
    assert not np.allclose(
        frame.view(swapped, frame="native").probs, b.probs, atol=1e-3
    )  # the native frame sees the context


def test_more_samples_converge(frame):
    from lexframe._closure import points

    f = lf.LexFrame(frame._m)
    f.calibration = frame.calibration
    p = {M: f.config(samples=M).view(TEXT, k=V, layers=1).probs for M in (16, 256)}
    ref = f.config(samples=1024).view(TEXT, k=V, layers=1).probs
    err = {M: np.abs(np.sort(p[M], -1) - np.sort(ref, -1)).max() for M in p}
    assert err[256] < err[16] and points(frame.calibration.fac, 8).shape == (8, L, frame._m.hidden_dim)
    small = lf.LexFrame(frame._m)
    small.calibration, small._rows = frame.calibration, 64  # chunks of one position: the same readings and directions
    s, t = small.view(TEXT, k=V, layers=[1, 2]), f.config(samples=64).view(TEXT, k=V, layers=[1, 2])
    assert np.allclose(s.probs, t.probs, atol=1e-6)
    assert torch.allclose(
        small.view(TEXT, layers=1).readout([PARIS], "closure"), t.readout([PARIS], "closure")[0], atol=1e-5
    )
    with pytest.raises(ValueError):
        f.config(samples=0)


def _logit(f, s, token):
    W = f._m.lm_head.weight.float()
    Wc = W[[token]] - W.mean(0)
    xs = s._context
    return lambda X: f._coeff(f._run(xs, [s.layers], [X[None].to(xs[s.layers].dtype)]), Wc)[0, 0, :, 0]


def test_basis_has_logit_units(frame):
    s = frame.config(steps=200).view(TEXT, layers=1)
    frame.config(steps=30)
    d = s.basis([PARIS, CITY])
    assert d.shape == (T, 2, frame._m.hidden_dim) and s.coef([PARIS, CITY]).shape == (T, 2)
    c, eps = _logit(frame, s, PARIS), 1e-3
    gain = (c(s.vector + eps * d[:, 0]) - c(s.vector - eps * d[:, 0])) / (2 * eps)
    assert torch.allclose(gain, torch.ones_like(gain), atol=0.01)


def test_readout_measures_displacements(frame):
    s = frame.view(TEXT, layers=1)
    r = s.readout([PARIS, CITY])
    delta = torch.randn(T, frame._m.hidden_dim, generator=torch.Generator().manual_seed(3))
    c, eps = _logit(frame, s, PARIS), 1e-3
    measured = (c(s.vector + eps * delta) - c(s.vector - eps * delta)) / (2 * eps)
    predicted = (r @ delta[..., None])[:, 0, 0]
    assert (measured - predicted).norm() / predicted.norm() < 1e-2
    moved = s + 0.5 * delta
    assert torch.allclose(r @ (moved - s).vector[..., None], 0.5 * (r @ delta[..., None]), atol=1e-5)


def test_closure_readout_measures_displacements(frame):
    s = frame.view(TEXT, layers=1)
    r, c = s.readout([PARIS, CITY], "closure"), s.coef([PARIS, CITY], "closure")
    assert r.shape == (T, 2, frame._m.hidden_dim) and c.shape == (T, 2)
    delta = torch.randn(T, frame._m.hidden_dim, generator=torch.Generator().manual_seed(3))
    eps = 1e-3
    up, down = (s + eps * delta).coef([PARIS], "closure"), (s - eps * delta).coef([PARIS], "closure")
    measured, predicted = (up - down)[:, 0] / (2 * eps), (r @ delta[..., None])[:, 0, 0]
    assert (measured - predicted)[1:].norm() / predicted[1:].norm() < 2e-2
    # at the output both frames read the model's own logits
    o = frame.view(TEXT, layers=L)
    assert torch.allclose(o.readout([PARIS], "closure"), o.readout([PARIS]), atol=1e-5)
    assert torch.allclose(o.coef([PARIS, CITY], "closure"), o.coef([PARIS, CITY]), atol=1e-4)


def test_gap(frame):
    s = frame.view(TEXT)
    g = s.gap([PARIS, CITY])
    assert g.shape == (L + 1, T, 2) and (g >= -1e-5).all() and (g <= 2 + 1e-5).all() and g[L].abs().max() < 1e-4
    assert g[1:L, 1:].mean() > 1e-3


def test_output_layer(frame):
    s = frame.view(TEXT, k=V, layers=L)
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
    assert torch.allclose(one.coef([PARIS], "closure"), s.coef([PARIS], "closure")[2], atol=1e-5)


def test_steering(frame):
    s = frame.view(TEXT, k=V, layers=2, frame="native")
    b = s.basis([PARIS, LONDON])
    moved = s + 2 * b[:, 0] - 2 * b[:, 1]
    assert moved.moved and not s.moved and moved.ids is None
    out = frame.view(moved, k=V, frame="native")
    logp = lambda v: np.log(np.take_along_axis(v.probs, np.argsort(v.ids, -1), -1))
    gain = (logp(out)[:, PARIS] - logp(out)[:, LONDON]) - (logp(s)[:, PARIS] - logp(s)[:, LONDON])
    assert out.moved and out.ids.shape == (T, V) and gain.mean() > 0
    c = moved.coef([PARIS, LONDON])
    assert np.allclose(c[:, 0] - c[:, 1], logp(out)[:, PARIS] - logp(out)[:, LONDON], atol=1e-4)
    assert moved.readout([PARIS]).shape == (T, 1, frame._m.hidden_dim)
    assert frame.view(moved, k=1).ids.shape == (T, 1) and frame.view(moved).reading == "closure"


def test_steer(frame):
    s = frame.view(TEXT, layers=2)
    b = s.basis([PARIS, LONDON])
    moved = s.steer({"paris": 2, LONDON: -2})
    # steer solves the basis again, and an iterative solve repeats only up to rounding
    d, want = (moved - s).vector, 2 * b[:, 0] - 2 * b[:, 1]
    assert moved.moved and (d - want).norm() < 1e-2 * want.norm()
    with pytest.raises(ValueError, match="moved"):
        moved.steer({"paris": 1})


def test_ledger(frame):
    led = frame.ledger(TEXT, target="paris")
    assert led.target == PARIS and led.token == "paris" and led.position == T - 1
    assert led.logp.shape == (L + 1,) and led.step.shape == led.innovation.shape == led.noise.shape == (L,)
    assert np.allclose(led.prob, np.exp(led.logp)) and led.check.max() < 1e-4
    v = frame.view(TEXT, k=V, positions=-1)
    p = np.take_along_axis(v.probs[:, 0], np.argsort(v.ids[:, 0], -1), -1)[:, PARIS]
    assert np.allclose(led.prob, p, atol=1e-5)  # the ledger reads what view() reads
    assert np.allclose(led.innovation + led.noise, led.step, atol=1e-6)
    assert abs(led.step[1:].sum() - (led.logp[L] - led.logp[1])) < 1e-6
    assert led.sources.shape == (L, T) and np.abs(led.residual).max() < 0.05 * np.abs(led.innovation).max() + 1e-4
    fine = frame.ledger(TEXT, target="paris", steps=32)
    assert np.abs(fine.residual).max() <= np.abs(led.residual).max() + 1e-6
    mid = frame.ledger(TEXT, position=3)
    assert mid.position == 3 and mid.sources.shape == (L, 4)
    with pytest.raises(ValueError, match="position 0"):
        frame.ledger(TEXT, position=0)


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
    u = [frame.view(t, layers=1).readout([PARIS])[:, 0] for t in texts]
    u = [x / x.norm(dim=-1, keepdim=True) for x in u]
    cos = [float((a @ b.T).mean()) for i, a in enumerate(u) for j, b in enumerate(u) if i != j]
    assert frame.bound(texts, PARIS).shape == (L,)
    assert bound.shape == (1,) and abs(bound[0] - max(sum(cos) / len(cos), 0) ** 0.5) < 1e-5


def test_refusals(frame, kind):
    with pytest.raises(RuntimeError):
        lf.LexFrame(hf_model(kind), tokenizer()).view(TEXT)
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
        s.readout([PARIS], frame="order1")
    with pytest.raises(ValueError):
        lf.LexFrame(hf_model(kind, seed=5), tokenizer()).init(TEXTS, cache=False).view(s)
    assert frame.view(s, frame="native").layers == 1
    with pytest.raises(TypeError, match="order was removed"):
        lf.LexFrame(hf_model(kind), tokenizer(), order=1)
    with pytest.raises(TypeError, match="unexpected"):
        lf.LexFrame(hf_model(kind), tokenizer(), orders=1)


def test_config(frame):
    f = lf.LexFrame(frame._m)
    f.calibration = frame.calibration
    assert f.solver == {"steps": 30, "ridge": 1e-3, "warm": False} and f.samples == 64
    short = f.config(steps=2).view(TEXT, layers=1).basis([PARIS])
    long_ = f.config(steps=60, warm=True).view(TEXT, layers=1).basis([PARIS])
    assert f.solver["steps"] == 60 and f.solver["warm"] and not torch.allclose(short, long_)
    for bad in ({"steps": 0}, {"steps": 2.5}, {"ridge": -1.0}, {"samples": 2.5}):
        with pytest.raises(ValueError):
            f.config(**bad)


def test_positions(frame):
    whole = frame.view(TEXT, k=2)
    last = frame.view(TEXT, k=2, positions=-1)
    assert last.positions == [T - 1] and np.array_equal(last.ids[:, 0], whole.ids[:, -1])
    assert np.allclose(last.kl[:, 0], whole.kl[:, -1], atol=1e-5) and last.vector.shape[-2] == 1
    two = frame.view(TEXT, positions=[2, 0], layers=1)
    assert two.positions == [0, 2] and np.allclose(two.kl, whole.kl[1, [0, 2]], atol=1e-5)
    native = frame.view(TEXT, positions=[2, 0], layers=1, frame="native")
    assert native.kl.shape == (2,) and native.kl.max() < 1e-5
    with pytest.raises(ValueError, match="every position"):
        last.steer({"paris": 1})
    with pytest.raises(ValueError, match="view the text again"):
        frame.view(last)
    with pytest.raises(ValueError, match="positions must lie"):
        frame.view(TEXT, positions=T)


def test_texts_of_different_lengths(frame):
    short = "the city of"
    vs = frame.view([TEXT, short], k=2)
    assert isinstance(vs, list) and [v.ids.shape[1] for v in vs] == [T, 3]
    assert np.array_equal(vs[1].ids, frame.view(short, k=2).ids)
    a = frame.attr([TEXT, short])
    assert isinstance(a, list) and a[1].shape == (L, 3) and np.allclose(a[1], frame.attr(short))
    assert frame.view([TEXT, TEXT]).kl.shape == (2, L + 1, T)


def test_positions_given_in_any_form(frame):
    short = "the city of"
    vs = frame.view([TEXT, short], positions=(p for p in [-1]))
    assert [v.positions for v in vs] == [[T - 1], [2]]
    assert frame.view(TEXT, positions=np.int64(-1)).positions == [T - 1]
    assert frame.view(TEXT, positions=torch.tensor(0)).positions == [0]
