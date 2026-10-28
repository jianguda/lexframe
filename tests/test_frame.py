import os

import numpy as np
import pytest
import torch

import lexframe as lf
from lexframe import _tools as TL
from lexframe._model import label_id
from lexframe import _trail
from lexframe._transport import run
from lexframe._model import forward
from conftest import FAMILIES, L, TEXT, TEXTS, V, fresh, hf_model, tokenizer

T = len(TEXT.split())
TOK = tokenizer()
PARIS, LONDON, CITY = (TOK.convert_tokens_to_ids(w) for w in ("paris", "london", "city"))


def _logp(dec, word):
    """The log-probability of one label in a Decoding, read from its full top(V)."""
    labels, probs = dec.top(V)
    return np.log((probs * (labels == word)).sum(-1))


def test_init_checks_the_identities(frame):
    i = frame._identities
    assert i["ok"] and i["visible_all_layers"] < 1e-4 and i["invisible_first_position"] < 1e-4


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
    a = fresh(model, cache=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1
    b = fresh(model, cache=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1 and torch.equal(a.calibration.Q, b.calibration.Q)
    assert torch.equal(a.calibration.fac, b.calibration.fac)
    lf.LexFrame(model, tokenizer()).config(cache=str(tmp_path), progress=False).init(TEXTS[:12])
    fresh(hf_model(kind, seed=5), cache=str(tmp_path))
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
    f = fresh(hf_model(family))
    assert f._identities["ok"] and max(v for k, v in f._identities.items() if k != "ok") < 1e-4
    s = f.state(TEXT)[2]
    r, b = s.gradient([PARIS]), s.direction([PARIS])
    assert torch.allclose(torch.diagonal(r @ b.mT, dim1=-2, dim2=-1), torch.ones(T, 1), atol=1e-3)
    sp = _trail.split(f, f._m.ids(TEXT), T - 1, PARIS)
    step = sp["innovation"] + sp["noise"]
    assert sp["check"].max() < 1e-4 and abs(step.sum() - (sp["logp"][-1] - sp["logp"][0])) < 1e-5


@pytest.mark.parametrize("context", [False, True])
def test_decode(frame, context):
    s = frame.state(TEXT)
    d = s.decode(context)
    labels, probs = d.top(3)
    assert isinstance(labels, np.ndarray) and labels.shape == probs.shape == (L + 1, T, 3) and isinstance(labels[0, 0, 0], str)
    assert d.kl.shape == (L + 1, T) and np.all((probs > 0) & (probs <= 1)) and d.kl[L].max() < 1e-5
    assert s.vector.shape == (L + 1, T, frame._m.hidden_dim) and s.layers == list(range(L + 1)) and s.positions == list(range(T))
    assert s._X is None
    one = s[2]
    ol, op = one.decode(context).top(3)
    assert ol.shape == (T, 3) and one.decode(context).kl.shape == (T,) and one.vector.shape == (T, frame._m.hidden_dim)
    assert np.array_equal(ol, labels[2]) and np.allclose(op, probs[2], atol=1e-6) and torch.equal(one.vector, s.vector[2])
    again = frame.state(TEXT)[2].decode(context)
    assert np.array_equal(again.top(3)[0], ol) and np.allclose(again.kl, one.decode(context).kl, atol=1e-6)
    cell = s[2, -1].decode(context)
    assert cell.top(3)[0].shape == (3,) and np.ndim(cell.kl) == 0 and np.array_equal(cell.top(3)[0], labels[2, -1])
    assert np.array_equal(d.top(2)[0], labels[..., :2])  # a smaller k is the same ranking


def test_native_reading_is_the_output(frame):
    d = frame.state(TEXT).decode(context=True)
    labels, _ = d.top(V)
    assert d.kl.max() < 1e-5 and (labels[:, :, 0] == labels[L, :, 0]).all()


def test_reading_without_context(frame):
    d = frame.state(TEXT).decode(context=False)
    _, probs = d.top(V)
    kl = d.kl
    assert kl[L].max() < 1e-5 and kl[:, 0].max() < 1e-5  # the output, and position 0, which sees only itself
    assert kl[1:L, 1:].min() > 1e-6  # elsewhere the reading is not the output: no tautology
    assert np.allclose(probs.sum(-1), 1, atol=1e-4)
    assert np.array_equal(frame.state(TEXT).decode(context=False).top(V)[1], probs)  # deterministic: the same points every time


def test_reading_without_context_depends_on_the_state_alone(frame):
    """The swap test: a state carried into another text decodes without context exactly as in its own text."""
    other = "the river is located near the city of"
    a, b = frame.state(TEXT)[2], frame.state(other)[2]
    swapped = a + (b.vector - a.vector)
    sl, sp = swapped.decode(context=False).top(5)
    bl, bp = b.decode(context=False).top(5)
    assert np.allclose(sp, bp, atol=1e-5) and np.array_equal(sl, bl)
    # with context all moved positions are carried together: swapping every state is the other text's own forward pass
    assert np.allclose(swapped.decode().top(5)[1], b.decode().top(5)[1], atol=1e-5)
    last = a[-1] + (b.vector[-1] - a.vector[-1])
    assert np.allclose(last.decode(context=False).top(5)[1], bp[-1], atol=1e-5)
    assert not np.allclose(
        last.decode().top(5)[1], bp[-1], rtol=0.02, atol=0
    )  # one swapped state: with context it sees its new context, without it does not


def test_more_samples_converge(frame):
    from lexframe._transport import points

    f = lf.LexFrame(frame._m)
    f.calibration = frame.calibration
    p = {M: f.config(samples=M).state(TEXT)[1].decode(False).top(V)[1] for M in (16, 256)}
    ref = f.config(samples=1024).state(TEXT)[1].decode(False).top(V)[1]
    err = {M: np.abs(np.sort(p[M], -1) - np.sort(ref, -1)).max() for M in p}
    assert err[256] < err[16] and points(frame.calibration.fac, 8).shape == (8, L, frame._m.hidden_dim)
    small = lf.LexFrame(frame._m)
    small.calibration, small._rows = frame.calibration, 64  # chunks of one position: the same readings and directions
    s, t = small.state(TEXT)[[1, 2]], f.config(samples=64).state(TEXT)[[1, 2]]
    assert np.allclose(s.decode(False).top(V)[1], t.decode(False).top(V)[1], atol=1e-6)
    assert torch.allclose(
        small.state(TEXT)[1].gradient([PARIS], context=False), t.gradient([PARIS], context=False)[0], atol=1e-5
    )
    with pytest.raises(ValueError):
        f.config(samples=0)


def _logit(f, s, token):
    """The centred logit of one label after the native transport of the State's layer, each position's column moved on its own."""
    W = f._m.lm_head.weight.float()
    Wc = W[[token]] - W.mean(0)
    xs, l = s._xs, s.layers[0]
    return lambda X: f._centred_logits(run(f._m, xs, [l], [X[None].to(xs[l].dtype)]), Wc)[0, 0, :, 0]


def test_direction_has_logit_units(frame):
    s = frame.state(TEXT)[1]
    d = s.direction([PARIS, CITY])
    assert d.shape == (T, 2, frame._m.hidden_dim) and s.decode().logits([PARIS, CITY]).shape == (T, 2)
    c, eps = _logit(frame, s, PARIS), 1e-3
    gain = (c(s.vector + eps * d[:, 0]) - c(s.vector - eps * d[:, 0])) / (2 * eps)
    assert torch.allclose(gain, torch.ones_like(gain), atol=0.01)


def test_gradient_measures_displacements(frame):
    s = frame.state(TEXT)[1]
    r = s.gradient([PARIS, CITY])
    delta = torch.randn(T, frame._m.hidden_dim, generator=torch.Generator().manual_seed(3))
    c, eps = _logit(frame, s, PARIS), 1e-3
    measured = (c(s.vector + eps * delta) - c(s.vector - eps * delta)) / (2 * eps)
    predicted = (r @ delta[..., None])[:, 0, 0]
    assert (measured - predicted).norm() / predicted.norm() < 1e-2
    moved = s + 0.5 * delta
    assert torch.allclose(r @ (moved - s).vector[..., None], 0.5 * (r @ delta[..., None]), atol=1e-5)


def test_coframe_without_context_measures_displacements(frame):
    s = frame.state(TEXT)[1]
    r, c = s.gradient([PARIS, CITY], context=False), s.decode(False).logits([PARIS, CITY])
    assert r.shape == (T, 2, frame._m.hidden_dim) and c.shape == (T, 2)
    delta = torch.randn(T, frame._m.hidden_dim, generator=torch.Generator().manual_seed(3))
    eps = 1e-3
    # the gradient without context is the derivative of the decoding without context
    up, down = (s + eps * delta).decode(False).logits([PARIS]), (s - eps * delta).decode(False).logits([PARIS])
    measured, predicted = (up - down)[:, 0] / (2 * eps), (r @ delta[..., None])[:, 0, 0]
    assert (measured - predicted)[1:].norm() / predicted[1:].norm() < 2e-2
    # at the output both read the model's own logits
    o = frame.state(TEXT)[L]
    assert torch.allclose(o.gradient([PARIS], context=False), o.gradient([PARIS]), atol=1e-5)
    assert torch.allclose(o.decode(False).logits([PARIS, CITY]), o.decode().logits([PARIS, CITY]), atol=1e-4)


def test_gradients_with_and_without_context(frame):
    """1 - cos of the coframes with and without context: zero at the output, positive inside."""
    s = frame.state(TEXT)
    gn, gc = s.gradient([PARIS, CITY]), s.gradient([PARIS, CITY], context=False)
    g = 1 - torch.nn.functional.cosine_similarity(gn, gc, dim=-1)
    assert g.shape == (L + 1, T, 2) and (g >= -1e-5).all() and (g <= 2 + 1e-5).all() and g[L].abs().max() < 1e-4
    assert g[1:L, 1:].mean() > 1e-3


def test_output_layer(frame):
    s = frame.state(TEXT)[L]
    dec = s.decode()
    c = dec.logits([PARIS, CITY])
    assert dec.kl.max() < 1e-5 and np.allclose(c[:, 0] - c[:, 1], _logp(dec, "paris") - _logp(dec, "city"), atol=1e-4)
    d, eps = s.direction([PARIS])[:, 0], 1e-3
    gain = ((s + eps * d).decode().logits([PARIS]) - (s - eps * d).decode().logits([PARIS]))[:, 0] / (2 * eps)
    assert torch.allclose(gain, torch.ones_like(gain), atol=0.01) and (s + d).decode(False).top(1)[0].shape == (T, 1)


def test_directions_at_every_layer(frame):
    s = frame.state(TEXT)
    b, r, c = s.direction([PARIS, LONDON]), s.gradient([PARIS, LONDON]), s.decode().logits([PARIS, LONDON])
    assert b.shape == r.shape == (L + 1, T, 2, frame._m.hidden_dim) and c.shape == (L + 1, T, 2)
    one = s[2]
    assert torch.allclose(one.decode().logits([PARIS, LONDON]), c[2]) and torch.allclose(
        one.gradient([PARIS, LONDON]), r[2], atol=1e-6
    )
    assert torch.allclose(one.decode(False).logits([PARIS]), s.decode(False).logits([PARIS])[2], atol=1e-5)
    assert torch.allclose(one.direction([PARIS, LONDON]), b[2], atol=1e-5)


def test_moving_a_state(frame):
    s = frame.state(TEXT)[2]
    b = s.direction([PARIS, LONDON])
    moved = s + 2 * b[:, 0] - 2 * b[:, 1]
    assert moved._X is not None and s._X is None and torch.allclose(moved.vector, s.vector + 2 * b[:, 0] - 2 * b[:, 1], atol=1e-5)
    out, base = moved.decode(), s.decode()
    gain = (_logp(out, "paris") - _logp(out, "london")) - (_logp(base, "paris") - _logp(base, "london"))
    assert out.top(V)[0].shape == (T, V) and gain.mean() > 0
    c = out.logits([PARIS, LONDON])
    assert np.allclose(c[:, 0] - c[:, 1], _logp(out, "paris") - _logp(out, "london"), atol=1e-4)
    assert moved.gradient([PARIS]).shape == (T, 1, frame._m.hidden_dim)
    assert moved.decode(False).top(1)[0].shape == (T, 1)


def test_steer(frame):
    """steer() reaches the logit changes it is given, with the directions' cross-talk removed, and continues by the path where one
    step misses; error and move report how well and how far, _info the steps."""
    s = frame.state(TEXT)[2]
    tokens = [PARIS, LONDON, CITY]
    w = torch.tensor([0.2, -0.2, 0.1])
    moved = s.steer(dict(zip(tokens, w.tolist())))
    actual = moved.decode().logits(tokens) - s.decode().logits(tokens)
    err = (actual - w).norm(dim=-1) / w.norm()
    tol = frame._settings["tolerance"]
    assert moved._X is not None and moved.error.shape == moved.move.shape == (T,)
    assert set(moved._info) == {2} and set(moved._info[2]) >= {"error", "steps", "move", "regime", "c2_ratio"}
    assert float(moved.error.max()) <= tol + 1e-6
    assert torch.allclose(err, torch.as_tensor(moved.error), atol=1e-4) and (moved._info[2]["steps"] >= 1).all()
    many = frame.state(TEXT)[[1, 3]].steer({PARIS: 0.1})
    assert set(many._info) == {1, 3} and many.error.shape == (2, T) and float(many._info[3]["error"].max()) <= tol + 1e-6
    with pytest.raises(ValueError, match="moved"):
        moved.steer({"paris": 1})
    with pytest.raises(TypeError):
        s.steer({"paris": 1}, exact=True)


def test_steer_small_move(frame):
    """For a small move, the model's logit change divided by the scale is the weight: the cross-talk is removed to first order."""
    s = frame.state(TEXT)[2]
    tokens = [PARIS, LONDON, CITY]
    w = torch.tensor([2.0, -2.0, 1.0])
    eps = 1e-4  # random tiny models are far from linear (and the converged early-layer direction is long): keep the move small
    small = s.steer(dict(zip(tokens, (eps * w).tolist())))
    actual = (small.decode().logits(tokens) - s.decode().logits(tokens)) / eps
    assert torch.allclose(actual, w.expand_as(actual), atol=0.05)


def test_coordinates(frame):
    """coordinates: a displacement on the labels' frame; the steps along the undamped frame vectors whose first-order logit changes
    equal the displacement's."""
    s = frame.state(TEXT)[2]
    tokens = [PARIS, LONDON, CITY]
    exact = TL.exact_frame(s, tokens)  # [T, label, d], each raising its own label's logit by one
    r = s.gradient(tokens)
    w = torch.tensor([1.5, -0.5, 2.0])
    c = s.coordinates((exact * w[:, None]).sum(-2), tokens)  # a combination of the frame vectors is recovered
    assert c.shape == (T, 3) and torch.allclose(c[1:], w.expand_as(c[1:]), atol=1e-2)
    delta = torch.randn(T, frame._m.hidden_dim, generator=torch.Generator().manual_seed(5))
    c = s.coordinates(delta, tokens)
    synth = (c[..., None] * exact).sum(-2)
    assert torch.allclose((r @ synth[..., None])[1:], (r @ delta[..., None])[1:], rtol=1e-2, atol=1e-3)
    nxt = frame.state(TEXT)[3]
    assert torch.isfinite(s.coordinates(nxt - s, tokens)).all()  # what layer 2 wrote, in label coordinates
    two = frame.state(TEXT)[[2, 3], -1].coordinates(torch.randn(2, frame._m.hidden_dim), tokens)
    assert two.shape == (2, 3)
    with pytest.raises(ValueError, match="moved"):
        (s + 0.0).coordinates(delta, tokens)


def test_trail(frame):
    t = frame.trail(TEXT, label="paris")
    assert t.label == "paris" and t._v == PARIS and t.position == T - 1 and t.logprobs.shape == (L + 1,)
    sp = _trail.split(frame, frame._m.ids(TEXT), T - 1, PARIS)
    assert np.allclose(sp["logp"], t.logprobs) and sp["check"].max() < 1e-4
    assert sp["innovation"].shape == sp["noise"].shape == (L,)
    d = frame.state(TEXT)[:, -1].decode(context=False)
    assert np.allclose(np.exp(t.logprobs), np.exp(_logp(d, "paris")), atol=1e-5)  # the trail reads what decode() reads
    step = np.diff(t.logprobs)
    assert np.allclose(sp["innovation"] + sp["noise"], step, atol=1e-6)
    assert abs((sp["innovation"] + sp["noise"])[1:].sum() - (t.logprobs[L] - t.logprobs[1])) < 1e-6
    assert t.upstream().shape == (L, T) and np.allclose(t.upstream(), sp["sources"])
    a = t.attribute()
    assert a.shape == (L, 3) and np.isfinite(a).all()
    mid = frame.trail(TEXT, position=3)
    assert mid.position == 3 and mid.upstream().shape == (L, 4) and mid.downstream().shape == (L, 4)
    assert frame.trail(TEXT, position=-1).downstream().shape == (L, 0)
    with pytest.raises(ValueError, match="position 0"):
        frame.trail(TEXT, position=0)


def test_split_by_source(frame, monkeypatch):
    """The split by source integrates over STEPS points; its residual is what it leaves of each innovation, and more points shrink it."""
    import lexframe._trail as F

    ids = frame._m.ids(TEXT)
    sp = _trail.split(frame, ids, T - 1, PARIS)
    assert np.allclose(sp["residual"], sp["innovation"] - sp["sources"].sum(-1), atol=1e-6)
    monkeypatch.setattr(F, "STEPS", 64)
    fine = _trail.split(frame, ids, T - 1, PARIS)
    assert np.allclose(fine["innovation"], sp["innovation"]) and np.allclose(fine["logp"], sp["logp"])
    assert np.abs(fine["residual"]).max() <= np.abs(sp["residual"]).max() + 1e-6


@pytest.mark.parametrize("position", [-1, 3])
def test_attribute_is_the_ablation(frame, position):
    """attribute() is the change of the label's centred logit, with context, when the layer's write at the position is removed:
    the State that ablate() returns, decoded with context."""
    t = frame.trail(TEXT, position=position, label="paris")
    a = t.attribute()
    s = frame.state(TEXT)
    for k in range(L):
        base = float(s[k + 1, t.position].decode().logits([PARIS])[0])
        for j, part in enumerate(("all", "attention", "mlp")):
            removed = t.ablate(k, part)
            assert removed.layers == [k + 1] and removed.positions == list(range(T))
            after = float(removed.decode().logits([PARIS])[t.position, 0])
            assert abs(a[k, j] - (base - after)) < 1e-4 * (1 + abs(a[k, j]))
    # the write is the sum of its two parts; to first order so is its effect
    ablated = t.ablate(L - 1).vector[t.position]
    xs = s.vector
    assert torch.allclose(ablated, xs[L - 1, t.position], atol=1e-4)  # removing the last layer's whole write gives its input


def test_ablate_a_source(frame):
    """Removing what a layer's attention read from one source position: the layer runs again with that source's share taken out of
    its attention output, exactly as the model does when the share is subtracted inside the attention module."""
    m = frame._m
    ids = m.ids(TEXT)
    t = frame.trail(TEXT, label="paris")
    k, tpos = 1, T - 1
    for p in (0, 3, tpos):
        got = t.ablate(k, source=p).vector[-1]
        u_in, a_out = {}, {}
        forward(m, ids, on_attention=lambda l, u, a: (u_in.__setitem__(l, u[0]), a_out.__setitem__(l, a[0, tpos])) if l == k else None)
        C = _trail.sources(m, k, u_in[k], tpos, a_out[k].float())[0]
        layer = m.layers[k]

        def hook(module, args, out, p=p):
            o = out[0] if isinstance(out, tuple) else out
            o = o.clone()
            o[0, tpos] = (o[0, tpos].float() - C[p]).to(o.dtype)
            return (o, *out[1:]) if isinstance(out, tuple) else o

        handle = layer.self_attn.register_forward_hook(hook)
        try:
            _, xs, _ = forward(m, ids)
            want = (xs[k + 1] if k + 1 < L else None)[0, tpos].float()
        finally:
            handle.remove()
        assert torch.allclose(got, want, atol=1e-4)
    with pytest.raises(ValueError):
        t.ablate(k, "mlp", source=0)
    with pytest.raises(ValueError):
        t.ablate(L)
    with pytest.raises(ValueError):
        t.ablate(0, "block")


def test_labels_as_strings(frame):
    s = frame.state(TEXT)[2]
    assert torch.equal(s.decode().logits(["paris", "london"]), s.decode().logits([PARIS, LONDON]))
    assert np.array_equal(frame.trail(TEXT, label="paris").attribute(), frame.trail(TEXT, label=PARIS).attribute())
    assert frame.trail(TEXT, label=PARIS).label == "paris"
    assert label_id(frame._m, "paris") == label_id(frame._m, PARIS) == PARIS
    with pytest.raises(ValueError, match="2 tokens"):
        s.decode().logits(["paris london"])


def test_bound(frame):
    g = torch.Generator().manual_seed(4)
    texts = [torch.randint(3, 20, (T,), generator=g) for _ in range(3)]
    bound = TL.bound(frame, texts, PARIS, layers=[1])
    u = [frame.state(t)[1].gradient([PARIS])[:, 0] for t in texts]
    u = [x / x.norm(dim=-1, keepdim=True) for x in u]
    cos = [float((a @ b.T).mean()) for i, a in enumerate(u) for j, b in enumerate(u) if i != j]
    assert TL.bound(frame, texts, PARIS).shape == (L,)
    assert bound.shape == (1,) and abs(bound[0] - max(sum(cos) / len(cos), 0) ** 0.5) < 1e-5


def test_refusals(frame, kind):
    bare = lf.LexFrame(hf_model(kind), tokenizer())
    with pytest.raises(RuntimeError):
        bare.state(TEXT).decode(context=False).top()
    with pytest.raises(RuntimeError):
        bare.trail(TEXT)
    assert bare.state(TEXT)[2].decode().top(1)[0].shape == (T, 1)  # decoding with context needs no calibration
    with pytest.raises(ValueError):
        frame.state([TEXT, TEXT])
    s = frame.state(TEXT)[1]
    with pytest.raises(ValueError):
        (s + s.gradient([PARIS])[:, 0]).direction([PARIS])
    with pytest.raises(IndexError):
        s[0, 0]  # one axis left
    with pytest.raises(IndexError):
        s[[]]
    for layers in ([3, 1], [2, 2]):  # layers are carried upwards: each once, in increasing order
        with pytest.raises(IndexError):
            frame.state(TEXT)[layers]
    assert s.layers == [1]
    for bad in ({"tolerance": 0}, {"tolerance": 1.5}, {"size": 1}, {"samples": -1}):
        with pytest.raises(ValueError):
            bare.config(**bad)


def test_config(frame):
    f = lf.LexFrame(frame._m)
    f.calibration = frame.calibration
    assert f._solver == {"steps": 50, "damping": 0.1, "tol": 1e-3}
    assert f._settings == {"samples": 64, "tolerance": 0.02, "size": 1024, "cache": "", "progress": True}
    assert f.config(samples=32, tolerance=0.05, size=8, cache=None, progress=False) is f
    assert f._settings == {"samples": 32, "tolerance": 0.05, "size": 8, "cache": None, "progress": False}
    # the default solver converges: it matches the exact solution of the same problem, which two steps do not
    s = f.state(TEXT)[1]
    ref = _reference_direction(f, s, [PARIS])
    cos = lambda a: float(torch.nn.functional.cosine_similarity(a.double(), ref, dim=-1).min())
    full = s.direction([PARIS])
    f._solver["steps"] = 2
    short = s.direction([PARIS])
    assert cos(full) > 0.999 and cos(short) < cos(full) and not torch.allclose(short, full)
    with pytest.raises(ValueError):
        f.config(samples=2.5)
    with pytest.raises(TypeError, match="unexpected"):
        f.config(steps=30)  # the solver is internal


def test_positions(frame):
    s = frame.state(TEXT)
    whole = s.decode(False)
    wl, _ = whole.top(2)
    last = s[:, -1]
    ll, _ = last.decode(False).top(2)
    assert last.positions == [T - 1] and np.array_equal(ll, wl[:, -1])
    assert np.allclose(last.decode(False).kl, whole.kl[:, -1], atol=1e-5) and last.vector.shape == (L + 1, frame._m.hidden_dim)
    assert s[:, [-1]].vector.shape == (L + 1, 1, frame._m.hidden_dim)
    two = s[1, [2, 0]]
    assert two.positions == [2, 0] and np.allclose(two.decode(False).kl, whole.kl[1, [2, 0]], atol=1e-5)
    native = two.decode()
    assert native.kl.shape == (2,) and native.kl.max() < 1e-5
    # steering a subset of the positions leaves the others where they are
    one = s[2, [T - 1]].steer({"paris": 0.1})
    assert one.error.shape == (1,) and float(one.error.max()) <= frame._settings["tolerance"] + 1e-6
    assert torch.equal(one._X[2][0, : T - 1], s._xs[2][0, : T - 1]) and not torch.equal(one._X[2][0, -1], s._xs[2][0, -1])
    with pytest.raises(IndexError):
        s[:, T]


def test_positions_given_in_any_form(frame):
    s = frame.state(TEXT)
    assert s[:, (p for p in [-1])].positions == [T - 1]
    assert s[:, np.int64(-1)].positions == [T - 1]
    assert s[:, torch.tensor(0)].positions == [0]
    assert s[np.int64(2)].layers == [2] and s[1:3].layers == [1, 2]


def _reference_direction(frame, s, tokens):
    """The damped least-squares direction from the full Jacobian of the native transport (in double precision), as a reference."""
    from torch.autograd import forward_ad as fwAD

    from lexframe._state import references, stream

    m, l, xs = frame._m, s.layers[0], s._xs
    d, n = m.hidden_dim, xs[l].shape[1]
    E = torch.eye(d)[:, None, :].expand(d, n, d).contiguous()
    with torch.no_grad(), fwAD.dual_level():
        out = stream(m, l, references(m, xs, l), fwAD.make_dual(xs[l][:1].expand(d, -1, -1).contiguous(), E))
        J = fwAD.unpack_dual(out).tangent.permute(1, 2, 0).double()
    W = m.lm_head.weight.double()
    Y = W[tokens] @ frame._gram_inverse().double()
    JtY = torch.einsum("toi,vo->tvi", J, Y)
    damp2 = frame._solver["damping"] * (JtY.norm(dim=-1) / Y.norm(dim=-1)) ** 2
    eye = torch.eye(d, dtype=torch.double)
    x = torch.stack([torch.linalg.solve(J.mT @ J + damp2[:, i, None, None] * eye, JtY[:, i, :, None])[..., 0] for i in range(len(tokens))], 1)
    gain = m.output_scale * torch.einsum("toi,tvi,vo->tv", J, x, W[tokens] - W.mean(0))
    return x / gain[..., None]


def test_direction_is_converged(frame):
    """The default solver reaches the exact damped solution at the earliest layer, where the transport is worst conditioned."""
    s = frame.state(TEXT)[1]
    ref = _reference_direction(frame, s, [PARIS, CITY])
    cos = torch.nn.functional.cosine_similarity(s.direction([PARIS, CITY]).double(), ref, dim=-1)
    assert float(cos.min()) > 0.999


def test_public_api():
    """The public surface: state() gives a State (decode, gradient, direction, coordinates, steer), trail() a Trail (attribute,
    upstream, downstream, ablate); research tools live in lexframe._tools."""
    from lexframe._frame import Decoding, State, Trail

    assert set(lf.__all__) == {"LexFrame", "__version__"}
    public = lambda cls: {n for n in vars(cls) if not n.startswith("_")}
    assert public(lf.LexFrame) == {"num_layers", "config", "init", "state", "trail"}
    assert public(State) == {"layers", "positions", "vector", "decode", "gradient", "direction", "coordinates", "steer"}
    assert public(Trail) == {"attribute", "upstream", "downstream", "ablate"}
    assert public(Decoding) == {"top", "logits", "kl"}
    for name in ("chart", "pairing", "exact_frame", "step_audit", "bound"):
        assert callable(getattr(TL, name))
