import copy
import os
import warnings

import pytest
import torch

import lexframe as lf
import lexframe._state as S
from lexframe import _ablate
from lexframe._model import forward, label_id, logits
from lexframe._transport import run
from conftest import FAMILIES, L, TEXT, V, fresh, hf_model, kl_of, tokenizer

T = len(TEXT.split())
TOK = tokenizer()
PARIS, LONDON, CITY = (TOK.convert_tokens_to_ids(w) for w in ("paris", "london", "city"))
LABELS = [PARIS, LONDON, CITY]


def _centred(z):
    return z - z.mean(-1, keepdim=True)


def _output(frame, xs, k, X, t):
    """The centred output logits at t after the moved states X of layer k, carried by the model's own forward pass, [V]."""
    return _centred(logits(frame._m, run(frame._m, xs, [k], [X], joint=True)[0, 0]))[t]


def _miss(before, moved, changes):
    """The relative error of a steer per selected position, from the public logits of the output layer."""
    labels = list(changes)
    w = torch.tensor([float(changes[t]) for t in labels])
    pos = before.positions
    whole = before._run(before._xs, before._moves)  # the whole run the state belongs to
    d = moved[-1, pos].logits(labels) - whole[-1, pos].logits(labels)
    return ((d.cpu() - w).norm(dim=-1) / w.norm()).float()


def _slope(st, k, t, delta, label, eps=1e-3):
    """The central difference of a label's output logit along delta at the state of layer k and position t."""
    out = lambda x: x[L, t].logits([label])[0]  # noqa: E731
    return (out(st[k, t] + eps * delta) - out(st[k, t] - eps * delta)) / (2 * eps)


# ---- the frame

def test_init(kind, tmp_path):
    """init() calibrates so that the frame reproduces the model, and caches the calibration under a fingerprint of the model and
    the calibration tokens."""
    model = hf_model(kind)
    a = fresh(model, cache=str(tmp_path))
    i = a._identities
    assert i["ok"] and i["visible_all_layers"] < 1e-4 and i["invisible_first_position"] < 1e-4
    b = fresh(model, cache=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 1 and torch.equal(a.calibration.fac, b.calibration.fac)
    fresh(hf_model(kind, seed=5), cache=str(tmp_path))
    assert len(os.listdir(tmp_path)) == 2


def test_calibration_takes_one_block_from_each_text(frame):
    from lexframe._calibrate import spread

    texts = [f"{w} " * 200 for w in ("the", "city", "tower", "river")]
    b = torch.cat(spread(frame._m, texts, 4, 128))
    assert b.shape == (4, 128) and len({int(r[0]) for r in b}) == 4


@pytest.mark.parametrize("family", list(FAMILIES))
def test_families(family):
    """In every family, the model's own transport reproduces the model, and logits, gradients, bases and steering work."""
    f = fresh(hf_model(family))
    m = f._m
    hL, xs, _ = forward(m, m.ids(TEXT))
    xs = xs + [hL]
    scale = float(hL.abs().max())
    assert float((run(m, xs, list(range(L)))[:, 0] - hL[0]).abs().max()) < 1e-4 * scale
    s = f.state(TEXT)[2]
    assert torch.isfinite(s.basis([PARIS])).all() and torch.isfinite(s.gradient([PARIS])).all()
    ch = {"paris": 0.05, "london": -0.05}
    assert float(_miss(s, s.steer(ch), ch).max()) <= S.TOLERANCE + 1e-6
    if family not in ("smollm3", "olmo3"):
        assert f._identities["ok"] and max(v for k, v in f._identities.items() if k != "ok") < 1e-4


def test_a_bf16_model_is_read_in_float32():
    """A model in bfloat16 runs its body in bfloat16; its final norm and head, and the frame's own algebra, run in float32."""
    m16 = hf_model("smollm3").to(torch.bfloat16)
    f = fresh(m16)
    s = f.state(TEXT)
    z = s[L].logits()
    norm = copy.deepcopy(m16.model.norm).float()
    want = _centred((norm(s._xs[L][0].float()) @ m16.lm_head.weight.float().T) * f._m.output_scale)
    assert z.dtype == torch.float32 and torch.allclose(z, want, atol=1e-4)
    assert m16.lm_head.weight.dtype == torch.bfloat16  # the model itself is untouched
    moved = s[2, -1].steer({PARIS: 0.5, LONDON: -0.5})
    assert torch.isfinite(moved[L, -1].logits()).all() and torch.isfinite(s[2, -1].basis([PARIS])).all()


def test_weight_changes_refresh_the_head_caches(tmp_path):
    f = fresh(hf_model("smollm3"), cache=str(tmp_path))
    f.state(TEXT)[-1, -1].basis([PARIS])
    wbar = f._head_mean().clone()
    with torch.no_grad():
        f._m.lm_head.weight.add_(1.0)
    f.state(TEXT)[-1, -1].basis([PARIS])
    assert torch.allclose(f._head_mean(), wbar + 1.0, atol=1e-4)


# ---- logits

def test_logits(frame):
    """At the output layer, the model's own centred logits; at position 0, which sees only itself, the output at every layer;
    elsewhere not. A selection of labels is a slice of the whole vocabulary's, and the logits are deterministic."""
    s = frame.state(TEXT)
    z = s.logits()
    assert z.shape == (L + 1, T, V)
    assert torch.allclose(z[L], _centred(logits(frame._m, s._xs[L][0])), atol=1e-5)
    assert torch.allclose(s.logits([PARIS, CITY]), z[..., [PARIS, CITY]], atol=1e-5)
    kl = kl_of(s)
    assert kl[L].max() < 1e-5 and kl[:, 0].max() < 1e-5 and kl[1:L, 1:].min() > 1e-6
    assert torch.equal(frame.state(TEXT).logits(), z)


def test_logits_depend_on_the_state_alone(frame):
    """A state carried into another text has the same logits as in its own text; carried forward, it sees its new context."""
    other = "the river is located near the city of"
    a, b = frame.state(TEXT)[2], frame.state(other)[2]
    swapped = a + (b.vector - a.vector)
    assert torch.allclose(swapped[2].logits(), b.logits(), atol=1e-4)
    assert torch.allclose(swapped[L].vector, frame.state(other)[L].vector, atol=1e-4)
    last = a[-1] + (b.vector[-1] - a.vector[-1])
    assert torch.allclose(last[2, -1].logits(), b.logits()[-1], atol=1e-4)
    assert not torch.allclose(last[L, -1].logits(), frame.state(other)[L, -1].logits(), atol=1e-2)


def test_more_samples_converge(frame):
    """More integration points converge; the batching of positions does not change the logits or the gradients."""
    f = lf.LexFrame(frame._m)
    f.calibration = frame.calibration
    p = {M: torch.softmax(f.config(samples=M).state(TEXT)[1].logits(), -1) for M in (16, 256)}
    ref = torch.softmax(f.config(samples=1024).state(TEXT)[1].logits(), -1)
    assert float((p[256] - ref).abs().max()) < float((p[16] - ref).abs().max())
    small = lf.LexFrame(frame._m)
    small.calibration, small._rows = frame.calibration, 64  # one position at a time
    s, t = small.state(TEXT)[[1, 2]], f.config(samples=64).state(TEXT)[[1, 2]]
    assert torch.allclose(s.logits(), t.logits(), atol=1e-5)
    assert torch.allclose(s[1].gradient([PARIS]), t[1].gradient([PARIS]), atol=1e-5)


# ---- gradient

def test_gradient_is_the_models(frame):
    """Its inner product with a change of the state is the change of the model's output logit; at the output it is the head's
    own; it needs no calibration."""
    s0 = frame.state(TEXT)
    d = s0[1, T - 1].gradient([PARIS, CITY])
    delta = torch.randn(frame._m.hidden_dim, generator=torch.Generator().manual_seed(3))
    measured = _slope(s0, 1, T - 1, delta, PARIS)
    assert abs(float(measured - d[0] @ delta)) < 2e-2 * abs(float(measured))
    x = s0._xs[L][0, T - 1].float().clone().requires_grad_(True)
    z = logits(frame._m, x)
    (own,) = torch.autograd.grad(z[PARIS] - z.mean(), x)
    assert torch.allclose(s0[L, T - 1].gradient([PARIS])[0], own, atol=1e-5)
    bare = lf.LexFrame(frame._m.hf, tokenizer())
    assert torch.allclose(bare.state(TEXT)[1, T - 1].gradient([PARIS]), d[:1], atol=1e-5)


# ---- basis

def test_bases_compose_the_state(frame):
    """Every state is the sum of the bases of the whole vocabulary weighted by its logits, and its projection on a basis is that
    logit, at every layer."""
    s = frame.state(TEXT)[:, [1, T - 1]]
    B = s.basis(None)
    h, c = s.vector, s.logits()
    assert B.shape == (L + 1, 2, V, frame._m.hidden_dim)
    assert torch.allclose((c[..., None] * B).sum(-2), h, atol=1e-4 * float(h.abs().max()))
    proj = (B @ h[..., None])[..., 0] * (c**2).sum(-1, keepdim=True) / (h * h).sum(-1, keepdim=True)
    assert torch.allclose(proj, c, atol=1e-4 * float(c.abs().max()))


def test_bases_have_unit_gain(frame):
    """A move along a label's basis changes that label's output logit by the step, to first order, and each label's basis is its
    own, whatever other labels are asked for."""
    s = frame.state(TEXT)[:, [1, T - 1]]
    B = s.basis(LABELS)
    gain = (s.gradient(LABELS) * B).sum(-1)
    assert torch.allclose(gain, torch.ones_like(gain), atol=1e-4)
    assert torch.allclose(s.basis([LONDON])[..., 0, :], B[..., 1, :], atol=1e-5 * float(B.abs().max()))


def test_bases_in_blocks(frame, monkeypatch):
    """The bases are the same whether the positions go together or apart."""
    import lexframe._basis as BA
    import lexframe._state as S

    s = frame.state(TEXT)[[1, L], [1, 4, T - 1]]
    whole = s.basis(LABELS)
    monkeypatch.setattr(S, "JFLOATS", 2 * frame._m.hidden_dim**2)
    monkeypatch.setattr(BA, "GFLOATS", 2 * V * frame._m.hidden_dim)
    assert torch.allclose(s.basis(LABELS), whole, atol=1e-5 * float(whole.abs().max()))


# ---- moves

def test_moving_a_state(frame):
    """A move is carried forward through the model: the moved run is the same below the moved layer, and its output is the model's
    output after the move; moves at two layers compose in increasing order of layer."""
    s0 = frame.state(TEXT)
    s = s0[2]
    g = s.gradient([PARIS, LONDON])
    v = 2 * g[:, 0] - 2 * g[:, 1]
    moved = s + v
    assert moved.layers == list(range(L + 1)) and moved.positions == list(range(T)) and repr(moved).endswith("moved=True)")
    assert torch.allclose(moved[2].vector, s.vector + v, atol=1e-5) and torch.equal(moved[1].vector, s0[1].vector)
    X = s._xs[2].clone()
    X[0] = (X[0].float() + v).to(X.dtype)
    assert torch.allclose(moved[L, T - 1].logits(), _output(frame, s._xs, 2, X, T - 1), atol=1e-4)
    two = s0[[1, 3], -1] + torch.ones(2, frame._m.hidden_dim)
    assert torch.allclose(two[1, -1].vector, s0[1, -1].vector + 1, atol=1e-5)
    assert not torch.allclose(two[3, -1].vector, s0[3, -1].vector + 1, atol=1e-3)


def test_moves_below_a_moved_layer(frame):
    """A moved run is the model with its moves added: every member works at every layer of it, and a further move, at any layer,
    keeps the earlier ones as the changes they made."""
    s = frame.state(TEXT)
    up = s[3, T - 1] + 0.1 * s[3, T - 1].vector
    kept = lambda a, b: torch.equal(a._moves[3], b._moves[3])  # noqa: E731
    delta = torch.randn(frame._m.hidden_dim, generator=torch.Generator().manual_seed(5))
    low, plain = up[1, T - 1] + 1e-3 * delta, s[1, T - 1] + 1e-3 * delta
    assert kept(low, up) and torch.allclose(low[3, T - 1].vector, plain[3, T - 1].vector + up._moves[3][0, T - 1], atol=1e-5)
    g = up[1, T - 1].gradient([PARIS])[0]  # the moved run's: it predicts the moved run's output
    measured = _slope(up, 1, T - 1, delta, PARIS)
    assert abs(float(measured - g @ delta)) < 2e-2 * abs(float(measured))
    assert not torch.allclose(g, s[1, T - 1].gradient([PARIS])[0], atol=1e-4)
    ch = {PARIS: 0.05}
    again = up[1, [T - 1]].steer(ch)
    assert kept(again, up) and float(_miss(up[1, [T - 1]], again, ch).max()) <= S.TOLERANCE + 1e-6
    cut = up[2, T - 1].ablate()
    assert kept(cut, up) and torch.allclose(cut[2, T - 1].vector, up[1, T - 1].vector, atol=1e-6)


def test_steer(frame):
    """steer() moves one layer so that the model's output logits change by the given amounts, at the selected positions only; the
    moved run can be steered again; a change beyond reach stops short with a warning."""
    s0 = frame.state(TEXT)
    s = s0[2]
    ch = dict(zip(LABELS, [0.1, -0.1, 0.05]))
    tol = S.TOLERANCE
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        moved = s.steer(ch)
    assert float(_miss(s, moved, ch).max()) <= tol + 1e-6
    again = moved[3].steer({PARIS: 0.05})
    assert float(_miss(moved[3], again, {PARIS: 0.05}).max()) <= tol + 1e-6 and torch.equal(again[2].vector, moved[2].vector)
    one = s0[2, [T - 1]].steer({PARIS: 0.1})
    assert torch.equal(one._xs[2][0, : T - 1], s0._xs[2][0, : T - 1]) and not torch.equal(one._xs[2][0, -1], s0._xs[2][0, -1])
    far = {PARIS: 2.0, LONDON: -2.0}  # beyond the reach of a rotation in these random models
    with pytest.warns(UserWarning, match="largest relative error"):
        short = s0[3, [T - 1]].steer(far)
    assert tol < float(_miss(s0[3, [T - 1]], short, far).max()) < 1


@pytest.mark.parametrize("position", [-1, 3])
def test_ablate(frame, position):
    """ablate() removes what the previous layer (or its attention or MLP part) wrote at the selected positions and carries the run
    forward: the output is the model's own after the removal."""
    m = frame._m
    hL, xs, mid = forward(m, m.ids(TEXT), mid=True)
    xs = xs + [hL]
    t = position % T
    s = frame.state(TEXT)
    for k in range(1, L + 1):
        for part in ("all", "attention", "mlp"):
            write = {"all": xs[k] - xs[k - 1], "attention": mid[k - 1] - xs[k - 1], "mlp": xs[k] - mid[k - 1]}[part]
            removed = s[k, t].ablate(part)
            X = xs[k].clone()
            X[0, t] -= write[0, t]
            assert torch.allclose(removed[k].vector, X[0].float(), atol=1e-5)
            assert torch.allclose(removed[L, t].logits(), _output(frame, xs, k, X, t), atol=1e-4)


def test_ablate_a_source(frame):
    """Removing what a layer's attention read from one source position: the layer runs again with that source's share taken out of
    its attention output, exactly as the model does when the share is subtracted inside the attention module."""
    m = frame._m
    ids = m.ids(TEXT)
    s = frame.state(TEXT)
    k, tpos = 1, T - 1
    for p in (0, 3, tpos):
        got = s[k + 1, tpos].ablate(source=p)[k + 1].vector[-1]
        u_in, a_out = {}, {}
        forward(m, ids, on_attention=lambda l, u, a: (u_in.__setitem__(l, u[0]), a_out.__setitem__(l, a[0, tpos])) if l == k else None)
        C = _ablate.sources(m, k, u_in[k], tpos, a_out[k].float())[0]

        def hook(module, args, out, p=p):
            o = (out[0] if isinstance(out, tuple) else out).clone()
            o[0, tpos] = (o[0, tpos].float() - C[p]).to(o.dtype)
            return (o, *out[1:]) if isinstance(out, tuple) else o

        handle = m.layers[k].self_attn.register_forward_hook(hook)
        try:
            want = forward(m, ids)[1][k + 1][0, tpos].float()
        finally:
            handle.remove()
        assert torch.allclose(got, want, atol=1e-4)


# ---- the surface

def test_indexing(frame):
    s = frame.state(TEXT)
    z = s[1:3].logits()
    assert s[1:3, -1].positions == [T - 1] and torch.allclose(s[1:3, -1].logits(), z[:, -1], atol=1e-5)
    assert s[:, -1].vector.shape == (L + 1, frame._m.hidden_dim) and s[:, [-1]].vector.shape == (L + 1, 1, frame._m.hidden_dim)
    two = s[1, [2, 0]]
    assert two.positions == [2, 0] and torch.allclose(two.logits(), z[0, [2, 0]], atol=1e-5)
    assert s[:, torch.tensor(0)].positions == [0] and s[1:3].layers == [1, 2]
    assert torch.equal(s[2].logits(["paris", "london"]), s[2].logits([PARIS, LONDON]))
    assert label_id(frame._m, "paris") == PARIS


def test_refusals(frame, kind):
    bare = lf.LexFrame(hf_model(kind), tokenizer())
    with pytest.raises(RuntimeError):
        bare.state(TEXT)[2].logits()
    with pytest.raises(RuntimeError):
        bare.state(TEXT)[2, -1].basis([PARIS])
    assert bare.state(TEXT)[L, -1].basis([PARIS]).shape == (1, frame._m.hidden_dim)  # the output layer needs no calibration
    assert bare.state(TEXT)[2, -1].ablate()[L, -1].logits().shape == (V,)
    s = frame.state(TEXT)
    for bad in (lambda: frame.state([TEXT, TEXT]), lambda: s[2].logits(["paris london"]),
                lambda: s[[1, 3]].steer({PARIS: 0.1}), lambda: s[[1, 2], -1].ablate(), lambda: s[0, -1].ablate(),
                lambda: s[1, -1].ablate("block"), lambda: s[2, -1].ablate("mlp", source=0)):  # fmt: skip
        with pytest.raises(ValueError):
            bad()
    for bad in (lambda: s[1][0, 0], lambda: s[[]], lambda: s[[3, 1]], lambda: s[[2, 2]], lambda: s[:, T]):
        with pytest.raises(IndexError):
            bad()
    with pytest.raises(ValueError, match="too short"):
        lf.LexFrame(hf_model(kind), tokenizer()).config(cache=None, length=200, progress=False).init(["paris london"])


def test_config(frame):
    f = lf.LexFrame(frame._m)
    assert f._settings == {"samples": 64, "size": 1024, "length": 256, "cache": "", "progress": True}
    assert f.config(samples=32, size=8, length=64, cache=None, progress=False) is f
    assert f._settings == {"samples": 32, "size": 8, "length": 64, "cache": None, "progress": False}
    for bad in ({"samples": 0}, {"samples": 2.5}, {"size": 1}, {"length": 1}):
        with pytest.raises(ValueError):
            f.config(**bad)
    with pytest.raises(TypeError, match="unexpected"):
        f.config(steps=30)  # no hidden knobs


def test_public_api():
    from lexframe._frame import State

    assert set(lf.__all__) == {"LexFrame", "__version__"}
    public = lambda cls: {n for n in vars(cls) if not n.startswith("_")}  # noqa: E731
    assert public(lf.LexFrame) == {"num_layers", "config", "init", "state"}
    assert public(State) == {"layers", "positions", "vector", "logits", "gradient", "basis", "steer", "ablate"}
