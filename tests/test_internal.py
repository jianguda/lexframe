"""Tests of the internal helpers and research tools (lexframe._tools; not part of the public API)."""

import numpy as np
import pytest
import torch
from conftest import TEXT, fresh, hf_model

from lexframe import _tools as TL

V, L = 256, 4
TOK = ["paris", "london", "city"]


def test_undamped_frame_pairing_is_layer_invariant(frame):
    """The pairings of the undamped frame vectors with the coframe are the same at every layer and equal their value at the output;
    the damping of the applied moves breaks that by design."""
    s = frame.state(TEXT)
    p = TL.pairing(s, TOK)
    assert p.realised.shape == p.ideal.shape == (L + 1, 8, 3, 3) and p.deviation.shape == (L + 1, 8)
    assert torch.allclose(p.ideal, p.ideal[-1].expand_as(p.ideal), atol=1e-5)  # the ideal is computed at the output
    assert torch.allclose(torch.diagonal(p.realised, dim1=-2, dim2=-1), torch.ones(L + 1, 8, 3), atol=1e-4)
    assert float(p.deviation[1:].max()) < 1e-3
    M = s.gradient(TOK) @ s.direction(TOK).mT  # the damped frame vectors: [layer, position, u, v]
    damped = (M / torch.diagonal(M, dim1=-2, dim2=-1)[..., None, :] - p.ideal).abs().amax((-1, -2))
    assert float(damped[1:L].min()) > 30 * float(p.deviation[1:L].max())  # the damping shrinks the weak directions of each chart differently


def _effect(frame, layer, tokens, exact=False):
    """The first-order change of all the labels' centred logits along each direction, gradient_u . direction_v [position, v, u]."""
    s = frame.state(TEXT)[layer]
    B = TL.exact_frame(s, tokens) if exact else s.direction(tokens)
    return B @ s.gradient(tokens).mT


def test_output_effect_is_the_same_at_every_layer(frame):
    e2, e3 = _effect(frame, 2, TOK, exact=True), _effect(frame, 3, TOK, exact=True)
    assert torch.allclose(torch.diagonal(e2, dim1=-2, dim2=-1), torch.ones(8, 3), atol=1e-4)
    assert float((e2 - e3)[1:].abs().max()) < 1e-3
    r2, r3 = _effect(frame, 2, TOK), _effect(frame, 3, TOK)  # the damped frame vectors: their cross-talk differs by layer
    assert float((r2 - r3)[1:].abs().max()) > 10 * float((e2 - e3)[1:].abs().max())


@pytest.mark.parametrize("family", [f for f in __import__("conftest").FAMILIES if f not in ("smollm3", "olmo3")])
def test_families(family):
    f = fresh(hf_model(family))
    s = f.state(TEXT)[[1, 2]]
    p = TL.pairing(s, TOK)
    assert torch.isfinite(p.realised).all() and float(p.deviation[-1].max()) < 1e-2
    mv = f.state(TEXT)[2].steer({"paris": 0.05, "london": -0.05})
    assert float(mv.error.max()) <= 0.02 + 1e-6


def test_steer_falls_back_to_the_path(frame):
    """Where the corrected step misses, it is discarded and the path from the model's own state is taken: no position ends above the
    tolerance without having followed the path."""
    s = frame.state(TEXT)[1]
    info = s.steer({"paris": 0.3, "london": -0.3})._info[1]
    path = torch.tensor([r == "path" for r in info["regime"]])
    assert bool(((info["error"] <= 0.02 + 1e-6) | path).all())
    assert set(info["regime"]) <= {"oneshot", "path"}

def test_weight_stamp_refreshes_the_reference_and_warns_on_a_stale_calibration(tmp_path):
    """The cached output-side factors (head mean, head Gram) follow in-place changes of the head; a change of any weight after init()
    makes the calibrated mean transport stale, which is reported."""
    f = fresh(hf_model("smollm3"), cache=str(tmp_path))
    f.state(TEXT)[1].decode(context=False).top()
    wbar = f._head_mean().clone()
    with torch.no_grad():
        f._m.lm_head.weight.add_(1.0)
    with pytest.warns(UserWarning, match="weights changed"):
        f.state(TEXT)[1].decode(context=False).top()
    assert torch.allclose(f._head_mean(), wbar + 1.0, atol=1e-4)


def test_downstream_mirrors_the_upstream(frame):
    """What later positions took from a position is the split by source of each later position, read from the source."""
    down = frame.trail(TEXT, position=3).downstream()
    assert down.shape == (L, 4)
    for later in (4, 6):
        assert np.allclose(down[:, later - 4], frame.trail(TEXT, position=later).upstream()[:, 3])


def test_reverse_mode_where_forward_mode_is_missing(frame, monkeypatch):
    """Without forward mode (Apple MPS), J V comes from two reverse passes and the column Jacobian by rows: the same numbers."""
    from lexframe import _state as _basis
    from lexframe._model import forward

    m = frame._m
    hL, xs, _ = forward(m, m.ids(TEXT))
    xs = xs + [hL]
    kv = _basis.references(m, xs, 2)
    V = torch.randn(3, *xs[2].shape[1:], generator=torch.Generator().manual_seed(8)).to(xs[2])
    f = lambda h: m.final_norm(h)
    monkeypatch.setattr(_basis, "FORWARD_AD", True)
    a, Ja = _basis.jvp(m, 2, kv, xs[2][:1], V), TL.column_jacobian(f, xs[L][:1], [0, 5], 16)
    monkeypatch.setattr(_basis, "FORWARD_AD", False)
    b, Jb = _basis.jvp(m, 2, kv, xs[2][:1], V), TL.column_jacobian(f, xs[L][:1], [0, 5], 16)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5) and torch.allclose(Ja, Jb, rtol=1e-4, atol=1e-5)


def test_step_audit(frame):
    names = [n for n, _ in frame._m.hf.named_parameters()]
    last = [n for n in names if f"layers.{L - 1}." in n and "down_proj" in n]
    first = [n for n in names if "layers.0." in n and "o_proj" in n]
    g = torch.Generator().manual_seed(2)
    P = dict(frame._m.hf.named_parameters())
    delta = {n: 1e-2 * P[n].detach().std() * torch.randn(P[n].shape, generator=g) for n in last + first}
    a = TL.step_audit(frame, TEXT, delta, target="next")
    assert set(a) == {"joint", 0, L - 1}
    assert abs(a[L - 1]["context_share"]) < 1e-2  # the last layer's write reaches no other position (to central-difference rounding)
    assert abs(a[L - 1]["curvature_ratio"] - 1) < 1e-2
    assert abs(a[0]["context_share"]) > 5 * abs(a[L - 1]["context_share"])
    assert abs(a[0]["context_share"]) > 1e-3 and a[0]["curvature_ratio"] > 0
    assert 0 <= a["joint"]["useful_fraction"] <= 1 + 1e-6
    # the gain is the derivative of the mean log-probability of the next tokens along the step
    from lexframe._model import forward, logprobs
    from lexframe._tools import swapped

    ids = frame._m.ids(TEXT)
    y = ids[0, 1:]

    def ll(sgn, e=1e-1):
        with swapped(frame._m.hf, {n: P[n].data + sgn * e * d for n, d in delta.items()}):
            return logprobs(frame._m, forward(frame._m, ids)[0][0])[:-1].gather(1, y[:, None]).mean()

    fd = float((ll(1) - ll(-1)) / 2e-1)
    assert abs(fd - a["joint"]["gain"]) < 2e-2 * abs(fd) + 1e-6
