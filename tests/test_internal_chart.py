"""Tests of the internal chart helpers and numerics in lexframe._tools."""

import torch
from conftest import TEXT

from lexframe import _tools as TL
from lexframe._model import forward, kl
from lexframe._tools import tail_mass

TOK = ["paris", "london", "city"]


def test_staged_chart_is_exact(frame):
    """Through the factors, the undamped frame vectors deliver exactly the orthogonal projection of each token's output vector at every
    layer (the pairings equal their output-side value); the scalars carry the conditioning (the triangular coupling is well conditioned)."""
    c = TL.chart(frame, TEXT)
    assert c.layers == list(range(frame.num_layers + 1))
    for k in c.layers:
        b = c.frame(TOK, k)
        assert b["deviation"] < 1e-4
        assert torch.allclose(torch.diagonal(b["realised"]), torch.ones(3, dtype=b["realised"].dtype), atol=1e-4)
        assert float((b["reach"] - 1).abs().max()) < 1e-6
        assert c.graded_cond[k] < 1e4
        assert float(c.lyapunov[k][0]) == 0.0 and bool((c.lyapunov[k][1:] <= c.lyapunov[k][:-1] + 1e-12).all())
    assert c.temperature > 0 and abs(float(c.direction.norm()) - 1) < 1e-5


def test_graded_truncation_is_a_projection(frame):
    c = TL.chart(frame, TEXT, layers=[0, 1])
    k = 0
    full = float(c.lyapunov[k][-1])  # log10 of the smallest scale relative to the largest
    prev = None
    for tol in (10 ** (full / 4), 10 ** (full / 2), 10 ** (3 * full / 4)):
        b = c.frame(TOK, k, tol=tol)
        assert float(b["reach"].max()) <= 1 + 1e-5
        if prev is not None:  # a smaller tolerance keeps more directions: reach does not fall
            assert bool((b["reach"] >= prev - 1e-5).all())
        prev = b["reach"]


def test_normal_space_is_the_left_null_space(frame):
    """The output directions no move makes are the orthogonal complement of the final norm's range: the projection of an output
    target onto what a move can reach is orthogonal, not oblique along h."""
    m = frame._m
    hL, xs, _ = forward(m, frame._one(TEXT))
    xs = xs + [hL]
    for t in (0, 3):
        JN, U, Nm, *_ = TL.tangent(m, xs, t)
        assert Nm.shape[1] == 1 and U.shape[1] == m.hidden_dim - 1
        sv = torch.linalg.svdvals(JN.double())
        assert abs(float((Nm.double().T @ JN.double()).norm()) - float(sv[-1])) < 1e-6 * float(sv[0])  # the epsilon-sized gauge value
        assert float(sv[-1]) < 1e-2 * float(sv[-2])


def test_kl_is_free_of_cancellation():
    """KL of two nearby outputs from float32 log-probabilities: the naive sum is rounding noise (often negative) below ~1e-6 nats, the
    centred form is accurate and never negative; identical inputs give exactly zero."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(16, 20000, generator=g, dtype=torch.float64) * 4
    dz = 1e-4 * torch.randn(16, 20000, generator=g, dtype=torch.float64)
    ref = kl(torch.log_softmax(z, -1), torch.log_softmax(z + dz, -1))
    lp, lq = torch.log_softmax(z.float(), -1), torch.log_softmax((z + dz).float(), -1)
    fixed = kl(lp, lq).double()
    naive = (lp.exp() * (lp - lq)).sum(-1).double()
    assert float(((fixed - ref).abs() / ref).max()) < 2e-2  # about 1%: the float32 log-probabilities limit it at ~1e-9 nats
    assert float(((naive - ref).abs() / ref).median()) > 10 * float(((fixed - ref).abs() / ref).median())
    assert bool((kl(lp, lp) == 0).all()) and bool((fixed >= 0).all())


def test_tail_mass_of_a_confident_label():
    z = torch.zeros(2, 1000)
    z[:, 7] = torch.tensor([25.0, 5.0])
    y = torch.tensor([7, 7])
    ref = 1 - torch.softmax(z.double(), -1)[:, 7]
    t = tail_mass(z, y).double()
    assert float(((t - ref).abs() / ref).max()) < 1e-5
    assert float(1 - torch.softmax(z, -1)[0, 7]) == 0.0  # the naive complement collapses to zero in float32


def test_factored_frame_closes(frame):
    """The undamped frame in factored form (orthonormal frame x shared log-scales x a finite token profile): it reproduces the frame vectors,
    decomposition gives finite coefficients that do not depend on the layer, and synthesis through the factors reproduces the
    displacement's first-order logit change exactly."""
    c = TL.chart(frame, TEXT)
    g = torch.Generator().manual_seed(0)
    delta = torch.randn(frame._m.hidden_dim, generator=g)
    cs = []
    for k in c.layers[:-1]:
        F = c.factored(TOK, k)
        s = F["Q"] @ ((F["sign"] * torch.exp(F["log_scale"]))[:, None] * F["profile"].T)  # rebuilt hidden vectors [d, n]
        b = c.frame(TOK, k)
        cos = torch.nn.functional.cosine_similarity(s.T.double(), b["vectors"].double(), dim=-1)
        assert float(cos.min()) > 0.999 and bool(torch.isfinite(F["profile"]).all())
        dec = c.decompose(delta, TOK, k)
        assert dec["check"] < 1e-4
        cs.append(dec["c"])
    # the pairing (hence the coefficient map from the coframe's change) is the output-side one at every layer
    G = [c.decompose(delta, TOK, k)["pairing"] for k in c.layers[:-1]]
    for Gk in G[1:]:
        assert torch.allclose(Gk / Gk.diagonal()[None], G[0] / G[0].diagonal()[None], atol=1e-4)


def test_chart_factors_kept_on_cpu(frame):
    a = TL.chart(frame, TEXT, layers=[1, 2])
    b = TL.chart(frame, TEXT, layers=[1, 2], store="cpu")
    assert all(r.device.type == "cpu" for r in b.staged.R.values())
    for k in (1, 2):
        assert torch.allclose(a.lyapunov[k], b.lyapunov[k], atol=1e-5)
        assert b.frame(TOK, k)["deviation"] < 1e-4
