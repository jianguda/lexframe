# Mechanism

What a `State` returns: the definition of each quantity, its properties and how it is computed.

At the output, the logit of a token $v$ is $z_v = w_v^\top N(x_L)$, with the last hidden state $x_L$, the final normalisation $N$ and
the unembedding row $w_v$. A centred logit subtracts the mean over the vocabulary; centring log-probabilities gives the same values.

The three quantities are `logits`, what a state $x_k(t)$ of layer $k$ and position $t$ carries in these coordinates; `gradient`, how
the model's output responds to it; and `basis`, their geometric carrier: a state's projections on its semantic bases are its logits,
and the bases' covectors are the gradients. The moves (`steer`, `ablate`, `+`, `-`) run the model itself.

## Quantities

### `logits`

**Definition.** The state runs on its own column through the remaining layers: norms, MLPs and residual connections as the model
computes them. Other positions enter a layer only through its attention write, so each write is replaced by its expectation given
the column's own attention input $u$, plus an independent Gaussian innovation for its spread:

$$a_\ell = \bar a_\ell + G_\ell\,(u - \bar u_\ell) + o_\ell[\beta(t)] + \varepsilon_\ell, \qquad \varepsilon_\ell \sim \mathcal N(0, \Sigma_\ell).$$

$R_k = \mathbb E_\varepsilon[\mathrm{softmax}(z(h_L))]$ is the mean next-token distribution over the innovations, and `logits` returns
$\log R_k$ centred over the vocabulary, so that its softmax is $R_k$.

**Properties.**
- They are a function of $x_k(t)$ and its position $t$ alone: a state carried into another text at the same position has the same
  logits.
- At the output, $R_L$ is the model's own distribution; below it, it is not. With the full context the state of any layer fixes the
  model's output exactly, so logits that tell layers apart must replace what the later layers read from the context, as above.
- Position 0 attends only to itself, so its writes are exact and its logits are the output at every layer.
- They are deterministic: the integration points are fixed (see Calibration).

**Computation.** `samples` columns per state, one per integration point, run through the remaining layers in batches; every layer
below the output is read in one pass.

### `gradient`

**Definition.** The gradient of each label's centred logit at the output with respect to the state, through the model's own
remaining layers with the context fixed: $\nabla_{x_k(t)} \bigl(z_v - \bar z\bigr)$, the other states held where they are.

**Properties.** Its inner product with a small change of the state is the change of the model's centred output logit.

**Computation.** The state is carried on its own column through the remaining layers, attending to the states of its prefix, and
differentiated by one backward pass for all labels and positions together.

### `basis`

**Definition.** One coefficient function per state $x_k$,

$$c(h) = \log R_k(x_k) + z(h) - z(x_k) \qquad \text{(centred over the vocabulary)},$$

with $R_k$ as in `logits` and $z$ the model's output log-probabilities, the context fixed. At the state the coefficients are its
logits, and their gradients $g_v = \nabla c_v$ are the model's own, as `gradient` returns. At the output layer $c$ is $z$.

The bases are the polar basis of $c$ at $x_k$, split along the state and across it. A state is read by its direction: the readout
covectors are almost orthogonal to it (less than 0.1% of their energy is radial in the models we measured). So each basis is a radial
part, which composes the state, and a tangential part, which moves it. With $\hat x = x_k / \lVert x_k \rVert$,
$\rho_v = g_v^\top \hat x$ and $p_v = g_v - \rho_v \hat x$,

$$b_v = r_v\, \hat x + \beta_v\, p_v + c_v\, \lambda, \qquad r_v = \frac{c_v \lVert x_k \rVert}{\lVert c \rVert^2},$$

where the tangential parts $\beta_v p_v + c_v \lambda$ are the shortest, over the whole vocabulary together, that satisfy the unit gain
and the cancellation below.

**Properties.**
- **Composition.** The state is the sum of the bases of the whole vocabulary weighted by its coefficients,
  $\sum_v c_v b_v = x_k$: the radial parts add up to the state, $\sum_v c_v r_v = \lVert x_k \rVert$, and the tangential parts
  cancel, $\sum_v c_v (\beta_v p_v + c_v \lambda) = 0$.
- **Projection.** The projection of the state on a basis is its coefficient, its logit:
  $\langle x_k, b_v \rangle \lVert c \rVert^2 / \lVert x_k \rVert^2 = c_v$.
- **Unit gain.** $\langle g_v, b_v \rangle = 1$: a move along $b_v$ changes the model's output logit of $v$ by the step, to first
  order.
- **Its own.** Each label's basis is the same whatever other labels are asked for, and every label of the vocabulary has one.
- **Not orthogonal.** A move along one basis also moves the other labels' logits a little: with a vocabulary much larger than the
  hidden size, no set of bases can avoid it.

**Computation.** The gradients of the whole vocabulary come from the Jacobian of the final normalisation's output with respect to the
state (one backward pass per hidden dimension, for several positions together) and the head. The tangential parts solve one system of
the hidden size,

$$\Bigl(\lVert c \rVert^2 I - \sum_v c_v^2\, \hat p_v \hat p_v^\top\Bigr) \lambda = -\sum_v \frac{c_v a_v}{\lVert p_v \rVert^2}\, p_v,
\qquad \beta_v = \frac{a_v - c_v\, p_v^\top \lambda}{\lVert p_v \rVert^2}, \qquad a_v = 1 - r_v \rho_v,$$

which the wide cone of the vocabulary's covectors keeps well conditioned, solved with one step of refinement (the sums over the
vocabulary lose float32 digits). A basis is returned as the vector itself, `[..., label, hidden]`.

## Operations

### Moves and the moved run

**Definition.** A move, by `steer`, `ablate`, `+` or `-`, changes some states of one layer, or of several, and carries the change
forward through the model's own layers, every position attending to the moved states of its prefix. The result is the State of the
moved run: the model with the move added. Each move is kept as the change it made to the states of its layer, and added again
wherever the run is carried through that layer.

**Properties.**
- The moved run is the same as the run below the moved layer. A position's move changes no earlier position.
- Every member works at every layer of a moved run. Below a moved layer, `gradient` and `basis` are those of the model with the
  moves added: a move is a fixed change, so it shifts the run and passes the gradient through unchanged.
- A further move, at any layer, keeps the earlier moves as the changes they made; it does not solve them again.

**Computation.** One forward pass from the lowest new move, the earlier moves added again above it.

### `steer`

**Definition.** Each selected state $x$ of one layer is moved on the sphere of its norm to solve

$$\min_x D(x) = \mathrm{KL}(q_0 \,\Vert\, q) \quad\text{subject to}\quad \Delta z_S = w,\ \ \lVert x \rVert = \lVert x_0 \rVert,$$

with $S$ the steered labels, $w$ their changes of centred output logit, $q$ the output's distribution over the other labels
(renormalised) and $q_0$ that before the move. The norm is kept because the layers read a state's direction, and its norm only weighs
it against later writes: scaling it is a channel the model's states do not use. The other labels are weighed by their probability.

**Properties.**
- Over the others' logits $z$, $D = -q_0^\top z + \log \sum e^{z}$ up to a constant: its gradient is $q - q_0$ and its curvature the
  Fisher metric of $q$, $\mathrm{diag}\,q - q q^\top$, exactly.
- The other positions stay as they are. The positions are moved from left to right: a position's move changes no earlier position's
  output, so each is solved exactly given the moves before it.
- A change beyond the reach of a rotation stops short, with a warning.

**Computation.** A trust-region SQP method (Byrd and Omojokun's composite step). With $J$ the Jacobian of the final normalisation's
output with respect to the state (the bases' Jacobian), $K$ the Fisher metric in that space (one pass over the head),
$M = J^\top K J$, $g$ the gradient of $D$ and $R$ the labels' gradients, all orthogonal to the state:
- **Step.** Within the radius $\Delta$, the normal step $n = R^+ r$, the shortest rotation towards the change $r$ still to make, cut
  to $0.8\,\Delta$, plus the tangential step $t$ with $R\,t = 0$ and $\lVert t \rVert \le (\Delta^2 - \lVert n \rVert^2)^{1/2}$
  that makes $(g + M n)^\top t + \tfrac12\, t^\top M t$ least, solved exactly in the eigenbasis of $M$; the sum is brought back to
  the sphere. The trust region keeps the problem well posed where the metric is flat (the moves that change only improbable labels).
- **Acceptance.** Each step is measured on the moved run's own output, the change first. While the change is not made within the
  tolerance (a relative error of 2%), a step is taken where it brings the change still to make down. Once it is, a step is taken
  where the change stays made and $D$ comes down.
- **Radius.** It follows the ratio of the actual to the predicted decrease of what is judged: quartered below 1/4, doubled above 3/4
  at the boundary, from the length of the shortest rotation up to the state's norm. $J$ and $M$ are taken again where a step with an
  older $J$ disappoints.
- **Stop.** Where the change is made within the tolerance and $D$'s predicted decrease is within the tolerance of $D$, or where the
  radius falls below the working precision.

### `ablate`

**Definition.** On the State of layer $k$, `ablate()` removes what layer $k-1$ wrote at the selected positions: its whole write, so
that the state of layer $k$ is that of layer $k-1$, or the write of its attention or its MLP sublayer. With a `source` position, layer
$k-1$ runs again at each selected position without what its attention read from that source. The result is carried forward like any
move.

**Properties.** The change of a label's output logit in the moved run is the write's effect on it.

**Computation.** The layer runs once on the run's own states, with its attention output and the state between its sublayers taken as
the model computes them. For a source, the attention write is split by source position (the attention weights times the values,
through the output projection), the source's share is taken out, and the layer runs again on what is left, so that whatever the layer
does to its attention output before the residual, and its MLP, respond as in the model.

## Calibration

`init()` fits the expected writes of `logits` on unlabelled text: `size` blocks of `length` tokens (1024 of 256 by default), one
from each of `size` evenly spaced texts.
- **Expected write.** One pass gives, per layer, the ridge regression $G_\ell$ of the attention write on its input, with the means
  $\bar u_\ell$ and $\bar a_\ell$. The ridge is the MacKay evidence fixed point, started from two-fold cross-validation.
- **Spread.** A second pass gives the residuals, position 0 excluded:
  - the offsets $o_\ell[\beta]$ are the mean residual per position bucket $\beta(t)$ (positions 0 to 7 one each, then buckets of
    doubling width);
  - $\Sigma_\ell$ is the posterior-predictive covariance, the pooled residual covariance times $n/(n-\gamma_\ell)$, with $\gamma_\ell$
    the regression's effective number of parameters.
- **Integration.** The expectation over innovations uses randomised quasi-Monte Carlo: scrambled Sobol points on the eigen-coordinates
  of every $\Sigma_\ell$, largest variances first, 64 by default. The same points serve every layer, position and text, so logits
  are deterministic and the logits of different layers are paired.
