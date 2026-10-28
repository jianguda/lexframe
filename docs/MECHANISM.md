# Mechanism

How LexFrame computes what a `State` and a `Trail` return. The notation follows the paper.

At the output, the logit of a token $v$ is $z_v = w_v^\top N(x_L)$, with the last hidden state $x_L$, the final normalisation $N$ and
the unembedding row $w_v$; its centred logit subtracts the mean over the vocabulary. Every state is read in these coordinates: it is
carried to the output through the remaining layers and read there. It can be carried in two ways.
- **With context**, through the model's own remaining layers with the true context: what the model makes of the state.
- **Without context**, on the state's own column, with every later attention write replaced by its expectation given the state:
  what the state carries by itself.

For a model that soft-caps its logits, probabilities use the capped logits; gradients, directions, coordinates and steering use the
logits before the cap, in which the head is linear.

## State

### `decode`

**With context.** The state $x_k(t)$ of layer $k$ and position $t$ runs through layers $k+1, \dots, L$, each with its own attention
module and position embeddings, attending to the true states of its prefix. Unmoved states reproduce the model exactly. Moved states
are carried together, each position attending to the moved states of its prefix: the model's own forward pass from the moved states.

**Without context.** The state runs through layers $k+1, \dots, L$ on its own column: norms, MLPs and residual connections as the
model computes them. Other positions enter a layer only through its attention write, so each write is replaced by its expectation
given the column's own attention input $u$, plus an independent Gaussian innovation for its spread:

$$a_\ell = \bar a_\ell + G_\ell\,(u - \bar u_\ell) + o_\ell[\beta(t)] + \varepsilon_\ell, \qquad \varepsilon_\ell \sim \mathcal N(0, \Sigma_\ell).$$

The reading is the mean next-token distribution over the innovations, $R_k = \mathbb E_\varepsilon[\mathrm{softmax}(W N(h_L))]$.
- It is a function of $x_k(t)$ and its position $t$ alone: a state carried into another text at the same position decodes the same.
- At the output, $R_L$ is the model's own distribution; elsewhere it is not.
- Position 0 attends only to itself, so its writes are exact and its reading is the output at every layer.

### `gradient`

The gradient of a label's centred logit with respect to the state, by reverse mode through the same carrying.
- **With context,** each position attends to the true states of its prefix and to its own key.
- **Without context,** the gradient is the derivative of the decoding without context: each replaced write contributes the
  regression matrix $G_\ell$.

### `direction`

With $J_k$ the Jacobian of the carrying with context followed by $N$, and $y_v = (W^\top W)^{-1} w_v$ the output change that raises
$v$'s centred logit by one with the least change of the head's coordinates, the direction of $v$ solves

$$\min_\delta \lVert J_k \delta - y_v \rVert^2 + \lambda \lVert \delta \rVert^2, \qquad \lambda = \rho\, \lVert J_k^\top y_v \rVert^2 / \lVert y_v \rVert^2, \quad \rho = 0.1,$$

by LSMR from Jacobian-vector products, without forming $J_k$, and is scaled so that one unit raises the label's logit by one.
- Without damping ($\rho = 0$) the solution is the minimum-norm move with exactly that output change.
- At shallow layers the remaining layers pass some directions on only very weakly, and that move becomes too long to apply. The
  damping keeps the move within the range where the model responds linearly; near the output it changes almost nothing.

### `coordinates`

A change $\delta$ of the state, written label by label: the amounts $c$ to move along each label's direction that make the same
first-order logit changes,

$$c = G^{-1} (B\,\delta),$$

with $B$ the labels' gradients and $G$ the pairing of the gradients with the undamped directions. That pairing is the same at every
layer: it equals its value at the output, computed from the head and from what a move can reach through the final normalisation at
the position. So the coordinates need only the gradients, are finite at every layer, and form no long vector.

### `steer`

The move is a combination of the labels' directions, solved with the inverse of their pairing with the gradients, so that the
cross-talk between related labels cancels to first order. Then:
1. a corrected step removes the second-order term;
2. the moved positions are carried together, the error is measured on the model's own forward pass, and the step is corrected there;
3. where the error still exceeds the tolerance, the step is discarded and the move approaches its target from the model's own state
   in steps, each within the linear range.

`error` and `move` report, per selected state, the relative error and the length of the move relative to the state.

## Trail

### `logprobs`

The label's log-probability decoded without context from the state of every layer at the position. At the last layer it is the
model's output.

### `upstream` and `downstream`

Between two consecutive layers, the log-probability without context changes because the true attention write $a_j$ replaces its
expectation $\hat a_j$. Every step uses the same integration points, so the steps add up exactly to the output. The change has two
parts:
- what the context brought in: the departure of the true write from its expectation;
- a noise term: the spread that the true write removes.

The true write is a sum over source positions, $a_j(t) = \sum_s c_s$, where $c_s$ is what the heads read at $s$, weighted by their
attention. Since the head-averaged attention weights $w_s$ sum to one, the departure splits exactly by source position,

$$a_j - \hat a_j = \sum_s (c_s - w_s \hat a_j).$$

Each source term is mapped to the log-probability by integrated gradients over 8 points along the straight path from $\hat a_j$ to
$a_j$, with the derivative of the readings themselves, so that the terms add up to the change up to
the discretisation's remainder.
- `upstream()` reports these terms by layer and source position.
- `downstream()` reads them from the source's side: what every later position took from this one at each layer.

### `attribute` and `ablate`

`attribute()` removes what each layer, or its attention or MLP sublayer, wrote at the position, carries both states to the output
with context, and reports the change of the label's centred logit. `ablate()` returns the state with such a write removed, or with
the layer run again without what its attention read from one source position, for the model to decode.

## Calibration

`init()` fits the expected writes of decoding without context on unlabelled text: `size` blocks of 128 tokens (1024 by default), one
from each of `size` evenly spaced texts.
- **Expected write.** One pass gives, per layer, the ridge regression $G_\ell$ of the attention write on its input, with the means
  $\bar u_\ell$ and $\bar a_\ell$. The ridge is the MacKay evidence fixed point, started from two-fold cross-validation.
- **Spread.** A second pass gives the residuals, position 0 excluded:
  - the offsets $o_\ell[\beta]$ are the mean residual per position bucket $\beta(t)$ (positions 0 to 7 one each, then buckets of
    doubling width);
  - $\Sigma_\ell$ is the posterior-predictive covariance, the pooled residual covariance times $n/(n-\gamma_\ell)$, with $\gamma_\ell$
    the regression's effective number of parameters.
- **Integration.** The expectation over innovations uses randomised quasi-Monte Carlo: scrambled Sobol points on the eigen-coordinates
  of every $\Sigma_\ell$, largest variances first, 64 by default. The same points serve every layer, position and text, so a
  reading is deterministic and the readings of different layers are paired.
