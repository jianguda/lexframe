# Mechanism

How LexFrame reads a hidden state. The notation follows the paper.

LexFrame answers three questions about the state $x_k(t)$ at layer $k$ and position $t$, each with its own tool:

| Question | Tool | Method |
|---|---|---|
| What does the state carry? | the closure reading $R_k$ | `view()` |
| What does the context add after layer $k$? | the ledger | `ledger()` |
| How does the model use the state here? | the native frame: readout, basis, coefficients | `readout()`, `basis()`, `coef()`, `steer()`, `attr()` |

Layers run from 1 to $L$. Layer 0 is the embedding, a fixed input frame, and is shown for diagnosis only.

## 1. Native semantics

A language model's vocabulary is its native semantics. At the output, the logit of a token $v$ is

$$z_v = w_v^\top N(x_L),$$

with the last hidden state $x_L$, the final normalisation $N$ and the unembedding row $w_v$. Its centred logit $c_v$ subtracts the mean over the vocabulary. Every state at the output is read in this one way. A state at an intermediate layer has no fixed reading, because what it means depends on the layers that remain.

## 2. The closure reading

The closure reads $x_k(t)$ by running it through the model's own layers $k+1, \dots, L$ on its own column. Norms, feed-forward blocks and residual connections are computed as the model computes them. Other positions enter a layer only through its attention write, so the closure replaces each write by its prediction from the column's own attention input $u$, plus an independent Gaussian innovation:

$$a_\ell = \bar a_\ell + G_\ell\,(u - \bar u_\ell) + o_\ell[\beta(t)] + \varepsilon_\ell, \qquad \varepsilon_\ell \sim \mathcal N(0, \Sigma_\ell).$$

The reading is the mean next-token distribution over the innovations,

$$R_k = \mathbb E_\varepsilon\big[\mathrm{softmax}(W N(h_L))\big].$$

After layer $k$ the column never reads another position. There is no reference context and no order to choose, and attention never runs on a context with parts missing. The reading is a function of $x_k(t)$ alone, given the model and the calibration, which the swap test checks exactly: a state carried into another text reads the same. At the output, $R_L$ is the model's own distribution. Elsewhere it is not, so the reading is not a tautology. Position 0 attends only to itself, so its writes are computed exactly and its reading equals the output at every layer.

**Calibration.** One forward pass over unlabelled text gives, per layer, the typical attention output: $G_\ell$ is the ridge regression of the write on its input, $\bar u_\ell$ and $\bar a_\ell$ are the means. The ridge is the MacKay evidence fixed point, a closed form without a data split, started from two-fold cross-validation. A second pass gives the residuals of this prediction, excluding position 0, which is exact:

- the offsets $o_\ell[\beta]$ are the mean residual in each position bucket $\beta(t)$: positions 0 to 7 one each, then buckets of doubling width;
- $\Sigma_\ell$ is the posterior-predictive covariance, the pooled residual covariance times $n/(n-\gamma_\ell)$, where $\gamma_\ell$ is the effective number of parameters of the regression.

The calibration is part of the reading's definition, a prior over contexts. `init()` takes 1024 blocks of 128 tokens, one from each of 1024 evenly spaced texts (WikiText-103 in the paper), cut by the same pipeline that cuts the texts it reads.

**Integration.** The expectation is computed by randomised quasi-Monte Carlo. The points are scrambled Sobol points on the eigen-coordinates of every $\Sigma_\ell$, with the largest variances first. By default there are 64. The same points serve every layer, position and text, so a reading is deterministic, and the readings of different layers are paired.

## 3. The ledger

The closure readings of one position telescope to the output:

$$\log p_L(v) - \log R_k(v) = \sum_{j \ge k} \big(\log R_{j+1}(v) - \log R_j(v)\big).$$

The sum is exact, because every term uses the same integration points. Step $j$ replaces layer $j$'s closure write, its prediction plus the innovation, by the true write $a_j$. The step splits into two terms:

- the **innovation** of the true write over its prediction;
- a **noise** term: the closure's own spread at layer $j$, which the true write removes.

The true write is a sum over source positions, $a_j(t) = \sum_s c_s$, where $c_s$ is what the heads read at $s$, weighted by their attention. Since the head-averaged attention weights $w_s$ sum to one, the innovation splits exactly by source position,

$$a_j - \hat a_j = \sum_s (c_s - w_s \hat a_j).$$

Each source term is mapped to the target's log reading by integrated gradients along the straight path from $\hat a_j$ to $a_j$. `ledger()` reports every term in log-probability units and the readings themselves as probabilities. Integrated gradients leave a small discretisation residual, which is reported as well.

## 4. The native frame

The native frame carries the state through the model's remaining layers with the true context. Each position attends to the true states of its prefix and to its own key, through the layer's own attention module and the model's own position embeddings. For an unmoved state this reproduces the model exactly. For a moved state it gives the model's output after the move. Let $J_k$ be the Jacobian of this transport followed by $N$. Each token then has two directions at the state:

- The **lexical readout** $b_k(v) = J_k^\top (w_v - \bar w)$ is the gradient of the token's centred logit. Its inner product with a displacement of the state is the logit change that the displacement causes, to first order.
- The **lexical basis** $s_k(v) = J_k^{+} s_v$ is the move of the state that raises the token alone, where $s_v = (W^\top W)^{-1} w_v$ is the output basis vector of $v$. It solves the damped least-squares problem
  $$\min_\delta \lVert J_k \delta - s_v \rVert^2 + \lambda \lVert \delta \rVert^2, \qquad \lambda = \rho\, \lVert J_k^\top s_v \rVert^2 / \lVert s_v \rVert^2,$$
  by LSMR, with forward-mode products $J_k \delta$ and reverse-mode products $J_k^\top y$ and without forming $J_k$. Each solution is scaled so that one unit adds exactly one logit.

The readout and basis of a token have an inner product of one. Attribution, steering and every other edit use the native frame. Edits along closure directions do not transfer better across paraphrases.

**The closure readout and the frame gap.** The gradient of the centred log reading, $\nabla_x \log R_k(v)$, is the closure readout. It is defined on the readout side only. The frame gap

$$1 - \cos\big(b_k(v),\ \nabla_x \log R_k(v)\big)$$

measures how much the context will still reinterpret the state. Fixed lenses fall short of both directions. The logit lens, the Jacobian lens and the best fixed lens (at most $\sqrt\kappa$, see `bound()`) agree with the native readout less than the closure readout does.

**Attribution.** `attr()` removes what a layer, or its attention or feed-forward sublayer, wrote at a position. It carries both states through the native frame and returns the change of the target's centred logit.
