# Mechanism

How LexFrame reads a hidden state. The notation follows the paper.

## 1. Native semantics

A language model's vocabulary is its native semantics. At the output, the logit of a token $v$ is

$$z_v = w_v^\top N(x_L),$$

with the last hidden state $x_L$, the final normalisation $N$ and the unembedding row $w_v$. Its centred logit $c_v$ subtracts the mean over the vocabulary. This output frame is fixed, as every state at the output is read in the same way.

A hidden state $x_\ell$ at an intermediate layer has no fixed frame, because its meaning depends on the layers that remain. Let $J_\ell$ be the Jacobian of the transport (section 2) followed by $N$. Each token then has two directions at the state.

- The **lexical readout** $b_\ell(v) = J_\ell^\top (w_v - \bar w)$ is the gradient of the token's centred logit, computed by reverse-mode differentiation through the transport. Its inner product with a displacement of the state is the logit change that the displacement causes to first order.
- The **lexical basis** $s_\ell(v) = J_\ell^{+} s_v$ is the move of the state that raises the token alone, where $s_v = (W^\top W)^{-1} w_v$ is the output basis vector of $v$. It solves the damped least-squares problem
  $$\min_\delta \lVert J_\ell \delta - s_v \rVert^2 + \lambda \lVert \delta \rVert^2, \qquad \lambda = \rho\, \lVert J_\ell^\top s_v \rVert^2 / \lVert s_v \rVert^2,$$
  by LSMR, with forward-mode products $J_\ell \delta$ and reverse-mode products $J_\ell^\top y$ and without forming $J_\ell$. The relative ridge $\rho$ damps the directions that the remaining layers barely pass on, and each solution is scaled so that one unit adds exactly one logit.

Both change with the layer and with the context, and the readout and basis of a token have an inner product of one.

## 2. Native transport

A frame reads $x_\ell(t)$, the state at layer $\ell$ and position $t$, by carrying it through the model's own remaining layers to the output. Every layer updates the state from the state one layer below and from its attention output, which brings in the context. The frame keeps the first input exactly, through each layer's own residual connections, normalisations and feed-forward block, and differs from the model only in the attention. Nothing is fitted to the model's output, which makes the frame **model-native**.

The attention is computed by the layer's own attention module. The module receives the prefix states followed by the transported states, with an additive mask, so that a transported state at position $t$ sees the prefix positions before $t$ and its own key and nothing else. The position embeddings come from the model itself, from a forward pass with the matching position ids, so they take whatever form the architecture uses.

After the transport $T_\ell$, the model's own inner product gives the coefficients of the state, its logits

$$z_\ell(v) = \langle w_v,\, N(T_\ell(x_\ell(t))) \rangle,$$

and their softmax is the reading of the state, a distribution over the next token. `coef()` returns them centred by their mean over the vocabulary, which leaves the reading unchanged.

## 3. Orders of Attention

The order $n$ counts the rounds of attention between positions that the transport computes, which makes a frame **context-aware**.

| Order | Attention of the state in each remaining layer | Reads |
|---|---|---|
| 0 | a typical attention output | what the state holds by itself |
| 1 | the layer's own attention over the prefix, transported at order 0 | what the state draws from its prefix |
| 2 | the layer's own attention over a prefix transported at order 1 | one exchange among the prefix positions as well |
| $n$ | the layer's own attention over a prefix transported at order $n-1$ | $n$ rounds of exchange between positions |
| `"inf"` | the layer's own attention over the true prefix states | the model's own output |

The attention that a frame withholds is replaced by a typical attention output, one linear map per layer,

$$\hat a_k(y) = \bar a_k + G_k\, (y - \bar y_k),$$

where $y$ is the attention input, $\bar y_k$ and $\bar a_k$ are the mean attention input and output of layer $k$, and $G_k$ is the ridge regression of the output on the input. It has a closed form, estimated from one forward pass over unlabelled text, and its ridge is chosen by two-fold cross-validation. The first position has no prefix and attends only to itself, so its attention output is exact at every order.

Each order adds one round of attention between positions. Two identities follow. The reading of every layer at order `"inf"` equals the model's output, and a frame of order $n$ is exact wherever at most $n$ layers remain.
