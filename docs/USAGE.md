# Usage

A frame is built on a loaded Hugging Face model and calibrated once on unlabelled text. `view()` returns a `Carrier`: the hidden states of a text at one or more layers, with their reading and the directions that move and measure them. `ledger()` returns a `Ledger`: what the context adds to the reading of one position after each layer.

```python
import lexframe as lf

frame = lf.LexFrame(model, tokenizer).init(texts)
c = frame.view("The Eiffel Tower is located in the city of", k=5, layers=16)
c.tokens                                                    # the closure reading: what each state carries by itself
b, r = c.basis([" Paris", " London"]), c.readout([" Paris", " London"])
moved = c.steer({" Paris": 2, " London": -2})               # the same as c + 2 * b[:, 0] - 2 * b[:, 1]
frame.view(moved, frame="native").tokens                    # the model's output after the move
frame.ledger("The Eiffel Tower is located in the city of").sources  # what each layer took from each position
```

## Frame

| Command | Returns |
|---|---|
| `LexFrame(model, tokenizer, samples=64)` | a frame whose closure reading averages over `samples` quasi-Monte Carlo points |
| `frame.init(texts, size=1024, spread=True, cache=True, cache_dir=None, tol=0.05, progress=True)` | the frame, calibrated on `size` sequences of 128 tokens (cached on disk) and checked: `frame.identities` |
| `frame.config(steps=30, ridge=1e-3, warm=False, samples=64)` | the frame, with these LSMR settings for the basis (`frame.solver`) and this number of integration points |
| `frame.view(text, k=1, layers=None, positions=None, frame="closure")` | a `Carrier` at every layer (or the given ones) and every position (or the given ones), or the reading of a (moved) `Carrier` |
| `frame.ledger(text, target=None, position=-1, steps=8)` | a `Ledger` of one position for the target token (default: the model's prediction) |
| `frame.attr(text, sublayer=None, target=None)` | each layer's (or its `"attn"` or `"mlp"` sublayer's) contribution to the target's logit, `[layer, position]` |
| `frame.bound(texts, token, layers=None)` | the best agreement that any fixed lens can reach with the native readout, per layer |
| `frame.token_id(token)` | the id of a token given as an id or as a string of exactly one token |

## Carrier

| Attribute or method | Returns |
|---|---|
| `vector` | the hidden states, `[..., position, hidden]` |
| `tokens`, `ids`, `probs` | the top `k` tokens of the reading, `[..., position, k]` |
| `kl` | the divergence of the reading from the model's output, in nats, `[..., position]` |
| `reading` | `"closure"` or `"native"`, the frame that `view` read the states in |
| `first_layer(token)` | the first read layer at which the token tops the reading, `[..., position]`, or -1 |
| `basis(tokens)` | vectors that move the state to add one logit to each token, in the native frame, `[..., position, token, hidden]` |
| `readout(tokens, frame="native")` | covectors that measure each token's logit, so `readout @ basis.mT` has ones on the diagonal; with `frame="closure"`, the gradient of each token's centred log reading |
| `coef(tokens, frame="native")` | the centred output logits of the tokens when the state is carried with its true context; with `frame="closure"`, their centred log reading, `[..., position, token]` |
| `gap(tokens)` | one minus the cosine of the native and the closure readout, `[..., position, token]` |
| `steer({token: weight})` | a moved `Carrier`, shifted along each token's basis by its weight in logits |
| `positions` | the positions that the `Carrier` holds, or `None` for every position |
| `carrier + v`, `carrier - v` | a moved `Carrier`, read again with `frame.view(moved)` |

## Ledger

| Attribute | Returns |
|---|---|
| `token`, `target`, `position` | the target token as text and as an id, and the position |
| `logp`, `prob` | the closure reading of the target at every layer, `[layer + 1]`; the last is the model's output |
| `step` | `logp[j + 1] - logp[j]`, so that `step[k:].sum()` is `logp[-1] - logp[k]` exactly, `[layer]` |
| `innovation`, `noise` | the two parts of each step: the true attention write against its closure prediction, and the closure's own spread |
| `sources` | the innovation split by source position, `[layer, position]`, or `None` where the attention module does not allow the split |
| `residual`, `check` | the integrated-gradients residual of `sources`, and the check that the true write rebuilds the next reading |

Tokens are given as ids or as strings that encode to exactly one token, such as `" Paris"`; a string of several tokens is refused. Layers run from 0, the embedding, to `frame.num_layers`, the output; the reading starts at layer 1, and layer 0 is shown for diagnosis. `view` and `attr` take a list of texts: texts of one token length are read together, which adds a leading axis, and texts of different lengths one at a time, which returns a list. `basis`, `readout`, `coef`, `gap` and `ledger` take one text.

## Notes

**Calibration.** `init` uses `size` sequences of 128 tokens, one from each of `size` evenly spaced texts, so that the calibration covers the whole corpus. With fewer texts than sequences, or with `spread=False`, the sequences are cut contiguously from the start. The calibration is part of the reading's definition: pass the whole corpus, e.g. all of WikiText-103's training split, rather than its first lines. It takes two passes over the sequences and is cached on disk, in `cache_dir`, `$LEXFRAME_CACHE` or `~/.cache/lexframe`, under a fingerprint of the model's configuration and weights and of the calibration tokens. `frame.calibration.stats` lists each layer's ridge, effective number of parameters and held-out fit. `frame.identities` reports how far the native frame and the closure reading of the first position are from the model, where both must be exact, and `init` warns if either exceeds `tol` nats.

**Reading.** The closure reading of a state depends on the state and its position alone. It is deterministic: the integration points are fixed. More `samples` integrate more accurately at a proportional cost. `frame="native"` reads states with their true context instead. For the model's own states this is the output at every layer, and for moved states the output after the move.

**Solver.** `frame.config` sets the LSMR solver behind `basis`, which solves all tokens and positions at once and the layers from the top down. `steps` is the number of iterations per layer, more of which give a more accurate basis, `ridge` the relative damping, and `warm` starts each layer from the solution of the layer above instead of from zero.

**Moving a carrier.** `carrier + v` moves every position on its own. `frame.view(moved)` gives the closure reading of the moved states, and `frame.view(moved, frame="native")` the model's output after the move, with the unmoved prefix. `coef` and `readout` of a moved carrier are taken at its new states, while its `basis` is refused, because the basis is taken at the model's own states.

**Ledger.** `ledger` integrates each source term along the path of the write with `steps` points. More points shrink `residual`. The split by source position needs the attention module's weights and an output projection `o_proj`. Where the weights do not sum to one, e.g. with attention sinks, `sources` is `None` and the steps, innovations and noise terms are still exact.

**Positions.** `positions` reads only the given positions of a text, and of each text in a list. As the model is causal, the text is cut after the last of them, which saves the computation of the later positions. Such a `Carrier` holds its reading only, so take directions or steer on a view of every position.

**Bound.** `bound` measures how much the native readouts of a token agree across at least two texts. Its square root is the best mean cosine that any lens with fixed directions per layer can reach.

**Cost.** The closure reading of one state runs `samples` columns through the remaining layers. These columns attend to nothing, so `view` over every layer of a text of $T$ tokens costs about `samples` $\cdot T (L + 1)/2$ single-position layer passes, batched together. `layers=` and `positions=` read less at a lower cost. `readout` adds one backward pass per token, `basis` one forward-mode and one reverse-mode pass per solver step and layer, and `ledger` about `steps + 2` closure readings per layer.
