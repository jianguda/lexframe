# Usage

A frame is built on a loaded Hugging Face model and calibrated once on unlabelled text. `view()` returns a `Carrier`: the hidden states of a text at one or more layers, with what the frame reads in them and the directions that move and measure them.

```python
import lexframe as lf

frame = lf.LexFrame(model, tokenizer, order=1).init(texts)
c = frame.view("The Eiffel Tower is located in the city of", k=5, layers=16)
b, r = c.basis([" Paris", " London"]), c.readout([" Paris", " London"])
moved = c.steer({" Paris": 2, " London": -2})               # the same as c + 2 * b[:, 0] - 2 * b[:, 1]
frame.view(moved).tokens
```

## Frame

| Command | Returns |
|---|---|
| `LexFrame(model, tokenizer, order=1)` | a frame with `order` rounds of attention between positions, where `"inf"` is the native frame |
| `frame.init(texts, size=256, cache=True, cache_dir=None, tol=0.05)` | the frame, calibrated on `size` sequences of 128 tokens (cached on disk) and checked: `frame.identities` |
| `frame.config(steps=30, ridge=1e-3, warm=False)` | the frame, with these LSMR settings for the basis: `frame.solver` |
| `frame.view(text, k=1, layers=None)` | a `Carrier` at every layer (or the given ones), or the reading of a (moved) `Carrier` |
| `frame.attr(text, sublayer=None, target=None)` | each layer's (or its `"attn"` or `"mlp"` sublayer's) contribution to the target's logit, `[layer, position]` |
| `frame.bound(texts, token, layers=None)` | the best agreement any fixed lens can reach, per layer |
| `frame.token_id(token)` | the id of a token given as an id or as a string of exactly one token |

## Carrier

| Attribute or method | Returns |
|---|---|
| `vector` | the hidden states, `[..., position, hidden]` |
| `tokens`, `ids`, `probs` | the top `k` tokens of the frame's reading, `[..., position, k]` |
| `kl` | the divergence of the reading from the model's output, in nats, `[..., position]` |
| `first_layer(token)` | the first read layer at which the token tops the reading, `[..., position]`, or -1 |
| `basis(tokens)` | vectors that move the state to add one logit to each token, `[..., position, token, hidden]` |
| `readout(tokens)` | covectors that measure each token's logit, so `readout @ basis.mT` has ones on the diagonal |
| `coef(tokens)` | how much of each token the state holds, its centred logits, `[..., position, token]` |
| `steer({token: weight})` | a moved `Carrier`, shifted along each token's basis by its weight in logits |
| `carrier + v`, `carrier - v` | a moved `Carrier`, read again with `frame.view(moved)` |

Tokens are given as ids or as strings that encode to exactly one token, such as `" Paris"`; a string of several tokens is refused. Layers run from 0, the embedding, to `frame.num_layers`, the output. `view` and `attr` take a list of texts of one token length, which adds a leading axis, and `basis`, `readout` and `coef` take the carrier of one text.

## Notes

**Calibration.** `init` uses `size` sequences of 128 tokens. The result is cached on disk, in `cache_dir`, `$LEXFRAME_CACHE` or `~/.cache/lexframe`, under a fingerprint of the model's configuration and weights and of the calibration tokens. `frame.identities` reports how far the native frame and the frames of order 1 and 2 are from the model where they must be exact, and `init` warns if any exceeds `tol` nats.

**Solver.** `frame.config` sets the LSMR solver behind `basis`, which solves all tokens and positions at once and the layers from the top down. `steps` is the number of iterations per layer, more of which give a more accurate basis, `ridge` the relative damping, and `warm` starts each layer from the solution of the layer above instead of from zero.

**Moving a carrier.** `carrier + v` moves every position on its own, and `frame.view(moved)` transports the moved states with the unmoved prefix. In the native frame, `order="inf"`, this reading is the model's own output. `coef` and `readout` of a moved carrier are taken at its new states, while its `basis` is refused, because the basis is taken at the model's own states.

**Attribution.** `attr` removes what a layer, or its attention or feed-forward sublayer, wrote from the state that follows it, transports both states and returns the difference of their coefficients on the target, by default the model's own prediction. A linear attribution is `readout @ write` instead.

**Bound.** `bound` measures how much the native frame's readouts of a token agree across at least two texts. Its square root is the best mean cosine that any lens with fixed directions per layer can reach, whatever the order of the frame.

**Cost.** Reading the states of one layer at order $n$ runs $n + 1$ streams through the remaining layers, so `view` over every layer of a text costs about $(n + 1)(L + 1)/2$ forward passes. `readout` adds one backward pass per token, and `basis` one forward-mode and one reverse-mode pass per solver step and layer.
