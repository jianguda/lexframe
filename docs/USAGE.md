# Usage

A frame is built on a loaded Hugging Face model and gives two objects:
- a **State**: the hidden states of a text at every layer and position. It tells what they decode to, how a change of them would move
  given logits, and how to move them.
- a **Trail**: one prediction, followed across layers and positions. It tells where the prediction comes from and what the position
  passes on.

```python
import lexframe as lf

frame = lf.LexFrame(model, tokenizer).init(texts)            # calibrate once on unlabelled texts
text = "The Eiffel Tower is located in the city of"

s = frame.state(text)                                         # every layer and position; index like an array
s[16, -1].decode().top(5)                                     # what the model outputs from layer 16's state at the last position
s[16, -1].decode(context=False).top(5)                        # what that state carries by itself
s[16, -1].gradient([" Paris", " London"])                     # how a change of the state moves each label's logit
moved = s[16, -1].steer({" Paris": -2, " London": 2})         # Paris loses two logits, London gains two
moved.decode().top(5)                                         # the model's output after the move
s[16, -1].coordinates(s[17, -1].vector - s[16, -1].vector, [" Paris", " London"])  # what layer 16 wrote, label by label

t = frame.trail(text)                                         # the prediction at the last position
t.logprobs                                                    # its log-probability from every layer's state by itself
t.attribute(), t.upstream(), t.downstream()                   # by layer; from which positions; to which positions
t.ablate(12, "attention").decode().top(5)                     # the output without layer 12's attention write here
```

## LexFrame

| Command | Returns |
|---|---|
| `LexFrame(model, tokenizer)` | a frame of a Hugging Face causal language model |
| `frame.config(samples=64, tolerance=0.02, size=1024, cache=..., progress=True)` | the frame with new settings (see below) |
| `frame.init(texts)` | the frame, calibrated on unlabelled `texts` (strings, or batches of token ids as tensors, used as given) |
| `frame.num_layers` | the number of layers; layer `num_layers` is the output |
| `frame.state(text)` | a `State` of one text at every layer and position |
| `frame.trail(text, position=-1, label=None)` | a `Trail` of the prediction at one position, for `label` (default: the model's prediction there) |

| Setting | Default | Effect |
|---|---|---|
| `samples` | 64 | integration points of decoding without context: more is more accurate, at a proportional cost |
| `tolerance` | 0.02 | the relative error at which `steer` stops |
| `size` | 1024 | calibration sequences of 128 tokens |
| `cache` | `$LEXFRAME_CACHE` or `~/.cache/lexframe` | the folder of calibrations; `None` for no cache |
| `progress` | `True` | show the calibration's progress |

## State

Index a `State` like an array: `s[layer]` or `s[layer, position]`, with ints (which drop the axis), slices or lists of ints; layers
are selected in increasing order, each once. Results have the leading axes `[layer, position]` of the selection.

| Member | Returns |
|---|---|
| `vector`, `layers`, `positions` | the selected hidden states `[layer, position, hidden]`, and their layers and positions |
| `decode(context=True)` | a `Decoding`: `top(k)` gives the k most probable labels and their probabilities, `logits(labels)` the labels' centred logits, `kl` the divergence from the model's output. With context: what the model outputs from the states, carried through the remaining layers with the true context. Without context: what each state carries by itself |
| `gradient(labels, context=True)` | the gradient of each label's centred logit with respect to the states, `[..., label, hidden]`, with or without context; its inner product with a change of a state is the logit change, to first order |
| `direction(labels)` | for each label, a move of the states that raises its logit by one and leaves the others' to first order, `[..., label, hidden]` |
| `coordinates(vector, labels)` | a change of the states written label by label, `[..., label]`: how far to move along each label's direction to make the same first-order logit changes; `vector` is `[..., hidden]` or a `State` |
| `steer({label: change})` | the states moved so that each label's logit changes by `change`; `error` and `move` report, per selected state, the relative error and the size of the move relative to the state |
| `state + v`, `state - v` | the selected states moved by `v` |

Moved states decode and give gradients like any others. `direction`, `coordinates` and `steer` start from the model's own states and
refuse moved ones.

## Trail

| Member | Returns |
|---|---|
| `label`, `position` | the label as text, and the position |
| `logprobs` | the label's log-probability from every layer's state by itself (decoded without context), `[layer + 1]`; the last is the model's output |
| `attribute()` | the change of the label's centred logit when what each layer wrote at this position is removed: the whole write, its attention part and its MLP part, `[layer, 3]` |
| `upstream()` | what each layer's attention brought in from each earlier position, `[layer, position]`; a row adds up to how much the context raised the label at that layer beyond what the state predicted by itself |
| `downstream()` | what each later position's own prediction took from this position, `[layer, later position]` |
| `ablate(layer, part="all", source=None)` | the `State` at layer `layer + 1` with what layer `layer` wrote here removed (`part` `"attention"` or `"mlp"` for one sublayer), or, with a `source` position, after the layer runs again without what its attention read from there |

## Notes

**Labels and layers.** Labels are vocabulary entries, given as ids or as strings that encode to exactly one token, such as `" Paris"`;
a string of several tokens is refused. Layers run from 0, the embedding, to `num_layers`, the output. Decoding without context starts
at layer 1; layer 0 is shown for diagnosis.

**Calibration.** Decoding without context and trails need `init`; everything with context does not.
- `init` uses `size` sequences of 128 tokens, one from each of `size` evenly spaced texts, so pass the whole corpus (e.g. all of
  WikiText-103's training split) rather than its first lines.
- It takes two passes over the sequences and is cached under a fingerprint of the model and the calibration tokens.
- It warns if the frame does not reproduce the model.

**Decoding.** Without context, a state's decoding depends on the state and its position alone, and it is deterministic: the
integration points are fixed. With context, the model's own states decode to the model's own output at every layer, and moved states
to the output after the move, with every moved position carried together.

**Steering.** `steer` changes each label's logit by its `change` at every selected position and layer, and leaves the other positions
as they are. It cancels the cross-talk between related labels and the logits' response beyond first order. The error is measured on
the model's own forward pass; where one step does not reach `tolerance`, `steer` approaches it in steps.

**Trail.** `upstream` and `downstream` need the attention module's weights and an output projection `o_proj`. Where the attention
output cannot be rebuilt from its weights and values (e.g. with attention sinks, whose weights do not sum to one), they return `None`,
and `ablate` with a `source` is refused.

**Cost.**
- Decoding without context runs `samples` single-position columns through the remaining layers for each selected state; select layers
  and positions to read less.
- `gradient` adds one backward pass per label; `direction` up to fifty pairs of forward and backward passes per layer; `coordinates`
  one gradient and the final norm's Jacobian at each position; `steer` one direction and gradient per step and layer.
- A trail decodes its label at every layer. `attribute` carries every layer's state, with its write removed, through the remaining
  layers, once for the whole write and once for each sublayer; `upstream` takes about ten decodings per layer, and `downstream` an
  `upstream` per later position.
- The calibration holds three `[hidden, hidden]` tables per layer (about 0.6 GB for a 1B model, 6 GB for a 7B one), kept on the CPU
  when large.
- LexFrame's own computations run in float32 and need no float64, so any accelerator works, Apple's MPS included.
