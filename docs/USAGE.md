# Usage

A frame is built on a loaded Hugging Face causal language model. `frame.state(text)` gives a **State**: the hidden states of the
text at every layer and position. A State gives three quantities and moves the model's run:
- `logits`: what each state carries in the model's vocabulary (calibrated once by `init`; at the output layer, the model's own);
- `gradient`: how the model's output responds to a change of each state;
- `basis`: the semantic bases, the geometric carrier of both: a state's projections on them are its logits, and their covectors
  are the gradients;
- `steer`, `ablate`, `+`, `-`: the moved run, a State like any other, whose output layer is the model's output after the move.

[MECHANISM.md](MECHANISM.md) gives the definitions and the computations.

```python
import lexframe as lf

frame = lf.LexFrame(model, tokenizer).init(texts)             # calibrate once on unlabelled texts
text = "The Eiffel Tower is located in the city of"

s = frame.state(text)                                          # every layer and position; index like an array
s[16, -1].logits().softmax(-1).topk(5)                         # what layer 16's state at the last position carries
s[-1, -1].logits().softmax(-1).topk(5)                         # at the output layer: the model's own prediction
s[16, -1].gradient([" Paris", " London"])                      # how a change of the state moves each label's output logit
b = s[16, -1].basis([" Paris", " London"])                     # the labels' semantic bases at the state

moved = s[16, -1].steer({" Paris": -2, " London": 2})          # the run moved at layer 16: Paris loses two logits, London gains two
moved[-1, -1].logits().softmax(-1).topk(5)                     # the model's output after the move
cut = s[13, -1].ablate("attention")                            # the run without layer 12's attention write at the last position
s[-1, -1].logits([" Paris"]) - cut[-1, -1].logits([" Paris"])  # what that write did to the prediction
s[13, -1].ablate(source=6)                                     # the run with layer 12's attention not reading position 6 there
```

## LexFrame

| Command | Returns |
|---|---|
| `LexFrame(model, tokenizer)` | a frame of a Hugging Face causal language model |
| `frame.config(samples=64, size=1024, length=256, cache=..., progress=True)` | the frame with new settings (see below) |
| `frame.init(texts)` | the frame, calibrated on unlabelled `texts` (strings, or batches of token ids as tensors, used as given) |
| `frame.num_layers` | the number of layers; layer `num_layers` is the output |
| `frame.state(text)` | the `State` of a text (a string or token ids) at every layer and position |

| Setting | Default | Effect |
|---|---|---|
| `samples` | 64 | integration points of `logits`: more is more accurate, at a proportional cost |
| `size` | 1024 | calibration sequences |
| `length` | 256 | tokens per calibration sequence |
| `cache` | `$LEXFRAME_CACHE` or `~/.cache/lexframe` | the folder of calibrations; `None` for no cache |
| `progress` | `True` | show the calibration's progress |

## State

Index a `State` like an array: `s[layer]` or `s[layer, position]`, with ints (which drop the axis), slices or lists of ints. Layers
are selected in increasing order, each once. Results have the leading axes `[layer, position]` of the selection.

| Member | Returns |
|---|---|
| `vector` | the selected hidden states, `[..., hidden]` |
| `layers`, `positions` | the selected layers and positions |
| `logits(labels=None)` | what the states carry: centred logits `[..., label]` (`None`: the whole vocabulary). At the output layer, the model's own centred logits |
| `gradient(labels)` | the model's own gradient of each label's centred output logit with respect to the states, the context fixed, `[..., label, hidden]` |
| `basis(labels)` | the labels' semantic bases at the states, `[..., label, hidden]` (`None`: the whole vocabulary): the state is the sum of the bases of the whole vocabulary weighted by its logits, and a move along a label's basis changes its output logit by the step, to first order |
| `steer({label: change})` | the run moved at the selected positions of one layer so that each label's centred output logit changes by `change`, to a relative error of 2%: each state rotated at its norm, the output's distribution over the other labels changed least |
| `ablate(part="all", source=None)` | the run without what the previous layer wrote at the selected positions of one layer (`part` `"attention"` or `"mlp"` for one sublayer), or, with a `source` position, with that layer run again without what its attention read from there |
| `state + v`, `state - v` | the run with the selected states moved by `v` |

A moved run is the model with the move added, and every member works on it at every layer. A further move, at any layer, keeps the
earlier moves as the changes they made.

Everything else follows from these members: probabilities, rankings and divergences from `logits`, and how well and how far a move
went from the moved run:

```python
moved = s[16, -1].steer({" Paris": -2, " London": 2})
done = moved[-1, -1].logits([" Paris", " London"]) - s[-1, -1].logits([" Paris", " London"])  # the changes made
size = (moved[16, -1].vector - s[16, -1].vector).norm() / s[16, -1].vector.norm()           # the move, relative to the state
```

## Notes

**Labels and layers.** Labels are vocabulary entries, given as ids or as strings that encode to exactly one token, such as `" Paris"`;
a string of several tokens is refused. Layers run from 0, the embedding, to `num_layers`, the output. `logits` start at layer 1;
layer 0 is shown for diagnosis.

**Calibration.** `logits` below the output layer, and so `basis`, need `init`; `gradient` and the moves do not.
- `init` uses `size` sequences of `length` tokens, one from each of `size` evenly spaced texts, so pass the whole corpus (e.g. all
  of WikiText-103's training split) rather than its first lines.
- The position statistics cover the first `length` positions; later positions read with those of the last positions seen.
- It takes two passes over the sequences and is cached under a fingerprint of the model and the calibration tokens.
- It warns if the frame does not reproduce the model.

**Logits.** A state's logits depend on the state and its position alone, and they are deterministic: the integration points are
fixed. At position 0, which sees only itself, they are the model's output at every layer.

**Bases.** A label's basis is the same whatever other labels are asked for. A move along one basis also moves the other labels'
logits a little; to change some labels and spare the rest, use `steer`.

**Steering.** `steer` leaves the other positions as they are. A change beyond the reach of a rotation stops short, with a warning
that names the largest relative error.

**Ablation by source.** `ablate` with a `source` needs the attention module's weights and an output projection `o_proj`. Where the
attention output cannot be rebuilt from its weights and values (e.g. with attention sinks, whose weights do not sum to one), it is
refused.

**Cost.**
- `logits` runs `samples` single-position columns through the remaining layers for each selected state. `logits()` of every
  layer and position holds layers × positions × vocabulary floats: select layers, positions or labels to read less.
- `gradient` is one backward pass through the remaining layers, all labels together.
- `basis` takes the logits of the whole vocabulary, the Jacobian of the final normalisation's output (one backward pass per hidden
  dimension) and one system of the hidden size per state, whatever the labels asked for; the whole vocabulary's bases of one state
  are vocabulary × hidden floats.
- `steer` takes, per linearisation, that Jacobian and the others' Fisher metric (one pass over the head), and per step one backward
  pass for the gradients and one forward pass from the moved layer. The selected positions are solved one after another.
- `ablate` runs the ablated layer once more on the run's states; it, `+` and `-` then run one forward pass from the moved layer.
- A moved run keeps one `[positions, hidden]` change per moved layer.
- The calibration holds three `[hidden, hidden]` tables per layer (about 0.6 GB for a 1B model, 6 GB for a 7B one), kept on the CPU
  when large.
- The model's layers run in the model's own precision (e.g. bfloat16); its final normalisation and head, and LexFrame's own
  computations, run in float32.
