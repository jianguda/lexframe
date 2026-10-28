"""Where does SmolLM3 decide that the Eiffel Tower is in Paris, what did the context add, who wrote it, and can we change it?"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import lexframe as lf

name = "HuggingFaceTB/SmolLM3-3B"
model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(name)
# the whole training split: the calibration takes 1024 blocks spread evenly over it
texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]

frame = lf.LexFrame(model, tokenizer).config(size=1024).init(texts)
print(frame, frame.num_layers)

text = "The Eiffel Tower is located in the city of"
answer, rival = " Paris", " London"

# 1. What does each state carry by itself? Decoding without context, at every layer of the last position.
s = frame.state(text)  # every layer and position; index it like an array, [layer, position]
d = s[:, -1].decode(context=False)
labels, probs = d.top(3)
for layer in s.layers:
    print(layer, labels[layer], probs[layer].round(2), d.kl[layer].round(2))
tops = labels[:, 0] == answer  # per layer: does the answer top the reading at the last position?
decided = int(np.argmax(tops)) if tops.any() else frame.num_layers // 2
print("the answer tops the reading from layer", decided)
print("one layer, every position", s[decided].decode(context=False).top(1)[0][:, 0])
print("texts of any length", [frame.state(t).decode(context=False).kl.shape for t in [text, "The Louvre is in"]])

# 2. Where does the answer come from? The trail of the last position: how the reading of the answer changes after each layer,
# what the context added there and from which positions, and what each layer's write contributed.
t = frame.trail(text, label=answer)
words = [tokenizer.decode([i]) for i in tokenizer.encode(text, add_special_tokens=False)]
print("reading of the answer per layer", np.exp(t.logprobs).round(3))
step = np.diff(t.logprobs)  # the change of the log reading that layer j makes
up = t.upstream()  # [layer, position]: what each layer's attention brought in from each earlier position
for j in np.argsort(-np.abs(step))[:3]:
    top = [] if up is None else [(words[p], round(float(up[j, p]), 3)) for p in np.argsort(-np.abs(up[j]))[:3]]
    added = "" if up is None else f", the context added {up[j].sum():+.3f}"
    print(f"layer {j}: step {step[j]:+.3f}{added}, from", top)
whole, attn, mlp = t.attribute().T  # removing each layer's whole write, its attention part, its feed-forward part
top = whole.argsort()[::-1][:3]
print("layers", top, "whole", whole[top].round(2), "attn", attn[top].round(2), "mlp", mlp[top].round(2))

# 3. What does a position pass on? What the later positions' predictions took from the word "Tower".
pos = words.index(" Tower") if " Tower" in words else 1
given = frame.trail(text, position=pos).downstream()
if given is not None:
    print("taken from", repr(words[pos]), "by the later positions", given.sum(0).round(3))

# 4. Each label at the state: its direction, its gradient and its logit, with and without context; one minus the cosine of the two
# gradients is what the context still reinterprets.
at = min(max(decided, frame.num_layers // 2), frame.num_layers - 1)  # from mid-depth on, the frame predicts the effect of a steer well
pair = [answer, rival]
c = s[at]  # every position at this layer
b, o = c.direction(pair), c.gradient(pair)
print("logit", c.decode().logits(pair)[-1], "without context", c.decode(context=False).logits(pair)[-1])
gap = 1 - torch.nn.functional.cosine_similarity(o, c.gradient(pair, context=False), dim=-1)
print("gradient . direction", (o @ b.mT)[-1], "gap between the gradients", gap[-1])
nxt = s[at + 1]
print("written by the layer, label by label", c[-1].coordinates(nxt[-1].vector - c[-1].vector, pair))

# 5. Change its mind: steer the states so that the answer's logit falls by 4 and the rival's rises by 4; decoding with context gives
# the model's output, decoding without context what the moved state carries by itself.
moved = c.steer({answer: -4, rival: 4})
actual = moved.decode().logits(pair) - c.decode().logits(pair)
print(moved, "actual", actual[-1], "error", moved.error[-1], "move", moved.move[-1])
print("model", moved.decode().top(5)[0][-1], "by itself", moved.decode(context=False).top(5)[0][-1])

# 6. More integration points for the reading without context.
frame.config(samples=256)
print(frame, s[:, -1].decode(context=False).top(3)[0][:, 0])
