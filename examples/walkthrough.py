"""Where does SmolLM3 decide that the Eiffel Tower is in Paris, who wrote it, which positions it read, and can we change it?"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import lexframe as lf

name = "HuggingFaceTB/SmolLM3-3B"
model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(name)
# the whole training split: the calibration takes 1024 blocks of 256 tokens spread evenly over it
texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]

frame = lf.LexFrame(model, tokenizer).init(texts)
print(frame, frame.num_layers)

text = "The Eiffel Tower is located in the city of"
answer, rival = " Paris", " London"
pair = [answer, rival]
L = frame.num_layers
names = lambda ids: [tokenizer.decode([int(i)]) for i in ids]  # noqa: E731
answer_id, rival_id = (tokenizer.encode(w, add_special_tokens=False)[0] for w in pair)

# 1. What does each state carry by itself? Its logits at every layer of the last position, as probabilities, and their divergence
# from the model's output; probabilities, rankings and divergences all follow from the logits.
s = frame.state(text)  # every layer and position; index it like an array, [layer, position]
logp = s[:, -1].logits().log_softmax(-1).cpu()  # [layer, vocabulary]
kl = (logp[L].exp() * (logp[L] - logp)).sum(-1)
top = logp.topk(3)
for layer in s.layers:
    print(layer, names(top.indices[layer]), top.values[layer].exp().numpy().round(2), round(float(kl[layer]), 2))
tops = (top.indices[:, 0] == answer_id).numpy()
decided = int(np.argmax(tops)) if tops.any() else L // 2
print("the answer tops the logits from layer", decided)

# 2. Who wrote the answer? How the answer's log-probability changes after each layer, and what each layer's write did to the model's
# output, by removing it (the whole write, its attention part, its MLP part).
words = names(tokenizer.encode(text, add_special_tokens=False))
lp_answer = logp[:, answer_id].numpy()
print("probability of the answer per layer", np.exp(lp_answer).round(3))
step = np.diff(lp_answer)  # the change of the answer's log-probability that layer j makes
print("the layers that raise it most", np.argsort(-step)[:3], step[np.argsort(-step)[:3]].round(3))
base = float(s[L, -1].logits([answer])[0])
effect = np.array([[base - float(s[k, -1].ablate(part)[L, -1].logits([answer])[0]) for part in ("all", "attention", "mlp")]
                   for k in range(1, L + 1)])  # [layer j, part]: removing layer j's whole write, its attention part, its MLP part
whole, attn, mlp = effect.T
best = whole.argsort()[::-1][:3]
print("layers", best, "whole", whole[best].round(2), "attn", attn[best].round(2), "mlp", mlp[best].round(2))

# 3. Which positions did it read? Removing what the attention of the layer that wrote the answer most read from each position.
j = int(attn.argmax())  # the layer whose attention wrote most of the answer
taken = [base - float(s[j + 1, -1].ablate(source=p)[L, -1].logits([answer])[0]) for p in range(len(words))]
print(f"layer {j}'s attention, by source", [(words[p], round(v, 2)) for p, v in sorted(enumerate(taken), key=lambda x: -abs(x[1]))[:3]])

# 4. The labels at a state: what it carries of them, their gradients and their semantic bases: the state is the sum of the bases
# of the whole vocabulary weighted by its logits, and a move along a label's basis changes that logit by the step, to first order.
# The gradient also predicts what removing a layer's write does.
at = min(max(decided, L // 2), L - 1)  # from mid-depth on
c = s[at, -1]
g, bases = c.gradient(pair), c.basis(pair)  # [label, hidden] each
print("logits", c.logits(pair), "gradients times bases", (g @ bases.T).round(decimals=3))
size, cosine = bases.norm(dim=-1) / c.vector.norm(), torch.cosine_similarity(bases, c.vector, dim=-1)
print("basis size (in state norms)", size, "cosine with the state", cosine)
write = c.vector - s[at - 1, -1].vector  # what layer at - 1 wrote at the last position
print(f"what layer {at - 1}'s write gave the answer: first order {float(g[0] @ write):+.3f}, exact {whole[at - 1]:+.3f}")

# 5. Change its mind: steer the run at this layer so that the answer's logit at the output falls by 4 and the rival's rises by 4.
# The move is carried forward through the model: the moved run's output layer is the model's output after the move, and its later
# layers show what they carry after it.
moved = c.steer({answer: -4, rival: 4})
change = moved[L, -1].logits(pair) - s[L, -1].logits(pair)
size = (moved[at, -1].vector - c.vector).norm() / c.vector.norm()  # the move, relative to the state
print(moved, "change", change, "move", size)
print("model", names(moved[L, -1].logits().topk(5).indices), "layer", at + 1, names(moved[at + 1, -1].logits().topk(5).indices))
again = moved[at + 1, -1].steer({answer: 2})  # a moved run can be steered again at a later layer
print("steered twice", names(again[L, -1].logits().topk(5).indices))
