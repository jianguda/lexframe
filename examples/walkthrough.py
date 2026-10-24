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

frame = lf.LexFrame(model, tokenizer).init(texts, size=1024, tol=0.05)
print(frame, frame.identities, frame.num_layers)

text = "The Eiffel Tower is located in the city of"
answer, rival = " Paris", " London"

# 1. What does each state carry? The closure reading of every layer at every position.
v = frame.view(text, k=3)
for layer in v.layers:
    print(layer, v.tokens[layer, -1], v.probs[layer, -1].round(2), v.kl[layer, -1].round(2))
decided = int(v.first_layer(answer)[-1])
decided = decided if 0 < decided < frame.num_layers else frame.num_layers // 2
print("the answer tops the reading from layer", decided)
print("last position only", frame.view(text, k=3, positions=-1).tokens[:, 0, 0])
print("texts of any length", [c.kl.shape for c in frame.view([text, "The Louvre is in"])])

# 2. What did the context add after each layer, and from which positions? The ledger of the last position.
led = frame.ledger(text, target=answer)
words = [tokenizer.decode([i]) for i in tokenizer.encode(text, add_special_tokens=False)]
print("reading of the answer per layer", led.prob.round(3))
for j in np.argsort(-np.abs(led.step))[:3]:
    src = led.sources[j] if led.sources is not None else None
    top = [] if src is None else [(words[s], round(float(src[s]), 3)) for s in np.argsort(-np.abs(src))[:3]]
    parts = f"innovation {led.innovation[j]:+.3f} + noise {led.noise[j]:+.3f}"
    print(f"layer {j}: step {led.step[j]:+.3f} = {parts}, from", top)

# 3. Who wrote it? Each layer's contribution to the logit of the answer, then split into its sublayers.
whole = frame.attr(text, target=answer)[:, -1]
top = whole.argsort()[::-1][:3]
attn = frame.attr(text, "attn", target=answer)[top, -1]
mlp = frame.attr(text, "mlp", target=answer)[top, -1]
print("layers", top, "whole", whole[top].round(2), "attn", attn.round(2), "mlp", mlp.round(2))

# 4. How far can any fixed lens go? The bound is the best mean cosine with the model's readouts.
prompts = ["The Louvre is a museum in the city of", "The Seine flows through the city of", text]
print("bound", frame.bound(prompts, answer).round(2))

# 5. The coordinates of the state: its basis, readout and coefficients in the native frame, and the frame gap to the closure.
at = max(decided, frame.num_layers // 2)  # from mid-depth on, the frame predicts the effect of a steer well
s = frame.view(text, layers=at)
b, o = s.basis([answer, rival]), s.readout([answer, rival])
print("coef", s.coef([answer, rival])[-1], "closure coef", s.coef([answer, rival], "closure")[-1])
print("duality", (o @ b.mT)[-1], "frame gap", s.gap([answer, rival])[-1])
nxt = frame.view(text, layers=at + 1)
print("written by the layer", (nxt.readout([answer, rival]) @ (nxt.vector - s.vector)[..., None])[-1, :, 0])

# 6. Change its mind: steer the state along the basis; the native frame gives the model's output, the closure what it now carries.
moved = s.steer({answer: -4, rival: 4})
actual = moved.coef([answer, rival]) - s.coef([answer, rival])
print(moved, "predicted", (o @ (moved - s).vector[..., None])[-1, :, 0], "actual", actual[-1])
print("model", frame.view(moved, frame="native").tokens[-1], "closure", frame.view(moved).tokens[-1])

# 7. More integration points and the solver.
frame.config(steps=100, ridge=1e-3, warm=True, samples=256)
print(frame.solver, frame.samples, "duality", (o @ frame.view(s).basis([answer, rival]).mT)[-1])
