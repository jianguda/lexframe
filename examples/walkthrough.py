"""Where does SmolLM3 decide that the Eiffel Tower is in Paris, who wrote that decision, and can we change it?"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import lexframe as lf

name = "HuggingFaceTB/SmolLM3-3B"
model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(name)
texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"][:1000]

frame = lf.LexFrame(model, tokenizer, order=1).init(texts, size=256, tol=0.05)
native = lf.LexFrame(model, tokenizer, order="inf").init(texts, size=256)  # the calibration comes from the cache
print(frame, frame.identities, frame.num_layers)

text = "The Eiffel Tower is located in the city of"
answer, rival = " Paris", " London"

# 1. Where does the answer appear? A Carrier holds the hidden states and what the frame reads in them.
v = frame.view(text, k=3)
for layer in v.layers:
    print(layer, v.tokens[layer, -1], v.probs[layer, -1].round(2), v.kl[layer, -1].round(2))
decided = int(v.first_layer(answer)[-1])
decided = decided if 0 <= decided < frame.num_layers else frame.num_layers // 2
print("the answer tops the reading from layer", decided)
at = max(decided, frame.num_layers // 2)  # from mid-depth on, the frame predicts the effect of a steer well
print("last position only", frame.view(text, k=3, positions=-1).tokens[:, 0, 0])
print("texts of any length", [c.kl.shape for c in frame.view([text, "The Louvre is in"])])

# 2. Who wrote it? Each layer's contribution to the logit of the answer, then split into its sublayers.
whole = frame.attr(text, target=answer)[:, -1]
top = whole.argsort()[::-1][:3]
attn = frame.attr(text, "attn", target=answer)[top, -1]
mlp = frame.attr(text, "mlp", target=answer)[top, -1]
print("layers", top, "whole", whole[top].round(2), "attn", attn.round(2), "mlp", mlp.round(2))

# 3. How far can any fixed lens go? The bound is the best mean cosine with the model's readouts.
prompts = ["The Louvre is a museum in the city of", "The Seine flows through the city of", text]
print("bound", frame.bound(prompts, answer).round(2))

# 4. The coordinates of the state at mid-depth or later: its basis, readout and coefficients.
s = frame.view(text, layers=at)
b, o = s.basis([answer, rival]), s.readout([answer, rival])
print("coef", s.coef([answer, rival])[-1], "duality", (o @ b.mT)[-1])
nxt = frame.view(text, layers=at + 1)
print("written by the layer", (nxt.readout([answer, rival]) @ (nxt.vector - s.vector)[..., None])[-1, :, 0])

# 5. Change its mind: steer the state along the basis and read it again; the native frame gives the model's output.
moved = s.steer({answer: -4, rival: 4})
actual = native.view(moved).coef([answer, rival]) - native.view(s).coef([answer, rival])
print(moved, "predicted", (o @ (moved - s).vector[..., None])[-1, :, 0], "actual", actual[-1])
print("model", native.view(moved).tokens[-1], "frame", frame.view(moved).tokens[-1])

# 6. Other orders and the solver.
print("order 0", lf.LexFrame(model, tokenizer, order=0).init(texts, size=256).view(text).kl.mean())
frame.config(steps=100, ridge=1e-3, warm=True)
print(frame.solver, "duality", (o @ frame.view(s).basis([answer, rival]).mT)[-1])
