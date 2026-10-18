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
twin = "The Eiffel Tower is found in the city of"
paris, london = (tokenizer.encode(t, add_special_tokens=False)[0] for t in (" Paris", " London"))

# 1. Where does the answer appear? A Carrier holds the hidden states and what the frame reads in them.
v = frame.view(text, k=3)
for layer in v.layers:
    print(layer, v.tokens[layer, -1], v.probs[layer, -1].round(2), v.kl[layer, -1].round(2))
decided = next((l for l in range(frame.num_layers - 1) if v.ids[l, -1, 0] == paris), frame.num_layers - 2)
s = frame.view(text, layers=decided)
print("decided at layer", decided, s.tokens[-1], s.vector.shape)
print("batch", frame.view([text, twin]).kl.shape, "model", native.view(text).kl.max())

# 2. Who wrote it? Each layer's contribution to the logit of the answer, then split into its sublayers.
whole = frame.attr(text, target=paris)[:, -1]
top = whole.argsort()[::-1][:3]
attn = frame.attr(text, "attn", target=paris)[top, -1]
mlp = frame.attr(text, "mlp", target=paris)[top, -1]
print("layers", top, "whole", whole[top].round(2), "attn", attn.round(2), "mlp", mlp.round(2))
print("batch", frame.attr([text, twin]).shape)

# 3. How far can any fixed lens go?
prompts = ["The Louvre is a museum in the city of", "The Seine flows through the city of"]
print("bound", frame.bound(prompts, paris).round(2), frame.bound(prompts, paris, layers=[decided]).round(2))

# 4. The coordinates of the state at the deciding layer: its basis, readout and coefficients.
b, o = s.basis([paris, london]), s.readout([paris, london])
print("coef", s.coef([paris, london])[-1], "duality", (o @ b.mT)[-1])
nxt = frame.view(text, layers=decided + 1)
print("written by the layer", (nxt.readout([paris, london]) @ (nxt.vector - s.vector)[..., None])[-1, :, 0])
print("every layer", v.coef([paris, london]).shape)

# 5. Change its mind: move the state and read it again; the native frame reads the model's own output.
moved = s - 4 * b[:, 0] + 4 * b[:, 1]
actual = native.view(moved).coef([paris, london]) - native.view(s).coef([paris, london])
print(moved, "predicted", (o @ (moved - s).vector[..., None])[-1, :, 0], "actual", actual[-1])
print("model", native.view(moved).tokens[-1], "frame", frame.view(moved).tokens[-1])

# 6. Other orders and the solver.
print("order 0", lf.LexFrame(model, tokenizer, order=0).init(texts, size=256).view(text).kl.mean())
frame.config(steps=100, ridge=1e-3, warm=True)
print(frame.solver, "duality", (o @ frame.view(s).basis([paris, london]).mT)[-1])
