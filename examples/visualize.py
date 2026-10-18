"""Plot what every layer of SmolLM3 reads about the Eiffel Tower: the top token at each position, the coefficients of
the answer and the runner-up, and each layer's contribution to the answer (visualize.png); then follow the lexical bases
around the state at the last position through the layers (visualize_bases.png). Needs matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import lexframe as lf

name = "HuggingFaceTB/SmolLM3-3B"
model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(name)
texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"][:1000]

frame = lf.LexFrame(model, tokenizer).init(texts)


def label(token: int, quote: bool = True, width: int = 12) -> str:
    """A token as printable text: escaped, shortened, and its id where it decodes to nothing (e.g. padding rows)."""
    text = repr(tokenizer.decode([token]))[1:-1]
    text = text[: width - 1] + "…" if len(text) > width else text
    return (f"'{text}'" if quote else text) if text else f"#{token}"


text = "The Eiffel Tower is located in the city of"
answer, rival = " Paris", " London"
top_k = 3  # how many of the bases nearest to the state to follow at every layer
extra = [rival]  # further labels whose bases to follow

v = frame.view(text, k=top_k)
coef = v.coef([answer, rival])[:, -1].cpu().numpy()
contribution = frame.attr(text, target=answer)[:, -1]
first = v.first_layer(answer)[-1]
layers = range(frame.num_layers + 1)

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(18, 9), gridspec_kw={"width_ratios": [3, 1, 1]}, sharey=True, constrained_layout=True
)

# 1. The top token of the reading at every layer and position, shaded by its probability.
image = ax1.imshow(v.probs[..., 0], aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1)
for (layer, position), token in np.ndenumerate(v.ids[..., 0]):
    ax1.text(position, layer, label(token, quote=False), ha="center", va="center", fontsize=6)
words = [label(i, quote=False) for i in tokenizer.encode(text, add_special_tokens=False)]
ax1.set_xticks(range(len(words)), words, rotation=45, ha="right")
ax1.set_ylabel("layer (0 is the embedding, the last is the output)")
ax1.set_title("top token of each layer's reading")
fig.colorbar(image, ax=ax1, label="probability", location="left")

# 2. The coefficients of the answer and the runner-up at the last position.
ax2.plot(coef[:, 0], layers, label=answer.strip())
ax2.plot(coef[:, 1], layers, label=rival.strip())
if first >= 0:
    ax2.axhline(first, color="grey", linestyle=":", label=f"{answer.strip()} on top")
ax2.set_xlabel("coefficient (centred logit)")
ax2.set_title("answer and runner-up")
ax2.legend()

# 3. Each layer's contribution to the answer's logit at the last position (layer l writes the state of layer l + 1).
ax3.barh([l + 1 for l in range(frame.num_layers)], contribution, color="tab:green")
ax3.set_xlabel("contribution to the logit")
ax3.set_title(f"who wrote {answer.strip()}")

fig.savefig("visualize.png", dpi=150)
print("saved visualize.png")

# 4. Where the state at the last position sits among the lexical bases, layer by layer: its cosine with each basis
# and its coefficient on it, its coordinate in the frame. Followed are the bases of the input token at this position,
# where the state starts, of the model's output, where it ends, of the extra labels, and at every layer the top_k
# nearest ones, those with the largest coefficients.
near = v.ids[:, -1]  # [layer, top_k]
start = tokenizer.encode(text, add_special_tokens=False)[-1]
target = int(near[-1, 0])
roles = {}
for token, role in [(start, "input"), (target, "output"), *((frame.token_id(t), "extra") for t in extra)]:
    roles.setdefault(token, role)
tracked = list(dict.fromkeys([*roles, *(int(t) for t in near.reshape(-1))]))
names = [label(t) for t in tracked]
b = v.basis(tracked)[:, -1].float()  # [layer, token, hidden]
cos = torch.nn.functional.cosine_similarity(b, v.vector[:, -1, None].float(), dim=-1).cpu().numpy()
coef = v.coef(tracked)[:, -1].cpu().numpy()

cell = lambda i, l: f"{names[i]} {coef[l, i]:+.1f} {cos[l, i]:+.2f}"
print(f"layer: the {top_k} nearest bases | " + ", ".join(f"{r} {names[tracked.index(t)]}" for t, r in roles.items()))
print("       (each as token, coefficient, cosine)")
for l in layers:
    nearest = "  ".join(cell(tracked.index(int(t)), l) for t in near[l])
    fixed = "  ".join(cell(i, l) for i in range(len(roles)))
    print(f"{l:>5}: {nearest} | {fixed}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
style = {"input": dict(color="tab:blue", linestyle="--"), "output": dict(color="tab:red", linewidth=2.5)}
others = iter(plt.cm.tab10.colors[1:])
for i, token in enumerate(tracked):
    role = roles.get(token)
    if role is None:
        kw = dict(color="grey", linewidth=0.8, alpha=0.6)
    else:
        kw = dict(style.get(role) or dict(color=next(others)))
        kw["label"] = f"{role} {names[i]}"
    ax1.plot(layers, cos[:, i], **kw)
    ax2.plot(layers, coef[:, i], **kw)
    if role is None:  # name each nearby basis where the state holds most of it
        l = int(coef[:, i].argmax())
        ax2.annotate(names[i], (l, coef[l, i]), fontsize=6, color="dimgrey")
ax1.plot([], [], color="grey", linewidth=0.8, label=f"top {top_k} nearest at some layer")
ax1.set_ylabel("cosine of the state with the basis")
ax1.set_title("the state's angle to each basis")
ax2.set_ylabel("coefficient (centred logit)")
ax2.set_title("the state's coordinate on each basis")
fig.legend(*ax1.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8, frameon=False)
for ax in (ax1, ax2):
    ax.set_xlabel("layer")
    ax.xaxis.get_major_locator().set_params(integer=True)
fig.savefig("visualize_bases.png", dpi=150)
print("saved visualize_bases.png")
