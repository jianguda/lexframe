"""Plot what every layer of SmolLM3 reads about the Eiffel Tower: the top token at each position, the logits of the answer and the
runner-up that each state carries by itself, and each layer's contribution to the answer (visualize.png); then follow the labels'
semantic bases around the state at the last position through the layers (visualize_frame.png); last, split each layer's
contribution into attention and MLP and steer the run, with the gradient's prediction next to the model's response
(visualize_steering.png). Needs matplotlib."""

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import lexframe as lf

name = "HuggingFaceTB/SmolLM3-3B"
model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(name)
texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]

frame = lf.LexFrame(model, tokenizer).init(texts)

# tokens in any script: a CJK font, where one is installed, covers the characters the default font lacks
cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "CJK" in f.name}, key=lambda n: "SC" not in n)
plt.rcParams["font.family"] = ["DejaVu Sans", *cjk[:1]]
# beige to orange, light enough for black text
sand = ListedColormap(plt.colormaps["YlOrBr"](np.linspace(0, 0.55, 256)))


def label(token: str, quote: bool = True, width: int = 12) -> str:
    """A label as printable text: escaped, shortened, and "#" where it decodes to nothing (e.g. padding rows)."""
    text = repr(token)[1:-1]
    text = text[: width - 3] + "..." if len(text) > width else text
    text = text if quote or text.strip() else "_" * bool(text)  # a bare space would leave its cell blank
    return (f"'{text}'" if quote else text) if text else "#"


text = "The Eiffel Tower is located in the city of"
answer, rival = " Paris", " London"
answer_id, rival_id = (tokenizer.encode(w, add_special_tokens=False)[0] for w in (answer, rival))
top_k = 3  # how many of the logits' top tokens to follow at every layer
extra = [rival]  # further labels whose directions to follow

state = frame.state(text)  # every layer and position
carried = state.logits().float().softmax(-1).topk(top_k)  # what each state carries by itself, [layer, position, top_k]
ids_of = lambda t: np.vectorize(lambda i: tokenizer.decode([int(i)]), otypes=[object])(t.cpu().numpy())  # noqa: E731
tokens, probs = ids_of(carried.indices), carried.values.cpu().numpy()
last = state[:, -1]  # every layer at the last position
logit0 = last.logits([answer, rival]).cpu().numpy()  # what each state carries of them: centred logits
base0 = float(state[-1, -1].logits([answer])[0])
attribution = np.array([[base0 - float(state[k, -1].ablate(part)[-1, -1].logits([answer])[0]) for part in ("all", "attention", "mlp")]
                        for k in range(1, frame.num_layers + 1)])  # [layer, 3]: removing each layer's whole write, attention, MLP
contribution = attribution[:, 0]  # each layer's whole write: the change of the answer's logit when it is removed
tops = tokens[:, -1, 0] == answer
first = int(np.argmax(tops)) if tops.any() else -1  # the first layer where the answer tops the logits
layers = range(frame.num_layers + 1)

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(18, 9), gridspec_kw={"width_ratios": [3, 1, 1]}, sharey=True, constrained_layout=True
)

# 1. The top token of the logits at every layer and position, shaded by its probability.
image = ax1.imshow(probs[..., 0], aspect="auto", origin="lower", cmap=sand, vmin=0, vmax=1, alpha=0.8)
for (layer, position), token in np.ndenumerate(tokens[..., 0]):
    ax1.text(position, layer, label(token, quote=False), ha="center", va="center", fontsize=6)
words = [label(tokenizer.decode([i]), quote=False) for i in tokenizer.encode(text, add_special_tokens=False)]
ax1.set_xticks(range(len(words)), words, rotation=45, ha="right")
ax1.set_ylabel("layer (0 is the embedding, the last is the output)")
ax1.set_title("top token of each layer's logits")
fig.colorbar(image, ax=ax1, label="probability", location="left")

# 2. The logits of the answer and the runner-up that each state carries at the last position.
ax2.plot(logit0[:, 0], layers, label=answer.strip())
ax2.plot(logit0[:, 1], layers, label=rival.strip())
if first >= 0:
    ax2.axhline(first, color="grey", linestyle=":", label=f"{answer.strip()} on top")
ax2.set_xlabel("centred logit")
ax2.set_title("answer and runner-up")
ax2.legend()

# 3. Each layer's contribution to the answer's logit at the last position (layer l writes the state of layer l + 1).
ax3.barh([l + 1 for l in range(frame.num_layers)], contribution, color="tab:green")
ax3.set_xlabel("contribution to the logit")
ax3.set_title(f"who wrote {answer.strip()}")

fig.savefig("visualize.png", dpi=150)
print("saved visualize.png")

# 4. Where the state at the last position sits among the labels' semantic bases, layer by layer: its cosine with each basis and
# the logit it carries of that label. Followed are the bases of the input token at this position, where the state starts, of the
# model's output, where it ends, of the extra labels, and at every layer the logits' top_k tokens.
near = tokens[:, -1]  # [layer, top_k]
start = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[-1:])
target = near[-1, 0]
# labels come as text: one that does not encode back to a single token (a special token, a partial character) is not followed
single = lambda token: len(tokenizer.encode(token, add_special_tokens=False)) == 1
roles = {}
for token, role in [(start, "input"), (target, "output"), *((t, "extra") for t in extra)]:
    if single(token):
        roles.setdefault(token, role)
tracked = [t for t in dict.fromkeys([*roles, *near.reshape(-1)]) if single(t)]
names = [label(t) for t in tracked]
b = last.basis(tracked)  # [layer, label, hidden]: the bases
cos = torch.nn.functional.cosine_similarity(b, last.vector[:, None].float(), dim=-1).cpu().numpy()
logit0 = last.logits(tracked).cpu().numpy()

cell = lambda i, l: f"{names[i]} {logit0[l, i]:+.1f} {cos[l, i]:+.2f}"
print(f"layer: the top {top_k} of the logits | " + ", ".join(f"{r} {names[tracked.index(t)]}" for t, r in roles.items()))
print("       (each as token, logit, cosine with its basis)")
for l in layers:
    leading = "  ".join(cell(tracked.index(t), l) for t in near[l] if t in tracked)
    fixed = "  ".join(cell(i, l) for i in range(len(roles)))
    print(f"{l:>5}: {leading} | {fixed}")

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
    ax2.plot(layers, logit0[:, i], **kw)
    if role is None:  # name each nearby direction where the state holds most of it
        l = int(logit0[:, i].argmax())
        ax2.annotate(names[i], (l, logit0[l, i]), fontsize=6, color="dimgrey")
ax1.plot([], [], color="grey", linewidth=0.8, label=f"in the top {top_k} at some layer")
ax1.set_ylabel("cosine of the state with the basis")
ax1.set_title("the state's angle to each basis")
ax2.set_ylabel("centred logit")
ax2.set_title("what the state carries of each token")
fig.legend(*ax1.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8, frameon=False)
for ax in (ax1, ax2):
    ax.set_xlabel("layer")
    ax.xaxis.get_major_locator().set_params(integer=True)
fig.savefig("visualize_frame.png", dpi=150)
print("saved visualize_frame.png")

# 5. Who wrote the answer, split into the attention and MLP sublayers of each layer, and what a steer does: move the run at the last
# position at mid-depth or later from the answer toward the runner-up by a growing amount, then compare the change in their logits
# that the gradients predict with the change in the model's output.
attn, mlp = attribution[:, 1], attribution[:, 2]
at = max(int(first), frame.num_layers // 2)
c = state[at, -1]
o = c.gradient([answer, rival])  # [label, hidden], the model's own
base = state[-1, -1].logits([answer, rival])
sizes = np.linspace(0, 8, 17)
predicted, actual, top = [], [], []
for size in sizes:
    moved = c.steer({answer: -size, rival: size})
    predicted.append((o @ (moved[at, -1].vector - c.vector)).float().cpu().numpy())
    out = moved[-1, -1].logits()  # the model's output after the move
    actual.append((out[[answer_id, rival_id]] - base).float().cpu().numpy())
    top.append(tokenizer.decode([int(out.argmax())]))
predicted, actual = np.stack(predicted), np.stack(actual)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
written = np.arange(1, frame.num_layers + 1)  # layer l writes the state of layer l + 1
ax1.bar(written - 0.2, attn, width=0.4, label="attention")
ax1.bar(written + 0.2, mlp, width=0.4, label="MLP")
ax1.axhline(0, color="grey", linewidth=0.8)
ax1.set_xlabel("layer")
ax1.set_ylabel(f"contribution to the logit of {answer.strip()}")
ax1.set_title(f"attribution: who wrote {answer.strip()}")
ax1.legend()
for i, (token, color) in enumerate([(answer, "tab:red"), (rival, "tab:blue")]):
    ax2.plot(sizes, actual[:, i], color=color, label=f"{token.strip()}, model")
    ax2.plot(sizes, predicted[:, i], color=color, linestyle="--", label=f"{token.strip()}, gradient")
flips = [size for size, t in zip(sizes, top) if t != answer]
if flips:
    ax2.axvspan(flips[0], sizes[-1], color="grey", alpha=0.15, label=f"model says {rival.strip()}")
ax2.set_xlabel(f"steer from {answer.strip()} toward {rival.strip()} at layer {at}")
ax2.set_ylabel("change in centred logit")
ax2.set_title("steering: what the gradient predicts and what the model does")
ax2.legend(fontsize=8)
fig.savefig("visualize_steering.png", dpi=150)
print("saved visualize_steering.png", "top token by steer", dict(zip(sizes.round(1).tolist(), top)))
