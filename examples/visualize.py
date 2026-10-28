"""Plot what every layer of SmolLM3 reads about the Eiffel Tower: the top token at each position, the logits without
context of the answer and the runner-up, and each layer's contribution to the answer (visualize.png); then follow the labels' directions
around the state at the last position through the layers (visualize_frame.png); and, as in the paper, compare the
gradients without context and of two lenses with the model's own gradients (with context), against the ceiling that no fixed lens
can pass (visualize_directions.png); last, split each layer's contribution into attention and MLP and steer the state, with
the frame's prediction next to the model's response (visualize_steering.png). Needs matplotlib."""

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import lexframe as lf
from lexframe.baselines import JacobianLens, LogitLens

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
top_k = 3  # how many of the reading's top tokens to follow at every layer
extra = [rival]  # further labels whose directions to follow

state = frame.state(text)  # every layer and position
tokens, probs = state.decode(context=False).top(top_k)  # the reading without context: what each state carries by itself
last = state[:, -1]  # every layer at the last position
logit0 = last.decode(context=False).logits([answer, rival]).cpu().numpy()  # what each state carries: centred logits without context
attribution = frame.trail(text, label=answer).attribute()  # [layer, 3]: the whole write, its attention part, its MLP part
contribution = attribution[:, 0]  # each layer's whole write: the change of the answer's logit when it is removed
tops = tokens[:, -1, 0] == answer
first = int(np.argmax(tops)) if tops.any() else -1  # the first layer where the answer tops the reading
layers = range(frame.num_layers + 1)

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(18, 9), gridspec_kw={"width_ratios": [3, 1, 1]}, sharey=True, constrained_layout=True
)

# 1. The top token of the reading at every layer and position, shaded by its probability.
image = ax1.imshow(probs[..., 0], aspect="auto", origin="lower", cmap=sand, vmin=0, vmax=1, alpha=0.8)
for (layer, position), token in np.ndenumerate(tokens[..., 0]):
    ax1.text(position, layer, label(token, quote=False), ha="center", va="center", fontsize=6)
words = [label(tokenizer.decode([i]), quote=False) for i in tokenizer.encode(text, add_special_tokens=False)]
ax1.set_xticks(range(len(words)), words, rotation=45, ha="right")
ax1.set_ylabel("layer (0 is the embedding, the last is the output)")
ax1.set_title("top token of each layer's reading")
fig.colorbar(image, ax=ax1, label="probability", location="left")

# 2. The logits without context of the answer and the runner-up at the last position.
ax2.plot(logit0[:, 0], layers, label=answer.strip())
ax2.plot(logit0[:, 1], layers, label=rival.strip())
if first >= 0:
    ax2.axhline(first, color="grey", linestyle=":", label=f"{answer.strip()} on top")
ax2.set_xlabel("logit without context (centred log reading)")
ax2.set_title("answer and runner-up")
ax2.legend()

# 3. Each layer's contribution to the answer's logit at the last position (layer l writes the state of layer l + 1).
ax3.barh([l + 1 for l in range(frame.num_layers)], contribution, color="tab:green")
ax3.set_xlabel("contribution to the logit")
ax3.set_title(f"who wrote {answer.strip()}")

fig.savefig("visualize.png", dpi=150)
print("saved visualize.png")

# 4. Where the state at the last position sits among the labels' directions, layer by layer: its cosine with each direction
# (the label's direction) and its logit without context, what the state carries
# of that token. Followed are the directions of the input token at this position, where the state starts, of the model's output,
# where it ends, of the extra labels, and at every layer the reading's top_k tokens.
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
b = last.direction(tracked).float()  # [layer, label, hidden]: the directions
cos = torch.nn.functional.cosine_similarity(b, last.vector[:, None].float(), dim=-1).cpu().numpy()
logit0 = last.decode(context=False).logits(tracked).cpu().numpy()

cell = lambda i, l: f"{names[i]} {logit0[l, i]:+.1f} {cos[l, i]:+.2f}"
print(f"layer: the reading's top {top_k} | " + ", ".join(f"{r} {names[tracked.index(t)]}" for t, r in roles.items()))
print("       (each as token, logit without context, cosine)")
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
ax1.set_ylabel("cosine of the state with the direction")
ax1.set_title("the state's angle to each direction")
ax2.set_ylabel("logit without context (centred log reading)")
ax2.set_title("what the state carries of each token")
fig.legend(*ax1.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8, frameon=False)
for ax in (ax1, ax2):
    ax.set_xlabel("layer")
    ax.xaxis.get_major_locator().set_params(integer=True)
fig.savefig("visualize_frame.png", dpi=150)
print("saved visualize_frame.png")

# 5. As in the paper: the cosine of each method's gradient of a label with the model's own gradient (with context),
# per layer, as a mean over the states of a few sequences and over labels, with a 95% interval over the sequences.
# The ceiling sqrt(kappa) is the best mean cosine that any lens with fixed directions per layer can reach, computed from the
# same gradients. The gradient without context is the gradient of the state's own reading; one minus its cosine is
# what the context still reinterprets. The first position, which attends only to itself, is left out.
logit, jacobian = LogitLens(model, tokenizer), JacobianLens(model, tokenizer).fit(texts[:1000], size=32)
methods = {
    "logit lens": lambda s: logit.state(s).gradient(labels),
    "Jacobian lens": lambda s: jacobian.state(s).gradient(labels),  # more sequences fit the lens better
    "without context": lambda s: frame.state(s).gradient(labels, context=False),
}
labels = [" the", " of", " and", " in", " city", " people", " world", " time"]
length, count = 32, 8  # tokens per sequence and sequences
seqs = [torch.tensor(i[:length]) for i in (tokenizer.encode(t, add_special_tokens=False) for t in texts[:1000])]
seqs = [s for s in seqs if len(s) == length][:count]

cosines = {title: [] for title in methods}
units = []
for s in seqs:
    truth = frame.state(s).gradient(labels)[:, 1:]  # [layer, position, label, hidden]
    units.append((truth / truth.norm(dim=-1, keepdim=True).clamp_min(1e-30)).sum(1).float().cpu())
    for title, method in methods.items():
        r = method(s)[:, 1:]
        c = torch.nn.functional.cosine_similarity(r, truth.to(r.device), dim=-1)
        cosines[title].append(c.mean((1, 2)).cpu().numpy())
U = torch.stack(units)  # [sequence, layer, label, hidden]: each sequence's sum of unit gradients
n = float(length - 1)
kappa = ((U.sum(0) ** 2).sum(-1) - (U**2).sum((0, -1))) / ((len(seqs) * n) ** 2 - len(seqs) * n**2)
bound = kappa.clamp_min(0).sqrt().mean(-1).cpu().numpy()  # [layer]: the mean over labels

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
ax.plot(layers, bound, color="grey", linestyle=":", label=r"ceiling $\sqrt{\kappa}$")
for title, rows in cosines.items():
    a = np.stack(rows)  # [sequence, layer], NaN where a lens does not read a layer
    boot = np.stack([a[rng.integers(0, len(a), len(a))].mean(0) for _ in range(300)])
    (line,) = ax.plot(layers, a.mean(0), label=title)
    ax.fill_between(layers, *np.percentile(boot, [2.5, 97.5], 0), color=line.get_color(), alpha=0.2, linewidth=0)
ax.set_xlabel("layer")
ax.set_ylabel("cosine with the model's own gradient")
ax.set_ylim(0, 1.03)
ax.xaxis.get_major_locator().set_params(integer=True)
ax.legend(fontsize=8)
fig.savefig("visualize_directions.png", dpi=150)
print("saved visualize_directions.png")

# 6. Who wrote the answer, split into the attention and MLP sublayers of each layer, and what a steer does: move the
# state at mid-depth or later from the answer toward the runner-up by a growing amount, then compare the change in
# their logits that the gradient predicts with the change in the model's own output.
attn, mlp = attribution[:, 1], attribution[:, 2]
at = max(int(first), frame.num_layers // 2)
c = state[at]  # every position at this layer
o = c.gradient([answer, rival])
base = c.decode().logits([answer, rival])[-1]
sizes = np.linspace(0, 8, 17)
predicted, actual, top = [], [], []
for size in sizes:
    moved = c.steer({answer: -size, rival: size})
    predicted.append((o @ (moved.vector - c.vector)[..., None])[-1, :, 0].float().cpu().numpy())
    out = moved.decode()  # with context: the model's output after the move
    actual.append((out.logits([answer, rival])[-1] - base).float().cpu().numpy())
    top.append(out.top(1)[0][-1, 0])
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
    ax2.plot(sizes, predicted[:, i], color=color, linestyle="--", label=f"{token.strip()}, frame")
flips = [size for size, t in zip(sizes, top) if t != answer]
if flips:
    ax2.axvspan(flips[0], sizes[-1], color="grey", alpha=0.15, label=f"model says {rival.strip()}")
ax2.set_xlabel(f"steer from {answer.strip()} toward {rival.strip()} at layer {at}")
ax2.set_ylabel("change in centred logit")
ax2.set_title("steering: what the frame predicts and what the model does")
ax2.legend(fontsize=8)
fig.savefig("visualize_steering.png", dpi=150)
print("saved visualize_steering.png", "top token by steer", dict(zip(sizes.round(1).tolist(), top)))
