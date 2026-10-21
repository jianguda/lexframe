"""Plot what every layer of SmolLM3 reads about the Eiffel Tower: the top token at each position, the coefficients of
the answer and the runner-up, and each layer's contribution to the answer (visualize.png); then follow the lexical bases
around the state at the last position through the layers (visualize_bases.png); and, as in the paper, compare the
directions of LexFrame and of two lenses with the model's own readouts, against the ceiling that no fixed lens can pass
(visualize_directions.png); last, split each layer's contribution into attention and MLP and steer the state, with the
frame's prediction next to the model's response (visualize_steering.png). Needs matplotlib."""

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
texts = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"][:1000]

frame = lf.LexFrame(model, tokenizer).init(texts)

# tokens in any script: a CJK font, where one is installed, covers the characters the default font lacks
cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "CJK" in f.name}, key=lambda n: "SC" not in n)
plt.rcParams["font.family"] = ["DejaVu Sans", *cjk[:1]]
# beige to orange, light enough for black text
sand = ListedColormap(plt.colormaps["YlOrBr"](np.linspace(0, 0.55, 256)))


def label(token: int, quote: bool = True, width: int = 12) -> str:
    """A token as printable text: escaped, shortened, and its id where it decodes to nothing (e.g. padding rows)."""
    text = repr(tokenizer.decode([token]))[1:-1]
    text = text[: width - 3] + "..." if len(text) > width else text
    text = text if quote or text.strip() else "_" * bool(text)  # a bare space would leave its cell blank
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
image = ax1.imshow(v.probs[..., 0], aspect="auto", origin="lower", cmap=sand, vmin=0, vmax=1, alpha=0.8)
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

# 5. As in the paper: the cosine of each method's readout of a label with the model's own readout (the native frame),
# per layer, as a mean over the states of a few sequences and over labels, with a 95% interval over the sequences.
# The ceiling sqrt(kappa) is the best mean cosine that any lens with fixed directions per layer can reach, as
# frame.bound() computes it, from the same readouts. The first position, which attends only to itself, is left out.
native = lf.LexFrame(model, tokenizer, order="inf").init(texts)  # the calibration comes from the cache
methods = {
    "logit lens": LogitLens(model, tokenizer),
    "Jacobian lens": JacobianLens(model, tokenizer).fit(texts, size=32),  # more sequences fit the lens better
    "LexFrame[1]": frame,
    "LexFrame[2]": lf.LexFrame(model, tokenizer, order=2).init(texts),
}
labels = [" the", " of", " and", " in", " city", " people", " world", " time"]
length, count = 32, 8  # tokens per sequence and sequences
seqs = [torch.tensor(i[:length]) for i in (tokenizer.encode(t, add_special_tokens=False) for t in texts)]
seqs = [s for s in seqs if len(s) == length][:count]

cosines = {title: [] for title in methods}
units = []
for s in seqs:
    truth = native.view(s).readout(labels)[:, 1:]  # [layer, position, label, hidden]
    units.append((truth / truth.norm(dim=-1, keepdim=True).clamp_min(1e-30)).sum(1).double())
    for title, method in methods.items():
        r = method.view(s).readout(labels)[:, 1:]
        c = torch.nn.functional.cosine_similarity(r, truth.to(r.device), dim=-1)
        cosines[title].append(c.mean((1, 2)).cpu().numpy())
U = torch.stack(units)  # [sequence, layer, label, hidden]: each sequence's sum of unit readouts
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
ax.set_ylabel("cosine with the model's own readout")
ax.set_ylim(0, 1.03)
ax.xaxis.get_major_locator().set_params(integer=True)
ax.legend(fontsize=8)
fig.savefig("visualize_directions.png", dpi=150)
print("saved visualize_directions.png")

# 6. Who wrote the answer, split into the attention and MLP sublayers of each layer, and what a steer does: move the
# state at mid-depth or later from the answer toward the runner-up by a growing amount, then compare the change in
# their coefficients that the frame predicts with the change in the model's own output, read in the native frame.
attn = frame.attr(text, "attn", target=answer)[:, -1]
mlp = frame.attr(text, "mlp", target=answer)[:, -1]
at = max(int(first), frame.num_layers // 2)
s = frame.view(text, layers=at)
o = s.readout([answer, rival])
base = native.view(s).coef([answer, rival])[-1]
sizes = np.linspace(0, 8, 17)
predicted, actual, top = [], [], []
for size in sizes:
    moved = s.steer({answer: -size, rival: size})
    predicted.append((o @ (moved - s).vector[..., None])[-1, :, 0].float().cpu().numpy())
    out = native.view(moved)
    actual.append((out.coef([answer, rival])[-1] - base).float().cpu().numpy())
    top.append(out.tokens[-1][0])
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
ax2.set_ylabel("change in coefficient (centred logit)")
ax2.set_title("steering: what the frame predicts and what the model does")
ax2.legend(fontsize=8)
fig.savefig("visualize_steering.png", dpi=150)
print("saved visualize_steering.png", "top token by steer", dict(zip(sizes.round(1).tolist(), top)))
