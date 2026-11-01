"""LexFrame next to two baselines, the logit lens and the Jacobian lens, on SmolLM3: the top token that each reads at
every second layer and every position (baselines_tokens.png), and, as in the paper, how far each method's logits lie from the
model's own output, layer by layer (baselines_divergence.png). Needs matplotlib."""

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
text = "The Eiffel Tower is located in the city of"
methods = {
    "logit lens": LogitLens(model, tokenizer),
    "Jacobian lens": JacobianLens(model, tokenizer).fit(texts[:1000], size=32),  # more sequences fit the lens better
    "LexFrame": frame,
}
L = frame.num_layers

# tokens in any script: a CJK font, where one is installed, covers the characters the default font lacks
cjk = sorted({f.name for f in font_manager.fontManager.ttflist if "CJK" in f.name}, key=lambda n: "SC" not in n)
plt.rcParams["font.family"] = ["DejaVu Sans", *cjk[:1]]
# beige to orange, light enough for black text
sand = ListedColormap(plt.colormaps["YlOrBr"](np.linspace(0, 0.55, 256)))


def label(token: str, width: int = 6) -> str:
    """A label as short printable text: control characters escaped, without its leading space and cut to width."""
    text = repr(token.lstrip(" ") or "_")[1:-1]
    return text if len(text) <= width else text[: width - 1] + "."


def logprobs_of(method, text, layers=slice(None)):
    """What a method reads from the states of `text` at `layers`, as log-probabilities [layer, position, vocabulary]: a lens its own
    decoding, LexFrame what each state carries by itself."""
    return method.state(text)[layers].logits().float().log_softmax(-1).cpu()


# 1. The top token of each method at every second layer and every position, shaded by its probability.
rows = sorted(set(range(0, L + 1, 2)) | {L})
words = [label(tokenizer.decode([i])) for i in tokenizer.encode(text, add_special_tokens=False)]
fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True, layout="constrained")
for ax, (title, method) in zip(axes, methods.items()):
    p, ids = logprobs_of(method, text, rows).exp().max(-1)
    probs = p.numpy()
    image = ax.imshow(probs, aspect="auto", origin="lower", cmap=sand, vmin=0, vmax=1, alpha=0.8)
    for (row, position), i in np.ndenumerate(ids.numpy()):
        ax.text(position, row, label(tokenizer.decode([int(i)])), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(words)), words, rotation=45, ha="right")
    ax.set_title(title)
axes[0].set_yticks(range(len(rows)), rows)
axes[0].set_ylabel("layer (0 is the embedding, the last is the output)")
fig.colorbar(image, ax=axes, label="probability of the top token", shrink=0.6)
fig.savefig("baselines_tokens.png", dpi=150)
print("saved baselines_tokens.png")

# 2. The divergence in nats of each method's logits from the model's output, per layer, as a mean over the positions of a few
# sequences. LexFrame's logits are not fitted to the output: they are what each state carries by itself, so their divergence measures
# what the context has yet to add, and it reaches zero only at the output.
length, count = 64, 8  # tokens per sequence and sequences
seqs = [torch.tensor(i[:length]) for i in (tokenizer.encode(t, add_special_tokens=False) for t in texts[:1000])]
seqs = [s for s in seqs if len(s) == length][:count]

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for title, reader in methods.items():
    def kl(s):  # [layer]: the divergence of each layer's logits from the output, averaged over the positions
        lp = logprobs_of(reader, s)
        return (lp[L].exp() * (lp[L] - lp[:L])).sum(-1).mean(-1).numpy()

    divergence = np.mean([kl(s) for s in seqs], 0)  # the output itself has none
    ax.plot(range(L), np.where(divergence > 1e-8, divergence, np.nan), label=title)
ax.set_yscale("log")
ax.set_xlabel("layer")
ax.set_ylabel("divergence from the model's output (nats)")
ax.xaxis.get_major_locator().set_params(integer=True)
ax.legend(fontsize=8)
fig.savefig("baselines_divergence.png", dpi=150)
print("saved baselines_divergence.png")
