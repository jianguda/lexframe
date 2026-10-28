<h1 align="center">LexFrame: Lexical Reference Frames</h1>

<p align="center">A toolkit for interpreting LM residual streams in their native semantics.</p>

<p align="center">
  <a href="https://pypi.org/project/lexframe/"><img src="https://img.shields.io/pypi/v/lexframe?cacheSeconds=3600" alt="PyPI"></a>
  <a href="https://github.com/jianguda/lexframe/actions/workflows/workflow.yml"><img src="https://img.shields.io/github/actions/workflow/status/jianguda/lexframe/workflow.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://arxiv.org/abs/2610.22169"><img src="https://img.shields.io/badge/arXiv-2610.22169-b31b1b" alt="arXiv"></a>
  <a href="https://jianguda.github.io/blogs/lexframe/"><img src="https://img.shields.io/badge/blog-lexframe-orange" alt="Blog"></a>
</p>

<p align="center"><img src="https://raw.githubusercontent.com/jianguda/lexframe/main/assets/overview.gif" width="720" alt="LexFrame overview"></p>

LexFrame reads a hidden state the way the model reads it, in the model's own vocabulary.
It is **model-native**: the state runs through the model's own remaining layers, with nothing fitted to the output.
It is **context-aware**: it separates what the state carries by itself from what the context adds later, layer by layer and position by position.

## Quickstart

```bash
pip install lexframe
```

```python
import lexframe as lf

frame = lf.LexFrame(model, tokenizer).init(texts)            # a loaded Hugging Face model and any unlabelled texts
text = "The Eiffel Tower is located in the city of"
s = frame.state(text)                                        # the hidden states of every layer and position
s[:, -1].decode(context=False).top(5)                        # decode: the top 5 labels that each layer's state carries
t = frame.trail(text)                                        # trail: where the prediction " Paris" comes from
t.attribute(), t.upstream()                                  # what each layer contributed, and from which positions
moved = s[16, -1].steer({" Paris": -4, " London": 4})        # steer: move layer 16 from Paris toward London
moved.decode().top(5)                                        # the model's output after the move
```

[docs/USAGE.md](https://github.com/jianguda/lexframe/blob/main/docs/USAGE.md) lists every command, and [examples/walkthrough.py](https://github.com/jianguda/lexframe/blob/main/examples/walkthrough.py) puts them to work: where SmolLM3 decides that the Eiffel Tower is in Paris, who wrote that decision, and how to change it. [examples/visualize.py](https://github.com/jianguda/lexframe/blob/main/examples/visualize.py) plots every layer's reading and where the state sits in the frame, layer by layer. The examples need `pip install "lexframe[demo]"`. To try LexFrame without installing anything, [open the quickstart in Colab](https://colab.research.google.com/github/jianguda/lexframe/blob/main/examples/quickstart.ipynb).

## Mechanism

Lenses such as the logit lens and the Jacobian lens read every state of a layer with one fixed map. LexFrame carries each state to the output through the model's remaining layers and reads it there: with context, as the model runs; without context, with every later attention write replaced by its expectation given the state, which shows what the state carries by itself. A trail follows one prediction across layers and tells what the context brought in at each layer and from which positions. [docs/MECHANISM.md](https://github.com/jianguda/lexframe/blob/main/docs/MECHANISM.md) explains the details.

<p align="center"><img src="https://raw.githubusercontent.com/jianguda/lexframe/main/assets/lenses-qwen.png" width="900" alt="The top token each method reads in Qwen3-0.6B"></p>
<p align="center"><img src="https://raw.githubusercontent.com/jianguda/lexframe/main/assets/lenses-minicpm.png" width="900" alt="The top token each method reads in MiniCPM5-1B"></p>
<p align="center"><sub>The top token that each method reads at every second layer of Qwen3-0.6B (top) and MiniCPM5-1B (bottom), shaded by its probability, with the Jacobian lens fitted on 100 sequences. At the last position, LexFrame reads " Paris" from layer 16 in MiniCPM5-1B, while both lenses read it only from layer 18 or later; in Qwen3-0.6B all three read it from layer 22.</sub></p>

## Compatibility

LexFrame supports any causal LM in 🤗 Transformers, including popular model families such as `Llama`, `Qwen`, `Gemma`, `Mistral` and `DeepSeek`, and compact or fully open ones such as `Phi`, `OLMo` and `MiniCPM`.

## Citation

```bibtex
@article{gu2026lexical,
  title={Lexical Reference Frames: Interpreting LM Residual Streams in Native Semantics},
  author={Gu, Jian and Zhang, Hongyu and Chen, Chunyang and Aleti, Aldeida},
  journal={arXiv preprint arXiv:2610.22169},
  year={2026}
}
```
