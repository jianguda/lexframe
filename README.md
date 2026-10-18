<h1 align="center">LexFrame: Lexical Reference Frames</h1>

<p align="center">A toolkit for interpreting LM residual streams in their native semantics.</p>

<p align="center">
  <a href="https://pypi.org/project/lexframe/"><img src="https://img.shields.io/pypi/v/lexframe?cacheSeconds=3600" alt="PyPI"></a>
  <a href="https://github.com/jianguda/lexframe/actions/workflows/workflow.yml"><img src="https://img.shields.io/github/actions/workflow/status/jianguda/lexframe/workflow.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://arxiv.org/abs/2610.22169"><img src="https://img.shields.io/badge/arXiv-2610.22169-b31b1b" alt="arXiv"></a>
  <a href="https://jianguda.github.io/blogs/lexframe/"><img src="https://img.shields.io/badge/blog-lexframe-orange" alt="Blog"></a>
</p>

<p align="center"><img src="https://raw.githubusercontent.com/jianguda/lexframe/main/assets/overview.png" width="720" alt="LexFrame overview"></p>

LexFrame reads a hidden state the way the model reads it, in the model's own vocabulary.
It is **model-native**: the state is transported layer by layer through the model's own remaining layers, with nothing fitted to the output.
It is **context-aware**: along the way the state attends to its prefix, for a chosen number of rounds of attention.

## Quickstart

```bash
pip install lexframe
```

```python
import lexframe as lf

frame = lf.LexFrame(model, tokenizer).init(texts)            # a loaded Hugging Face model and any unlabelled texts
text = "The Eiffel Tower is located in the city of"
v = frame.view(text, k=5)
v.tokens                                                     # read: the top 5 tokens of every layer at every position
v.first_layer(" Paris")                                      # the first layer where " Paris" tops the reading
frame.attr(text)                                             # attribute: each layer's contribution to the prediction
c = frame.view(text, layers=16)
frame.view(c.steer({" Paris": -4, " London": 4})).tokens     # steer: move layer 16 from Paris toward London
```

[docs/USAGE.md](https://github.com/jianguda/lexframe/blob/main/docs/USAGE.md) lists every command, and [examples/walkthrough.py](https://github.com/jianguda/lexframe/blob/main/examples/walkthrough.py) puts them to work: where SmolLM3 decides that the Eiffel Tower is in Paris, who wrote that decision, and how to change it. [examples/visualize.py](https://github.com/jianguda/lexframe/blob/main/examples/visualize.py) plots every layer's reading and where the state sits among the lexical bases, layer by layer.

## Mechanism

Lenses such as the logit lens and the Jacobian lens read every hidden state of a layer through one fixed frame, so they share a ceiling, and in middle layers they often miss what the model already knows. A frame instead transports a hidden state through the model's own remaining layers, with the context the state attends to, and reads the result in lexical semantics, over the model's own vocabulary. In this frame, predictions can be read, attributed and steered at every layer. [docs/MECHANISM.md](https://github.com/jianguda/lexframe/blob/main/docs/MECHANISM.md) explains the details, and the [blog post](https://jianguda.github.io/blogs/lexframe/) explains why this frame matters.

## Compatibility

LexFrame supports the same causal language models as 🤗 Transformers, including popular model families such as `Llama`, `Qwen`, `Gemma`, `Mistral` and `DeepSeek`.

## Citation

```bibtex
@article{gu2026lexframe,
  title   = {Lexical Reference Frames: Interpreting LM Residual Streams in Native Semantics},
  author  = {Gu, Jian and Zhang, Hongyu and Chen, Chunyang and Aleti, Aldeida},
  journal = {arXiv preprint arXiv:2610.22169},
  year    = {2026}
}
```
