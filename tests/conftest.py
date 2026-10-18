"""Tiny random models of several families with a small word-level tokenizer, in 32 bit so that the exact identities
hold to rounding."""

import pytest
import torch

import lexframe as lf

V, D, L = 256, 64, 4
WORDS = "the a of in is to and city tower river paris london rome berlin france was built located near old new".split()
TEXTS = [" ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(40)) for i in range(24)]
TEXT = "the tower is located in the city of"

COMMON = dict(
    vocab_size=V,
    hidden_size=D,
    intermediate_size=128,
    num_hidden_layers=L,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=256,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
MLA = dict(  # multi-head latent attention, whose key and value heads equal the attention heads
    num_key_value_heads=4,
    q_lora_rank=16,
    kv_lora_rank=16,
    qk_rope_head_dim=8,
    qk_nope_head_dim=8,
    v_head_dim=16,
    n_routed_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    first_k_dense_replace=1,
    n_group=1,
    topk_group=1,
)
FAMILIES = {
    "smollm3": ("SmolLM3Config", dict(no_rope_layers=[1, 1, 1, 0], tie_word_embeddings=True)),
    "olmo3": ("Olmo3Config", dict(sliding_window=4)),
    "llama": ("LlamaConfig", {}),
    "qwen3": ("Qwen3Config", dict(head_dim=16)),
    "qwen3_moe": ("Qwen3MoeConfig", dict(head_dim=16, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)),
    "mistral": ("MistralConfig", dict(sliding_window=4)),
    "gemma3": ("Gemma3TextConfig", dict(head_dim=16, sliding_window=4)),
    "phi3": ("Phi3Config", {}),
    "granite": ("GraniteConfig", dict(embedding_multiplier=2.0, residual_multiplier=0.5, logits_scaling=4.0)),
    "gpt_oss": ("GptOssConfig", dict(head_dim=16, num_local_experts=4, num_experts_per_tok=2, sliding_window=4)),
    "deepseek_v2": ("DeepseekV2Config", MLA),
    "deepseek_v3": ("DeepseekV3Config", MLA),
}


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>")


def hf_model(kind, seed=0):
    import transformers

    name, extra = FAMILIES[kind]
    torch.manual_seed(seed)
    cfg = getattr(transformers, name)(**{**COMMON, **extra})
    return transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)


@pytest.fixture(scope="session", params=["smollm3", "olmo3"])
def kind(request):
    return request.param


@pytest.fixture(scope="session")
def frame(kind, tmp_path_factory):
    return lf.LexFrame(hf_model(kind), tokenizer()).init(TEXTS, cache_dir=str(tmp_path_factory.mktemp("cache")))


def with_order(frame, order):
    f = lf.LexFrame(frame._m, order=order)
    f.calibration = frame.calibration
    return f
