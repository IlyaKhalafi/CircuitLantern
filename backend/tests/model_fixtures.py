import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def assert_no_forward_hooks(modules):
    """Allow native output recorders while rejecting leaked research hooks."""
    for module in modules:
        for hook in module._forward_hooks.values():
            assert (
                hook.__module__ == "transformers.utils.output_capturing"
                and hook.__name__ == "output_capturing_hook"
            ), f"Unexpected retained hook {hook}"


def tiny_gpt2(*, vocab_size=16, context_length=64, hidden_size=16, layers=2, heads=2):
    """Build deterministic real weights and a small tokenizer without Hub access."""
    torch.manual_seed(42)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=vocab_size,
            n_positions=context_length,
            n_embd=hidden_size,
            n_layer=layers,
            n_head=heads,
            bos_token_id=0,
            eos_token_id=1,
            attn_implementation="eager",
        )
    ).eval()
    words = ["<pad>", "<eos>", "<unk>", "the", "capital", "of", "France", "is", "The", "Poland"]
    vocabulary = {words[i] if i < len(words) else f"token_{i}": i for i in range(vocab_size)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<eos>", unk_token="<unk>", pad_token="<pad>"
    )
    return model, tokenizer


def checkpoint_metadata(model, checkpoint, model_id="local-test", revision="local-revision"):
    from circuit_lantern.services.architecture import describe_model_layout

    config = model.config
    return {
        "id": model_id,
        "revision": revision,
        **describe_model_layout(model),
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "dtype": "float32",
        "has_safetensors": (checkpoint / "model.safetensors").exists(),
        "weight_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "quantization": None,
        "config": config.to_dict(),
    }
