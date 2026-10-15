import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


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
