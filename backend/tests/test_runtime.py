from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from circuit_lantern.config import Settings
from circuit_lantern.services.runtime import ModelRuntime


@pytest.fixture
def runtime(tmp_path):
    torch.manual_seed(42)
    instance = ModelRuntime(Settings(data_dir=tmp_path))
    instance.model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=64,
            n_embd=16,
            n_layer=2,
            n_head=2,
            bos_token_id=0,
            eos_token_id=1,
            attn_implementation="eager",
        )
    ).eval()
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<eos>": 1,
                "<unk>": 2,
                "the": 3,
                "capital": 4,
                "of": 5,
                "France": 6,
                "is": 7,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    instance.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<eos>", unk_token="<unk>", pad_token="<pad>"
    )
    instance.metadata = {"id": "local-test", "revision": "test-revision", "context_length": 64}
    yield instance
    instance.executor.shutdown(wait=True)


def test_trace_last_residual_is_before_final_normalization(runtime):
    captured = []
    handle = runtime.model.transformer.h[-1].register_forward_hook(
        lambda m, a, o: captured.append(o[0].detach().clone())
    )
    trace = runtime.trace("the capital of France is", 1, True, None)
    handle.remove()
    expected = float(captured[0][0, -1].norm())
    assert trace["layers"][-1]["norm"] == pytest.approx(expected)
    assert trace["final_state_normalized"] is False
    assert len(trace["attention"]) == 2


def test_block_bypass_preserves_input_residual_and_hooks_are_removed(runtime):
    trace = runtime.trace("the capital", 1, False, 1)
    assert trace["layers"][1]["norm"] == pytest.approx(trace["layers"][2]["norm"])
    assert all(not block._forward_hooks for block in runtime.model.transformer.h)


def test_hook_cleanup_even_when_generation_fails(runtime):
    with (
        patch.object(runtime.model, "generate", side_effect=RuntimeError("generation failed")),
        pytest.raises(RuntimeError, match="generation failed"),
    ):
        runtime.trace("the capital", 1, False, 0)
    assert all(not block._forward_hooks for block in runtime.model.transformer.h)


def test_failed_download_preserves_active_model(runtime):
    active_model = runtime.model
    job = {"status": "queued"}
    with patch(
        "circuit_lantern.services.runtime.snapshot_download", side_effect=OSError("network down")
    ):
        runtime._load(job, {"id": "missing", "revision": "main"}, "cpu")
    assert runtime.model is active_model
    assert job["status"] == "failed"


def test_invalid_interventions_are_rejected(runtime):
    with pytest.raises(ValueError, match="cannot be bypassed"):
        runtime.trace("the capital", 1, False, -1)
    with pytest.raises(ValueError, match="between 1 and 64"):
        runtime.trace("the capital", 0, False, None)


def test_model_context_reserves_generated_tokens(runtime):
    with pytest.raises(ValueError, match="too long"):
        runtime.trace("the " * 60, 8, False, None)
