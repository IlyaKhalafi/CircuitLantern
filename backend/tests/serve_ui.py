"""Offline API harness for browser tests. Never imported by the production app."""

import argparse
import hashlib
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from model_fixtures import tiny_gpt2


def serve(port: int):
    with tempfile.TemporaryDirectory(prefix="lantern-browser-") as temporary:
        directory = Path(temporary)
        os.environ["LANTERN_DATA_DIR"] = str(directory / "records")
        os.environ["HF_HUB_OFFLINE"] = "1"
        from circuit_lantern import main
        from circuit_lantern.services import runtime as runtime_module
        from circuit_lantern.services.architecture import describe_model_layout
        from circuit_lantern.services.jobs import JobRecord

        model, tokenizer = tiny_gpt2(
            vocab_size=1000, context_length=512, hidden_size=32, layers=5, heads=4
        )
        checkpoint = directory / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        weight = checkpoint / "model.safetensors"
        digest = hashlib.sha256(weight.read_bytes()).hexdigest()
        metadata = {
            "id": "hf-internal-testing/tiny-random-gpt2",
            "revision": "fixture-" + digest[:32],
            "architecture": "gpt2",
            **describe_model_layout(model),
            "parameter_estimated": False,
            "weight_bytes": weight.stat().st_size,
            "has_safetensors": True,
            "hidden_size": 32,
            "num_layers": 5,
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 8,
            "context_length": 512,
            "vocab_size": 1000,
            "dtype": "float32",
            "quantization": None,
            "gated": False,
            "downloads": 0,
            "config": model.config.to_dict(),
        }
        del model, tokenizer
        native_metadata = None
        native_checkpoint = directory / "fp4-checkpoint"
        native_lock = threading.Lock()

        def fp4_metadata():
            nonlocal native_metadata
            with native_lock:
                if native_metadata is None:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                    from circuit_lantern.services.execution import prepare_model_execution

                    packed = AutoModelForCausalLM.from_pretrained(
                        checkpoint,
                        dtype=torch.float32,
                        device_map={"": "cpu"},
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="fp4",
                            bnb_4bit_compute_dtype=torch.float32,
                        ),
                        attn_implementation="eager",
                        local_files_only=True,
                    )
                    prepare_model_execution(packed, "int4", "cpu")
                    packed.save_pretrained(native_checkpoint, safe_serialization=True)
                    AutoTokenizer.from_pretrained(
                        checkpoint, local_files_only=True
                    ).save_pretrained(native_checkpoint)
                    packed_weight = native_checkpoint / "model.safetensors"
                    fingerprint = hashlib.sha256(packed_weight.read_bytes()).hexdigest()
                    native_metadata = dict(
                        metadata,
                        revision="fixture-fp4-" + fingerprint[:32],
                        weight_bytes=packed_weight.stat().st_size,
                        quantization=packed.config.quantization_config.to_dict(),
                        config=packed.config.to_dict(),
                    )
                return dict(native_metadata)

        def fixture_metadata(model_id: str, revision: str = "main"):
            if model_id != metadata["id"]:
                raise ValueError(
                    "The browser test server only provides the deterministic GPT-2 fixture."
                )
            if revision == "fp4" or (
                native_metadata is not None and revision == native_metadata["revision"]
            ):
                return fp4_metadata()
            if revision not in ("main", "step1000", metadata["revision"]):
                raise ValueError("The requested test revision does not exist.")
            return dict(metadata)

        def fixture_download(*args, **kwargs):
            if (
                native_metadata is not None
                and kwargs.get("revision") == native_metadata["revision"]
            ):
                return str(native_checkpoint)
            return str(checkpoint)

        main.model_metadata = fixture_metadata
        runtime_module.snapshot_download = fixture_download
        original_lifespan = main.app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with original_lifespan(app):
                initial_job = JobRecord({"status": "queued", "progress": 0})
                app.state.runtime._load(initial_job, metadata, "cpu")
                if initial_job["status"] != "ready":
                    raise RuntimeError("Could not load browser fixture. " + initial_job["message"])
                yield

        main.app.router.lifespan_context = lifespan
        uvicorn.run(main.app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the offline browser test API")
    parser.add_argument("--port", type=int, default=8000)
    serve(parser.parse_args().port)
