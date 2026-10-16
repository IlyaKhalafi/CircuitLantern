"""Shared dictionary provenance for artifact tests."""


def manifest():
    return {
        "name": "Layer 0 dictionary",
        "model_id": "research/model",
        "revision": "pinned-revision",
        "layer": 0,
        "execution_precision": "float32",
        "train_metrics": {"mse": 0.25},
        "validation_metrics": {"mse": 0.5},
        "training_mean": [0.0] * 4,
        "execution_id": "a" * 32,
        "execution_config": {"precision": "float32"},
        "corpus": {
            "fingerprint": "b" * 64,
            "documents": [{"text": "test"}],
            "samples": [{"document": 0, "position": 1}],
            "split_method": "document groups",
            "special_tokens_excluded": True,
            "first_position_excluded": True,
        },
        "training_summary": {"completed_steps": 20, "tokens_seen": 40, "effective_batch_size": 2},
    }
