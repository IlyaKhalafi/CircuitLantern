import pytest
from model_fixtures import tiny_gpt2

from circuit_lantern.config import Settings
from circuit_lantern.services.runtime import ModelRuntime


@pytest.fixture
def runtime(tmp_path):
    instance = ModelRuntime(Settings(data_dir=tmp_path))
    instance.model, instance.tokenizer = tiny_gpt2()
    instance.metadata = {"id": "local-test", "revision": "test-revision", "context_length": 64}
    instance._state = {"loaded": True, "metadata": instance.metadata, "device": "cpu"}
    yield instance
    instance.executor.shutdown(wait=True)
