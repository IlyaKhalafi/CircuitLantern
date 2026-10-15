from circuit_lantern.services.store import ExperimentStore


def test_experiment_lifecycle(tmp_path):
    store = ExperimentStore(tmp_path / "test.db")
    created = store.create("Paris probe", "initial", {"model_id": "gpt2", "prompt": "Paris"})
    assert store.list()[0]["name"] == "Paris probe"
    assert store.get(created["id"])["trace"]["prompt"] == "Paris"
    assert store.update(created["id"], None, "updated")
    assert store.get(created["id"])["notes"] == "updated"
    reopened = ExperimentStore(tmp_path / "test.db")
    assert reopened.get(created["id"])["notes"] == "updated"
    assert store.delete(created["id"])
    assert store.get(created["id"]) is None
    assert not store.delete(created["id"])
