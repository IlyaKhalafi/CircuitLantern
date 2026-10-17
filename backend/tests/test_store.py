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


def test_payload_size_counts_unicode_storage_bytes_without_decoding(tmp_path):
    import json

    from circuit_lantern.services.store import ExperimentStore

    store = ExperimentStore(tmp_path / "unicode.sqlite3")
    record = store.create_artifact("Unicode evidence", "", "induction_probe", {"model_id": "model"})
    encoded = (
        json.dumps({"model_id": "model", "text": "日本語 فارسی 😀"}, ensure_ascii=False, indent=3)
        + "  "
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE experiments SET payload = ? WHERE id = ?", (encoded, record["id"])
        )
    assert store.payload_size(record["id"]) == len(encoded.encode("utf-8"))
    assert store.payload_size("missing") is None
    assert store.get(record["id"])["result"]["text"] == json.loads(encoded)["text"]
