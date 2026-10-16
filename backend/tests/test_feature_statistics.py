import torch

from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig
from circuit_lantern.services.features.statistics import feature_statistics


def test_streamed_feature_statistics_match_direct_measurements_across_batches():
    model = SparseAutoencoder(4, TrainingConfig(features=8))
    generator = torch.Generator().manual_seed(15)
    activations = torch.randn(701, 4, generator=generator)
    actual = feature_statistics(model, activations, top_examples=6)
    with torch.inference_mode():
        encoded = model.encode(activations)
    for feature, row in enumerate(actual):
        column = encoded[:, feature]
        values, indices = column.topk(6)
        assert row["activation_count"] == int(column.gt(0).sum())
        assert row["frequency"] == float(column.gt(0).sum()) / len(activations)
        torch.testing.assert_close(torch.tensor(row["mean_activation"]), column.mean())
        assert [item["sample"] for item in row["examples"]] == [
            int(index) for value, index in zip(values, indices, strict=True) if value > 0
        ]
        torch.testing.assert_close(
            torch.tensor([item["activation"] for item in row["examples"]]), values[values > 0]
        )


def test_inactive_features_have_no_invented_activating_examples():
    model = SparseAutoencoder(4, TrainingConfig(features=8))
    with torch.no_grad():
        model.encoder.zero_()
        model.encoder_bias.fill_(-1)
    actual = feature_statistics(model, torch.ones(3, 4))
    assert len(actual) == 8
    assert all(row["examples"] == [] for row in actual)
    assert all(row["frequency"] == 0 and row["max_activation"] == 0 for row in actual)


def test_equal_activation_examples_use_earliest_sample_indices():
    model = SparseAutoencoder(2, TrainingConfig(features=2))
    with torch.no_grad():
        model.encoder.copy_(torch.eye(2))
        model.center.zero_()
        model.encoder_bias.zero_()
    values = torch.ones(701, 2)
    rows = feature_statistics(model, values)
    assert [item["sample"] for item in rows[0]["examples"]] == list(range(6))


def test_feature_statistics_separate_split_frequencies_and_tag_examples():
    model = SparseAutoencoder(2, TrainingConfig(features=2))
    with torch.no_grad():
        model.encoder.copy_(torch.eye(2))
        model.encoder_bias.zero_()
        model.center.zero_()
    values = torch.tensor([[1.0, 0], [1, 1], [0, 1], [0, 2]])
    rows = feature_statistics(model, values, splits=["train", "train", "validation", "validation"])
    assert rows[0]["frequency"] == 0.5
    assert rows[0]["train_frequency"] == 1
    assert rows[0]["validation_frequency"] == 0
    assert all(example["split"] == "train" for example in rows[0]["examples"])
    assert rows[1]["train_frequency"] == 0.5
    assert rows[1]["validation_frequency"] == 1


def test_streamed_feature_batches_preserve_global_ties_and_split_counts():
    from circuit_lantern.services.features.statistics import summarize_feature_batches

    values = torch.tensor([[2.0, 0], [2.0, 3], [0, 3], [2.0, 0], [1.0, 1]])
    rows = summarize_feature_batches(
        values.split(2),
        2,
        5,
        top_examples=3,
        splits=["train", "validation", "train", "validation", "train"],
    )
    assert [example["sample"] for example in rows[0]["examples"]] == [0, 1, 3]
    assert [example["sample"] for example in rows[1]["examples"]] == [1, 2, 4]
    assert rows[0]["frequency"] == 4 / 5
    assert rows[0]["train_frequency"] == 2 / 3
    assert rows[0]["validation_frequency"] == 1
    assert rows[1]["mean_activation"] == 7 / 5
