"""Validate retained evidence before a saved probe comparison."""

import math
import re
from dataclasses import dataclass
from statistics import pstdev

from circuit_lantern.services.induction.identity import identity_digest

INVALID_EVIDENCE = (
    "A saved probe is missing or has inconsistent comparison evidence. "
    "Capture new probes to compare them. The original evidence remains available."
)


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("At least one measured value is required.")
    return math.fsum(value / len(values) for value in values)


def _expect(condition):
    if not condition:
        raise ValueError(INVALID_EVIDENCE)


def _integer(value, lower=0, upper=2**53 - 1):
    return type(value) is int and lower <= value <= upper


def _number(value, lower=-math.inf, upper=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _close(value, expected):
    return _number(value) and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12)


@dataclass(frozen=True)
class ProbeEvidence:
    result: dict
    heads: dict
    probes: dict
    causal: dict


def _dataset(result, layout):
    data = result["dataset"]
    config = data["config"]
    length, repeats, examples = (config[key] for key in ("block_length", "repeats", "examples"))
    _expect(_integer(length, 4, 64) and _integer(repeats, 2, 4) and _integer(examples, 1, 32))
    _expect(_integer(config["seed"], 0, 2**32 - 1))
    start = data["start_token_id"]
    _expect(start is None or _integer(start, 0, layout["input_vocabulary_rows"] - 1))
    prefix = int(start is not None)
    count = prefix + length * repeats
    _expect(
        _integer(data["vocabulary_pool_size"], length) and _digest(data["vocabulary_pool_sha256"])
    )
    _expect(isinstance(data["probes"], list) and len(data["probes"]) == examples)
    for probe in data["probes"]:
        tokens, targets = probe["token_ids"], probe["target_ids"]
        _expect(
            isinstance(tokens, list)
            and isinstance(targets, list)
            and len(tokens) == len(targets) == count
        )
        _expect(all(_integer(token, 0, layout["input_vocabulary_rows"] - 1) for token in tokens))
        _expect(all(_integer(token, 0, layout["output_vocabulary_rows"] - 1) for token in targets))
        block = tokens[prefix : prefix + length]
        _expect(len(set(block)) == length and (start is None or start not in block))
        _expect(tokens == ([] if start is None else [start]) + block * repeats)
        _expect(targets == [block[(position - prefix + 1) % length] for position in range(count)])
        for field in ("first_queries", "repeated_queries"):
            _expect(
                isinstance(probe[field], list)
                and all(_integer(value, 0, count - 1) for value in probe[field])
            )
        _expect(probe["first_queries"] == list(range(prefix, prefix + length)))
        _expect(probe["repeated_queries"] == list(range(prefix + length, count)))
    _expect(
        _digest(result["dataset_fingerprint"])
        and identity_digest(data) == result["dataset_fingerprint"]
    )
    _expect(
        _integer(result["requested_examples"], 1, 32) and result["requested_examples"] == examples
    )
    return examples, repeats


def _layout(identity):
    _expect(identity["format"] == "circuit-lantern-probe-identity-v1")
    _expect(
        _digest(identity["vocabulary_mapping_sha256"])
        and _integer(identity["mapped_token_entries"], 1)
    )
    layout = identity["layout"]
    _expect(isinstance(layout["model_type"], str) and bool(layout["model_type"].strip()))
    for key in (
        "layers",
        "hidden_size",
        "input_vocabulary_rows",
        "output_vocabulary_rows",
        "key_value_heads",
    ):
        _expect(_integer(layout[key], 1))
    available = {}
    _expect(
        isinstance(layout["attention_heads"], list)
        and len(layout["attention_heads"]) == layout["layers"]
    )
    for row in layout["attention_heads"]:
        layer = row["layer"]
        _expect(_integer(layer, 0, layout["layers"] - 1) and layer not in available)
        _expect(_integer(row["heads"], 1) and _integer(row["head_dim"], 1))
        available[layer] = row
    return layout, available


def _heads(result, available, indices):
    targets = result["layer_targets"]
    _expect(isinstance(targets, list) and 1 <= len(targets) <= 256)
    coordinates = set()
    selected = set()
    for row in targets:
        layer = row["layer"]
        _expect(
            _integer(layer)
            and layer in available
            and layer not in selected
            and row == available[layer]
        )
        selected.add(layer)
        _expect(row["heads"] + len(coordinates) <= 8192)
        coordinates.update((layer, head) for head in range(row["heads"]))
    rows = result["heads"]
    _expect(isinstance(rows, list) and len(rows) == (len(coordinates) if indices else 0))
    heads = {}
    for row in rows:
        _expect(_integer(row["layer"]) and _integer(row["head"]))
        coordinate = row["layer"], row["head"]
        _expect(coordinate in coordinates and coordinate not in heads)
        for field in ("prefix", "duplicate", "previous"):
            values = row[field + "_by_probe"]
            _expect(isinstance(values, list) and len(values) == len(indices))
            _expect(all(_number(value, 0, 1 + 1e-12) for value in values))
            _expect(_close(row[field + "_attention"], mean(values)))
        _expect(_close(row["prefix_std"], pstdev(row["prefix_by_probe"])))
        heads[coordinate] = row
    return heads


def _probe_losses(result, indices, repeats):
    rows = result["probe_metrics"]
    _expect(isinstance(rows, list) and len(rows) == len(indices))
    probes = {}
    for row in rows:
        index, values = row["probe"], row["repeat_nll"]
        _expect(_integer(index) and index in indices and index not in probes)
        _expect(
            isinstance(values, list)
            and len(values) == repeats
            and all(_number(value, 0) for value in values)
        )
        _expect(
            _close(row["first_repeat_nll"], values[0])
            and _close(row["repeated_nll"], mean(values[1:]))
        )
        probes[index] = row
    if probes:
        _expect(
            _close(result["mean_first_repeat_nll"], mean(row["first_repeat_nll"] for row in rows))
        )
        _expect(_close(result["mean_repeated_nll"], mean(row["repeated_nll"] for row in rows)))
    return probes


def _causal(result, heads, probes, examples):
    rows = result["causal_results"]
    _expect(isinstance(rows, list) and len(rows) <= 12)
    _expect(
        _integer(
            result["requested_causal_heads"],
            0,
            min(12, sum(row["heads"] for row in result["layer_targets"])),
        )
    )
    _expect(not rows or len(probes) == examples)
    captured = {}
    for row in rows:
        _expect(_integer(row["layer"]) and _integer(row["head"]))
        coordinate = row["layer"], row["head"]
        _expect(coordinate in heads and coordinate not in captured)
        _expect(
            _integer(row["requested_examples"], 1, 32)
            and row["requested_examples"] == examples
            and isinstance(row["examples"], list)
        )
        count = len(row["examples"])
        _expect(
            1 <= count <= examples
            and _integer(row["completed_examples"], 1, 32)
            and row["completed_examples"] == count
        )
        _expect(type(row["complete"]) is bool and row["complete"] == (count == examples))
        measurements = {}
        for value in row["examples"]:
            index = value["probe"]
            _expect(_integer(index) and index in probes and index not in measurements)
            _expect(
                _number(value["baseline_log_probability"], upper=0)
                and _number(value["ablated_log_probability"], upper=0)
            )
            _expect(_close(value["baseline_log_probability"], -probes[index]["repeated_nll"]))
            _expect(
                _close(
                    value["delta_log_probability"],
                    value["ablated_log_probability"] - value["baseline_log_probability"],
                )
            )
            _expect(_number(value["kl_divergence"], 0))
            measurements[index] = value
        _expect(
            _close(
                row["mean_delta_log_probability"],
                mean(value["delta_log_probability"] for value in row["examples"]),
            )
        )
        _expect(
            _close(
                row["mean_kl_divergence"], mean(value["kl_divergence"] for value in row["examples"])
            )
        )
        captured[coordinate] = measurements
    _expect(len(captured) <= result["requested_causal_heads"])
    _expect(
        _integer(result["completed_causal_heads"], 0, 12)
        and result["completed_causal_heads"] == sum(row["complete"] for row in rows)
    )
    return captured


def validate_probe(result: dict) -> ProbeEvidence:
    try:
        _expect(
            isinstance(result, dict) and result["format"] == "circuit-lantern-induction-probe-v2"
        )
        _expect(
            all(
                isinstance(result[key], str) and result[key].strip()
                for key in ("model_id", "revision", "device")
            )
        )
        _expect(
            isinstance(result.get("execution_id"), str)
            and re.fullmatch("[0-9a-f]{32}", result["execution_id"]) is not None
        )
        identity = result["comparison_identity"]
        _expect(identity_digest(identity) == result["comparison_identity_sha256"])
        layout, available = _layout(identity)
        examples, repeats = _dataset(result, layout)
        _expect(
            isinstance(identity["decoded_probe_sha256"], list)
            and len(identity["decoded_probe_sha256"]) == examples
        )
        _expect(all(_digest(value) for value in identity["decoded_probe_sha256"]))
        count = result["measured_examples"]
        indices = result["attention_probe_indices"]
        _expect(
            _integer(count, 0, examples) and isinstance(indices, list) and len(indices) == count
        )
        _expect(all(_integer(index, 0, examples - 1) for index in indices))
        _expect(indices == sorted(set(indices)))
        heads = _heads(result, available, indices)
        probes = _probe_losses(result, indices, repeats)
        causal = _causal(result, heads, probes, examples)
        _expect(type(result["complete"]) is bool)
        if result["complete"]:
            _expect(
                count == examples
                and result["completed_causal_heads"] == result["requested_causal_heads"]
                and not result.get("cancelled")
                and not result.get("error")
            )
        _expect(
            isinstance(result["method_record"], dict)
            and result["method_record"]["implementation"]
            == "circuit_lantern_repeated_token_probe_v1"
        )
        _expect(
            all(
                isinstance(result["method_record"].get(field), str)
                and result["method_record"][field].strip()
                for field in (
                    "sampling",
                    "prefix_score",
                    "other_scores",
                    "numerics",
                    "targets",
                    "causal_test",
                    "scope",
                )
            )
        )
        return ProbeEvidence(result, heads, probes, causal)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as error:
        raise ValueError(INVALID_EVIDENCE) from error
