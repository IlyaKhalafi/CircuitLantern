"""Bind a new probe to retained inputs and candidate heads from a reference."""

from dataclasses import dataclass

from circuit_lantern.services.induction.comparison_compatibility import assert_probe_alignment
from circuit_lantern.services.induction.comparison_validation import validate_probe
from circuit_lantern.services.induction.identity import identity_digest
from circuit_lantern.services.induction.probes import ProbeBatch, ProbeConfig, RepeatedTokenProbe


class ReferenceProbeNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class MatchedProbe:
    candidates: list[dict]
    summary: dict
    signature: str


class ProbeReferences:
    def __init__(self, store):
        self.store = store

    def size(self, identifier: str | None) -> int:
        if identifier is None:
            return 0
        if self.store is None:
            raise ValueError("Saved probe references are unavailable in this runtime.")
        size = self.store.payload_size(identifier)
        if size is None:
            raise ReferenceProbeNotFoundError(
                "The reference probe was not found. It may have been deleted."
            )
        return size

    def restore(self, request):
        identifier = request.reference_experiment_id
        if identifier is None:
            return None
        record = self.store.get(identifier)
        if record is None:
            raise ReferenceProbeNotFoundError(
                "The reference probe was not found. It may have been deleted."
            )
        if record.get("kind") != "induction_probe" or not isinstance(record.get("result"), dict):
            raise ValueError("Choose a saved induction probe as the matching reference.")
        evidence = validate_probe(record["result"])
        data = evidence.result["dataset"]
        config = ProbeConfig(**data["config"])
        config.validate()
        requested = ProbeConfig(
            request.block_length, request.repeats, request.examples, request.seed
        )
        if config != requested or request.include_start_token != (
            data["start_token_id"] is not None
        ):
            raise ValueError(
                "The saved probes use different inputs or sampling settings. Repeat the probe with the same dataset configuration."
            )
        prefix = int(data["start_token_id"] is not None)
        batch = ProbeBatch(
            config=config,
            probes=tuple(
                RepeatedTokenProbe(
                    tuple(probe["token_ids"]),
                    tuple(probe["target_ids"]),
                    config.block_length,
                    prefix,
                )
                for probe in data["probes"]
            ),
            vocabulary_pool_size=data["vocabulary_pool_size"],
            vocabulary_pool_sha256=data["vocabulary_pool_sha256"],
            start_token_id=data["start_token_id"],
        )
        if batch.fingerprint != evidence.result["dataset_fingerprint"]:
            raise ValueError("The reference probe uses incompatible sampling metadata.")
        return batch, record, evidence

    def prepare(self, restored, candidate_count, active, targets) -> MatchedProbe | None:
        if restored is None:
            return None
        _, record, evidence = restored
        identifier = record["id"]
        assert_probe_alignment(active, evidence.result)
        allowed = {(row["layer"], head) for row in targets for head in range(row["heads"])}
        ranked = sorted(
            (row for coordinate, row in evidence.heads.items() if coordinate in allowed),
            key=lambda row: (-row["prefix_attention"], row["layer"], row["head"]),
        )
        if candidate_count and not ranked:
            raise ValueError(
                "The reference probe has no measured candidate heads in the selected layers."
            )
        chosen = ranked[:candidate_count]
        candidates = [{"layer": row["layer"], "head": row["head"]} for row in chosen]
        summary = {
            "experiment_id": identifier,
            "name": record["name"],
            "model_id": evidence.result["model_id"],
            "revision": evidence.result["revision"],
            "dataset_fingerprint": evidence.result["dataset_fingerprint"],
            "comparison_identity_sha256": evidence.result["comparison_identity_sha256"],
            "measured_examples": evidence.result["measured_examples"],
            "requested_examples": evidence.result["requested_examples"],
            "candidate_scores": [
                {
                    "layer": row["layer"],
                    "head": row["head"],
                    "prefix_attention": row["prefix_attention"],
                }
                for row in chosen
            ],
        }
        signature = identity_digest({key: value for key, value in summary.items() if key != "name"})
        return MatchedProbe(candidates, summary, signature)
