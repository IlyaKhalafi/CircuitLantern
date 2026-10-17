"""Deterministic repeated token inputs with explicit model vocabulary provenance."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ProbeConfig:
    block_length: int = 25
    repeats: int = 4
    examples: int = 10
    seed: int = 0

    def validate(self):
        if any(type(value) is not int for value in asdict(self).values()):
            raise ValueError("Probe sizes and seed must be integers.")
        if not 4 <= self.block_length <= 64 or not 2 <= self.repeats <= 4:
            raise ValueError("Choose a block of 4 to 64 tokens and 2 to 4 repeats.")
        if not 1 <= self.examples <= 32 or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Choose 1 to 32 examples and a valid random seed.")


@dataclass(frozen=True)
class RepeatedTokenProbe:
    token_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    block_length: int
    prefix_length: int

    @property
    def first_queries(self) -> tuple[int, ...]:
        return tuple(range(self.prefix_length, self.prefix_length + self.block_length))

    @property
    def repeated_queries(self) -> tuple[int, ...]:
        return tuple(range(self.prefix_length + self.block_length, len(self.token_ids)))

    def description(self) -> dict:
        return {
            "token_ids": list(self.token_ids),
            "target_ids": list(self.target_ids),
            "first_queries": list(self.first_queries),
            "repeated_queries": list(self.repeated_queries),
        }


@dataclass(frozen=True)
class ProbeBatch:
    config: ProbeConfig
    probes: tuple[RepeatedTokenProbe, ...]
    vocabulary_pool_size: int
    vocabulary_pool_sha256: str
    start_token_id: int | None

    def description(self) -> dict:
        return {
            "config": asdict(self.config),
            "start_token_id": self.start_token_id,
            "vocabulary_pool_size": self.vocabulary_pool_size,
            "vocabulary_pool_sha256": self.vocabulary_pool_sha256,
            "sampling": "Uniform distinct nonspecial token IDs within each block.",
            "target_policy": "The next token in the known periodic sequence, including the final prompt position.",
            "probes": [probe.description() for probe in self.probes],
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.description())


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _token_set(values: Iterable[int]) -> set[int]:
    result = set()
    for value in values:
        if type(value) is not int or not 0 <= value <= 2**63 - 1:
            raise ValueError("Vocabulary and special token IDs must be nonnegative int64 values.")
        result.add(value)
    return result


def build_probe_batch(
    vocabulary_ids: Iterable[int],
    special_token_ids: Iterable[int],
    start_token_id: int | None,
    config: ProbeConfig,
    *,
    context_limit: int,
) -> ProbeBatch:
    config.validate()
    if type(context_limit) is not int or context_limit < 1:
        raise ValueError("The probe requires a positive context limit.")
    vocabulary = _token_set(vocabulary_ids)
    excluded = _token_set(special_token_ids)
    if start_token_id is not None and (
        type(start_token_id) is not int or start_token_id not in vocabulary
    ):
        raise ValueError("The start token must belong to the model vocabulary.")
    prefix = () if start_token_id is None else (start_token_id,)
    if config.block_length * config.repeats + len(prefix) > context_limit:
        raise ValueError("The repeated token probe exceeds the active context allowance.")
    if start_token_id is not None:
        excluded.add(start_token_id)
    pool = sorted(vocabulary - excluded)
    if len(pool) < config.block_length:
        raise ValueError("The nonspecial vocabulary is too small for a distinct token block.")
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    pool_tensor = torch.tensor(pool, dtype=torch.int64, device="cpu")
    probes = []
    for _ in range(config.examples):
        indices = torch.randperm(len(pool), generator=generator, device="cpu")[
            : config.block_length
        ]
        block = tuple(pool_tensor[indices].tolist())
        tokens = prefix + block * config.repeats
        targets = tuple(
            block[(position - len(prefix) + 1) % len(block)] for position in range(len(tokens))
        )
        probes.append(RepeatedTokenProbe(tokens, targets, config.block_length, len(prefix)))
    return ProbeBatch(config, tuple(probes), len(pool), _digest(pool), start_token_id)
