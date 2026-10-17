"""Retain aligned residual captures and train shared features without changing teachers."""

import json

import torch

from circuit_lantern.crosscoder_schemas import (
    ActivationCaptureRequest,
    CrosscoderTrainingRequest,
    MatchedCaptureRequest,
)
from circuit_lantern.services.crosscoders.model import CrosscoderConfig, SparseCrosscoder
from circuit_lantern.services.crosscoders.planning import capture_budget, training_budget
from circuit_lantern.services.crosscoders.statistics import crosscoder_statistics
from circuit_lantern.services.crosscoders.training import train_crosscoder
from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.features.corpus import (
    CaptureCancelled,
    capture_activations,
    prepare_corpus,
)
from circuit_lantern.services.features.planning import forward_memory
from circuit_lantern.services.hardware import hardware_snapshot
from circuit_lantern.services.tensor_artifacts import ArtifactNotFoundError, TensorArtifactStore

REFERENCES = [
    {
        "title": "Sparse Crosscoders for Cross-Layer Features and Model Diffing",
        "year": 2024,
        "url": "https://transformer-circuits.pub/2024/crosscoders/index.html",
    },
    {
        "title": "Insights on Crosscoder Model Diffing",
        "year": 2025,
        "url": "https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html",
    },
]


def capture_evidence(manifest: dict) -> dict:
    return {
        key: manifest[key]
        for key in (
            "id",
            "name",
            "created_at",
            "source",
            "layer",
            "input_dim",
            "capture_point",
            "sample_count",
            "tensors_sha256",
            "manifest_sha256",
        )
    } | {"corpus_fingerprint": manifest["corpus"]["fingerprint"]}


def capture_summary(manifest: dict) -> dict:
    return capture_evidence(manifest) | {
        "matched_to": manifest.get("matched_to"),
        "train_samples": sum(row["split"] == "train" for row in manifest["corpus"]["samples"]),
        "validation_samples": sum(
            row["split"] == "validation" for row in manifest["corpus"]["samples"]
        ),
    }


def crosscoder_summary(manifest: dict) -> dict:
    return {
        key: manifest[key]
        for key in (
            "id",
            "name",
            "created_at",
            "features",
            "training_summary",
            "train_metrics",
            "validation_metrics",
            "tensors_sha256",
            "manifest_sha256",
        )
    } | {"banks": bank_summaries(manifest["banks"])}


def bank_summaries(banks: list[dict]) -> list[dict]:
    return [{key: value for key, value in bank.items() if key != "training_mean"} for bank in banks]


class CrosscoderLab:
    def __init__(self, runtime, store: TensorArtifactStore):
        self.runtime = runtime
        self.store = store

    def _capture(self, identifier: str) -> dict:
        result = self.store.get(identifier, kind="capture")
        if result is None:
            raise ArtifactNotFoundError("The retained activation capture was not found.")
        return result

    def _load_capture(self, identifier: str) -> tuple[dict, torch.Tensor]:
        manifest, tensors = self.store.load(identifier, kind="capture")
        values = tensors.get("activations")
        if (
            set(tensors) != {"activations"}
            or values is None
            or values.ndim != 2
            or tuple(values.shape) != (len(manifest["corpus"]["samples"]), manifest["input_dim"])
        ):
            raise ValueError("The saved activation matrix does not match its recorded corpus.")
        return manifest, values

    def _prepare_capture(self, request: ActivationCaptureRequest):
        self.runtime._require_model()
        self.runtime._verify_execution(request.expected_execution_id)
        blocks = self.runtime._blocks()
        if (
            not request.name.strip()
            or blocks is None
            or not 0 <= request.layer < len(blocks)
            or self.runtime.model.base_model is self.runtime.model
        ):
            raise ValueError("Choose a named capture and an available transformer block.")
        width = int(self.runtime.metadata["hidden_size"])
        corpus = prepare_corpus(
            self.runtime.tokenizer,
            self.runtime.metadata,
            request.documents,
            seed=request.split_seed,
            max_samples=request.max_samples,
            max_tokens=min(request.max_tokens, self.runtime.settings.max_prompt_tokens),
            exclude_special_tokens=request.exclude_special_tokens,
            exclude_first_position=request.exclude_first_position,
        )
        hardware = hardware_snapshot(self.store.directory)
        budget = capture_budget(
            width,
            len(corpus.samples),
            len(json.dumps(corpus.description(), ensure_ascii=False).encode()),
            forward_memory(
                self.runtime.model,
                max(len(row["token_ids"]) for row in corpus.documents),
                logits=False,
            ),
            self.runtime.device,
            hardware,
            self.store.max_bytes,
        )
        budget.update(
            input_dim=width,
            samples=len(corpus.samples),
            train_samples=len(corpus.train_indices),
            validation_samples=len(corpus.validation_indices),
            layer=request.layer,
            corpus_fingerprint=corpus.fingerprint,
            token_limit=corpus.token_limit,
            teacher_forward_passes=len({row["document"] for row in corpus.samples}),
            truncated_documents=sum(row["truncated"] for row in corpus.documents),
            source=execution_provenance(self.runtime.metadata),
        )
        return corpus, budget

    def capture_plan(self, request: ActivationCaptureRequest) -> dict:
        with self.runtime.operation():
            return self._prepare_capture(request)[1]

    def _matched_request(self, identifier: str, request: MatchedCaptureRequest):
        reference = self._capture(identifier)
        values = dict(
            reference["capture_config"],
            name=request.name,
            layer=request.layer,
            max_tokens=reference["corpus"]["token_limit"],
            expected_execution_id=request.expected_execution_id,
        )
        matched = ActivationCaptureRequest(**values)
        corpus, budget = self._prepare_capture(matched)
        if corpus.fingerprint != reference["corpus"]["fingerprint"]:
            raise ValueError(
                "Tokenization or capture limits differ from the reference. Use matching model inputs and context limits."
            )
        return matched, corpus, budget

    def matched_plan(self, identifier: str, request: MatchedCaptureRequest) -> dict:
        with self.runtime.operation():
            return self._matched_request(identifier, request)[2]

    def start_capture(self, request: ActivationCaptureRequest) -> dict:
        with self.runtime.operation():
            _, plan = self._prepare_capture(request)
            if not plan["compatible"]:
                raise ValueError(" ".join(plan["blockers"]))
            payload = request.model_dump()
            payload["expected_execution_id"] = self.runtime.metadata["execution_id"]
        return self.runtime._start_sweep("activation_capture", payload, self._capture_worker)

    def start_matched(self, identifier: str, request: MatchedCaptureRequest) -> dict:
        with self.runtime.operation():
            matched, _, plan = self._matched_request(identifier, request)
            if not plan["compatible"]:
                raise ValueError(" ".join(plan["blockers"]))
            payload = matched.model_dump()
            payload["expected_execution_id"] = self.runtime.metadata["execution_id"]
            payload["reference_capture_id"] = identifier
        return self.runtime._start_sweep("activation_capture", payload, self._capture_worker)

    def _capture_worker(self, payload: dict, job) -> dict:
        payload = dict(payload)
        reference_id = payload.pop("reference_capture_id", None)
        request = ActivationCaptureRequest(**payload)
        corpus, plan = self._prepare_capture(request)
        if not plan["compatible"]:
            raise ValueError(" ".join(plan["blockers"]))
        reference = self._capture(reference_id) if reference_id else None
        if reference and reference["corpus"]["fingerprint"] != corpus.fingerprint:
            raise ValueError("The reference corpus changed before capture.")
        result = {"kind": "activation_capture", "captures": [], "complete": False}

        def cancelled():
            return bool(job.get("cancel_requested"))

        def progress(done, total):
            job.update(
                progress=90 * done / total, message="Collecting aligned residual activations"
            )

        try:
            values = capture_activations(
                self.runtime.model,
                self.runtime._blocks(),
                corpus,
                request.layer,
                plan["input_dim"],
                device=self.runtime.device,
                progress=progress,
                cancelled=cancelled,
            )
        except CaptureCancelled:
            return result
        manifest = {
            "format": "circuit-lantern-activation-capture-v1",
            "name": request.name,
            "source": execution_provenance(self.runtime.metadata)
            | {
                "architecture": self.runtime.metadata["architecture"],
            },
            "capture_point": "residual_post",
            "layer": request.layer,
            "input_dim": plan["input_dim"],
            "sample_count": len(corpus.samples),
            "corpus": corpus.description(),
            "capture_config": request.model_dump(exclude={"name", "expected_execution_id"}),
            "matched_to": capture_evidence(reference) if reference else None,
        }
        brief = {
            "source": manifest["source"],
            "layer": request.layer,
            "input_dim": plan["input_dim"],
            "sample_count": len(corpus.samples),
            "corpus_fingerprint": corpus.fingerprint,
            "capture_point": "residual_post",
            "matched_to": capture_evidence(reference) if reference else None,
            "train_samples": len(corpus.train_indices),
            "validation_samples": len(corpus.validation_indices),
        }
        saved = self.store.save("capture", manifest, {"activations": values}, brief)
        result.update(complete=True, captures=[capture_summary(saved)])
        return result

    def _training_inputs(self, request: CrosscoderTrainingRequest):
        if not request.name.strip():
            raise ValueError("Give the crosscoder a nonempty name.")
        captures = [self._capture(identifier) for identifier in request.capture_ids]
        if (
            captures[0]["corpus"]["fingerprint"] != captures[1]["corpus"]["fingerprint"]
            or captures[0]["corpus"]["samples"] != captures[1]["corpus"]["samples"]
        ):
            raise ValueError(
                "Choose captures with identical token positions, model inputs and dataset splits."
            )
        widths = tuple(capture["input_dim"] for capture in captures)
        config = CrosscoderConfig(
            features=max(widths) * request.expansion,
            steps=request.steps,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            l1_coefficient=request.l1_coefficient,
            seed=request.seed,
        )
        config.validate(widths)
        hardware = hardware_snapshot(self.store.directory)
        plan = training_budget(
            widths,
            captures[0]["sample_count"],
            config,
            sum(
                len(json.dumps(capture["corpus"], ensure_ascii=False).encode())
                for capture in captures
            ),
            request.device,
            hardware,
            self.store.max_bytes,
        )
        plan.update(
            sources=[capture_evidence(capture) for capture in captures],
            corpus_fingerprint=captures[0]["corpus"]["fingerprint"],
            requested_steps=request.steps,
        )
        return captures, config, plan

    def training_plan(self, request: CrosscoderTrainingRequest) -> dict:
        return self._training_inputs(request)[2]

    def start_training(self, request: CrosscoderTrainingRequest) -> dict:
        _, _, plan = self._training_inputs(request)
        if not plan["compatible"]:
            raise ValueError(" ".join(plan["blockers"]))
        return self.runtime._start_sweep(
            "crosscoder_training",
            request.model_dump(),
            self._train_worker,
            requires_model=False,
        )

    def _train_worker(self, payload: dict, job) -> dict:
        request = CrosscoderTrainingRequest(**payload)
        _, config, plan = self._training_inputs(request)
        if not plan["compatible"]:
            raise ValueError(" ".join(plan["blockers"]))
        loaded = [self._load_capture(identifier) for identifier in request.capture_ids]
        captures = [item[0] for item in loaded]
        values = [item[1] for item in loaded]
        corpus = captures[0]["corpus"]
        train_indices = [
            index for index, row in enumerate(corpus["samples"]) if row["split"] == "train"
        ]
        validation_indices = [
            index for index, row in enumerate(corpus["samples"]) if row["split"] == "validation"
        ]
        train = [value[train_indices] for value in values]
        validation = [value[validation_indices] for value in values]
        result = {"kind": "crosscoder_training", "crosscoders": [], "complete": False}

        def progress(row):
            job.update(
                progress=10 + 75 * row["step"] / config.steps,
                message="Training shared sparse features",
                training_progress=row,
            )

        trained = train_crosscoder(
            train,
            validation,
            config,
            device=request.device,
            progress=progress,
            cancelled=lambda: bool(job.get("cancel_requested")),
        )
        if not trained.completed_steps:
            return result
        job.update(progress=90, message="Measuring joint feature evidence")
        statistics = crosscoder_statistics(
            trained.model,
            values,
            [row["split"] for row in corpus["samples"]],
        )
        banks = [
            capture_evidence(capture)
            | {
                "input_scale": float(trained.model.input_scales[bank]),
                "training_mean": trained.training_means[bank].tolist(),
            }
            for bank, capture in enumerate(captures)
        ]
        summary = {
            "completed_steps": trained.completed_steps,
            "requested_steps": config.steps,
            "tokens_seen": trained.tokens_seen,
            "effective_batch_size": trained.effective_batch_size,
            "unique_train_tokens": len(train_indices),
            "unique_validation_tokens": len(validation_indices),
            "cancelled": trained.cancelled,
        }
        manifest = {
            "format": "circuit-lantern-crosscoder-v1",
            "name": request.name,
            "features": config.features,
            "banks": banks,
            "corpus": corpus,
            "training": config.description(),
            "training_summary": summary,
            "history": trained.history,
            "train_metrics": trained.train_metrics,
            "validation_metrics": trained.validation_metrics,
            "feature_statistics": statistics,
            "analysis_environment": {
                "device": request.device,
                "torch_version": str(torch.__version__),
                "cuda_device": torch.cuda.get_device_name(0) if request.device == "cuda" else None,
            },
            "method_record": {
                "implementation": "circuit_lantern_crosscoder_v1",
                "architecture": "Joint ReLU encoder and one decoder per retained residual bank.",
                "preprocessing": "Each bank is divided by its training-only RMS L2 norm. Zero scales use one. Learned per-bank centering biases start at training geometric medians.",
                "objective": "Mean per-sample squared normalized L2 reconstruction error summed across banks, plus shared feature activation weighted by the sum of the individual decoder column norms.",
                "decoder_constraints": "Joint unit column norm initialization only. Decoder banks remain unconstrained during optimization.",
                "optimizer": "Adam with epsilon 6.25e-10 and gradient norm limit one. Seeded shuffled full batches.",
                "scope": "Acausal joint representation learning. Decoder norm differences do not establish semantic exclusivity or causal model differences. Independent runs do not share feature indices.",
                "references": REFERENCES,
            },
        }
        brief = {
            key: manifest[key]
            for key in (
                "features",
                "banks",
                "training_summary",
                "train_metrics",
                "validation_metrics",
            )
        }
        brief["banks"] = bank_summaries(banks)
        saved = self.store.save("crosscoder", manifest, trained.model.state_dict(), brief)
        result.update(
            complete=not trained.cancelled and trained.completed_steps == config.steps,
            crosscoders=[crosscoder_summary(saved)],
        )
        return result

    def overview(self, identifier: str) -> dict:
        manifest = self.store.get(identifier, kind="crosscoder")
        if manifest is None:
            raise ArtifactNotFoundError("The shared feature comparison was not found.")
        corpus = manifest["corpus"]
        return {
            key: value
            for key, value in manifest.items()
            if key not in ("feature_statistics", "corpus", "banks")
        } | {
            "banks": bank_summaries(manifest["banks"]),
            "corpus": {
                "fingerprint": corpus["fingerprint"],
                "documents": len(corpus["documents"]),
                "samples": len(corpus["samples"]),
                "train_samples": sum(row["split"] == "train" for row in corpus["samples"]),
                "validation_samples": sum(
                    row["split"] == "validation" for row in corpus["samples"]
                ),
                "split_method": corpus["split_method"],
            },
            "manifest_scope": "The checksum identifies the complete stored manifest available through export.",
        }

    def feature_rows(
        self,
        identifier: str,
        *,
        query: str = "",
        sort: str = "frequency",
        direction: str = "highest",
        offset: int = 0,
        limit: int = 50,
    ) -> dict:
        manifest = self.store.get(identifier, kind="crosscoder")
        if manifest is None:
            raise ArtifactNotFoundError("The shared feature comparison was not found.")
        if (
            sort not in ("frequency", "reference", "compared", "cosine")
            or direction not in ("highest", "lowest")
            or offset < 0
            or not 1 <= limit <= 100
        ):
            raise ValueError("Choose a supported feature ranking and page size.")
        rows = list(manifest["feature_statistics"])
        if query.strip():
            needle = query.strip().casefold()
            matches = self.store.search_annotations(identifier, needle)
            rows = [row for row in rows if needle in str(row["index"]) or row["index"] in matches]

        def order(row):
            value = (
                row["frequency"]
                if sort == "frequency"
                else row["decoder_cosine"]
                if sort == "cosine"
                else row["relative_decoder_norms"][0 if sort == "reference" else 1]
                if row["relative_decoder_norms"] is not None
                else None
            )
            return (
                value is None,
                0 if value is None else -value if direction == "highest" else value,
                row["index"],
            )

        rows.sort(key=order)
        selected = rows[offset : offset + limit]
        labels = self.store.annotations(identifier, [row["index"] for row in selected])
        return {
            "crosscoder_id": identifier,
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "features": [
                dict(
                    row,
                    **{
                        key: value
                        for key, value in labels.get(row["index"], {}).items()
                        if key != "feature"
                    },
                )
                for row in selected
            ],
        }

    def feature(self, identifier: str, feature: int) -> dict:
        manifest = self.store.get(identifier, kind="crosscoder")
        if manifest is None:
            raise ArtifactNotFoundError("The shared feature comparison was not found.")
        if not 0 <= feature < manifest["features"]:
            raise ValueError("Choose an existing crosscoder feature.")
        row = dict(manifest["feature_statistics"][feature])
        row.update(
            {
                key: value
                for key, value in self.store.annotations(identifier, [feature])
                .get(feature, {})
                .items()
                if key != "feature"
            }
        )
        examples = []
        for example in row["examples"]:
            sample = manifest["corpus"]["samples"][example["sample"]]
            document = manifest["corpus"]["documents"][sample["document"]]
            position = sample["position"]
            start = max(0, position - 8)
            examples.append(
                example
                | {
                    "document": sample["document"],
                    "position": position,
                    "tokens": document["tokens"][start : position + 9],
                    "active_offset": position - start,
                }
            )
        return row | {"crosscoder_id": identifier, "examples": examples}

    def load_model(self, identifier: str) -> tuple[dict, SparseCrosscoder]:
        manifest, tensors = self.store.load(identifier, kind="crosscoder")
        config = CrosscoderConfig(**manifest["training"])
        model = SparseCrosscoder(
            tuple(bank["input_dim"] for bank in manifest["banks"]), config, initialize=False
        )
        expected = model.state_dict()
        if set(tensors) != set(expected) or any(
            tensors[key].shape != expected[key].shape for key in expected
        ):
            raise ValueError(
                "The saved crosscoder weights do not match their recorded bank layout."
            )
        model.load_state_dict(tensors, assign=True)
        if (model.input_scales <= 0).any():
            raise ValueError("The saved crosscoder scales must be positive.")
        model.eval()
        return manifest, model
