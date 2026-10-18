"""Coordinate bounded model captures, dictionary training and artifact provenance."""

import gc
import json

import torch

from circuit_lantern.feature_schemas import DictionaryTrainingRequest
from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.features.autoencoder import TrainingConfig
from circuit_lantern.services.features.corpus import (
    CaptureCancelled,
    capture_activations,
    prepare_corpus,
)
from circuit_lantern.services.features.examples import DEMONSTRATION_DOCUMENTS
from circuit_lantern.services.features.methods import REFERENCES
from circuit_lantern.services.features.planning import dictionary_plan, forward_memory
from circuit_lantern.services.features.statistics import feature_statistics
from circuit_lantern.services.features.store import DictionaryStore
from circuit_lantern.services.features.training import train_dictionary
from circuit_lantern.services.hardware import hardware_snapshot
from circuit_lantern.services.tokens import capture_token_limit


class FeatureLab:
    def __init__(self, runtime, store: DictionaryStore):
        self.runtime = runtime
        self.store = store

    def capabilities(self) -> dict:
        state = self.runtime.state()
        metadata = state.get("metadata") or {}
        layers = int(metadata.get("num_layers") or 0)
        return {
            "loaded": state["loaded"],
            "supported": bool(
                state["loaded"] and metadata.get("architecture_supported", True) and layers
            ),
            "layers": list(range(layers)),
            "input_dim": metadata.get("hidden_size"),
            "max_capture_tokens": capture_token_limit(
                metadata, self.runtime.settings.max_prompt_tokens
            )
            if state["loaded"]
            else self.runtime.settings.max_prompt_tokens,
            "execution": execution_provenance(metadata) if state["loaded"] else None,
            "example_documents": DEMONSTRATION_DOCUMENTS,
            "references": REFERENCES,
        }

    def _input_dim(self):
        self.runtime._require_model()
        return int(self.runtime.model.get_input_embeddings().weight.shape[1])

    def _prepare(self, request: DictionaryTrainingRequest):
        runtime = self.runtime
        runtime._require_model()
        runtime._verify_execution(request.expected_execution_id)
        blocks = runtime._blocks()
        if blocks is None or runtime.model.base_model is runtime.model:
            raise ValueError("This architecture does not support residual dictionary capture.")
        if not request.name.strip():
            raise ValueError("Give the dictionary a nonempty name.")
        layers = sorted(set(request.layers))
        if any(layer < 0 or layer >= len(blocks) for layer in layers):
            raise ValueError("Choose transformer layers that exist in the loaded model.")
        width = self._input_dim()
        config = TrainingConfig(
            features=width * request.expansion,
            method=request.method,
            active_features=request.active_features,
            steps=request.steps,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            l1_coefficient=request.l1_coefficient,
            seed=request.seed,
        )
        config.validate(width)
        corpus = prepare_corpus(
            runtime.tokenizer,
            runtime.metadata,
            request.documents,
            seed=request.split_seed,
            max_samples=request.max_samples,
            max_tokens=runtime.settings.max_prompt_tokens,
            exclude_special_tokens=request.exclude_special_tokens,
            exclude_first_position=request.exclude_first_position,
        )
        if runtime.device.startswith("cuda"):
            torch.cuda.empty_cache()
        hardware = hardware_snapshot(runtime.settings.data_dir)
        plan = dictionary_plan(
            width,
            config,
            len(corpus.samples),
            len(layers),
            runtime.device,
            hardware,
            self.store.max_weight_bytes,
            corpus_bytes=len(json.dumps(corpus.description(), ensure_ascii=False).encode()),
        )
        capture = forward_memory(
            runtime.model,
            max(len(document["token_ids"]) for document in corpus.documents),
            logits=False,
        )
        if runtime.device.startswith("cuda"):
            plan["estimated_gpu_bytes"] += capture["device_bytes"]
            available = (hardware.get("execution_gpu") or {}).get("free_bytes", 0)
            if plan["estimated_gpu_bytes"] > available * 0.75:
                plan["blockers"].append(
                    "Insufficient free GPU memory for residual capture and dictionary training."
                )
        else:
            plan["estimated_host_bytes"] += capture["device_bytes"]
        if plan["estimated_host_bytes"] > hardware["ram_available_bytes"] * 0.75 and not any(
            "RAM" in reason for reason in plan["blockers"]
        ):
            plan["blockers"].append(
                "Insufficient available RAM for residual capture and dictionary training."
            )
        plan["compatible"] = not plan["blockers"]
        plan["warnings"] = []
        if len(corpus.train_indices) < config.batch_size:
            plan["warnings"].append(
                "The training corpus is smaller than one requested batch. Each step reuses all available training tokens."
            )
        if len(corpus.validation_indices) > len(corpus.train_indices):
            plan["warnings"].append(
                "Held out tokens outnumber training tokens. Add more independent training documents before interpreting dictionary quality."
            )

        plan.update(
            requested_steps=config.steps,
            optimizer_steps_total=config.steps * len(layers),
            teacher_forward_passes=len({sample["document"] for sample in corpus.samples})
            * len(layers),
            train_document_groups=len(
                {
                    document["split_group"]
                    for document in corpus.documents
                    if document["split"] == "train"
                }
            ),
            validation_document_groups=len(
                {
                    document["split_group"]
                    for document in corpus.documents
                    if document["split"] == "validation"
                }
            ),
            unique_train_tokens=len(corpus.train_indices),
            unique_validation_tokens=len(corpus.validation_indices),
            corpus_fingerprint=corpus.fingerprint,
            document_count=len(corpus.documents),
            truncated_documents=sum(document["truncated"] for document in corpus.documents),
            duplicate_documents=corpus.duplicate_documents,
            selected_layers=layers,
        )
        return config, corpus, plan

    def plan(self, request: DictionaryTrainingRequest) -> dict:
        with self.runtime.operation():
            return self._prepare(request)[2]

    def start_training(self, request: DictionaryTrainingRequest) -> dict:
        # Validate and bind before scheduling. The worker repeats resource checks
        # under the model lock immediately before allocating activation storage.
        with self.runtime.operation():
            _, _, plan = self._prepare(request)
            if not plan["compatible"]:
                raise ValueError(" ".join(plan["blockers"]))
            payload = request.model_dump()
            payload["expected_execution_id"] = self.runtime.metadata["execution_id"]
        return self.runtime._start_sweep("sae_training", payload, self._train)

    def _train(self, payload: dict, job) -> dict:
        request = DictionaryTrainingRequest(**payload)
        config, corpus, plan = self._prepare(request)
        if not plan["compatible"]:
            raise ValueError(" ".join(plan["blockers"]))
        layers = plan["selected_layers"]
        completed = []
        result = {
            **execution_provenance(self.runtime.metadata),
            "kind": "sae_training",
            "requested_layers": layers,
            "dictionaries": completed,
            "complete": False,
        }

        def cancelled():
            return bool(job.get("cancel_requested"))

        try:
            for layer_number, layer in enumerate(layers):
                if cancelled():
                    break
                base_progress = 100 * layer_number / len(layers)

                def capturing(done, total, base_progress=base_progress, layer=layer):
                    job.update(
                        progress=base_progress + 15 * done / total / len(layers),
                        message="Collecting residual activations",
                        current_layer=layer,
                    )

                values = capture_activations(
                    self.runtime.model,
                    self.runtime._blocks(),
                    corpus,
                    layer,
                    self._input_dim(),
                    device=self.runtime.device,
                    progress=capturing,
                    cancelled=cancelled,
                )

                def training(row, base_progress=base_progress, layer=layer):
                    job.update(
                        progress=base_progress
                        + (15 + 75 * row["step"] / config.steps) / len(layers),
                        message="Training sparse autoencoder",
                        current_layer=layer,
                        training_progress=row,
                    )

                trained = train_dictionary(
                    values[corpus.train_indices],
                    values[corpus.validation_indices],
                    config,
                    device=self.runtime.device,
                    progress=training,
                    cancelled=cancelled,
                )
                if trained.completed_steps < 1:
                    break
                job.update(message="Measuring feature frequencies and activating examples")
                statistics = feature_statistics(
                    trained.model, values, splits=[sample["split"] for sample in corpus.samples]
                )
                manifest = {
                    **execution_provenance(self.runtime.metadata),
                    "name": request.name if len(layers) == 1 else f"{request.name} · {layer}",
                    "layer": layer,
                    "corpus": corpus.description(),
                    "history": trained.history,
                    "feature_statistics": statistics,
                    "training_mean": values[corpus.train_indices].mean(dim=0).tolist(),
                    "train_metrics": trained.train_metrics,
                    "validation_metrics": trained.validation_metrics,
                    "training_summary": {
                        "completed_steps": trained.completed_steps,
                        "requested_steps": config.steps,
                        "tokens_seen": trained.tokens_seen,
                        "effective_batch_size": trained.effective_batch_size,
                        "unique_train_tokens": len(corpus.train_indices),
                        "unique_validation_tokens": len(corpus.validation_indices),
                        "auxiliary_steps": trained.auxiliary_steps,
                        "cancelled": trained.cancelled,
                    },
                }
                completed.append(self.store.save(manifest, trained.model))
                job.update(
                    result=dict(result, dictionaries=list(completed)),
                    progress=100 * (layer_number + 1) / len(layers),
                )
                stopped = trained.cancelled
                del values, trained, statistics, manifest
                gc.collect()
                if stopped:
                    break
        except CaptureCancelled:
            pass
        except Exception as error:  # noqa: BLE001 - retain completed dictionaries on partial failure
            result["error"] = str(error)[:1000]
        result["complete"] = len(completed) == len(layers) and all(
            not item["training_summary"]["cancelled"] for item in completed
        )
        return result

    def compatible_dictionary(self, manifest: dict):
        self.runtime._require_model()
        current = self.runtime.metadata
        same = manifest["model_id"] == current["id"] and manifest["revision"] == current["revision"]
        saved_config = manifest["execution_config"]
        current_config = current.get("execution_config") or {}
        keys = ("precision", "residual_dtype", "quantization")
        if not same or any(saved_config.get(key) != current_config.get(key) for key in keys):
            raise ValueError(
                "Load the dictionary's recorded checkpoint revision and execution format before using it."
            )
        blocks = self.runtime._blocks()
        if (
            blocks is None
            or manifest["layer"] >= len(blocks)
            or manifest["input_dim"] != self._input_dim()
        ):
            raise ValueError("The dictionary capture point does not match the loaded model.")

    def load_dictionaries(self, identifiers: list[str], token_count: int) -> list[tuple]:
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Choose each dictionary only once.")
        manifests = [self.store.get(identifier) for identifier in identifiers]
        if any(manifest is None for manifest in manifests):
            raise ValueError("A selected dictionary was not found.")
        for manifest in manifests:
            self.compatible_dictionary(manifest)
        if self.runtime.device.startswith("cuda"):
            torch.cuda.empty_cache()
        hardware = hardware_snapshot(self.runtime.settings.data_dir)
        try:
            sizes = [
                self.store._weights_path(identifier).stat().st_size for identifier in identifiers
            ]
        except OSError as error:
            raise ValueError(
                "A selected dictionary weight file is missing or unreadable."
            ) from error
        work = forward_memory(
            self.runtime.model, token_count, sum(manifest["features"] for manifest in manifests)
        )
        host_work = work["host_bytes"]
        if not self.runtime.device.startswith("cuda"):
            host_work += work["device_bytes"]
        if sum(sizes) * 4 + host_work > hardware["ram_available_bytes"] * 0.75:
            raise ValueError("Insufficient available RAM to load the selected dictionaries.")
        if (
            self.runtime.device.startswith("cuda")
            and sum(sizes) * 2 + work["device_bytes"]
            > (hardware.get("execution_gpu") or {}).get("free_bytes", 0) * 0.75
        ):
            raise ValueError("Insufficient free GPU memory to inspect the selected dictionaries.")
        return [
            (manifest, model.to(self.runtime.device))
            for manifest, model in (self.store.load_model(identifier) for identifier in identifiers)
        ]
