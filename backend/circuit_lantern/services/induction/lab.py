"""Execution bound planning and background jobs for repeated token probes."""

from circuit_lantern.induction_schemas import InductionProbeRequest
from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.hardware import hardware_snapshot
from circuit_lantern.services.head_sweep import select_head_targets
from circuit_lantern.services.induction.engine import METHOD_RECORD, REFERENCES, measure_induction
from circuit_lantern.services.induction.identity import comparison_identity, identity_digest
from circuit_lantern.services.induction.planning import output_vocabulary_size, probe_plan
from circuit_lantern.services.induction.probes import ProbeConfig, build_probe_batch
from circuit_lantern.services.induction.reference import ProbeReferences
from circuit_lantern.services.interventions import attention_head_targets
from circuit_lantern.services.tokens import capture_token_limit, special_token_ids

MAX_PATTERN_HEADS = 8192


class InductionLab:
    def __init__(self, runtime, store=None):
        self.runtime = runtime
        self.references = ProbeReferences(store)

    def _layout(self):
        runtime = self.runtime
        runtime._require_model()
        if runtime.model.training:
            raise ValueError("Repeated token probes require the model in evaluation mode.")
        if getattr(runtime.model.config, "_attn_implementation", None) != "eager":
            raise ValueError("Load the model with eager attention to capture verified patterns.")
        blocks = runtime._blocks()
        heads = attention_head_targets(blocks)
        if not heads or len(heads) != len(blocks):
            raise ValueError("This architecture has no complete verified attention head layout.")
        return blocks, heads

    def _start_token(self, include: bool) -> int | None:
        if not include:
            return None
        token = self.runtime.tokenizer.bos_token_id
        if token is None:
            token = getattr(self.runtime.model.config, "bos_token_id", None)
        if token is not None and (
            type(token) is not int
            or not 0 <= token < self.runtime.model.get_input_embeddings().weight.shape[0]
        ):
            raise ValueError("The configured start token is outside the input vocabulary.")
        return token

    def _vocabulary(self, start, vocabulary=None):
        runtime = self.runtime
        limit = min(
            int(runtime.model.get_input_embeddings().weight.shape[0]),
            output_vocabulary_size(runtime.model),
        )
        vocabulary = runtime.tokenizer.get_vocab() if vocabulary is None else vocabulary
        vocabulary = {
            token for token in vocabulary.values() if type(token) is int and 0 <= token < limit
        }
        # Some tokenizers omit a model supplied start token from their vocabulary map.
        if start is not None:
            vocabulary.add(start)
        return vocabulary

    def capabilities(self) -> dict:
        runtime = self.runtime
        with runtime.operation():
            loaded = runtime.model is not None
            result = {
                "loaded": loaded,
                "supported": False,
                "execution": execution_provenance(runtime.metadata) if loaded else None,
                "layers": [],
                "recommended": None,
                "references": REFERENCES,
                "max_pattern_heads": MAX_PATTERN_HEADS,
            }
            if not loaded:
                return result
            try:
                _, heads = self._layout()
                start = self._start_token(True)
                # Reject an excessive vocabulary allocation before calling get_vocab.
                config = ProbeConfig(block_length=4, repeats=2, examples=1)
                minimal = probe_plan(
                    runtime.model,
                    config,
                    prefix_length=int(start is not None),
                    causal_heads=0,
                    all_heads=heads,
                    selected_heads=heads[:1],
                    device=runtime.device,
                    hardware=hardware_snapshot(runtime.settings.data_dir),
                )
                if not minimal["compatible"]:
                    raise ValueError(" ".join(minimal["blockers"]))
                vocabulary = self._vocabulary(start)
                excluded = special_token_ids(runtime.tokenizer)
                if start is not None:
                    excluded.add(start)
                pool = len(vocabulary - excluded)
                limit = capture_token_limit(runtime.metadata, runtime.settings.max_prompt_tokens)
                recommended = None
                for repeats in (4, 3, 2):
                    length = min(25, pool, (limit - int(start is not None)) // repeats)
                    if length >= 4:
                        recommended = {
                            "block_length": length,
                            "repeats": repeats,
                            "examples": 10,
                            "seed": 0,
                            "causal_heads": 3,
                            "include_start_token": True,
                            "layers": None,
                        }
                        break
                if recommended is None:
                    raise ValueError(
                        "The context or nonspecial vocabulary is too small for a probe."
                    )
                result.update(
                    supported=True,
                    layers=heads,
                    recommended=recommended,
                    token_limit=limit,
                    start_token_id=start,
                    vocabulary_pool_size=pool,
                )
            except (ValueError, AttributeError, NotImplementedError) as error:
                result["reason"] = str(error)
            return result

    def _prepare(self, request: InductionProbeRequest):
        runtime = self.runtime
        runtime._require_model()
        runtime._verify_execution(request.expected_execution_id)
        blocks, heads = self._layout()
        selected = select_head_targets(blocks, request.layers, max_heads=MAX_PATTERN_HEADS)
        config = ProbeConfig(request.block_length, request.repeats, request.examples, request.seed)
        start = (
            self._start_token(request.include_start_token)
            if request.reference_experiment_id is None
            else None
        )
        prefix_length = (
            int(request.include_start_token)
            if request.reference_experiment_id
            else int(start is not None)
        )
        limit = capture_token_limit(runtime.metadata, runtime.settings.max_prompt_tokens)
        if config.block_length * config.repeats + prefix_length > limit:
            raise ValueError("The repeated token probe exceeds the active context allowance.")
        plan = probe_plan(
            runtime.model,
            config,
            prefix_length=prefix_length,
            causal_heads=request.causal_heads,
            all_heads=heads,
            selected_heads=selected,
            device=runtime.device,
            hardware=hardware_snapshot(runtime.settings.data_dir),
            retained_reference_bytes=self.references.size(request.reference_experiment_id),
        )
        plan.update(**execution_provenance(runtime.metadata), selected_layers=selected)
        if not plan["compatible"]:
            return None, selected, plan
        restored = self.references.restore(request)
        vocabulary = runtime.tokenizer.get_vocab()
        if restored is not None:
            batch = restored[0]
            start = batch.start_token_id
        else:
            batch = build_probe_batch(
                self._vocabulary(start, vocabulary),
                special_token_ids(runtime.tokenizer),
                start,
                config,
                context_limit=limit,
            )
        identity = comparison_identity(
            runtime.model, runtime.tokenizer, blocks, batch, vocabulary=vocabulary
        )
        plan.update(
            comparison_identity=identity,
            comparison_identity_sha256=identity_digest(identity),
            dataset_fingerprint=batch.fingerprint,
            vocabulary_pool_size=batch.vocabulary_pool_size,
            start_token_id=start,
        )
        matched = self.references.prepare(
            restored,
            request.causal_heads,
            {
                **execution_provenance(runtime.metadata),
                "device": runtime.device,
                "dataset": batch.description(),
                "dataset_fingerprint": batch.fingerprint,
                "comparison_identity": identity,
                "method_record": METHOD_RECORD,
            },
            selected,
        )
        if matched is not None:
            plan.update(
                reference_probe=matched.summary,
                reference_signature=matched.signature,
                causal_candidates=matched.candidates,
                causal_heads=len(matched.candidates),
                requested_forward_passes=batch.config.examples * (1 + len(matched.candidates)),
            )
        return batch, selected, plan

    def plan(self, request: InductionProbeRequest) -> dict:
        with self.runtime.operation():
            return self._prepare(request)[2]

    def start(self, request: InductionProbeRequest) -> dict:
        with self.runtime.operation():
            batch, _, plan = self._prepare(request)
            if not plan["compatible"]:
                raise ValueError(" ".join(plan["blockers"]))
            payload = request.model_dump()
            payload.update(
                expected_execution_id=self.runtime.metadata["execution_id"],
                expected_probe_fingerprint=batch.fingerprint,
                expected_comparison_identity=plan["comparison_identity_sha256"],
                expected_reference_signature=plan.get("reference_signature"),
            )
        return self.runtime._start_sweep("induction_probe", payload, self._measure)

    def _measure(self, payload: dict, job) -> dict:
        values = dict(payload)
        expected = values.pop("expected_probe_fingerprint")
        expected_identity = values.pop("expected_comparison_identity")
        expected_reference = values.pop("expected_reference_signature", None)
        request = InductionProbeRequest(**values)
        batch, targets, plan = self._prepare(request)
        if not plan["compatible"]:
            raise ValueError(" ".join(plan["blockers"]))
        if batch.fingerprint != expected:
            raise ValueError("The probe vocabulary changed. Start the measurement again.")
        if plan["comparison_identity_sha256"] != expected_identity:
            raise ValueError(
                "The probe token mapping or layout changed. Start the measurement again."
            )
        if plan.get("reference_signature") != expected_reference:
            raise ValueError("The reference probe changed. Start the measurement again.")
        runtime = self.runtime
        result = measure_induction(
            runtime.model,
            runtime.tokenizer,
            runtime._blocks(),
            runtime.metadata,
            runtime.device,
            batch,
            targets,
            plan["causal_heads"],
            job,
            plan,
            causal_candidates=plan.get("causal_candidates"),
        )
        if request.reference_experiment_id:
            result.update(
                reference_experiment_id=request.reference_experiment_id,
                reference_probe=plan["reference_probe"],
                causal_candidate_policy=plan["reference_probe"]["candidate_policy"],
                input_policy="Exact saved reference IDs replayed without resampling. The reference sampling distribution is retained even when this tokenizer designates different special tokens.",
            )
        return result
