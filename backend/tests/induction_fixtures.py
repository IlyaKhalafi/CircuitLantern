import torch

from circuit_lantern.services.execution import describe_execution
from circuit_lantern.services.induction.engine import measure_induction
from circuit_lantern.services.induction.probes import ProbeConfig, build_probe_batch
from circuit_lantern.services.interventions import attention_head_targets


def capture(runtime, *, fine=False, causal=0):
    runtime.metadata.update(
        id="fine-checkpoint" if fine else "base-checkpoint",
        revision="fine-revision" if fine else "base-revision",
        execution_id=("b" if fine else "a") * 32,
        execution_precision="float32",
        execution_config={
            **describe_execution(runtime.model, "float32", "cpu"),
            "context_length": 64,
        },
    )
    batch = build_probe_batch(
        runtime.tokenizer.get_vocab().values(),
        runtime.tokenizer.all_special_ids,
        0,
        ProbeConfig(block_length=4, repeats=2, examples=3, seed=19),
        context_limit=64,
    )
    return measure_induction(
        runtime.model,
        runtime.tokenizer,
        runtime._blocks(),
        runtime.metadata,
        runtime.device,
        batch,
        attention_head_targets(runtime._blocks()),
        causal,
        {},
        {"requested_forward_passes": 3 * (1 + causal)},
    )


def paired_captures(runtime):
    base = capture(runtime, causal=4)
    with torch.no_grad():
        runtime.model.transformer.h[0].attn.c_attn.weight[0, 0].add_(0.2)
        runtime.model.transformer.h[1].mlp.c_fc.weight.mul_(1.05)
    fine = capture(runtime, fine=True, causal=4)
    return fine, base
