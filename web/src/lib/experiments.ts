import type { Experiment } from "@/lib/types";

export function experimentLabel(kind: Experiment["kind"]) {
  if (kind === "causal") return "Residual patching";
  if (kind === "head_sweep") return "Head ablation";
  return "Activation trace";
}

export function experimentArtifact(experiment: Experiment) {
  return experiment.kind === "causal" || experiment.kind === "head_sweep"
    ? experiment.result
    : experiment.trace;
}
