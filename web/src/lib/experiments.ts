import type { Experiment } from "@/lib/types";

export function experimentLabel(kind: Experiment["kind"]) {
  if (kind === "causal_batch") return "Prompt-set patching";
  if (kind === "causal") return "Residual patching";
  if (kind === "head_sweep") return "Head ablation";
  if (kind === "matrix") return "Matrix analysis";
  if (kind === "induction_probe") return "Induction probe";
  return "Activation trace";
}

export function experimentArtifact(experiment: Experiment) {
  return experiment.kind === "causal_batch" ||
    experiment.kind === "causal" ||
    experiment.kind === "head_sweep" ||
    experiment.kind === "matrix" ||
    experiment.kind === "induction_probe"
    ? experiment.result
    : experiment.trace;
}

export function notebookHref(
  current: string,
  changes: Record<string, string | null>,
) {
  const params = new URLSearchParams(current);
  for (const [key, value] of Object.entries(changes)) {
    if (value === null) params.delete(key);
    else params.set(key, value);
  }
  const query = params.toString();
  return "/experiments" + (query ? "?" + query : "");
}
