import type { Experiment } from "@/lib/types";
import type { InductionResult } from "@/lib/induction";

export type ProbeComparisonSource = Pick<
  InductionResult,
  | "model_id"
  | "revision"
  | "execution_id"
  | "execution_precision"
  | "execution_config"
  | "device"
  | "complete"
  | "cancelled"
  | "error"
  | "requested_examples"
  | "measured_examples"
  | "requested_causal_heads"
  | "completed_causal_heads"
  | "dataset_fingerprint"
>;

export interface ProbeComparisonHead {
  layer: number;
  head: number;
  pattern_paired_examples: number;
  selected_prefix: number | null;
  reference_prefix: number | null;
  prefix_difference: number | null;
  selected_duplicate: number | null;
  reference_duplicate: number | null;
  duplicate_difference: number | null;
  selected_previous: number | null;
  reference_previous: number | null;
  previous_difference: number | null;
  causal_paired_examples: number;
  selected_causal_examples: number;
  reference_causal_examples: number;
  selected_causal_delta: number | null;
  reference_causal_delta: number | null;
  causal_delta_difference: number | null;
  selected_kl: number | null;
  reference_kl: number | null;
  kl_difference: number | null;
  causal_probe_indices: number[];
}

export interface ProbeComparison {
  format: string;
  selected_record: Experiment;
  reference_record: Experiment;
  selected_source: ProbeComparisonSource;
  reference_source: ProbeComparisonSource;
  shared_probe_indices: number[];
  shared_examples: number;
  selected_only_examples: number[];
  reference_only_examples: number[];
  shared_pattern_heads: number;
  shared_causal_heads: number;
  first_loss_difference: number;
  repeated_loss_difference: number;
  loss_by_repeat: {
    repeat: number;
    selected: number;
    reference: number;
    difference: number;
  }[];
  heads: ProbeComparisonHead[];
  convention: string;
}

export const probeReferenceStorage = "lantern-induction-reference";

export function matchingProbeHref(identifier: string) {
  if (!/^[a-f0-9]{32}$/.test(identifier))
    throw new Error("The matching probe reference is invalid.");
  try {
    sessionStorage.setItem(probeReferenceStorage, identifier);
  } catch {}
  return "/causal/heads/induction?reference=" + identifier;
}

export function matchingModelLibraryHref(
  identifier: string,
  result: InductionResult,
) {
  const query = new URLSearchParams({
    return: "induction",
    reference: identifier,
    device: result.device,
  });
  if (result.execution_precision)
    query.set("precision", result.execution_precision);
  const context = result.execution_config?.context_length;
  if (typeof context === "number" && Number.isSafeInteger(context))
    query.set("context", String(context));
  return "/?" + query;
}
