import type { Trace } from "@/lib/types";
import {
  executionProfileKnown,
  sameExecutionProfile,
} from "@/lib/comparison-profile";

export function hasComparisonMeasurements(trace: Trace): boolean {
  return (
    !!trace &&
    typeof trace === "object" &&
    ["model_id", "revision", "prompt", "device"].every(
      (field) =>
        typeof (trace as unknown as Record<string, unknown>)[field] ===
        "string",
    ) &&
    Array.isArray(trace.token_ids) &&
    trace.token_ids.every((id) => Number.isSafeInteger(id) && id >= 0) &&
    Array.isArray(trace.layers) &&
    trace.layers.every(
      (layer) =>
        !!layer &&
        typeof layer === "object" &&
        Number.isSafeInteger(layer.layer) &&
        typeof layer.norm === "number" &&
        Number.isFinite(layer.norm) &&
        typeof layer.entropy === "number" &&
        Number.isFinite(layer.entropy) &&
        Array.isArray(layer.predictions) &&
        (!layer.predictions.length ||
          typeof layer.predictions[0]?.token === "string"),
    )
  );
}

export function traceComparisonProblem(
  selected: Trace,
  reference: Trace,
): string | null {
  if (
    !hasComparisonMeasurements(selected) ||
    !hasComparisonMeasurements(reference)
  )
    return "A saved trace is missing finite residual norms, entropy or prediction measurements required for comparison.";
  if (
    !selected.model_id ||
    !selected.revision ||
    selected.model_id !== reference.model_id ||
    selected.revision !== reference.revision
  )
    return "These traces use different checkpoints. Choose a capture of the same model and revision.";
  if (
    selected.prompt !== reference.prompt ||
    !selected.token_ids.length ||
    selected.token_ids.length !== reference.token_ids.length ||
    selected.token_ids.some((id, index) => id !== reference.token_ids[index])
  )
    return "These prompts do not align. Comparison requires the same prompt and vocabulary IDs at every token position.";
  if (!executionProfileKnown(selected) || !executionProfileKnown(reference))
    return "Execution settings are missing from an older capture. Save new traces with recorded precision and library versions to compare them.";
  if (!sameExecutionProfile(selected, reference))
    return "Execution settings differ. Comparison requires matching precision, quantization, device, context allowance and library versions.";
  if (
    typeof selected.capture_attention !== "boolean" ||
    selected.capture_attention !== reference.capture_attention
  )
    return "Attention capture settings differ or are unavailable. Use the same attention capture option for both traces.";
  if (
    typeof selected.final_state_normalized !== "boolean" ||
    selected.final_state_normalized !== reference.final_state_normalized ||
    !selected.method ||
    selected.method !== reference.method
  )
    return "The residual measurement conventions differ or are unavailable. Save captures with the same normalization and logit lens method.";
  if (
    !selected.layers.length ||
    selected.layers.length !== reference.layers.length ||
    selected.layers.some(
      (layer, index) => layer.layer !== reference.layers[index].layer,
    )
  )
    return "The captured layer positions do not align. Choose traces with the same residual capture layout.";
  return null;
}

export function traceDifferences(selected: Trace, reference: Trace) {
  return selected.layers.map((layer, index) => ({
    layer: layer.layer,
    label: layer.label,
    selected_norm: layer.norm,
    reference_norm: reference.layers[index].norm,
    norm_difference: layer.norm - reference.layers[index].norm,
    entropy_difference: layer.entropy - reference.layers[index].entropy,
    selected_prediction: layer.predictions[0]?.token ?? null,
    reference_prediction: reference.layers[index].predictions[0]?.token ?? null,
  }));
}
