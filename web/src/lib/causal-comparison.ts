import {
  executionProfileKnown,
  sameExecutionProfile,
} from "@/lib/comparison-profile";
import type { CausalResult } from "@/lib/types";

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function tokenIds(ids: unknown, length: number): ids is number[] {
  return (
    Array.isArray(ids) &&
    ids.length === length &&
    ids.every((id) => Number.isSafeInteger(id) && id >= 0)
  );
}

function sameIds(selected: number[], reference: number[]) {
  return (
    selected.length === reference.length &&
    selected.every((id, index) => id === reference[index])
  );
}

export function hasCausalComparisonMeasurements(result: CausalResult): boolean {
  if (!result || typeof result !== "object") return false;
  const count = result.tokens?.length;
  return (
    [
      "model_id",
      "revision",
      "clean_prompt",
      "corrupted_prompt",
      "method",
      "device",
    ].every(
      (key) =>
        typeof result[key as keyof CausalResult] === "string" &&
        (result[key as keyof CausalResult] as string).length > 0,
    ) &&
    Array.isArray(result.tokens) &&
    count > 0 &&
    result.tokens.every((token) => typeof token === "string") &&
    Array.isArray(result.corrupted_tokens) &&
    result.corrupted_tokens.length === count &&
    result.corrupted_tokens.every((token) => typeof token === "string") &&
    tokenIds(result.token_ids, count) &&
    tokenIds(result.corrupted_token_ids, count) &&
    Number.isSafeInteger(result.target_token_id) &&
    result.target_token_id >= 0 &&
    Number.isSafeInteger(result.total_layers) &&
    result.total_layers > 0 &&
    (result.position === null ||
      (Number.isSafeInteger(result.position) &&
        result.position >= -count &&
        result.position < count)) &&
    [result.clean, result.corrupted].every(
      (score) =>
        !!score &&
        finite(score.probability) &&
        score.probability >= 0 &&
        score.probability <= 1 &&
        finite(score.log_probability),
    ) &&
    finite(result.baseline_log_probability_gap) &&
    finite(result.recovery_minimum_gap) &&
    result.recovery_minimum_gap > 0 &&
    typeof result.recovery_defined === "boolean" &&
    Array.isArray(result.layers) &&
    result.layers.every(
      (row, index) =>
        !!row &&
        Number.isSafeInteger(row.layer) &&
        row.layer >= 0 &&
        row.layer < result.total_layers &&
        (index === 0 || row.layer > result.layers[index - 1].layer) &&
        finite(row.probability) &&
        row.probability >= 0 &&
        row.probability <= 1 &&
        finite(row.log_probability) &&
        finite(row.delta_log_probability) &&
        (result.recovery_defined
          ? finite(row.recovery)
          : row.recovery === null),
    ) &&
    (result.complete !== true ||
      (result.layers.length === result.total_layers &&
        !result.cancelled &&
        !result.error))
  );
}

export function causalComparisonProblem(
  selected: CausalResult,
  reference: CausalResult,
): string | null {
  if (
    !hasCausalComparisonMeasurements(selected) ||
    !hasCausalComparisonMeasurements(reference)
  )
    return "A saved sweep is missing finite patch measurements or prompt vocabulary IDs. Capture new sweeps to compare them. The original evidence remains available.";
  if (
    selected.model_id !== reference.model_id ||
    selected.revision !== reference.revision
  )
    return "These sweeps use different checkpoints. Choose captures of the same model and revision.";
  if (
    selected.clean_prompt !== reference.clean_prompt ||
    selected.corrupted_prompt !== reference.corrupted_prompt ||
    !sameIds(selected.token_ids!, reference.token_ids!) ||
    !sameIds(selected.corrupted_token_ids!, reference.corrupted_token_ids!)
  )
    return "The clean or corrupted prompts do not align. Comparison requires the same prompt text and vocabulary IDs at every position.";
  if (selected.target_token_id !== reference.target_token_id)
    return "These sweeps score different target tokens. Choose captures of the same target vocabulary ID.";
  if (!executionProfileKnown(selected) || !executionProfileKnown(reference))
    return "Execution settings are missing from an older capture. Save new sweeps with recorded precision and library versions to compare them.";
  if (!sameExecutionProfile(selected, reference))
    return "Execution settings differ. Comparison requires matching precision, quantization, device, context allowance and library versions.";
  if (
    selected.total_layers !== reference.total_layers ||
    selected.method !== reference.method ||
    selected.recovery_minimum_gap !== reference.recovery_minimum_gap
  )
    return "Patching or scoring conventions differ. Choose captures with the same block layout, method and recovery threshold.";
  if (
    selected.clean.log_probability !== reference.clean.log_probability ||
    selected.clean.probability !== reference.clean.probability ||
    selected.corrupted.log_probability !==
      reference.corrupted.log_probability ||
    selected.corrupted.probability !== reference.corrupted.probability ||
    selected.baseline_log_probability_gap !==
      reference.baseline_log_probability_gap ||
    selected.recovery_defined !== reference.recovery_defined
  )
    return "Baseline scores differ. Repeat the captures under matching deterministic settings before comparing patch effects.";
  return null;
}

export function patchPositionLabel(result: CausalResult) {
  if (result.position === null) return "All token positions";
  const position =
    result.position < 0
      ? result.tokens.length + result.position
      : result.position;
  return "Position " + position + " · " + (result.tokens[position] || "∅");
}

function difference(selected: number | null, reference: number | null) {
  if (selected === null || reference === null) return null;
  const value = selected - reference;
  return Number.isFinite(value) ? value : null;
}

export function causalDifferences(
  selected: CausalResult,
  reference: CausalResult,
) {
  const selectedLayers = new Map(
    selected.layers.map((row) => [row.layer, row]),
  );
  const referenceLayers = new Map(
    reference.layers.map((row) => [row.layer, row]),
  );
  return Array.from({ length: selected.total_layers }, (_, layer) => {
    const value = selectedLayers.get(layer),
      other = referenceLayers.get(layer);
    const selectedDelta = value?.delta_log_probability ?? null;
    const referenceDelta = other?.delta_log_probability ?? null;
    const selectedRecovery = value?.recovery ?? null;
    const referenceRecovery = other?.recovery ?? null;
    return {
      layer,
      selected_delta: selectedDelta,
      reference_delta: referenceDelta,
      delta_difference: difference(selectedDelta, referenceDelta),
      selected_recovery: selectedRecovery,
      reference_recovery: referenceRecovery,
      recovery_difference: difference(selectedRecovery, referenceRecovery),
    };
  });
}
