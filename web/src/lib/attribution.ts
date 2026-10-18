import type { TrackedTarget } from "@/lib/types";

export interface AttributionTarget extends TrackedTarget {
  patchable: boolean;
  source_types: string[];
  prediction_rank: number | null;
}
export interface AttributionComponent {
  kind: "embedding" | "block" | "norm_bias" | "head_bias";
  layer: number | null;
  values: number[];
  update_norm?: number;
}
export interface AttributionChecks {
  actual_logits: number[];
  reconstructed_logits: number[];
  absolute_errors: number[];
  strict_tolerances: number[];
  rounding_allowances: number[];
  head_projected_logits: number[];
  head_absolute_errors: number[];
  native_epsilon: number;
  tf32_allowed: boolean;
  output_dtype: string;
  observed_readout_magnitudes: number[];
  maximum_normalization_error: number;
  residual_maximum_error: number;
  status: "ok" | "rounding";
  tolerance_method: string;
}
export interface LogitAttribution {
  schema_version: 1;
  status: "available" | "unavailable";
  reason: string | null;
  position: number;
  token_id: number;
  targets?: AttributionTarget[];
  components?: AttributionComponent[];
  checks?: AttributionChecks;
  selection_mode?: "top_predictions" | "tracked_and_predictions";
  normalization?: {
    kind: "layer_norm" | "rms_norm" | "identity";
    module: string;
    epsilon: number;
    denominator: number;
    statistic_dtype: string;
    weight_mode: string;
    has_bias: boolean;
  };
  captured_dtype?: string;
  readout_dtype?: string;
  compute_dtype?: string;
  retained_vectors?: boolean;
  scope?: string;
  method?: string;
}
export function contributionLabel(row: AttributionComponent): string {
  return row.kind === "block"
    ? "Block {layer} update"
    : {
        embedding: "Embedding input",
        norm_bias: "Normalization bias",
        head_bias: "Readout bias",
      }[row.kind];
}
export function selectedAttribution(
  result: LogitAttribution,
  target: number,
  alternative: number,
) {
  if (
    result.schema_version !== 1 ||
    result.status !== "available" ||
    !Array.isArray(result.targets) ||
    !result.targets[target] ||
    !Array.isArray(result.components) ||
    !result.components.length ||
    !result.checks
  )
    return null;
  const count = result.targets.length;
  if (
    ![
      result.checks.actual_logits,
      result.checks.strict_tolerances,
      result.checks.rounding_allowances,
    ].every(
      (values) =>
        Array.isArray(values) &&
        values.length === count &&
        values.every(Number.isFinite),
    )
  )
    return null;
  if (
    result.checks.strict_tolerances.some((value) => value <= 0) ||
    result.checks.rounding_allowances.some(
      (value, index) => value < result.checks!.strict_tolerances[index],
    )
  )
    return null;
  if (
    result.components.some(
      (row) =>
        !Array.isArray(row.values) ||
        row.values.length !== count ||
        !row.values.every(Number.isFinite),
    )
  )
    return null;
  const compared =
    alternative >= 0 && alternative !== target && !!result.targets[alternative];
  const difference = (values: number[]) =>
    values[target] - (compared ? values[alternative] : 0);
  const rows = result.components.map((component, index) => ({
    ...component,
    index,
    value: difference(component.values),
  }));
  const actual = difference(result.checks.actual_logits);
  const reconstructed = rows.reduce((total, row) => total + row.value, 0);
  if (
    ![actual, reconstructed, ...rows.map((row) => row.value)].every(
      Number.isFinite,
    )
  )
    return null;
  const error = Math.abs(reconstructed - actual);
  const recordedStrict =
    result.checks.strict_tolerances[target] +
    (compared ? result.checks.strict_tolerances[alternative] : 0);
  // A small contrast can be much less reliable than either large raw logit.
  const strict = Math.min(
    recordedStrict,
    1e-5 + 1e-5 * Math.max(1, Math.abs(actual)),
  );
  const allowance =
    result.checks.rounding_allowances[target] +
    (compared ? result.checks.rounding_allowances[alternative] : 0);
  if (![strict, recordedStrict, allowance].every(Number.isFinite)) return null;
  const relative = actual === 0 ? null : error / Math.abs(actual);
  const nearZero = compared && Math.abs(actual) <= strict;
  const health =
    error > allowance
      ? "mismatch"
      : error > strict && (relative === null || relative > 0.1)
        ? "contrast_error"
        : error > strict || result.checks.status === "rounding"
          ? "rounding"
          : nearZero
            ? "near_zero"
            : "ok";
  return {
    rows,
    actual,
    reconstructed,
    error,
    strict,
    recordedStrict,
    allowance,
    relative,
    compared,
    health,
    patchingAllowed:
      compared && !["mismatch", "contrast_error"].includes(health),
  };
}
export function attributionUnavailable(reason?: string | null): string {
  const reasons: Record<string, string> = {
    unsupported_norm:
      "This final normalization does not have a verified attribution adapter.",
    unsupported_head:
      "This output head does not have a verified linear readout adapter.",
    quantized_head:
      "Attribution of this quantized output head is not supported yet.",
    output_transform:
      "A transform after the residual readout prevents this additive decomposition.",
    residual_discontinuity:
      "An operation between block boundaries prevents a verified block-update decomposition.",
    normalization_mismatch:
      "The normalization adapter did not reproduce the captured model output.",
    readout_mismatch:
      "The linear readout did not reproduce the model logits within the recorded allowance.",
    reconstruction_mismatch:
      "The contribution sum did not reproduce the model logits within the recorded allowance.",
    capture_budget:
      "This attribution capture exceeds the bounded diagnostic memory allowance.",
    missing_blocks:
      "This architecture does not expose verified block boundaries for attribution.",
    padded_prompt:
      "The last captured position is padded. Remove trailing padding and capture again.",
  };
  return (
    reasons[reason ?? ""] ??
    "Score contributions could not be verified for this capture. Other trace measurements remain available."
  );
}
