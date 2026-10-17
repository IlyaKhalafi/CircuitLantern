import type { CausalResult, ExecutionProvenance } from "@/lib/types";

export const maxPromptPairs = 64;
export type PromptSetMetric = "logit_difference" | "log_probability";
export type PatchPositions = "last" | "changed" | "all";
export interface PromptPair {
  label: string;
  clean_prompt: string;
  corrupted_prompt: string;
  target: string;
  alternative: string;
}
export interface PromptSetPlan extends ExecutionProvenance {
  compatible: boolean;
  blockers: string[];
  pair_errors: { index: number; label: string; message: string }[];
  pairs: {
    index: number;
    token_count: number;
    positions: number[];
    target_token_id: number;
    alternative_token_id: number | null;
  }[];
  pair_count: number;
  layers: number[];
  forward_passes: number;
  prompt_limit: number;
  estimated_host_bytes: number;
  estimated_gpu_bytes: number;
  estimated_disk_bytes: number;
  input_fingerprint: string;
}
export interface SetLayer {
  layer: number;
  mean_delta: number;
  median_delta: number;
  minimum_delta: number;
  maximum_delta: number;
  mean_interval: [number, number] | null;
  positive_fraction: number;
  negative_fraction: number;
  near_zero_fraction: number;
  recovery: number | null;
  recovery_interval: [number, number] | null;
}
export interface MeasuredPair {
  index: number;
  label: string;
  positions: number[];
  changed_positions: number[];
  baseline_gap: number;
  metric_recovery_defined: boolean;
  metric_recovery_warning: string | null;
  clean_score: number;
  corrupted_score: number;
  result: Omit<CausalResult, "position" | "layers"> & {
    position: number[] | null;
    alternative_token?: string;
    alternative_token_id?: number;
    layers: (CausalResult["layers"][number] & {
      metric_delta: number;
      metric_recovery: number | null;
    })[];
  };
}
export interface PromptSetResult extends ExecutionProvenance {
  model_id: string;
  revision: string;
  device: string;
  metric: PromptSetMetric;
  position_mode: PatchPositions;
  selected_layers: number[];
  requested_pairs: PromptPair[];
  pairs: MeasuredPair[];
  layers: SetLayer[];
  completed_pairs: number;
  total_pairs: number;
  recovery_pairs: number;
  clean_target_preferred_pairs?: number | null;
  input_fingerprint: string;
  complete: boolean;
  cancelled?: boolean;
  error: string | null;
  failed_pair: number | null;
  duration_ms: number;
  method: string;
  statistics: {
    seed: number;
    bootstrap_samples: number;
    confidence: number;
    sign_epsilon: number;
    minimum_recovery_gap: number;
    cohort: string;
    interval_method: string;
  };
}

export const examplePairs: PromptPair[] = ["bag", "book", "keys", "letter"].map(
  (object) => ({
    label: "",
    clean_prompt: `When John and Mary went to the shop, John gave the ${object} to`,
    corrupted_prompt: `When John and Mary went to the shop, Mary gave the ${object} to`,
    target: " Mary",
    alternative: " John",
  }),
);

export class PairImportError extends Error {
  constructor(
    message: string,
    public row = 0,
  ) {
    super(message);
  }
}
export function parsePromptPairs(
  text: string,
  { allowIncomplete = false }: { allowIncomplete?: boolean } = {},
): PromptPair[] {
  let values: unknown;
  try {
    values = text.trim().startsWith("[")
      ? JSON.parse(text)
      : text
          .trim()
          .split(/\r?\n/)
          .filter((line) => line.trim())
          .map((line) => JSON.parse(line));
  } catch {
    throw new PairImportError(
      "Import a JSON array or one JSON object per line.",
    );
  }
  if (
    !Array.isArray(values) ||
    !values.length ||
    values.length > maxPromptPairs
  )
    throw new PairImportError("Import between 1 and 64 prompt pairs.");
  return values.map((value, index) => {
    if (!value || typeof value !== "object" || Array.isArray(value))
      throw new PairImportError(
        "Pair {count} must be a JSON object.",
        index + 1,
      );
    const pair = value as Record<string, unknown>;
    if (
      Object.keys(pair).some(
        (key) =>
          ![
            "label",
            "clean_prompt",
            "corrupted_prompt",
            "target",
            "alternative",
          ].includes(key),
      )
    )
      throw new PairImportError(
        "Pair {count} contains unknown fields.",
        index + 1,
      );
    if (
      ["clean_prompt", "corrupted_prompt", "target"].some(
        (key) =>
          typeof pair[key] !== "string" || (!allowIncomplete && !pair[key]),
      )
    )
      throw new PairImportError(
        "Pair {count} needs a clean prompt, corrupted prompt and target.",
        index + 1,
      );
    const result = {
      label: "",
      alternative: "",
      ...pair,
    } as unknown as PromptPair;
    if (
      Object.values(result).some((item) => typeof item !== "string") ||
      result.clean_prompt.length > 8000 ||
      result.corrupted_prompt.length > 8000 ||
      result.target.length > 100 ||
      result.alternative.length > 100 ||
      result.label.length > 100
    )
      throw new PairImportError(
        "Pair {count} has invalid fields or exceeds the text limits.",
        index + 1,
      );
    return result;
  });
}

export function parseSetLayers(value: string): number[] | null {
  if (!value.trim()) return null;
  const values = value.split(",").map((part) => part.trim());
  if (values.length > 128 || values.some((part) => !/^\d+$/.test(part)))
    throw new Error(
      "Enter distinct layer indices separated by commas, or leave blank for all layers.",
    );
  const numbers = values.map(Number);
  if (
    numbers.some((n) => !Number.isSafeInteger(n)) ||
    new Set(numbers).size !== numbers.length
  )
    throw new Error(
      "Enter distinct layer indices separated by commas, or leave blank for all layers.",
    );
  return numbers;
}
