import type { InductionSettings } from "@/lib/induction";

const bounds = [
  ["block_length", 4, 64],
  ["repeats", 2, 4],
  ["examples", 1, 32],
  ["causal_heads", 0, 12],
  ["seed", 0, 4294967295],
] as const;

export function normalizeProbeDraft(
  value: unknown,
  fallback: InductionSettings,
  availableLayers: number[],
  vocabularyPool = 64,
): InductionSettings {
  const result = { ...fallback };
  if (!value || typeof value !== "object" || Array.isArray(value))
    return result;
  const recorded = value as Record<string, unknown>;
  for (const [key, min, max] of bounds) {
    const candidate = recorded[key];
    if (
      typeof candidate === "number" &&
      Number.isSafeInteger(candidate) &&
      candidate >= min &&
      candidate <=
        (key === "block_length" ? Math.min(max, vocabularyPool) : max)
    )
      result[key] = candidate;
  }
  if (typeof recorded.include_start_token === "boolean")
    result.include_start_token = recorded.include_start_token;
  if (recorded.layers === null) result.layers = null;
  else if (
    Array.isArray(recorded.layers) &&
    recorded.layers.length > 0 &&
    recorded.layers.length <= 256 &&
    recorded.layers.every(
      (layer) => Number.isSafeInteger(layer) && availableLayers.includes(layer),
    ) &&
    new Set(recorded.layers).size === recorded.layers.length
  )
    result.layers = [...recorded.layers];
  return result;
}

export function validProbeNumbers(
  value: InductionSettings,
  vocabularyPool = 64,
) {
  return bounds.every(
    ([key, min, max]) =>
      Number.isSafeInteger(value[key]) &&
      value[key] >= min &&
      value[key] <=
        (key === "block_length" ? Math.min(max, vocabularyPool) : max),
  );
}
