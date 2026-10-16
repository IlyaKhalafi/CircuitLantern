import { number } from "@/lib/api";
import type { MatrixMetrics } from "@/lib/types";

const definitions = [
  ["frobenius_norm", "Frobenius norm"],
  ["spectral_norm", "Spectral norm"],
  ["spectral_radius", "Spectral radius"],
  ["stable_rank", "Stable rank"],
  ["condition_number", "Condition number"],
  ["operator_1_norm", "Operator 1 norm"],
  ["l1_norm", "Entrywise L1 norm"],
  ["infinity_norm", "Infinity norm"],
  ["numerical_rank", "Numerical rank"],
  ["effective_rank", "Effective rank"],
  ["nuclear_norm", "Nuclear norm"],
  ["sparsity", "Exact zeros"],
  ["mean", "Mean"],
  ["std", "Standard deviation"],
  ["min", "Minimum"],
  ["max", "Maximum"],
] as const satisfies readonly (readonly [keyof MatrixMetrics, string])[];

export function matrixProperties(metrics: MatrixMetrics) {
  return definitions.map(([key, label]) => {
    const value = metrics[key];
    const completeOnly = [
      "condition_number",
      "numerical_rank",
      "effective_rank",
      "nuclear_norm",
    ].includes(key);
    return {
      key,
      label,
      value:
        typeof value === "number" &&
        Number.isFinite(value) &&
        (!completeOnly || metrics.spectrum_exact)
          ? value
          : null,
      estimated:
        !metrics.spectrum_exact &&
        (key === "spectral_norm" || key === "stable_rank"),
      unit: key === "sparsity" ? ("fraction" as const) : ("number" as const),
    };
  });
}

export function matrixValue(
  value: number | null,
  unit: "fraction" | "number",
  difference = false,
) {
  if (value === null) return "N/A";
  return unit === "fraction"
    ? number(value * 100, 5) + (difference ? " pp" : "%")
    : number(value, 5);
}
