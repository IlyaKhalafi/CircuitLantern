import { matrixProperties } from "@/lib/matrix-properties";
import type { MatrixMetrics } from "@/lib/types";

const required = [
  "frobenius_norm",
  "l1_norm",
  "infinity_norm",
  "mean",
  "std",
  "min",
  "max",
  "sparsity",
  "spectral_norm",
  "stable_rank",
] as const;

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function hasMatrixComparisonMeasurements(
  matrix: MatrixMetrics,
): boolean {
  return (
    !!matrix &&
    ["model_id", "revision", "name", "spectrum_method"].every(
      (key) =>
        typeof matrix[key as keyof MatrixMetrics] === "string" &&
        (matrix[key as keyof MatrixMetrics] as string).trim().length > 0,
    ) &&
    Array.isArray(matrix.shape) &&
    matrix.shape.length === 2 &&
    matrix.shape.every((size) => Number.isSafeInteger(size) && size > 0) &&
    matrix.elements === matrix.shape[0] * matrix.shape[1] &&
    Array.isArray(matrix.spectrum_shape) &&
    matrix.spectrum_shape.length === 2 &&
    matrix.spectrum_shape.every(
      (size, index) => size === matrix.shape[index],
    ) &&
    typeof matrix.spectrum_exact === "boolean" &&
    required.every((key) => finite(matrix[key])) &&
    Array.isArray(matrix.singular_values) &&
    matrix.singular_values.length > 0 &&
    matrix.singular_values.length <= Math.min(...matrix.shape) &&
    matrix.singular_values.every((value) => finite(value) && value >= 0) &&
    Array.isArray(matrix.cumulative_energy) &&
    matrix.cumulative_energy.length === matrix.singular_values.length &&
    matrix.cumulative_energy.every(
      (value) => finite(value) && value >= 0 && value <= 1 + 1e-9,
    )
  );
}

export function matrixComparisonProblem(
  selected: MatrixMetrics,
  reference: MatrixMetrics,
): string | null {
  if (
    !hasMatrixComparisonMeasurements(selected) ||
    !hasMatrixComparisonMeasurements(reference)
  )
    return "A saved analysis is missing finite matrix measurements or a spectrum of the full logical matrix. Capture new analyses to compare them.";
  if (selected.model_id !== reference.model_id)
    return "These analyses use different model repositories. Choose the same model repository to compare revisions or execution formats.";
  if (selected.name !== reference.name)
    return "These analyses describe different weights. Choose the same weight name in both captures.";
  if (selected.shape.some((size, index) => size !== reference.shape[index]))
    return "The logical matrix dimensions differ. Choose captures with the same matrix shape.";
  return null;
}

function difference(selected: number | null, reference: number | null) {
  if (selected === null || reference === null) return null;
  const value = selected - reference;
  return Number.isFinite(value) ? value : null;
}

export function matrixDifferences(
  selected: MatrixMetrics,
  reference: MatrixMetrics,
) {
  const other = matrixProperties(reference);
  return matrixProperties(selected).map((property, index) => ({
    key: property.key,
    label: property.label,
    unit: property.unit,
    selected: property.value,
    reference: other[index].value,
    difference: difference(property.value, other[index].value),
    selected_estimated: property.estimated,
    reference_estimated: other[index].estimated,
    difference_estimated: property.estimated || other[index].estimated,
  }));
}

export function matrixSpectrumComparison(
  selected: MatrixMetrics,
  reference: MatrixMetrics,
) {
  return Array.from(
    {
      length: Math.max(
        selected.singular_values.length,
        reference.singular_values.length,
      ),
    },
    (_, index) => {
      const value = selected.singular_values[index] ?? null;
      const other = reference.singular_values[index] ?? null;
      return {
        index: index + 1,
        selected: value,
        reference: other,
        difference: difference(value, other),
        selected_energy: selected.cumulative_energy[index] ?? null,
        reference_energy: reference.cumulative_energy[index] ?? null,
      };
    },
  );
}
