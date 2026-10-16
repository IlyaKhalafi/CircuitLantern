import type { Trace, TrackedTarget } from "@/lib/types";

export function trackedTargets(trace: Trace): TrackedTarget[] {
  const targets = trace.tracked_targets;
  if (
    !Array.isArray(targets) ||
    !targets.length ||
    targets.length > 8 ||
    targets.some(
      (target) =>
        !target ||
        !Number.isSafeInteger(target.token_id) ||
        target.token_id < 0 ||
        typeof target.token !== "string" ||
        typeof target.requested_text !== "string",
    ) ||
    new Set(targets.map((target) => target.token_id)).size !== targets.length
  )
    return [];
  return targets;
}

export function targetTrajectory(
  trace: Trace,
  position: number,
  selected: number,
  reference: number,
) {
  const targets = trackedTargets(trace);
  return trace.layers.map((layer) => {
    const matrix = layer.token_target_log_probabilities;
    const row =
      Array.isArray(matrix) && matrix.length === trace.tokens.length
        ? matrix[position]
        : null;
    const known = Array.isArray(row) && row.length === targets.length;
    const score = (index: number) => {
      const value = known ? row[index] : null;
      return typeof value === "number" && Number.isFinite(value) && value <= 0
        ? value
        : null;
    };
    const value = score(selected),
      other = score(reference);
    const difference = value === null || other === null ? null : value - other;
    return {
      index: layer.layer,
      label: layer.label,
      selected: value,
      reference: other,
      difference:
        difference !== null && Number.isFinite(difference) ? difference : null,
    };
  });
}

export function targetLabel(target: TrackedTarget) {
  return JSON.stringify(target.token) + " · ID " + target.token_id;
}
