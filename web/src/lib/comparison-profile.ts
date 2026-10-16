import type { ExecutionProvenance } from "@/lib/types";

type Profile = ExecutionProvenance & { device?: string };

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, canonical(item)]),
    );
  return value;
}

export function executionProfileKnown(profile: Profile) {
  const config = profile.execution_config;
  return (
    typeof profile.execution_precision === "string" &&
    profile.execution_precision.length > 0 &&
    !!config &&
    !Array.isArray(config) &&
    [
      "precision",
      "device",
      "residual_dtype",
      "torch_version",
      "transformers_version",
    ].every(
      (field) => typeof config[field] === "string" && config[field] !== "",
    )
  );
}

export function sameExecutionProfile(selected: Profile, reference: Profile) {
  return (
    selected.device === reference.device &&
    selected.execution_precision === reference.execution_precision &&
    JSON.stringify(canonical(selected.execution_config)) ===
      JSON.stringify(canonical(reference.execution_config))
  );
}
