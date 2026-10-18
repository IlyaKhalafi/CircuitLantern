import { selectedAttribution } from "@/lib/attribution";
import {
  executionProfileKnown,
  sameExecutionProfile,
} from "@/lib/comparison-profile";
import type { ExecutionProvenance, RuntimeState, Trace } from "@/lib/types";
import type { AttributionTarget } from "@/lib/attribution";

const prefix = "lantern.patching-hypothesis.v1.";
const pending = new Map<string, PatchingHypothesis>();
export interface PatchingHypothesis {
  version: 1;
  id: string;
  prompt: string;
  token_ids: number[];
  position: number;
  layer: number;
  evidence: { schema_version: 1; value: number; score: number; error: number };
  target: AttributionTarget;
  alternative: AttributionTarget;
  source: ExecutionProvenance & {
    model_id: string;
    revision: string;
    device: string;
    trace_id: string;
  };
}
export function hypothesisIdValid(id: string | null): id is string {
  return (
    typeof id === "string" &&
    /^[a-f0-9]{32}-\d{1,6}-\d{1,10}-\d{1,10}-[a-f0-9]{16}$/.test(id)
  );
}
function valid(value: unknown): value is PatchingHypothesis {
  if (!value || typeof value !== "object") return false;
  const draft = value as PatchingHypothesis;
  return (
    draft.version === 1 &&
    hypothesisIdValid(draft.id) &&
    typeof draft.prompt === "string" &&
    draft.prompt.length > 0 &&
    draft.prompt.length <= 8000 &&
    Array.isArray(draft.token_ids) &&
    draft.token_ids.length > 0 &&
    draft.token_ids.length <= 131072 &&
    draft.token_ids.every((id) => Number.isSafeInteger(id) && id >= 0) &&
    Number.isSafeInteger(draft.layer) &&
    draft.layer >= 0 &&
    draft.position === draft.token_ids.length - 1 &&
    !!draft.evidence &&
    draft.evidence.schema_version === 1 &&
    [draft.evidence.value, draft.evidence.score, draft.evidence.error].every(
      Number.isFinite,
    ) &&
    draft.evidence.error >= 0 &&
    !!draft.target &&
    !!draft.alternative &&
    [draft.target, draft.alternative].every(
      (target) =>
        Number.isSafeInteger(target.token_id) &&
        target.token_id >= 0 &&
        target.patchable === true &&
        typeof target.requested_text === "string" &&
        target.requested_text.length > 0 &&
        target.requested_text.length <= 100,
    ) &&
    draft.target.token_id !== draft.alternative.token_id &&
    !!draft.source &&
    typeof draft.source.model_id === "string" &&
    typeof draft.source.revision === "string" &&
    typeof draft.source.trace_id === "string" &&
    draft.id.startsWith(
      `${draft.source.trace_id}-${draft.layer}-${draft.target.token_id}-${draft.alternative.token_id}-`,
    ) &&
    executionProfileKnown(draft.source)
  );
}
export function preparePatchingHypothesis(
  trace: Trace,
  layer: number,
  target: AttributionTarget,
  alternative: AttributionTarget,
): string {
  const suffix = Array.from(crypto.getRandomValues(new Uint8Array(8)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  const id = `${trace.trace_id}-${layer}-${target.token_id}-${alternative.token_id}-${suffix}`;
  const result = trace.logit_attribution;
  const values =
    result &&
    selectedAttribution(
      result,
      result.targets?.findIndex((item) => item.token_id === target.token_id) ??
        -1,
      result.targets?.findIndex(
        (item) => item.token_id === alternative.token_id,
      ) ?? -1,
    );
  const component = values?.rows.find(
    (row) => row.kind === "block" && row.layer === layer,
  );
  if (
    !values?.patchingAllowed ||
    !component ||
    result?.position !== trace.token_ids.length - 1 ||
    result.token_id !== trace.token_ids.at(-1) ||
    result.scope !== "final_prompt_token"
  )
    throw new Error(
      "Resolve the reconstruction warning before preparing a patching hypothesis.",
    );
  const draft: PatchingHypothesis = {
    version: 1,
    id,
    evidence: {
      schema_version: 1,
      value: component.value,
      score: values.actual,
      error: values.error,
    },
    prompt: trace.prompt,
    token_ids: trace.token_ids.slice(),
    position: trace.token_ids.length - 1,
    layer,
    target,
    alternative,
    source: {
      model_id: trace.model_id,
      revision: trace.revision,
      trace_id: trace.trace_id ?? "",
      device: trace.device,
      execution_id: trace.execution_id,
      execution_precision: trace.execution_precision,
      execution_context_length: trace.execution_context_length,
      execution_config: trace.execution_config,
    },
  };
  if (!valid(draft))
    throw new Error(
      "This capture does not contain a verified source and answer pair for patching.",
    );
  pending.set(id, draft);
  try {
    sessionStorage.setItem(prefix + id, JSON.stringify(draft));
  } catch {
    /* Client navigation can retain this draft in memory. */
  }
  return `/causal?hypothesis=${id}`;
}
export function readPatchingHypothesis(id: string): PatchingHypothesis {
  let draft: unknown = pending.get(id);
  if (!valid(draft)) {
    try {
      draft = JSON.parse(sessionStorage.getItem(prefix + id) ?? "null");
    } catch {
      draft = null;
    }
  }
  if (!valid(draft) || draft.id !== id)
    throw new Error(
      "This patching hypothesis is no longer available. Return to the capture to prepare it again.",
    );
  pending.set(id, draft);
  return draft;
}
export function hypothesisSourceProblem(
  draft: PatchingHypothesis,
  runtime: RuntimeState | null,
): string | null {
  if (!runtime?.loaded || !runtime.metadata)
    return "Load the recorded checkpoint and execution format to test this hypothesis.";
  if (
    runtime.metadata.id !== draft.source.model_id ||
    runtime.metadata.revision !== draft.source.revision
  )
    return "This hypothesis belongs to another checkpoint. Load its recorded model and revision before testing it.";
  const current = {
    ...runtime.metadata,
    device: runtime.device,
    execution_precision:
      runtime.metadata.execution_precision ?? runtime.precision,
  };
  if (
    !executionProfileKnown(current) ||
    !sameExecutionProfile(draft.source, current)
  )
    return "The active execution profile differs from the source capture. Restore its format, device and library versions before testing it.";
  if (draft.layer >= runtime.metadata.num_layers)
    return "The captured block is unavailable in the active model.";
  return null;
}

export interface PatchingSource extends ExecutionProvenance {
  trace_id: string;
  model_id: string;
  revision: string;
  device: string;
  execution_precision: string;
  execution_config: Record<string, unknown>;
  clean_prompt: string;
  token_ids: number[];
  position: number;
  layer: number;
  target_token_id: number;
  alternative_token_id: number;
  attribution_schema_version: 1;
  attribution_value: number;
  attribution_score: number;
  attribution_error: number;
}
export function hypothesisSource(draft: PatchingHypothesis): PatchingSource {
  return {
    attribution_schema_version: draft.evidence.schema_version,
    attribution_value: draft.evidence.value,
    attribution_score: draft.evidence.score,
    attribution_error: draft.evidence.error,
    trace_id: draft.source.trace_id,
    model_id: draft.source.model_id,
    revision: draft.source.revision,
    device: draft.source.device,
    execution_id: draft.source.execution_id,
    execution_precision: draft.source.execution_precision!,
    execution_config: draft.source.execution_config!,
    clean_prompt: draft.prompt,
    token_ids: draft.token_ids,
    position: draft.position,
    layer: draft.layer,
    target_token_id: draft.target.token_id,
    alternative_token_id: draft.alternative.token_id,
  };
}
