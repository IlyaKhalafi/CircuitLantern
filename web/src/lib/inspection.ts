import type {
  ExecutionProvenance,
  HeadSweepResult,
  HeadSweepRow,
  RuntimeState,
} from "@/lib/types";

const STORAGE_KEY = "lantern.head-inspection.v1";
const UNAVAILABLE =
  "This inspection draft is no longer available. Open the sweep and select the head again.";

export interface HeadInspectionDraft {
  version: 1;
  id: string;
  prompt: string;
  layer: number;
  head: number;
  source: ExecutionProvenance & {
    model_id: string;
    revision: string;
    device: string;
  };
}

let pending: HeadInspectionDraft | null = null;

function record(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function textWithin(value: unknown, limit: number): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    Array.from(value).length <= limit
  );
}

function index(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function valid(value: unknown): value is HeadInspectionDraft {
  if (!record(value) || !record(value.source)) return false;
  const source = value.source;
  return (
    value.version === 1 &&
    typeof value.id === "string" &&
    /^[a-f0-9]{32}$/.test(value.id) &&
    textWithin(value.prompt, 8000) &&
    index(value.layer) &&
    index(value.head) &&
    textWithin(source.model_id, 180) &&
    textWithin(source.revision, 128) &&
    textWithin(source.device, 32) &&
    (source.execution_precision === undefined ||
      textWithin(source.execution_precision, 32)) &&
    (source.execution_config == null || record(source.execution_config))
  );
}

export function prepareHeadInspection(
  result: HeadSweepResult,
  head: Pick<HeadSweepRow, "layer" | "head">,
) {
  const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  const draft: HeadInspectionDraft = {
    version: 1,
    id,
    prompt: result.prompt,
    layer: head.layer,
    head: head.head,
    source: {
      model_id: result.model_id,
      revision: result.revision,
      execution_id: result.execution_id,
      execution_precision: result.execution_precision,
      execution_config: result.execution_config,
      device: result.device,
    },
  };
  if (!valid(draft))
    throw new Error("This sweep does not contain a usable prompt and head.");
  pending = draft;
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(draft));
  } catch {
    // Client navigation can use the in-memory draft when storage is unavailable.
  }
  return `/playground?inspect=${id}`;
}

export function readHeadInspection(id?: string): HeadInspectionDraft | null {
  let draft: unknown = pending;
  if (!valid(draft) || (id !== undefined && draft.id !== id)) {
    try {
      draft = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? "null");
    } catch {
      draft = null;
    }
  }
  if (!valid(draft) || (id !== undefined && draft.id !== id)) {
    if (id !== undefined) throw new Error(UNAVAILABLE);
    completeHeadInspection();
    return null;
  }
  pending = draft;
  return draft;
}

export function completeHeadInspection(id?: string) {
  if (id === undefined || pending?.id === id) pending = null;
  try {
    const stored: unknown = JSON.parse(
      window.sessionStorage.getItem(STORAGE_KEY) ?? "null",
    );
    if (id === undefined || !valid(stored) || stored.id === id)
      window.sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // Client navigation can complete a draft when browser storage is blocked.
    }
  }
}

export function inspectionProblem(
  draft: HeadInspectionDraft,
  runtime: RuntimeState | null,
): { kind: "loading" | "checkpoint" | "unsupported"; message: string } | null {
  if (!runtime)
    return {
      kind: "loading",
      message: "Checking the active checkpoint for this head.",
    };
  if (!runtime.loaded || !runtime.metadata)
    return {
      kind: "checkpoint",
      message:
        "Load the captured checkpoint from the model library to inspect this head.",
    };
  if (
    runtime.metadata.id !== draft.source.model_id ||
    runtime.metadata.revision !== draft.source.revision
  )
    return {
      kind: "checkpoint",
      message:
        "This head belongs to a different checkpoint. Load its recorded model and revision before preparing the capture.",
    };
  const target = runtime.attention_head_targets?.find(
    (item) => item.layer === draft.layer,
  );
  if (!target || draft.head >= target.heads)
    return {
      kind: "unsupported",
      message:
        "This head is unavailable in the active architecture. Its layer and head must have a verified ablation adapter.",
    };
  return null;
}
