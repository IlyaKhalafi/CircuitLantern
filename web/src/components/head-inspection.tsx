"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ErrorNotice } from "@/components/ui";
import { executionLabel } from "@/lib/execution";
import {
  completeHeadInspection,
  readHeadInspection,
  inspectionProblem,
  type HeadInspectionDraft,
} from "@/lib/inspection";
import type { RuntimeState } from "@/lib/types";

function clearInspectionParameter() {
  const url = new URL(window.location.href);
  if (!url.searchParams.has("inspect")) return;
  url.searchParams.delete("inspect");
  window.history.replaceState(
    window.history.state,
    "",
    url.pathname + url.search + url.hash,
  );
}

export function useHeadInspection(
  runtime: RuntimeState | null,
  apply: (draft: HeadInspectionDraft) => void,
) {
  const [draft, setDraft] = useState<HeadInspectionDraft | null>(null);
  const [applied, setApplied] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestedId, setRequestedId] = useState<string | undefined>(undefined);
  const [alternate, setAlternate] = useState<HeadInspectionDraft | null>(null);
  useEffect(() => {
    const url = new URL(window.location.href);
    const id = url.searchParams.get("inspect");
    setRequestedId(id ?? undefined);
    try {
      const stored = readHeadInspection(id ?? undefined);
      if (!stored) return;
      setDraft(stored);
      setApplied(null);
      setAlternate(null);
      setError(null);
    } catch (reason) {
      setDraft(null);
      setApplied(null);
      setError((reason as Error).message);
      const available = readHeadInspection();
      setAlternate(available?.id !== id ? available : null);
    }
  }, []);
  const pending = !!draft && applied !== draft.id;
  const problem = draft && pending ? inspectionProblem(draft, runtime) : null;
  useEffect(() => {
    if (!draft || !pending || problem) return;
    apply(draft);
    completeHeadInspection(draft.id);
    clearInspectionParameter();
    setApplied(draft.id);
  }, [draft, pending, problem, apply]);
  const dismiss = useCallback(() => {
    completeHeadInspection(draft?.id ?? requestedId);
    clearInspectionParameter();
    setAlternate(null);
    setDraft(null);
    setApplied(null);
    setError(null);
  }, [draft, requestedId]);
  const resume = useCallback(() => {
    if (!alternate) return;
    clearInspectionParameter();
    setDraft(alternate);
    setApplied(null);
    setError(null);
    setRequestedId(alternate.id);
    setAlternate(null);
  }, [alternate]);
  return {
    draft,
    pending,
    problem,
    error,
    dismiss,
    runtime,
    alternate,
    resume,
  };
}

export function HeadInspectionNotice({
  inspection,
}: {
  inspection: ReturnType<typeof useHeadInspection>;
}) {
  const {
    draft,
    pending,
    problem,
    error,
    dismiss,
    runtime,
    alternate,
    resume,
  } = inspection;
  if (!draft && !error) return null;
  if (error)
    return (
      <div className="inspection-notice">
        <ErrorNotice
          message={
            error +
            (alternate ? " A different head inspection draft is pending." : "")
          }
        />
        {alternate && (
          <button className="button secondary" onClick={resume}>
            Resume pending draft
          </button>
        )}
        <button className="button secondary" onClick={dismiss}>
          Dismiss
        </button>
      </div>
    );
  if (!draft) return null;
  return (
    <div className="notice info inspection-notice" role="status">
      <div>
        <strong>
          Draft from head sweep · layer {draft.layer} · head {draft.head}
        </strong>
        <p>
          {draft.source.model_id} · revision {draft.source.revision}
          {" · "}
          {executionLabel(
            draft.source.execution_precision,
            draft.source.execution_config,
          )}
          {" · "}
          {draft.source.device.toUpperCase()}
        </p>
        <p>
          {problem?.message ??
            "The captured prompt and selected head were copied into the controls. Edit them or run a new capture."}
        </p>
        {runtime?.loaded && (
          <p>
            Active runtime uses{" "}
            {executionLabel(
              runtime.precision,
              runtime.metadata?.execution_config,
            )}{" "}
            on {runtime.device.toUpperCase()}.
          </p>
        )}
        {problem?.kind === "checkpoint" && (
          <Link href="/" className="text-link">
            Open model library
          </Link>
        )}
      </div>
      <button
        className="button secondary"
        onClick={dismiss}
        aria-label={
          pending ? "Discard head inspection draft" : "Dismiss head inspection"
        }
      >
        {pending ? "Discard draft" : "Dismiss"}
      </button>
    </div>
  );
}
