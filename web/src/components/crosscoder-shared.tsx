"use client";

import { useEffect, useRef, useState } from "react";
import { Square } from "lucide-react";
import { ErrorNotice, Metric } from "@/components/ui";
import { useI18n } from "@/i18n/provider";
import { bytes, number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import { useBackgroundJob } from "@/lib/jobs";
import type { CaptureEvidence, CrosscoderPlan } from "@/lib/crosscoders";

export type CrosscoderTask<T> = ReturnType<typeof useBackgroundJob<T>>;

export function useSessionDraft<T extends Record<string, string | boolean>>(
  key: string,
  initial: T,
) {
  const [draft, setDraft] = useState<T>(initial);
  const [ready, setReady] = useState(false);
  const defaults = useRef(initial);
  useEffect(() => {
    try {
      const value = JSON.parse(sessionStorage.getItem(key) || "{}");
      if (value && typeof value === "object") {
        const restored = { ...defaults.current };
        for (const field of Object.keys(restored) as (keyof T)[]) {
          if (typeof value[field] === typeof restored[field])
            restored[field] = value[field];
        }
        setDraft(restored);
      }
    } catch {
      /* Browser persistence is optional. */
    }
    setReady(true);
  }, [key]);
  useEffect(() => {
    if (ready) {
      try {
        sessionStorage.setItem(key, JSON.stringify(draft));
      } catch {
        /* Optional persistence. */
      }
    }
  }, [key, ready, draft]);
  return [draft, setDraft, ready] as const;
}

export function useCompletedJob<T>(
  task: ReturnType<typeof useBackgroundJob<T>>,
  key: string,
  onComplete: (result: T) => void,
) {
  const delivered = useRef<string | null>(null);
  useEffect(() => {
    const job = task.job;
    if (
      !job ||
      !["ready", "cancelled", "failed"].includes(job.status) ||
      !job.result ||
      delivered.current === job.id
    )
      return;
    delivered.current = job.id;
    try {
      if (sessionStorage.getItem(key) === job.id) return;
      sessionStorage.setItem(key, job.id);
    } catch {
      /* Optional persistence. */
    }
    onComplete(job.result);
  }, [task.job, key, onComplete]);
}

export function CrosscoderProgress<T>({
  task,
}: {
  task: ReturnType<typeof useBackgroundJob<T>>;
}) {
  const { t } = useI18n();
  return (
    <>
      <ErrorNotice message={task.error} />
      {task.job && (
        <div className="feature-job" role="status">
          <span>{t(task.job.message)}</span>
          <progress
            aria-label={t("Job progress")}
            value={task.job.progress}
            max={100}
          />
          <bdi>{number(task.job.progress, 0)}%</bdi>
          {task.canCancel && (
            <button
              className="button secondary"
              disabled={task.cancelling}
              onClick={() => task.cancel()}
            >
              <Square size={14} />
              {t(task.cancelling ? "Stopping" : "Stop")}
            </button>
          )}
        </div>
      )}
    </>
  );
}

export function CaptureCard({
  capture,
  role,
}: {
  capture: CaptureEvidence;
  role: string;
}) {
  const { t } = useI18n();
  return (
    <div className="crosscoder-source">
      <span className="eyebrow">{t(role)}</span>
      <strong dir="auto">{capture.name}</strong>
      <code dir="ltr">{capture.source.model_id}</code>
      <span>
        {t("Layer {v0}", { v0: capture.layer })} ·{" "}
        {t("{v0} dimensions", { v0: capture.input_dim })} ·{" "}
        {t("{v0} token positions", { v0: capture.sample_count })}
      </span>
      <small dir="ltr">
        {capture.source.revision} ·{" "}
        {t(
          executionLabel(
            capture.source.execution_precision,
            capture.source.execution_config,
          ),
        )}
      </small>
    </div>
  );
}

export function CrosscoderBudget({ plan }: { plan: CrosscoderPlan | null }) {
  const { t } = useI18n();
  if (!plan) return null;
  return (
    <div className="feature-plan">
      <strong>
        {t(plan.compatible ? "Resources available" : "Resources unavailable")}
      </strong>
      <div className="metrics-grid">
        <Metric
          label="Additional RAM"
          value={bytes(plan.estimated_host_bytes)}
        />
        <Metric
          label="Additional VRAM"
          value={bytes(plan.estimated_gpu_bytes)}
        />
        <Metric
          label="Disk estimate"
          value={bytes(plan.estimated_disk_bytes)}
        />
      </div>
      <ErrorNotice message={plan.blockers.length ? plan.blockers : null} />
    </div>
  );
}
