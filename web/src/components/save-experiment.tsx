"use client";
import { useI18n } from "@/i18n/provider";
import { useRef, useState } from "react";
import Link from "next/link";
import { Bookmark, Check } from "lucide-react";
import { ErrorNotice } from "@/components/ui";
import { truncateText } from "@/lib/text";
import type { Experiment, Trace } from "@/lib/types";
import { post } from "@/lib/api";
type ExperimentSource =
  | {
      kind: "job";
      id: string;
    }
  | {
      kind: "matrix";
      id: string;
      executionId?: string;
    }
  | {
      kind: "trace";
      trace: Trace;
    };
export function SaveExperiment({
  source,
  defaultName,
  disabled = false,
}: {
  source: ExperimentSource;
  defaultName: string;
  disabled?: boolean;
}) {
  const { t } = useI18n();
  const [name, setName] = useState(truncateText(defaultName, 100));
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<Experiment | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef(false);
  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (pending.current || saved || disabled || !name.trim()) return;
    pending.current = true;
    setSaving(true);
    setError(null);
    try {
      const record = await post<Experiment>(
        source.kind === "job"
          ? "/experiments/from-job"
          : source.kind === "matrix"
            ? "/experiments/from-matrix"
            : "/experiments",
        {
          ...(source.kind === "job"
            ? { job_id: source.id }
            : source.kind === "matrix"
              ? {
                  analysis_id: source.id,
                  expected_execution_id: source.executionId,
                }
              : { trace: source.trace }),
          name: name.trim(),
        },
      );
      setName(record.name);
      setSaved(record);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      pending.current = false;
      setSaving(false);
    }
  }
  return (
    <>
      <form className="save-form" onSubmit={save}>
        <input
          dir="auto"
          className="form-input"
          aria-label={t("Experiment name")}
          value={name}
          onChange={(event) => setName(truncateText(event.target.value, 100))}
          placeholder={t("Name this experiment")}
          maxLength={200}
          disabled={disabled || saving || !!saved}
          required
        />
        <button
          className="button secondary"
          disabled={disabled || saving || !!saved || !name.trim()}
        >
          {saved ? <Check size={14} /> : <Bookmark size={14} />}
          {saved
            ? t("Saved to experiments")
            : saving
              ? t("Saving experiment")
              : t("Save experiment")}
        </button>
        {saved && (
          <Link
            className="button secondary"
            href={`/experiments?experiment=${saved.id}`}
          >
            {t("View saved experiment")}
          </Link>
        )}
      </form>
      <ErrorNotice message={error} />
    </>
  );
}
