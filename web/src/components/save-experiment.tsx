"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import { Bookmark, Check } from "lucide-react";
import { ErrorNotice } from "@/components/ui";
import { truncateText } from "@/lib/text";
import type { Experiment } from "@/lib/types";
import { post } from "@/lib/api";

type ExperimentSource =
  | { kind: "job"; id: string }
  | { kind: "matrix"; id: string; executionId?: string };

export function SaveExperiment({
  source,
  defaultName,
  disabled = false,
}: {
  source: ExperimentSource;
  defaultName: string;
  disabled?: boolean;
}) {
  const [name, setName] = useState(truncateText(defaultName, 100));
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
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
          : "/experiments/from-matrix",
        {
          ...(source.kind === "job"
            ? { job_id: source.id }
            : {
                analysis_id: source.id,
                expected_execution_id: source.executionId,
              }),
          name: name.trim(),
        },
      );
      setName(record.name);
      setSaved(true);
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
          className="form-input"
          aria-label="Experiment name"
          value={name}
          onChange={(event) => setName(truncateText(event.target.value, 100))}
          placeholder="Name this experiment"
          maxLength={200}
          disabled={disabled || saving || saved}
          required
        />
        <button
          className="button secondary"
          disabled={disabled || saving || saved || !name.trim()}
        >
          {saved ? <Check size={14} /> : <Bookmark size={14} />}
          {saved
            ? "Saved to experiments"
            : saving
              ? "Saving experiment"
              : "Save experiment"}
        </button>
        {saved && (
          <Link className="button secondary" href="/experiments">
            Open notebook
          </Link>
        )}
      </form>
      <ErrorNotice message={error} />
    </>
  );
}
