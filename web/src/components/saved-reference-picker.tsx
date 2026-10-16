"use client";
import { useI18n } from "@/i18n/provider";
import type { ReactNode } from "react";
import { ErrorNotice, Loading } from "@/components/ui";
import { notebookHref } from "@/lib/experiments";
import type { Experiment } from "@/lib/types";
export function SavedReferencePicker({
  label,
  referenceId,
  reference,
  candidates,
  libraryLoading,
  libraryError,
  onRetryLibrary,
  emptyText,
  retryLabel = "Retry saved evidence",
  loadingLabel = "Finding saved evidence",
  children,
}: {
  label: string;
  referenceId: string | null;
  reference: Experiment | null;
  candidates: Experiment[];
  libraryLoading: boolean;
  libraryError: string | null;
  onRetryLibrary: () => void;
  emptyText: string;
  retryLabel?: string;
  loadingLabel?: string;
  children?: ReactNode;
}) {
  const { t, dateTime } = useI18n();
  const linked =
    referenceId && !candidates.some((item) => item.id === referenceId);
  return (
    <>
      <div className="comparison-controls">
        <label>
          {t(label)}
          <select
            aria-label={t(label)}
            value={referenceId ?? ""}
            onChange={(event) =>
              window.history.pushState(
                null,
                "",
                notebookHref(window.location.search, {
                  compare: event.target.value || null,
                }),
              )
            }
            disabled={libraryLoading && !referenceId}
          >
            <option value="">{t("No reference")}</option>
            {linked && (
              <option value={referenceId}>
                {reference?.name ?? t("Linked reference")}
              </option>
            )}
            {candidates.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} · {dateTime(item.created_at)} ·{" "}
                {item.id.slice(0, 8)}
              </option>
            ))}
          </select>
        </label>
        {children}
      </div>
      {libraryLoading && <Loading label={loadingLabel} />}
      <ErrorNotice message={libraryError} />
      {libraryError && (
        <div className="comparison-controls">
          <button className="button secondary" onClick={onRetryLibrary}>
            {t(retryLabel)}
          </button>
        </div>
      )}
      {!libraryLoading &&
        !libraryError &&
        !candidates.length &&
        !referenceId && <p className="inline-note">{t(emptyText)}</p>}
    </>
  );
}
