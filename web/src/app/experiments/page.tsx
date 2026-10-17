"use client";
import { useI18n } from "@/i18n/provider";
import {
  ArrowLeft,
  Check,
  Download,
  FlaskConical,
  Plus,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { ExperimentLibrary } from "@/components/experiment-library";
import { useExperiment } from "@/lib/use-experiment";
import { ExperimentResult } from "@/components/experiment-result";
import { experimentArtifact, notebookHref } from "@/lib/experiments";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import { api, downloadJSON } from "@/lib/api";
import { truncateText } from "@/lib/text";
import type { Experiment } from "@/lib/types";
export default function ExperimentsPage() {
  const { t } = useI18n();
  return (
    <Suspense fallback={<Loading label={t("Opening experiment notebook")} />}>
      <ExperimentNotebook />
    </Suspense>
  );
}
function ExperimentNotebook() {
  const { t, rich, dateTime } = useI18n();
  const params = useSearchParams();
  const recordId = params.get("experiment");
  const query = params.get("q") ?? "";
  const requestedKind = params.get("kind");
  const kind = [
    "trace",
    "causal",
    "head_sweep",
    "matrix",
    "induction_probe",
  ].includes(requestedKind ?? "")
    ? requestedKind!
    : "all";
  const {
    selected,
    error: selectionError,
    loading: opening,
    replace,
  } = useExperiment(recordId);
  const pending = useRef(false);
  function navigate(changes: Record<string, string | null>, replace = false) {
    const href = notebookHref(window.location.search, changes);
    window.history[replace ? "replaceState" : "pushState"](null, "", href);
  }
  function isCurrent(id: string) {
    return new URLSearchParams(window.location.search).get("experiment") === id;
  }
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [name, setName] = useState("");
  const [saved, setSaved] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const selectedArtifact = selected ? experimentArtifact(selected) : undefined;
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setListError(null);
    api<Experiment[]>("/experiments", { signal: controller.signal })
      .then((items) => {
        if (!controller.signal.aborted) setExperiments(items);
      })
      .catch((reason: Error) => {
        if (!controller.signal.aborted) setListError(reason.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [loadAttempt]);
  useEffect(() => {
    setError(null);
    setSaved(false);
    setConfirmDelete(false);
  }, [recordId]);
  useEffect(() => {
    setNotes(selected?.notes ?? "");
    setName(selected?.name ?? "");
  }, [selected]);
  async function save() {
    if (!selected || !name.trim() || pending.current) return;
    pending.current = true;
    setBusy(true);
    setError(null);
    try {
      const result = await api<Experiment>(`/experiments/${selected.id}`, {
        method: "PATCH",
        body: JSON.stringify({ name: name.trim(), notes }),
      });
      if (isCurrent(result.id)) replace(result);
      setExperiments((items) =>
        items.map((item) => (item.id === result.id ? result : item)),
      );
      setLoadAttempt((attempt) => attempt + 1);
      if (isCurrent(result.id)) setSaved(true);
    } catch (e) {
      if (isCurrent(selected.id)) setError((e as Error).message);
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }
  async function remove() {
    if (!selected || pending.current) return;
    pending.current = true;
    setBusy(true);
    setError(null);
    try {
      await api(`/experiments/${selected.id}`, { method: "DELETE" });
      setExperiments((items) =>
        items.filter((item) => item.id !== selected.id),
      );
      setLoadAttempt((attempt) => attempt + 1);
      if (isCurrent(selected.id))
        navigate({ experiment: null, compare: null }, true);
      setConfirmDelete(false);
    } catch (e) {
      if (isCurrent(selected.id)) setError((e as Error).message);
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader
        eyebrow={t("A RECORD OF YOUR CURIOSITY")}
        title={selected ? selected.name : t("Experiments")}
        description={
          selected
            ? t("{v0} \u00B7 saved {v1}", {
                v0: selected.model_id,
                v1: dateTime(selected.created_at),
              })
            : t(
                "Save traces, weight analyses and sweeps, then record what you observe.",
              )
        }
        actions={
          selected ? (
            <button
              className="button secondary"
              onClick={() =>
                downloadJSON(
                  selected,
                  `experiment-${selected.id.slice(0, 8)}.json`,
                )
              }
            >
              <Download size={14} />
              {t("Export experiment")}
            </button>
          ) : (
            <Link className="button primary" href="/playground">
              <Plus size={14} />
              {t("New experiment")}
            </Link>
          )
        }
      />
      <ErrorNotice message={error || selectionError} />
      {!selected && !opening && <ErrorNotice message={listError} />}
      {!selected && !opening && listError && (
        <button
          className="button secondary"
          onClick={() => setLoadAttempt((attempt) => attempt + 1)}
        >
          {t("Retry notebook")}
        </button>
      )}
      {recordId && (
        <Link
          className="button secondary"
          style={{ marginBottom: 20 }}
          href={notebookHref(params.toString(), {
            experiment: null,
            compare: null,
          })}
        >
          <ArrowLeft size={14} />
          {t("All experiments")}
        </Link>
      )}
      {(recordId ? opening : loading) ? (
        <Loading label={t("Opening experiment notebook")} />
      ) : selected ? (
        <>
          <Panel
            title={t("Research notes")}
            subtitle={t(
              "The checkpoint revision and capture settings are preserved with every experiment.",
            )}
          >
            <div className="experiment-detail">
              <label>
                {t("Experiment name")}
                <input
                  dir="auto"
                  className="form-input"
                  value={name}
                  onChange={(e) => {
                    setName(truncateText(e.target.value, 100));
                    setSaved(false);
                  }}
                  maxLength={200}
                  disabled={busy}
                />
              </label>
              <label style={{ marginTop: 17 }}>
                {t("Observations")}
                <textarea
                  dir="auto"
                  value={notes}
                  onChange={(e) => {
                    setNotes(truncateText(e.target.value, 10000));
                    setSaved(false);
                  }}
                  placeholder={t("What changed? What would you test next?")}
                  maxLength={20000}
                  disabled={busy}
                />
              </label>
              <div className="prompt-toolbar">
                <div style={{ display: "flex", gap: 10 }}>
                  <button
                    className="button secondary"
                    onClick={save}
                    disabled={busy || !name.trim()}
                  >
                    <Check size={14} />
                    {saved ? t("Changes saved") : t("Save notes")}
                  </button>
                  <button
                    className="button danger"
                    onClick={() =>
                      confirmDelete ? remove() : setConfirmDelete(true)
                    }
                    disabled={busy}
                  >
                    <Trash2 size={14} />
                    {confirmDelete ? t("Confirm delete") : t("Delete")}
                  </button>
                  {confirmDelete && (
                    <button
                      className="button secondary"
                      onClick={() => setConfirmDelete(false)}
                    >
                      {t("Cancel")}
                    </button>
                  )}
                </div>
                <span
                  className="mono"
                  style={{ fontSize: 9, color: "var(--muted)" }}
                >
                  {rich("revision {v0}", {
                    v0:
                      selectedArtifact?.revision?.slice(0, 10) ??
                      t("unavailable"),
                  })}
                </span>
              </div>
            </div>
          </Panel>
          <ExperimentResult
            key={selected.id}
            experiment={selected}
            experiments={experiments}
            libraryLoading={loading}
            libraryError={listError}
            onRetryLibrary={() => setLoadAttempt((attempt) => attempt + 1)}
          />
        </>
      ) : loading ? (
        <Loading label={t("Opening experiment notebook")} />
      ) : listError ? null : experiments.length ? (
        <ExperimentLibrary
          experiments={experiments}
          query={query}
          kind={kind}
          params={params.toString()}
          update={(changes) => navigate(changes, true)}
        />
      ) : (
        <EmptyState
          icon={<FlaskConical size={28} />}
          title={t("Start collecting evidence.")}
          href="/playground"
          action={t("Run your first trace")}
        >
          {t(
            "Save an activation trace, matrix analysis, residual patch sweep or head ablation sweep to keep its measurements and your observations together.",
          )}
        </EmptyState>
      )}
    </>
  );
}
