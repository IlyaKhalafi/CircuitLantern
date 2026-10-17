"use client";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, RefreshCw, Sparkles } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { api } from "@/lib/api";
import { useBackgroundJob } from "@/lib/jobs";
import { CrosscoderCaptures } from "@/components/crosscoder-captures";
import { CrosscoderTraining } from "@/components/crosscoder-training";
import { CrosscoderStorage } from "@/components/crosscoder-storage";
import { CrosscoderExplorer } from "@/components/crosscoder-explorer";
import {
  useCompletedJob,
  useSessionDraft,
} from "@/components/crosscoder-shared";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import type {
  CaptureJobResult,
  CaptureSummary,
  DamagedArtifact,
  CrosscoderCapabilities,
  CrosscoderJobResult,
  CrosscoderSummary,
} from "@/lib/crosscoders";

export function CrosscoderWorkbench() {
  const { t } = useI18n();
  const [capabilities, setCapabilities] =
    useState<CrosscoderCapabilities | null>(null);
  const [captureRecords, setCaptures] = useState<
    (CaptureSummary | DamagedArtifact)[]
  >([]);
  const [comparisonRecords, setComparisons] = useState<
    (CrosscoderSummary | DamagedArtifact)[]
  >([]);
  const [selection, setSelection, restored] = useSessionDraft(
    "lantern-crosscoder-selection",
    { reference: "", compared: "", comparison: "", mode: "prepare" },
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const request = useRef(0);
  const [refreshToken, setRefreshToken] = useState(0);
  const refresh = useCallback(async () => {
    const sequence = ++request.current;
    const results = await Promise.allSettled([
      api<CrosscoderCapabilities>("/crosscoders/capabilities"),
      api<(CaptureSummary | DamagedArtifact)[]>("/crosscoders/captures"),
      api<(CrosscoderSummary | DamagedArtifact)[]>("/crosscoders"),
    ]);
    if (sequence !== request.current) return;
    const errors: string[] = [];
    if (results[0].status === "fulfilled") setCapabilities(results[0].value);
    else errors.push(results[0].reason.message);
    if (results[1].status === "fulfilled") setCaptures(results[1].value);
    else errors.push(results[1].reason.message);
    if (results[2].status === "fulfilled") setComparisons(results[2].value);
    else errors.push(results[2].reason.message);
    setError(errors.length ? errors.join(" ") : null);
    setLoading(false);
  }, []);
  useEffect(() => {
    void refresh();
    return () => {
      request.current++;
    };
  }, [refresh]);
  const captured = useCallback(
    (result: CaptureJobResult) => {
      const capture = result.captures[0];
      if (capture) {
        setCaptures((current) => [
          capture,
          ...current.filter((item) => item.id !== capture.id),
        ]);
        setSelection((current) =>
          capture.matched_to
            ? {
                ...current,
                reference: capture.matched_to.id,
                compared: capture.id,
              }
            : { ...current, reference: capture.id, compared: "" },
        );
      }
      void refresh();
    },
    [refresh, setSelection],
  );
  const trained = useCallback(
    (result: CrosscoderJobResult) => {
      const comparison = result.crosscoders[0];
      if (comparison) {
        setComparisons((current) => [
          comparison,
          ...current.filter((item) => item.id !== comparison.id),
        ]);
        setSelection((current) => ({
          ...current,
          comparison: comparison.id,
          mode: "explore",
        }));
      }
      void refresh();
    },
    [refresh, setSelection],
  );
  const captureTask = useBackgroundJob<CaptureJobResult>(
    "lantern-crosscoder-capture-job",
  );
  const trainingTask = useBackgroundJob<CrosscoderJobResult>(
    "lantern-crosscoder-training-job",
  );
  useCompletedJob(
    captureTask,
    "lantern-crosscoder-capture-delivered",
    captured,
  );
  useCompletedJob(
    trainingTask,
    "lantern-crosscoder-training-delivered",
    trained,
  );
  const captures = captureRecords.filter(
    (item): item is CaptureSummary => !("integrity_error" in item),
  );
  const comparisons = comparisonRecords.filter(
    (item): item is CrosscoderSummary => !("integrity_error" in item),
  );
  const damaged =
    captureRecords.length !== captures.length ||
    comparisonRecords.length !== comparisons.length;
  const reference =
    captures.find((item) => item.id === selection.reference) || null;
  const compared =
    captures.find(
      (item) =>
        item.id === selection.compared &&
        item.id !== reference?.id &&
        item.corpus_fingerprint === reference?.corpus_fingerprint,
    ) || null;
  const comparison =
    comparisons.find((item) => item.id === selection.comparison) || null;
  const mode = selection.mode === "explore" ? "explore" : "prepare";
  return (
    <div className="crosscoder-workbench">
      <PageHeader
        eyebrow={t("MODEL DIFFING")}
        title={t("Shared feature comparisons")}
        help={t(
          "Retain matching activations from two checkpoints, train a sparse crosscoder and inspect the evidence without writing code.",
        )}
        actions={
          <>
            <Link className="button secondary" href="/features">
              <ArrowLeft size={14} />
              {t("Feature lab")}
            </Link>
            <button
              className="button secondary"
              onClick={() => {
                setRefreshToken((value) => value + 1);
                void refresh();
              }}
            >
              <RefreshCw size={14} />
              {t("Refresh")}
            </button>
          </>
        }
      />
      <ErrorNotice message={error} />
      {damaged && (
        <ErrorNotice
          message={t(
            "Some saved artifact summaries failed integrity checks. Remove affected records under Manage retained data. They are excluded from training and comparison selection.",
          )}
        />
      )}
      <div
        className="crosscoder-modes"
        role="group"
        aria-label={t("Shared feature workflow")}
      >
        <button
          className={"button " + (mode === "prepare" ? "primary" : "secondary")}
          aria-pressed={mode === "prepare"}
          onClick={() =>
            setSelection((current) => ({ ...current, mode: "prepare" }))
          }
        >
          {t("Prepare comparison")}
        </button>
        <button
          className={"button " + (mode === "explore" ? "primary" : "secondary")}
          aria-pressed={mode === "explore"}
          onClick={() =>
            setSelection((current) => ({ ...current, mode: "explore" }))
          }
        >
          {t("Explore comparisons")}
        </button>
      </div>
      {(loading || !restored) && <Loading />}
      <div hidden={mode !== "prepare" || loading || !restored}>
        <CrosscoderCaptures
          capabilities={capabilities}
          captures={captures}
          reference={reference}
          compared={compared}
          onReference={(id) =>
            setSelection((current) => ({
              ...current,
              reference: id,
              compared: "",
            }))
          }
          onCompared={(id) =>
            setSelection((current) => ({ ...current, compared: id }))
          }
          task={captureTask}
          blocked={trainingTask.busy}
        />
        {reference && compared && (
          <CrosscoderTraining
            key={reference.id + compared.id}
            reference={reference}
            compared={compared}
            capabilities={capabilities}
            task={trainingTask}
            blocked={captureTask.busy}
          />
        )}
      </div>
      <div hidden={mode !== "explore" || loading || !restored}>
        <Panel title={t("Saved shared feature comparisons")}>
          <div className="feature-form">
            <label>
              {t("Choose a comparison")}
              <select
                aria-label={t("Choose a comparison")}
                value={comparison?.id || ""}
                onChange={(event) =>
                  setSelection((current) => ({
                    ...current,
                    comparison: event.target.value,
                  }))
                }
              >
                <option value="">{t("Select a comparison")}</option>
                {comparisons.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </Panel>
        {comparison ? (
          <CrosscoderExplorer
            key={comparison.id}
            identifier={comparison.id}
            refreshToken={refreshToken}
          />
        ) : (
          <EmptyState
            icon={<Sparkles />}
            title={t("No shared feature comparison selected")}
          >
            {t(
              "Prepare two activation captures and train a joint sparse dictionary to start exploring.",
            )}
          </EmptyState>
        )}
      </div>
      <CrosscoderStorage
        captures={captureRecords}
        comparisons={comparisonRecords}
        blocked={captureTask.busy || trainingTask.busy}
        onDeleted={(kind, identifier) => {
          if (kind === "capture")
            setCaptures((current) =>
              current.filter((item) => item.id !== identifier),
            );
          else
            setComparisons((current) =>
              current.filter((item) => item.id !== identifier),
            );
          setSelection((current) => ({
            ...current,
            reference:
              current.reference === identifier ? "" : current.reference,
            compared: current.compared === identifier ? "" : current.compared,
            comparison:
              current.comparison === identifier ? "" : current.comparison,
          }));
        }}
      />
      <details className="feature-methods">
        <summary>{t("Methods and research papers")}</summary>
        <p>
          {t(
            "The crosscoder sums both normalized banks in a shared ReLU encoder. Its loss combines reconstruction with feature activation weighted by the sum of individual decoder norms. Each bank uses a training only RMS L2 scale. Decoder columns remain unconstrained after initialization.",
          )}
        </p>
        <p>
          {t(
            "Held out documents are separated before training. Dense or polysemantic features can still occur. The reduced penalty shared feature variant from the 2025 paper is not implemented.",
          )}
        </p>
        <ul>
          {capabilities?.references.map((reference) => (
            <li key={reference.url}>
              <a href={reference.url} target="_blank" rel="noreferrer">
                {reference.title}
              </a>{" "}
              · {reference.year}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
