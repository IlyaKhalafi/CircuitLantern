"use client";
import { useI18n } from "@/i18n/provider";
import { ExecutionContext } from "@/components/execution-context";
import {
  ArrowRight,
  Download,
  FlaskConical,
  Play,
  Square,
  RefreshCw,
} from "lucide-react";
import { useEffect, useState } from "react";
import { CausalNavigation } from "@/components/causal-navigation";
import { CausalResults } from "@/components/causal-results";
import { SaveJobExperiment } from "@/components/save-job-experiment";
import { TokenPreview } from "@/components/token-preview";
import { useTokenPreview } from "@/lib/token-preview";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import { downloadJSON } from "@/lib/api";
import { useBackgroundJob } from "@/lib/jobs";
import type { CausalResult } from "@/lib/types";
import { truncateText } from "@/lib/text";
import { useRuntime } from "@/lib/runtime";
export default function CausalPage() {
  const { t } = useI18n();
  const {
    runtime,
    loading,
    error: runtimeError,
    refresh: refreshRuntime,
  } = useRuntime();
  const [clean, setClean] = useState("The capital of France is");
  const [corrupted, setCorrupted] = useState("The capital of Poland is");
  const [target, setTarget] = useState("");
  const [position, setPosition] = useState("last");
  const [customPosition, setCustomPosition] = useState(0);
  const { job, error, clearError, start, cancel, busy, canCancel, cancelling } =
    useBackgroundJob<CausalResult>("lantern-causal-job");
  const result = job?.result;
  const tokenPreview = useTokenPreview(
    [clean, corrupted],
    0,
    !!runtime?.loaded && !busy,
    runtime?.metadata?.execution_id,
  );
  const previews = tokenPreview.data?.prompts;
  const promptsFit =
    !previews ||
    (previews.every((prompt) => prompt.fits) &&
      previews[0].token_count === previews[1].token_count);
  const captureLimit = Math.min(
    runtime?.max_prompt_tokens ?? 128,
    runtime?.context_length ?? runtime?.metadata?.context_length ?? 2048,
  );
  const positionFits =
    position !== "custom" ||
    (customPosition >= 0 &&
      customPosition <
        Math.min(captureLimit, previews?.[0]?.token_count ?? captureLimit));
  useEffect(() => {
    if (!result) return;
    setClean(result.clean_prompt);
    setCorrupted(result.corrupted_prompt);
    setTarget(result.requested_target || "");
    setPosition(
      result.position === null
        ? "all"
        : result.position === -1
          ? "last"
          : "custom",
    );
    if (result.position !== null && result.position >= 0)
      setCustomPosition(result.position);
  }, [result]);
  async function run(event: React.FormEvent) {
    event.preventDefault();
    await start("/causal", {
      expected_execution_id: runtime?.metadata?.execution_id ?? undefined,
      clean_prompt: clean,
      corrupted_prompt: corrupted,
      target,
      position:
        position === "all" ? null : position === "last" ? -1 : customPosition,
    });
  }
  return (
    <>
      <PageHeader
        eyebrow={t("TEST THE PATH OF INFORMATION")}
        title={t("Residual patching")}
        description={<ExecutionContext runtime={runtime} />}
        help={t(
          "Patch clean residual activations into a corrupted prompt, one transformer block at a time.",
        )}
        actions={
          <>
            <button
              className="button secondary"
              disabled={loading || busy}
              onClick={() => {
                clearError();
                refreshRuntime();
              }}
            >
              <RefreshCw size={14} />
              {t("Refresh runtime")}
            </button>
            {result && (
              <button
                className="button secondary"
                onClick={() => downloadJSON(result, "causal-patching.json")}
              >
                <Download size={14} />
                {t("Export sweep")}
              </button>
            )}
          </>
        }
      />
      <CausalNavigation />
      <ErrorNotice
        message={runtimeError || (error === result?.error ? null : error)}
      />
      {result?.execution_id &&
        runtime?.metadata?.execution_id &&
        result.execution_id !== runtime.metadata.execution_id && (
          <div className="notice info">
            {t(
              "These results belong to a previous model session. Their recorded model and settings are preserved.",
            )}
          </div>
        )}
      {loading ? (
        <Loading label={t("Connecting to model runtime")} />
      ) : !runtime?.loaded ? (
        <EmptyState
          icon={<FlaskConical size={28} />}
          title={t("Turn an observation into a test.")}
          href="/"
          action={t("Load a model")}
        >
          {t(
            "Choose a checkpoint to compare clean and corrupted prompts and identify where restored activations change a prediction.",
          )}
        </EmptyState>
      ) : (
        <>
          <Panel
            title={t("Design a controlled comparison")}
            subtitle={t(
              "Prompts must have equal token counts so positions align. Up to {v0} tokens per prompt within the loaded context allowance.",
              {
                v0: captureLimit,
              },
            )}
          >
            <form className="panel-body" onSubmit={run}>
              <div className="comparison-grid">
                <label>
                  {t("Clean prompt")}
                  <textarea
                    dir="auto"
                    aria-label={t("Clean prompt")}
                    value={clean}
                    onChange={(e) => setClean(e.target.value)}
                    rows={3}
                    maxLength={8000}
                    disabled={busy}
                    required
                  />
                </label>
                <label>
                  {t("Corrupted prompt")}
                  <textarea
                    dir="auto"
                    aria-label={t("Corrupted prompt")}
                    value={corrupted}
                    onChange={(e) => setCorrupted(e.target.value)}
                    rows={3}
                    maxLength={8000}
                    disabled={busy}
                    required
                  />
                </label>
              </div>
              <TokenPreview
                preview={tokenPreview}
                labels={["Clean", "Corrupted"]}
              />
              {position === "custom" && !positionFits && (
                <p className="token-preview-warning" role="status">
                  {t("Choose a patch position within the tokenized prompts.")}
                </p>
              )}
              <div className="form-grid">
                <label>
                  {t("Target next token")}
                  <input
                    dir="auto"
                    className="form-input mono"
                    aria-label={t("Target next token")}
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                    placeholder={t("Auto \u00B7 clean top prediction")}
                    maxLength={100}
                    disabled={busy}
                  />
                </label>
                <label>
                  {t("Positions to restore")}
                  <select
                    aria-label={t("Positions to restore")}
                    value={position}
                    onChange={(e) => setPosition(e.target.value)}
                    disabled={busy}
                  >
                    <option value="last">{t("Last prompt token")}</option>
                    <option value="all">
                      {t("All prompt tokens \u00B7 sanity check")}
                    </option>
                    <option value="custom">
                      {t("A specific token position")}
                    </option>
                  </select>
                </label>
                {position === "custom" && (
                  <label>
                    {t("Token position \u00B7 zero based")}
                    <input
                      dir="auto"
                      className="form-input mono"
                      aria-label={t("Patch token position")}
                      type="number"
                      min={0}
                      max={Math.max(
                        0,
                        (previews?.[0]?.token_count ?? captureLimit) - 1,
                      )}
                      value={customPosition}
                      onChange={(e) => setCustomPosition(+e.target.value)}
                      disabled={busy}
                    />
                  </label>
                )}
              </div>
              <div className="prompt-toolbar">
                <p className="control-note">
                  {t(
                    "Target text must encode to one token. A leading space may matter.",
                  )}
                </p>
                <div className="action-row">
                  {busy && canCancel && (
                    <button
                      type="button"
                      className="button secondary"
                      onClick={cancel}
                      disabled={cancelling}
                    >
                      <Square size={12} />{" "}
                      {cancelling ? t("Cancelling") : t("Cancel sweep")}
                    </button>
                  )}
                  <button
                    className="button primary"
                    type="submit"
                    disabled={
                      busy ||
                      !clean.trim() ||
                      !corrupted.trim() ||
                      !promptsFit ||
                      !positionFits
                    }
                  >
                    <Play size={13} />
                    {busy ? t("Running sweep") : t("Run patch sweep")}
                    <ArrowRight size={14} />
                  </button>
                </div>
              </div>
            </form>
            {job && (
              <div className="job-progress" role="status" aria-live="polite">
                <div>
                  <span>{t(job.message)}</span>
                  <strong>{job.progress}%</strong>
                </div>
                <progress
                  aria-label={t("Residual patch sweep progress")}
                  value={job.progress}
                  max={100}
                />
              </div>
            )}
          </Panel>
          {!result && !busy && (
            <div className="guide-callout method-explanation">
              <h2>{t("What the sweep measures")}</h2>
              <p>
                {t(
                  "First the model runs each prompt unchanged. Then each block is tested in a separate corrupted run with clean residual outputs restored at the selected positions. Recovery measures the change in target log probability relative to the clean and corrupted baseline difference. Negative values and values above one are meaningful.",
                )}
              </p>
            </div>
          )}
        </>
      )}
      {result && (
        <>
          <CausalResults result={result} />
          {job && (
            <SaveJobExperiment
              key={job.id}
              jobId={job.id}
              defaultName={
                "Residual patching · " + truncateText(result.clean_prompt, 60)
              }
              disabled={busy}
            />
          )}
        </>
      )}
    </>
  );
}
