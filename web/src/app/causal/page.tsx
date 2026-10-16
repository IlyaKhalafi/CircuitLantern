"use client";

import { executionLabel } from "@/lib/execution";

import {
  ArrowRight,
  Download,
  FlaskConical,
  Play,
  Square,
  RefreshCw,
} from "lucide-react";
import { useEffect, useState } from "react";
import { SeriesChart } from "@/components/charts";
import { TokenPreview } from "@/components/token-preview";
import { useTokenPreview } from "@/lib/token-preview";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  Metric,
  PageHeader,
  Panel,
} from "@/components/ui";
import { downloadJSON, number } from "@/lib/api";
import { useBackgroundJob } from "@/lib/jobs";
import type { CausalResult } from "@/lib/types";
import { useRuntime } from "@/lib/runtime";

export default function CausalPage() {
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
  const { job, error, clearError, start, cancel, busy } =
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
        eyebrow="TEST THE PATH OF INFORMATION"
        title="Restore a signal. Measure the effect."
        description="Patch clean residual activations into a corrupted prompt, one transformer block at a time."
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
              <RefreshCw size={14} /> Refresh runtime
            </button>
            {result && (
              <button
                className="button secondary"
                onClick={() => downloadJSON(result, "causal-patching.json")}
              >
                <Download size={14} /> Export sweep
              </button>
            )}
          </>
        }
      />
      <ErrorNotice message={runtimeError || error} />
      {result?.execution_id &&
        runtime?.metadata?.execution_id &&
        result.execution_id !== runtime.metadata.execution_id && (
          <div className="notice info">
            These results belong to a previous model session. Their recorded
            model and settings are preserved.
          </div>
        )}
      {loading ? (
        <Loading label="Connecting to model runtime" />
      ) : !runtime?.loaded ? (
        <EmptyState
          icon={<FlaskConical size={28} />}
          title="Turn an observation into a test."
          href="/"
          action="Load a model"
        >
          Choose a checkpoint to compare clean and corrupted prompts and
          identify where restored activations change a prediction.
        </EmptyState>
      ) : (
        <>
          <Panel
            title="Design a controlled comparison"
            subtitle={`Prompts must have equal token counts so positions align. Up to ${captureLimit} tokens per prompt within the loaded context allowance.`}
          >
            <form className="panel-body" onSubmit={run}>
              <div className="comparison-grid">
                <label>
                  Clean prompt
                  <textarea
                    aria-label="Clean prompt"
                    value={clean}
                    onChange={(e) => setClean(e.target.value)}
                    rows={3}
                    maxLength={8000}
                    disabled={busy}
                    required
                  />
                </label>
                <label>
                  Corrupted prompt
                  <textarea
                    aria-label="Corrupted prompt"
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
                  Choose a patch position within the tokenized prompts.
                </p>
              )}
              <div className="form-grid">
                <label>
                  Target next token
                  <input
                    className="form-input mono"
                    aria-label="Target next token"
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                    placeholder="Auto · clean top prediction"
                    maxLength={100}
                    disabled={busy}
                  />
                </label>
                <label>
                  Positions to restore
                  <select
                    aria-label="Positions to restore"
                    value={position}
                    onChange={(e) => setPosition(e.target.value)}
                    disabled={busy}
                  >
                    <option value="last">Last prompt token</option>
                    <option value="all">
                      All prompt tokens · sanity check
                    </option>
                    <option value="custom">A specific token position</option>
                  </select>
                </label>
                {position === "custom" && (
                  <label>
                    Token position · zero based
                    <input
                      className="form-input mono"
                      aria-label="Patch token position"
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
                  Target text must encode to one token. A leading space may
                  matter.
                </p>
                <div className="action-row">
                  {busy && (
                    <button
                      type="button"
                      className="button secondary"
                      onClick={cancel}
                    >
                      <Square size={12} /> Cancel sweep
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
                    {busy ? "Running sweep" : "Run patch sweep"}
                    <ArrowRight size={14} />
                  </button>
                </div>
              </div>
            </form>
            {job && (
              <div className="job-progress" role="status" aria-live="polite">
                <div>
                  <span>{job.message}</span>
                  <strong>{job.progress}%</strong>
                </div>
                <progress value={job.progress} max={100} />
              </div>
            )}
          </Panel>
          {!result && !busy && (
            <div className="guide-callout">
              <h2>What the sweep measures</h2>
              <p>
                First the model runs each prompt unchanged. Then each block is
                tested in a separate corrupted run with clean residual outputs
                restored at the selected positions. Recovery measures the change
                in target log probability relative to the clean and corrupted
                baseline difference. Negative values and values above one are
                meaningful.
              </p>
            </div>
          )}
        </>
      )}
      {result && (
        <>
          <div className="trace-meta">
            <span
              className={`pill ${job?.status === "cancelled" ? "orange" : "green"}`}
            >
              {job?.status === "cancelled" ? "Partial sweep" : "Sweep complete"}
            </span>
            <span>{result.model_id}</span>
            <span>{number(result.duration_ms, 0)} ms</span>
            <span>revision {result.revision.slice(0, 8)}</span>
            <span>{executionLabel(result.execution_precision)}</span>
          </div>
          <div className="metrics-grid">
            <Metric
              label="Target token"
              value={
                <span className="target-token">
                  {result.target_token || "∅"}
                </span>
              }
              note={`Vocabulary ID ${result.target_token_id}`}
            />
            <Metric
              label="Clean probability"
              value={`${(result.clean.probability * 100).toFixed(3)}%`}
              note={`Log probability ${number(result.clean.log_probability, 4)}`}
            />
            <Metric
              label="Corrupted probability"
              value={`${(result.corrupted.probability * 100).toFixed(3)}%`}
              note={`Log probability ${number(result.corrupted.log_probability, 4)}`}
            />
            <Metric
              label="Blocks tested"
              value={result.layers.length}
              note={
                result.position === null
                  ? "All token positions restored"
                  : result.position === -1
                    ? "Last token position restored"
                    : `Position ${result.position} restored`
              }
            />
          </div>
          {result.position === null && (
            <div className="notice info">
              All position restoration is a sanity check. Later layers receive
              the complete clean residual stream, so recovery near one is
              expected at every block. Use a single token position to
              investigate localization.
            </div>
          )}
          {result.recovery_warning && (
            <div className="notice info">{result.recovery_warning}</div>
          )}
          {!result.recovery_defined && (
            <div className="notice info">
              The corrupted prompt did not lower the target log probability by
              more than {result.recovery_minimum_gap} nats. Normalized recovery
              is undefined. The raw log probability change remains available.
            </div>
          )}
          <div className="two-column">
            <Panel
              title={
                result.position === null
                  ? "All position restoration check"
                  : "Recovery by transformer block"
              }
              subtitle="One is clean baseline recovery. Zero is no change from the corrupted baseline."
            >
              <SeriesChart
                data={result.layers.map((l) => ({
                  index: l.layer,
                  recovery: l.recovery,
                  delta: l.delta_log_probability,
                }))}
                series={[
                  {
                    key: result.recovery_defined ? "recovery" : "delta",
                    label: result.recovery_defined
                      ? "Normalized recovery"
                      : "Log probability change",
                    color: "#b68159",
                  },
                ]}
              />
              <p className="inline-note">{result.method}</p>
            </Panel>
            <Panel
              title="Token alignment"
              subtitle="Columns preserve the positions used in the patching intervention."
            >
              <div className="alignment-table table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>POSITION</th>
                      <th>CLEAN TOKEN</th>
                      <th>CORRUPTED TOKEN</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.tokens.map((token, i) => (
                      <tr key={i}>
                        <td className="code">{i}</td>
                        <td>
                          <span className="lens-token">{token || "∅"}</span>
                        </td>
                        <td>
                          <span className="lens-token">
                            {result.corrupted_tokens[i] || "∅"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>
          <Panel
            title="Per block patching results"
            subtitle="Each row is an independent forward pass with exactly one block patched."
          >
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>BLOCK</th>
                    <th>TARGET PROBABILITY</th>
                    <th>LOG PROBABILITY</th>
                    <th>CHANGE FROM CORRUPTED</th>
                    <th>NORMALIZED RECOVERY</th>
                  </tr>
                </thead>
                <tbody>
                  {result.layers.map((layer) => (
                    <tr key={layer.layer}>
                      <td className="code">Layer {layer.layer}</td>
                      <td className="code">
                        {(layer.probability * 100).toFixed(4)}%
                      </td>
                      <td className="code">
                        {number(layer.log_probability, 5)}
                      </td>
                      <td className="code">
                        {number(layer.delta_log_probability, 5)}
                      </td>
                      <td className="code">{number(layer.recovery, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
