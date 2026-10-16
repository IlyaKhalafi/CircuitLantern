"use client";

import { useEffect, useState } from "react";
import { Download, FlaskConical, Play, RefreshCw, Square } from "lucide-react";
import { CausalNavigation } from "@/components/causal-navigation";
import { HeadSweepResults } from "@/components/head-sweep-results";
import { TokenPreview } from "@/components/token-preview";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import { downloadJSON } from "@/lib/api";
import { useBackgroundJob } from "@/lib/jobs";
import { useRuntime } from "@/lib/runtime";
import { useTokenPreview } from "@/lib/token-preview";
import type { HeadSweepResult } from "@/lib/types";

export default function HeadSweepPage() {
  const { runtime, loading, error: runtimeError, refresh } = useRuntime();
  const [prompt, setPrompt] = useState("The capital of France is");
  const [target, setTarget] = useState("");
  const [layer, setLayer] = useState("0");
  const { job, error, clearError, start, cancel, busy, canCancel, cancelling } =
    useBackgroundJob<HeadSweepResult>("lantern-head-sweep-job");
  const result = job?.result;
  const targets = runtime?.attention_head_targets ?? [];
  const selected = targets.find((item) => item.layer === +layer) ?? targets[0];
  const selectedTargets =
    layer === "all" ? targets : selected ? [selected] : [];
  const count = selectedTargets.reduce((sum, item) => sum + item.heads, 0);
  const maximum = runtime?.max_head_sweep_heads ?? 512;
  const allHeads = targets.reduce((sum, item) => sum + item.heads, 0);
  const preview = useTokenPreview(
    [prompt],
    0,
    !!runtime?.loaded && !busy,
    runtime?.metadata?.execution_id,
  );
  const fits = preview.data?.prompts[0]?.fits ?? true;

  useEffect(() => {
    if (!targets.length) return;
    if (
      (layer === "all" && allHeads > maximum) ||
      (layer !== "all" && !targets.some((item) => item.layer === +layer))
    )
      setLayer(String(targets[0].layer));
  }, [layer, runtime]);

  useEffect(() => {
    if (!result) return;
    setPrompt(result.prompt);
    setTarget(result.requested_target);
    setLayer(
      result.layer_targets.length === 1
        ? String(result.layer_targets[0].layer)
        : "all",
    );
  }, [result]);

  async function run(event: React.FormEvent) {
    event.preventDefault();
    await start("/head-sweep", {
      expected_execution_id: runtime?.metadata?.execution_id,
      prompt,
      target,
      layers: layer === "all" ? null : [selected?.layer],
    });
  }

  return (
    <>
      <PageHeader
        eyebrow="TEST INDIVIDUAL ATTENTION HEADS"
        title="Remove a head. Measure what changes."
        description="Zero each attention head independently and compare its effect on the next token prediction."
        actions={
          <>
            <button
              className="button secondary"
              disabled={loading || busy}
              onClick={() => {
                clearError();
                refresh();
              }}
            >
              <RefreshCw size={14} /> Refresh runtime
            </button>
            {result && (
              <button
                className="button secondary"
                onClick={() => downloadJSON(result, "head-ablation-sweep.json")}
              >
                <Download size={14} /> Export head sweep
              </button>
            )}
          </>
        }
      />
      <CausalNavigation />
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
          title="Test a model's attention heads."
          href="/"
          action="Load a model"
        >
          Choose a checkpoint to measure how individual head outputs influence a
          prediction.
        </EmptyState>
      ) : !targets.length ? (
        <div className="notice info">
          Head sweeps are unavailable for this architecture. Verified GPT-2,
          Llama, Qwen2 and GPT-NeoX attention layouts are supported.
        </div>
      ) : (
        <Panel
          title="Design a head ablation sweep"
          subtitle="The baseline and every ablated run use the same prompt. No tokens are generated."
        >
          <form className="panel-body" onSubmit={run}>
            <label>
              Prompt
              <textarea
                aria-label="Head sweep prompt"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                maxLength={8000}
                rows={3}
                disabled={busy}
                required
              />
            </label>
            <TokenPreview preview={preview} />
            <div className="head-sweep-controls">
              <label>
                Layers to test
                <select
                  aria-label="Layers to test"
                  value={
                    layer === "all" ? "all" : String(selected?.layer ?? "")
                  }
                  disabled={busy}
                  onChange={(event) => setLayer(event.target.value)}
                >
                  {targets.map((item) => (
                    <option value={item.layer} key={item.layer}>
                      Layer {item.layer} · {item.heads} heads
                    </option>
                  ))}
                  <option value="all" disabled={allHeads > maximum}>
                    All layers · {allHeads} heads
                    {allHeads > maximum ? " · exceeds sweep limit" : ""}
                  </option>
                </select>
              </label>
              <label className="head-target-field">
                Target token
                <input
                  type="text"
                  className="form-input"
                  aria-label="Head sweep target token"
                  value={target}
                  onChange={(event) => setTarget(event.target.value)}
                  placeholder="Automatic baseline top prediction"
                  maxLength={100}
                  disabled={busy}
                />
              </label>
            </div>
            <p className="control-note">
              Leave the target empty to use the baseline top prediction, or
              enter exactly one token with its leading space when needed.
            </p>
            <div className="head-sweep-actions">
              <p className="control-note">
                {count} heads selected. The sweep runs one baseline and one
                forward pass per head, with a limit of {maximum} heads per job.
                Cancellation takes effect after the current forward pass.
              </p>
              <div className="action-row">
                {busy && canCancel && (
                  <button
                    type="button"
                    className="button secondary"
                    onClick={cancel}
                    disabled={cancelling}
                  >
                    <Square size={13} />{" "}
                    {cancelling ? "Cancelling" : "Cancel sweep"}
                  </button>
                )}
                <button
                  type="submit"
                  className="button primary"
                  disabled={
                    busy || !prompt.trim() || !fits || !count || count > maximum
                  }
                >
                  <Play size={14} />{" "}
                  {busy ? "Running head sweep" : "Run head sweep"}
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
              <progress
                aria-label="Head sweep progress"
                value={job.progress}
                max={100}
              />
            </div>
          )}
        </Panel>
      )}
      {result ? (
        <div className="section-gap">
          <HeadSweepResults key={job?.id} result={result} />
        </div>
      ) : (
        !busy && (
          <div className="guide-callout">
            <h2>What the sweep measures</h2>
            <p>
              Each intervention zeros one query head output before the attention
              output projection at every prompt position. Negative target log
              probability changes mean removal reduced the target probability.
              The distribution divergence also captures changes to other tokens.
              Start with one layer before expanding the sweep.
            </p>
          </div>
        )
      )}
    </>
  );
}
