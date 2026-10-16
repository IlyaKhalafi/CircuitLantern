"use client";
import { ResearchNavigation } from "@/components/research-navigation";
import { useI18n } from "@/i18n/provider";
import {
  Activity,
  ArrowRight,
  Download,
  Play,
  RefreshCw,
  RotateCcw,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import { TraceResults } from "@/components/trace-results";
import { TrackedTokenInputs } from "@/components/tracked-token-inputs";
import { SaveExperiment } from "@/components/save-experiment";
import {
  HeadInspectionNotice,
  useHeadInspection,
} from "@/components/head-inspection";
import type { HeadInspectionDraft } from "@/lib/inspection";
import { TokenPreview } from "@/components/token-preview";
import { useTokenPreview } from "@/lib/token-preview";
import { useTargetPreview } from "@/lib/target-preview";
import { downloadJSON, post } from "@/lib/api";
import { interventionLabel } from "@/lib/interventions";
import type { Trace } from "@/lib/types";
import { truncateText } from "@/lib/text";
import { useRuntime } from "@/lib/runtime";
export default function PlaygroundPage() {
  const { t, rich } = useI18n();
  const {
    runtime,
    loading,
    error: runtimeError,
    refresh: refreshRuntime,
  } = useRuntime();
  const [prompt, setPrompt] = useState("The capital of France is");
  const [tokens, setTokens] = useState(16);
  const [captureAttention, setCaptureAttention] = useState(true);
  const [trackedTokens, setTrackedTokens] = useState<string[]>([]);
  const [attentionQuery, setAttentionQuery] = useState(-1);
  const [ablate, setAblate] = useState("");
  const [headLayer, setHeadLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [baseline, setBaseline] = useState<Trace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [traceSequence, setTraceSequence] = useState(0);
  const tokenPreview = useTokenPreview(
    [prompt],
    tokens,
    !!runtime?.loaded && !busy,
    runtime?.metadata?.execution_id,
  );
  const targetPreview = useTargetPreview(
    trackedTokens,
    !!runtime?.loaded && !busy,
    runtime?.metadata?.execution_id,
  );
  const targetsFit = targetPreview.data?.valid ?? true;
  const promptFits = tokenPreview.data?.prompts[0]?.fits ?? true;
  const queryTokens = tokenPreview.data?.prompts[0]?.tokens ?? [];
  const attentionQueryFits =
    !captureAttention ||
    attentionQuery === -1 ||
    (!tokenPreview.pending &&
      !!tokenPreview.data &&
      attentionQuery < queryTokens.length);
  useEffect(() => {
    setAttentionQuery(-1);
  }, [prompt, runtime?.metadata?.execution_id]);
  useEffect(() => {
    if (!runtime) return;
    const targets = runtime.attention_head_targets || [];
    if (targets.length) {
      setHeadLayer((current) =>
        targets.some((target) => target.layer === current)
          ? current
          : targets[0].layer,
      );
    }
    setAblate((current) =>
      current === "head"
        ? targets.length
          ? current
          : ""
        : current !== "" && +current >= (runtime.metadata?.num_layers ?? 0)
          ? ""
          : current,
    );
    const allowance =
      runtime.context_length ?? runtime.metadata?.context_length ?? 2048;
    setTokens((current) =>
      current < allowance
        ? current
        : ([64, 32, 16, 8, 4, 1, 0].find((n) => n < allowance) ?? 0),
    );
  }, [runtime]);
  async function run(event: React.FormEvent) {
    event.preventDefault();
    if (inspection.pending) return;
    setBusy(true);
    setError(null);
    try {
      const result = await post<Trace>("/trace", {
        prompt,
        expected_execution_id: runtime?.metadata?.execution_id ?? undefined,
        max_new_tokens: tokens,
        capture_attention: captureAttention,
        attention_query_position: attentionQuery,
        tracked_tokens: trackedTokens.filter((token) => token.length > 0),
        ablate_layer: ablate === "" || ablate === "head" ? null : +ablate,
        ablate_head: ablate === "head" ? { layer: headLayer, head } : null,
      });
      setTrace(result);
      setTraceSequence((sequence) => sequence + 1);
      if (result.ablate_layer === null && !result.ablate_head)
        setBaseline(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const contextLimit =
    runtime?.context_length ?? runtime?.metadata?.context_length ?? 2048;
  const promptLimit = Math.max(
    0,
    Math.min(runtime?.max_prompt_tokens ?? 128, contextLimit - tokens),
  );
  const headTarget = runtime?.attention_head_targets?.find(
    (target) => target.layer === headLayer,
  );
  useEffect(() => {
    setHead((current) =>
      Math.min(current, Math.max(0, (headTarget?.heads ?? 1) - 1)),
    );
  }, [headTarget?.heads]);
  const applyInspection = useCallback((draft: HeadInspectionDraft) => {
    setPrompt(draft.prompt);
    setTokens(0);
    setCaptureAttention(true);
    setAttentionQuery(-1);
    setAblate("head");
    setHeadLayer(draft.layer);
    setHead(draft.head);
    setTrace(null);
    setBaseline(null);
    setError(null);
  }, []);
  const inspection = useHeadInspection(runtime, applyInspection);
  const compatibleBaseline =
    baseline?.prompt === trace?.prompt &&
    baseline?.revision === trace?.revision &&
    baseline?.model_id === trace?.model_id &&
    baseline?.execution_precision === trace?.execution_precision &&
    baseline?.device === trace?.device &&
    baseline?.execution_id === trace?.execution_id &&
    JSON.stringify(baseline?.execution_config) ===
      JSON.stringify(trace?.execution_config) &&
    (trace?.ablate_layer != null || !!trace?.ablate_head)
      ? baseline
      : null;
  return (
    <>
      <PageHeader
        eyebrow={t("FOLLOW THE COMPUTATION")}
        title={t("From a prompt to a prediction.")}
        description={
          runtime?.loaded
            ? t("{v0} \u00B7 live activations from your local model", {
                v0: runtime.metadata?.id,
              })
            : t(
                "Watch information move through the model, one layer at a time.",
              )
        }
        actions={
          <>
            <button
              className="button secondary"
              disabled={loading || busy}
              onClick={() => {
                setError(null);
                refreshRuntime();
              }}
            >
              <RefreshCw size={14} />
              {t("Refresh runtime")}
            </button>
            {trace && (
              <button
                className="button secondary"
                onClick={() => downloadJSON(trace, "activation-trace.json")}
              >
                <Download size={14} />
                {t("Export trace")}
              </button>
            )}
          </>
        }
      />
      <ResearchNavigation />
      <ErrorNotice message={runtimeError || error} />
      <HeadInspectionNotice inspection={inspection} />
      {trace?.execution_id &&
        runtime?.metadata?.execution_id &&
        trace.execution_id !== runtime.metadata.execution_id && (
          <div className="notice info">
            {t(
              "This capture belongs to a previous model session. Its recorded model and settings are preserved.",
            )}
          </div>
        )}
      {loading ? (
        <Loading label={t("Connecting to model runtime")} />
      ) : !runtime?.loaded ? (
        <EmptyState
          icon={<Activity size={28} />}
          title={t("Give a prediction some context.")}
          href="/"
          action={t("Load a model")}
        >
          {t(
            "Choose a checkpoint, then run a prompt to capture residual streams, layer predictions, and attention patterns.",
          )}
        </EmptyState>
      ) : (
        <>
          <Panel
            title={t("Prompt your model")}
            subtitle={
              tokens === 0
                ? t(
                    "Capture the prompt without generating a continuation. Up to {v0} prompt tokens.",
                    {
                      v0: promptLimit,
                    },
                  )
                : t(
                    "Deterministic generation with bounded activation capture. Up to {v0} prompt tokens with this generation length.",
                    {
                      v0: promptLimit,
                    },
                  )
            }
          >
            <form className="prompt-box" onSubmit={run}>
              <label htmlFor="prompt" className="sr-only">
                {t("Prompt")}
              </label>
              <textarea
                dir="auto"
                id="prompt"
                aria-label={t("Prompt")}
                disabled={inspection.pending}
                value={prompt}
                onChange={(e) => setPrompt(truncateText(e.target.value, 8000))}
                placeholder={t("Ask a question about how a model thinks\u2026")}
                maxLength={16000}
                required
              />
              <TokenPreview preview={tokenPreview} />
              <div className="prompt-suggestions">
                {[
                  "The capital of France is",
                  "When Mary and John went to the store, John gave a drink to",
                  "The opposite of hot is",
                ].map((p) => (
                  <button
                    type="button"
                    key={p}
                    disabled={inspection.pending}
                    onClick={() => setPrompt(p)}
                  >
                    {p.length > 42 ? `${p.slice(0, 42)}…` : p}
                  </button>
                ))}
              </div>
              <div className="prompt-toolbar">
                <div className="prompt-controls">
                  <label>
                    {t("Generate")}
                    <select
                      aria-label={t("Generated tokens")}
                      disabled={busy || inspection.pending}
                      value={tokens}
                      onChange={(e) => setTokens(+e.target.value)}
                    >
                      {[0, 1, 4, 8, 16, 32, 64]
                        .filter((n) => n < contextLimit)
                        .map((n) => (
                          <option key={n} value={n}>
                            {n === 0
                              ? t("Capture only")
                              : t("{v0} tokens", {
                                  v0: n,
                                })}
                          </option>
                        ))}
                    </select>
                  </label>
                  <label>
                    {t("Intervention")}
                    <select
                      aria-label={t("Layer intervention")}
                      disabled={inspection.pending}
                      value={ablate}
                      onChange={(e) => setAblate(e.target.value)}
                    >
                      <option value="">{t("None \u00B7 baseline")}</option>
                      {!!runtime.attention_head_targets?.length && (
                        <option value="head">{t("Zero attention head")}</option>
                      )}
                      {Array.from(
                        { length: runtime.metadata?.num_layers || 0 },
                        (_, i) => (
                          <option key={i} value={i}>
                            {rich("Bypass block {v0}", {
                              v0: i,
                            })}
                          </option>
                        ),
                      )}
                    </select>
                  </label>
                  {ablate === "head" && (
                    <>
                      <label>
                        {t("Layer")}
                        <select
                          aria-label={t("Head ablation layer")}
                          disabled={inspection.pending}
                          value={headLayer}
                          onChange={(event) => {
                            setHeadLayer(+event.target.value);
                            setHead(0);
                          }}
                        >
                          {runtime.attention_head_targets?.map((target) => (
                            <option key={target.layer} value={target.layer}>
                              {rich("Layer {v0}", {
                                v0: target.layer,
                              })}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        {t("Head")}
                        <select
                          aria-label={t("Attention head to zero")}
                          disabled={inspection.pending}
                          value={head}
                          onChange={(event) => setHead(+event.target.value)}
                        >
                          {Array.from(
                            { length: headTarget?.heads ?? 0 },
                            (_, i) => (
                              <option key={i} value={i}>
                                {rich("Head {v0}", {
                                  v0: i,
                                })}
                              </option>
                            ),
                          )}
                        </select>
                      </label>
                    </>
                  )}
                  <label>
                    <input
                      dir="auto"
                      type="checkbox"
                      disabled={inspection.pending}
                      checked={captureAttention}
                      onChange={(e) => setCaptureAttention(e.target.checked)}
                    />
                    {t("Capture attention")}
                  </label>
                </div>
                <button
                  type="submit"
                  className="button primary"
                  disabled={
                    busy ||
                    !prompt.trim() ||
                    !promptFits ||
                    !targetsFit ||
                    !attentionQueryFits ||
                    inspection.pending ||
                    promptLimit < 1 ||
                    (ablate === "head" && !headTarget)
                  }
                >
                  <Play size={14} />
                  {busy ? t("Tracing activations") : t("Run & trace")}
                  <ArrowRight size={14} />
                </button>
              </div>
              {captureAttention && (
                <details className="attention-capture-options">
                  <summary>{t("Attention capture options")}</summary>
                  <label>
                    {t("Routing query")}
                    <select
                      aria-label={t("Attention routing query")}
                      value={attentionQuery}
                      disabled={busy || inspection.pending}
                      onChange={(event) =>
                        setAttentionQuery(+event.target.value)
                      }
                    >
                      <option value={-1}>{t("Final prompt token")}</option>
                      {attentionQuery >= 0 && !queryTokens.length && (
                        <option value={attentionQuery}>
                          {rich("Position {v0} \u00B7 checking prompt tokens", {
                            v0: attentionQuery,
                          })}
                        </option>
                      )}
                      {queryTokens.map((token, index) => (
                        <option key={index} value={index}>
                          {index} · {token || "∅"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p>
                    {t(
                      "Capture per-head routing from one prompt token across layers. Run a new trace after choosing a position. Average patterns still cover the whole prompt.",
                    )}
                  </p>
                </details>
              )}
              <TrackedTokenInputs
                tokens={trackedTokens}
                preview={targetPreview}
                onChange={setTrackedTokens}
                disabled={busy || inspection.pending}
              />
            </form>
          </Panel>
          {ablate !== "" && ablate !== "head" && (
            <div className="notice info section-gap">
              {t(
                "Block bypass replaces the selected transformer block output with its input residual stream. Run a baseline in the same model session with the same prompt to compare residual magnitudes.",
              )}
            </div>
          )}
          {busy && (
            <Loading
              label={
                tokens === 0
                  ? t("Running the forward pass and capturing prompt states")
                  : t(
                      "Running the forward pass, capturing states and generating a continuation",
                    )
              }
            />
          )}
          {trace && (
            <div key={traceSequence} hidden={busy || inspection.pending}>
              <TraceResults trace={trace} comparison={compatibleBaseline} />
              <SaveExperiment
                source={{ kind: "trace", trace }}
                defaultName={`${interventionLabel(trace)} · ${truncateText(trace.prompt, 45)}`}
                disabled={busy || inspection.pending}
              />
            </div>
          )}
          {!trace && !busy && (
            <div className="how-it-works">
              <div>
                <span>
                  <Activity size={15} />
                </span>
                <div>
                  <h3>{t("Residual stream")}</h3>
                  <p>{t("Track activation magnitudes across layers.")}</p>
                </div>
              </div>
              <div>
                <span>
                  <Play size={15} />
                </span>
                <div>
                  <h3>{t("Logit lens")}</h3>
                  <p>{t("See the prediction at each stage of processing.")}</p>
                </div>
              </div>
              <div>
                <span>
                  <RotateCcw size={15} />
                </span>
                <div>
                  <h3>{t("Causal interventions")}</h3>
                  <p>
                    {t(
                      "Bypass a block or zero a head and measure what changes.",
                    )}
                  </p>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
