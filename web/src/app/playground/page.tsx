"use client";

import {
  Activity,
  ArrowRight,
  Bookmark,
  Check,
  Download,
  Play,
  RefreshCw,
  RotateCcw,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  Panel,
} from "@/components/ui";
import { TraceResults } from "@/components/trace-results";
import { TokenPreview } from "@/components/token-preview";
import { useTokenPreview } from "@/lib/token-preview";
import { downloadJSON, post } from "@/lib/api";
import { interventionLabel } from "@/lib/interventions";
import type { Trace } from "@/lib/types";
import { useRuntime } from "@/lib/runtime";

export default function PlaygroundPage() {
  const {
    runtime,
    loading,
    error: runtimeError,
    refresh: refreshRuntime,
  } = useRuntime();
  const [prompt, setPrompt] = useState("The capital of France is");
  const [tokens, setTokens] = useState(16);
  const [captureAttention, setCaptureAttention] = useState(true);
  const [ablate, setAblate] = useState("");
  const [headLayer, setHeadLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [baseline, setBaseline] = useState<Trace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const tokenPreview = useTokenPreview(
    [prompt],
    tokens,
    !!runtime?.loaded && !busy,
    runtime?.metadata?.execution_id,
  );
  const promptFits = tokenPreview.data?.prompts[0]?.fits ?? true;
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
        : [64, 32, 16, 8, 4, 1].find((n) => n < allowance) || 1,
    );
  }, [runtime]);
  async function run(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      const result = await post<Trace>("/trace", {
        prompt,
        expected_execution_id: runtime?.metadata?.execution_id ?? undefined,
        max_new_tokens: tokens,
        capture_attention: captureAttention,
        ablate_layer: ablate === "" || ablate === "head" ? null : +ablate,
        ablate_head: ablate === "head" ? { layer: headLayer, head } : null,
      });
      setTrace(result);
      setName(`${interventionLabel(result)} · ${prompt.slice(0, 45)}`);
      if (result.ablate_layer === null && !result.ablate_head)
        setBaseline(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function save() {
    if (!trace || !name.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await post("/experiments", { name: name.trim(), notes: "", trace });
      setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
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
        eyebrow="FOLLOW THE COMPUTATION"
        title="From a prompt to a prediction."
        description={
          runtime?.loaded
            ? `${runtime.metadata?.id} · live activations from your local model`
            : "Watch information move through the model, one layer at a time."
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
              <RefreshCw size={14} /> Refresh runtime
            </button>
            {trace && (
              <button
                className="button secondary"
                onClick={() => downloadJSON(trace, "activation-trace.json")}
              >
                <Download size={14} /> Export trace
              </button>
            )}
          </>
        }
      />
      <ErrorNotice message={runtimeError || error} />
      {trace?.execution_id &&
        runtime?.metadata?.execution_id &&
        trace.execution_id !== runtime.metadata.execution_id && (
          <div className="notice info">
            This capture belongs to a previous model session. Its recorded model
            and settings are preserved.
          </div>
        )}
      {loading ? (
        <Loading label="Connecting to model runtime" />
      ) : !runtime?.loaded ? (
        <EmptyState
          icon={<Activity size={28} />}
          title="Give a prediction some context."
          href="/"
          action="Load a model"
        >
          Choose a checkpoint, then run a prompt to capture residual streams,
          layer predictions, and attention patterns.
        </EmptyState>
      ) : (
        <>
          <Panel
            title="Prompt your model"
            subtitle={`Deterministic generation with bounded activation capture. Up to ${promptLimit} prompt tokens with this generation length.`}
          >
            <form className="prompt-box" onSubmit={run}>
              <label htmlFor="prompt" className="sr-only">
                Prompt
              </label>
              <textarea
                id="prompt"
                aria-label="Prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask a question about how a model thinks…"
                maxLength={8000}
                required
              />
              <TokenPreview preview={tokenPreview} />
              <div className="prompt-suggestions">
                {[
                  "The capital of France is",
                  "When Mary and John went to the store, John gave a drink to",
                  "The opposite of hot is",
                ].map((p) => (
                  <button type="button" key={p} onClick={() => setPrompt(p)}>
                    {p.length > 42 ? `${p.slice(0, 42)}…` : p}
                  </button>
                ))}
              </div>
              <div className="prompt-toolbar">
                <div className="prompt-controls">
                  <label>
                    Generate
                    <select
                      aria-label="Generated tokens"
                      value={tokens}
                      onChange={(e) => setTokens(+e.target.value)}
                    >
                      {[1, 4, 8, 16, 32, 64]
                        .filter((n) => n < contextLimit)
                        .map((n) => (
                          <option key={n} value={n}>
                            {n} tokens
                          </option>
                        ))}
                    </select>
                  </label>
                  <label>
                    Intervention
                    <select
                      aria-label="Layer intervention"
                      value={ablate}
                      onChange={(e) => setAblate(e.target.value)}
                    >
                      <option value="">None · baseline</option>
                      {!!runtime.attention_head_targets?.length && (
                        <option value="head">Zero attention head</option>
                      )}
                      {Array.from(
                        { length: runtime.metadata?.num_layers || 0 },
                        (_, i) => (
                          <option key={i} value={i}>
                            Bypass block {i}
                          </option>
                        ),
                      )}
                    </select>
                  </label>
                  {ablate === "head" && (
                    <>
                      <label>
                        Layer
                        <select
                          aria-label="Head ablation layer"
                          value={headLayer}
                          onChange={(event) => {
                            setHeadLayer(+event.target.value);
                            setHead(0);
                          }}
                        >
                          {runtime.attention_head_targets?.map((target) => (
                            <option key={target.layer} value={target.layer}>
                              Layer {target.layer}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Head
                        <select
                          aria-label="Attention head to zero"
                          value={head}
                          onChange={(event) => setHead(+event.target.value)}
                        >
                          {Array.from(
                            { length: headTarget?.heads ?? 0 },
                            (_, i) => (
                              <option key={i} value={i}>
                                Head {i}
                              </option>
                            ),
                          )}
                        </select>
                      </label>
                    </>
                  )}
                  <label>
                    <input
                      type="checkbox"
                      checked={captureAttention}
                      onChange={(e) => setCaptureAttention(e.target.checked)}
                    />{" "}
                    Capture attention
                  </label>
                </div>
                <button
                  type="submit"
                  className="button primary"
                  disabled={
                    busy ||
                    !prompt.trim() ||
                    !promptFits ||
                    promptLimit < 1 ||
                    (ablate === "head" && !headTarget)
                  }
                >
                  <Play size={14} />
                  {busy ? "Tracing activations" : "Run & trace"}
                  <ArrowRight size={14} />
                </button>
              </div>
            </form>
          </Panel>
          {ablate !== "" && ablate !== "head" && (
            <div className="notice info section-gap">
              Block bypass replaces the selected transformer block output with
              its input residual stream. Run a baseline in the same model
              session with the same prompt to compare residual magnitudes.
            </div>
          )}
          {busy && (
            <Loading label="Running the forward pass, capturing states and generating a continuation" />
          )}
          {trace && !busy && (
            <>
              <TraceResults
                key={`${trace.duration_ms}-${interventionLabel(trace)}-${trace.prompt}`}
                trace={trace}
                comparison={compatibleBaseline}
              />
              <div className="save-form">
                <input
                  className="form-input"
                  aria-label="Experiment name"
                  placeholder="Give this experiment a name"
                  value={name}
                  onChange={(e) => {
                    setName(e.target.value);
                    setSaved(false);
                  }}
                  maxLength={100}
                />
                <button
                  className="button secondary"
                  onClick={save}
                  disabled={saving || saved || !name.trim()}
                >
                  {saved ? <Check size={14} /> : <Bookmark size={14} />}
                  {saved
                    ? "Saved to experiments"
                    : saving
                      ? "Saving"
                      : "Save experiment"}
                </button>
              </div>
            </>
          )}
          {!trace && !busy && (
            <div className="how-it-works">
              <div>
                <span>
                  <Activity size={15} />
                </span>
                <div>
                  <h3>Residual stream</h3>
                  <p>Track activation magnitudes across layers.</p>
                </div>
              </div>
              <div>
                <span>
                  <Play size={15} />
                </span>
                <div>
                  <h3>Logit lens</h3>
                  <p>See the prediction at each stage of processing.</p>
                </div>
              </div>
              <div>
                <span>
                  <RotateCcw size={15} />
                </span>
                <div>
                  <h3>Causal interventions</h3>
                  <p>Bypass a block or zero a head and measure what changes.</p>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
