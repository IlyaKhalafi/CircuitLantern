"use client";

import {
  Activity,
  ArrowRight,
  Bookmark,
  Check,
  Download,
  Play,
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
import { api, downloadJSON, post } from "@/lib/api";
import type { RuntimeState, Trace } from "@/lib/types";

export default function PlaygroundPage() {
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [loading, setLoading] = useState(true);
  const [prompt, setPrompt] = useState("The capital of France is");
  const [tokens, setTokens] = useState(16);
  const [captureAttention, setCaptureAttention] = useState(true);
  const [ablate, setAblate] = useState("");
  const [trace, setTrace] = useState<Trace | null>(null);
  const [baseline, setBaseline] = useState<Trace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    api<RuntimeState>("/runtime")
      .then(setRuntime)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);
  async function run(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      const result = await post<Trace>("/trace", {
        prompt,
        max_new_tokens: tokens,
        capture_attention: captureAttention,
        ablate_layer: ablate === "" ? null : +ablate,
      });
      setTrace(result);
      setName(
        `${ablate ? `Block ${ablate} bypass` : "Baseline"} · ${prompt.slice(0, 45)}`,
      );
      if (ablate === "") setBaseline(result);
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
  const compatibleBaseline =
    baseline?.prompt === trace?.prompt &&
    baseline?.revision === trace?.revision &&
    baseline?.model_id === trace?.model_id &&
    baseline?.execution_precision === trace?.execution_precision &&
    baseline?.device === trace?.device &&
    trace?.ablate_layer !== null
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
          trace && (
            <button
              className="button secondary"
              onClick={() => downloadJSON(trace, "activation-trace.json")}
            >
              <Download size={14} /> Export trace
            </button>
          )
        }
      />
      <ErrorNotice message={error} />
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
            subtitle="Deterministic generation with bounded activation capture. Up to 128 prompt tokens."
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
                      {[8, 16, 32, 64].map((n) => (
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
                  disabled={busy || !prompt.trim()}
                >
                  <Play size={14} />
                  {busy ? "Tracing activations" : "Run & trace"}
                  <ArrowRight size={14} />
                </button>
              </div>
            </form>
          </Panel>
          {ablate !== "" && (
            <div className="notice info section-gap">
              Block bypass replaces the selected transformer block output with
              its input residual stream. Run a baseline with the same prompt
              first to compare residual magnitudes.
            </div>
          )}
          {busy && (
            <Loading label="Running the forward pass, capturing states and generating a continuation" />
          )}
          {trace && !busy && (
            <>
              <TraceResults
                key={`${trace.duration_ms}-${trace.ablate_layer}-${trace.prompt}`}
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
                  <p>Bypass a block and measure what changes.</p>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
