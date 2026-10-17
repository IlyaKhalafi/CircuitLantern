"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Check, Layers, Play } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Panel } from "@/components/ui";
import {
  CaptureCard,
  CrosscoderBudget,
  CrosscoderProgress,
  type CrosscoderTask,
  useSessionDraft,
} from "@/components/crosscoder-shared";
import { post } from "@/lib/api";
import {
  validInteger,
  type CaptureJobResult,
  type CaptureSummary,
  type CrosscoderCapabilities,
  type CrosscoderPlan,
} from "@/lib/crosscoders";

export function CrosscoderCaptures({
  capabilities,
  captures,
  reference,
  compared,
  onReference,
  onCompared,
  task,
  blocked,
}: {
  capabilities: CrosscoderCapabilities | null;
  captures: CaptureSummary[];
  reference: CaptureSummary | null;
  compared: CaptureSummary | null;
  onReference: (id: string) => void;
  onCompared: (id: string) => void;
  task: CrosscoderTask<CaptureJobResult>;
  blocked: boolean;
}) {
  const { t } = useI18n();
  const [draft, setDraft, ready] = useSessionDraft(
    "lantern-crosscoder-capture-draft",
    {
      referenceName: "",
      comparedName: "",
      documents: "",
      layer: "0",
      comparedLayer: "0",
      samples: "4096",
      tokens: "128",
      splitSeed: "0",
      excludeSpecial: true,
      excludeFirst: true,
    },
  );
  const [plan, setPlan] = useState<{
    role: string;
    value: CrosscoderPlan;
  } | null>(null);
  const [planning, setPlanning] = useState(false);
  const planSequence = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const tokenLimit = Math.min(2048, capabilities?.max_capture_tokens || 128);
  useEffect(() => {
    if (!ready || !capabilities) return;
    setDraft((current) => ({
      ...current,
      documents: current.documents || capabilities.example_documents.join("\n"),
      layer: capabilities.layers.includes(Number(current.layer))
        ? current.layer
        : String(capabilities.layers[0] ?? 0),
      comparedLayer: capabilities.layers.includes(Number(current.comparedLayer))
        ? current.comparedLayer
        : String(capabilities.layers[0] ?? 0),
      tokens:
        Number(current.tokens) > tokenLimit
          ? String(tokenLimit)
          : current.tokens,
    }));
  }, [ready, capabilities, tokenLimit, setDraft]);
  useEffect(() => {
    planSequence.current++;
    setPlan(null);
  }, [draft, reference?.id, capabilities?.execution?.execution_id]);
  const documents = draft.documents
    .split(/\r?\n/)
    .filter((value) => value.trim());
  const validCorpus =
    documents.length >= 4 &&
    documents.length <= 256 &&
    documents.every((value) => value.length <= 8000) &&
    validInteger(draft.samples, 3, 32768) &&
    validInteger(draft.tokens, 1, tokenLimit) &&
    validInteger(draft.splitSeed, 0, 4294967295);
  const enabled =
    ready &&
    !!capabilities?.supported &&
    !!capabilities.execution?.execution_id;
  const busy = task.busy || planning || blocked;
  function edit(field: keyof typeof draft, value: string | boolean) {
    setDraft((current) => ({ ...current, [field]: value }));
  }
  function payload(role: "reference" | "compared") {
    const bound = {
      name:
        (role === "reference"
          ? draft.referenceName
          : draft.comparedName
        ).trim() ||
        t(role === "reference" ? "Reference capture" : "Compared capture"),
      layer: Number(role === "reference" ? draft.layer : draft.comparedLayer),
      expected_execution_id: capabilities?.execution?.execution_id,
    };
    return role === "reference"
      ? {
          ...bound,
          documents,
          max_samples: Number(draft.samples),
          max_tokens: Number(draft.tokens),
          split_seed: Number(draft.splitSeed),
          exclude_special_tokens: draft.excludeSpecial,
          exclude_first_position: draft.excludeFirst,
        }
      : bound;
  }
  function path(role: "reference" | "compared") {
    return role === "reference"
      ? "/crosscoders/captures"
      : "/crosscoders/captures/" + reference?.id + "/match";
  }
  async function check(role: "reference" | "compared") {
    setPlanning(true);
    setError(null);
    setPlan(null);
    try {
      const sequence = planSequence.current;
      const value = await post<CrosscoderPlan>(
        path(role) + "/plan",
        payload(role),
      );
      if (sequence === planSequence.current) setPlan({ role, value });
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setPlanning(false);
    }
  }
  function controls(role: "reference" | "compared") {
    const valid = enabled && (role === "reference" ? validCorpus : !!reference);
    return (
      <>
        <div className="feature-toolbar">
          <button
            className="button primary"
            disabled={!valid || busy}
            onClick={() => {
              setError(null);
              void task.start(path(role), payload(role));
            }}
          >
            <Play size={14} />
            {t(
              role === "reference"
                ? "Capture reference"
                : "Capture matched activations",
            )}
          </button>
          <button
            className="button secondary"
            disabled={!valid || busy}
            onClick={() => check(role)}
          >
            {t("Check resources")}
          </button>
        </div>
        {plan?.role === role && <CrosscoderBudget plan={plan.value} />}
      </>
    );
  }
  function layerPicker(role: "reference" | "compared") {
    const key = role === "reference" ? "layer" : "comparedLayer";
    return (
      <label>
        {t("Transformer layer")}
        <select
          aria-label={t("Transformer layer")}
          value={draft[key]}
          disabled={busy || !enabled}
          onChange={(event) => edit(key, event.target.value)}
        >
          {(capabilities?.layers || [0]).map((layer) => (
            <option key={layer} value={layer}>
              {t("Layer {v0}", { v0: layer })}
            </option>
          ))}
        </select>
      </label>
    );
  }
  const matching = captures.filter(
    (item) =>
      item.id !== reference?.id &&
      item.corpus_fingerprint === reference?.corpus_fingerprint,
  );
  return (
    <>
      <div className="crosscoder-session">
        <Layers size={20} />
        <div>
          <span>{t("Current checkpoint")}</span>
          <strong dir="auto">
            {capabilities?.execution?.model_id || t("No model loaded")}
          </strong>
        </div>
        <Link className="button secondary" href="/">
          {t("Open model library")}
        </Link>
      </div>
      {!enabled && (
        <p className="notice">
          {t(
            "Load a supported model to capture activations. Retained captures can be trained without a loaded model.",
          )}
        </p>
      )}
      <ErrorNotice message={error} />
      <CrosscoderProgress task={task} />
      <Panel
        title={t("1 · Reference activations")}
        subtitle={t(
          "Choose a retained capture or record residual activations from the loaded checkpoint.",
        )}
        action={
          reference ? <Check size={18} className="crosscoder-check" /> : null
        }
      >
        <div className="feature-form">
          {captures.length > 0 && (
            <label>
              {t("Reference capture")}
              <select
                aria-label={t("Reference capture")}
                disabled={busy}
                value={reference?.id || ""}
                onChange={(event) => onReference(event.target.value)}
              >
                <option value="">{t("Record a new capture")}</option>
                {captures.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} · {item.source.model_id} ·{" "}
                    {t("Layer {v0}", { v0: item.layer })}
                  </option>
                ))}
              </select>
            </label>
          )}
          {reference && <CaptureCard capture={reference} role="Reference" />}
          <details open={!reference} className="feature-advanced">
            <summary>{t("Record a new reference")}</summary>
            <div className="crosscoder-details feature-form">
              <div className="feature-form-grid">
                <label>
                  {t("Capture name")}
                  <input
                    maxLength={100}
                    disabled={busy}
                    value={draft.referenceName}
                    placeholder={t("Reference capture")}
                    onChange={(event) =>
                      edit("referenceName", event.target.value)
                    }
                  />
                </label>
                {layerPicker("reference")}
              </div>
              <label>
                {t("Training corpus")}
                <textarea
                  aria-label={t("Training corpus")}
                  rows={7}
                  value={draft.documents}
                  disabled={busy}
                  onChange={(event) => edit("documents", event.target.value)}
                />
              </label>
              <small className="muted">
                {t(
                  "Use 4 to 256 documents, one per line. Both checkpoints will use these exact model inputs and document splits.",
                )}
              </small>
              <details className="feature-advanced">
                <summary>{t("Advanced capture settings")}</summary>
                <div className="crosscoder-details feature-form">
                  <div className="feature-form-grid">
                    {(
                      [
                        ["samples", "Maximum token positions", 3, 32768],
                        ["tokens", "Tokens per document", 1, tokenLimit],
                        ["splitSeed", "Dataset split seed", 0, 4294967295],
                      ] as const
                    ).map(([field, label, min, max]) => (
                      <label key={field}>
                        {t(label)}
                        <input
                          type="number"
                          min={min}
                          max={max}
                          step={1}
                          disabled={busy}
                          value={draft[field]}
                          onChange={(event) => edit(field, event.target.value)}
                        />
                      </label>
                    ))}
                  </div>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={draft.excludeSpecial}
                      disabled={busy}
                      onChange={(event) =>
                        edit("excludeSpecial", event.target.checked)
                      }
                    />
                    {t("Exclude special tokens")}
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={draft.excludeFirst}
                      disabled={busy}
                      onChange={(event) =>
                        edit("excludeFirst", event.target.checked)
                      }
                    />
                    {t("Exclude first position")}
                  </label>
                </div>
              </details>
              {controls("reference")}
            </div>
          </details>
        </div>
      </Panel>
      {reference && (
        <Panel
          title={t("2 · Matched activations")}
          subtitle={t(
            "Load the checkpoint to compare, refresh, then capture the same token positions.",
          )}
          action={
            compared ? <Check size={18} className="crosscoder-check" /> : null
          }
        >
          <div className="feature-form">
            {matching.length > 0 && (
              <label>
                {t("Compared capture")}
                <select
                  aria-label={t("Compared capture")}
                  disabled={busy}
                  value={compared?.id || ""}
                  onChange={(event) => onCompared(event.target.value)}
                >
                  <option value="">{t("Record a new capture")}</option>
                  {matching.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name} · {item.source.model_id} ·{" "}
                      {t("Layer {v0}", { v0: item.layer })}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {compared && <CaptureCard capture={compared} role="Compared" />}
            <details open={!compared} className="feature-advanced">
              <summary>{t("Record a new matched capture")}</summary>
              <div className="crosscoder-details feature-form">
                <p className="muted">
                  {t(
                    "The reference corpus and capture rules are reused automatically. Different tokenization is rejected before running the model.",
                  )}
                </p>
                <div className="feature-form-grid">
                  <label>
                    {t("Capture name")}
                    <input
                      maxLength={100}
                      value={draft.comparedName}
                      placeholder={t("Compared capture")}
                      disabled={busy}
                      onChange={(event) =>
                        edit("comparedName", event.target.value)
                      }
                    />
                  </label>
                  {layerPicker("compared")}
                </div>
                {capabilities?.execution?.execution_id ===
                  reference.source.execution_id && (
                  <p className="notice">
                    {t(
                      "The reference checkpoint is still loaded. Load another version for a model comparison, or keep it for a repeatability check.",
                    )}
                  </p>
                )}
                {controls("compared")}
              </div>
            </details>
          </div>
        </Panel>
      )}
    </>
  );
}
