"use client";
import { useEffect, useRef, useState } from "react";
import { Download, Play } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Metric, Panel } from "@/components/ui";
import { Heatmap } from "@/components/charts";
import { downloadJSON, number, post } from "@/lib/api";
import type {
  DictionaryEvaluation,
  DictionaryManifest,
  FeatureInspection,
  FeaturePrediction,
} from "@/lib/features";

function Predictions({ rows }: { rows: FeaturePrediction[] }) {
  return (
    <div className="feature-predictions">
      {rows.slice(0, 5).map((row) => (
        <div key={row.token_id}>
          <span className="mono feature-captured-prompt" dir="auto">
            {row.token || "∅"}
          </span>
          <strong>{number(row.probability * 100, 2)}%</strong>
        </div>
      ))}
    </div>
  );
}
export function FeatureInspectionPanel({
  dictionary,
  feature,
  executionId,
  enabled,
}: {
  dictionary: DictionaryManifest;
  feature: number;
  executionId?: string;
  enabled: boolean;
}) {
  const { t } = useI18n();
  const [prompt, setPrompt] = useState("The capital of France is");
  const [position, setPosition] = useState("all");
  const [result, setResult] = useState<FeatureInspection | null>(null);
  const [evaluation, setEvaluation] = useState<DictionaryEvaluation | null>(
    null,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sequence = useRef(0);
  useEffect(() => {
    sequence.current++;
    setBusy(false);
    setError(null);
    return () => {
      sequence.current++;
    };
  }, [dictionary.id, executionId]);
  async function run(multiplier?: number) {
    const id = ++sequence.current;
    setBusy(true);
    setError(null);
    try {
      const value = await post<FeatureInspection>(
        "/features/dictionaries/" + dictionary.id + "/inspect",
        {
          prompt,
          expected_execution_id: executionId,
          feature: multiplier === undefined ? null : feature,
          multiplier: multiplier ?? 0,
          position: position === "all" ? null : Number(position),
        },
      );
      if (id === sequence.current) setResult(value);
    } catch (reason) {
      if (id === sequence.current) setError((reason as Error).message);
    } finally {
      if (id === sequence.current) setBusy(false);
    }
  }
  async function evaluate() {
    const id = ++sequence.current;
    setBusy(true);
    setError(null);
    try {
      const value = await post<DictionaryEvaluation>(
        "/features/dictionaries/" + dictionary.id + "/evaluate",
        { prompt, expected_execution_id: executionId },
      );
      if (id === sequence.current) setEvaluation(value);
    } catch (reason) {
      if (id === sequence.current) setError((reason as Error).message);
    } finally {
      if (id === sequence.current) setBusy(false);
    }
  }
  return (
    <Panel
      title={t("Test features on a prompt")}
      subtitle={t(
        "Inspect activation patterns and measure how a feature changes the next prediction.",
      )}
    >
      <div className="feature-form">
        <ErrorNotice message={error} />
        <label>
          {t("Feature prompt")}
          <textarea
            aria-label={t("Feature prompt")}
            rows={3}
            value={prompt}
            dir="auto"
            maxLength={8000}
            onChange={(e) => {
              setPrompt(e.target.value);
              setPosition("all");
            }}
            disabled={busy}
          />
        </label>
        <div className="feature-toolbar">
          <button
            className="button primary"
            onClick={() => run()}
            disabled={busy || !enabled || !prompt.trim()}
          >
            <Play size={14} />
            {t("Inspect features")}
          </button>
          <button
            className="button secondary"
            onClick={evaluate}
            disabled={busy || !enabled || !prompt.trim()}
          >
            {t("Evaluate reconstruction")}
          </button>
        </div>
        <div className="feature-intervention-controls">
          <span>
            {t("Selected feature")} <strong>{feature}</strong>
          </span>
          <label>
            {t("Intervention positions")}
            <select
              aria-label={t("Intervention positions")}
              value={position}
              onChange={(e) => setPosition(e.target.value)}
              disabled={busy}
            >
              <option value="all">{t("All eligible tokens")}</option>
              {result?.prompt === prompt &&
                result.eligible_positions.map((index) => (
                  <option key={index} value={index}>
                    {index} · {result.tokens[index] || "∅"}
                  </option>
                ))}
            </select>
          </label>
          <button
            className="button secondary"
            onClick={() => run(0)}
            disabled={busy || !enabled}
          >
            {t("Ablate feature")}
          </button>
          <button
            className="button secondary"
            onClick={() => run(2)}
            disabled={busy || !enabled}
          >
            {t("Amplify feature")} ×2
          </button>
        </div>
        <p className="inline-note">
          {t(
            "Interventions change the selected decoded contribution and preserve the original reconstruction error.",
          )}
        </p>
        {result && (
          <div className="feature-inspection-result">
            <div className="feature-toolbar">
              <h3>{t("Captured prompt")}</h3>
              <button
                className="button secondary"
                onClick={() =>
                  downloadJSON(
                    result,
                    "feature-inspection-" + result.dictionary_id + ".json",
                  )
                }
              >
                <Download size={14} />
                {t("Export feature inspection")}
              </button>
            </div>
            <p className="inline-note">
              {t(
                result.prompt_split === "train"
                  ? "Training text"
                  : result.prompt_split === "validation"
                    ? "Held out text"
                    : "No matching corpus document",
              )}{" "}
              · {t("Excluded token positions remain untouched.")}
            </p>

            <p className="mono feature-captured-prompt" dir="auto">
              {result.prompt}
            </p>
            {result.features.length > 0 && (
              <Heatmap
                data={result.features.map((row) => row.activations)}
                sequential
                label={t(
                  "Columns are prompt tokens. Rows are sparse features. Values are normalized SAE coefficients.",
                )}
                xLabels={result.tokens}
                yLabels={result.features.map((row) =>
                  t("Feature {v0}", { v0: row.index }),
                )}
                height={Math.min(
                  320,
                  Math.max(96, result.features.length * 14),
                )}
              />
            )}
            <div className="feature-heatmap-legend">
              <span aria-hidden="true" />
              {t("Gray cells mark excluded token positions.")}
            </div>
            <div className="two-column">
              <div>
                <h3>{t("Baseline predictions")}</h3>
                <Predictions rows={result.baseline_predictions} />
              </div>
              {result.intervention && (
                <div>
                  <h3>{t("Intervened predictions")}</h3>
                  <p className="inline-note feature-intervention-evidence">
                    {t("Feature {v0}", { v0: result.intervention.feature })}
                    {" · ×"}
                    {result.intervention.multiplier}
                    {" · "}
                    {result.intervention.position === null
                      ? t("All eligible tokens")
                      : t("Token position") +
                        " " +
                        result.intervention.position}
                  </p>
                  <p dir="auto">{result.intervention.target_token}</p>
                  <Predictions rows={result.intervention.predictions} />
                </div>
              )}
            </div>
            {result.intervention && (
              <div className="metrics-grid">
                <Metric
                  label={t("Target log probability change")}
                  value={number(
                    result.intervention.target_log_probability_change,
                    5,
                  )}
                />
                <Metric
                  label={t("KL divergence")}
                  value={number(result.intervention.kl_baseline_to_patched, 5)}
                />
              </div>
            )}
          </div>
        )}
        {evaluation && (
          <div className="feature-evaluation">
            <div className="feature-toolbar">
              <h3>{t("Reconstruction faithfulness")}</h3>
              <button
                className="button secondary"
                onClick={() =>
                  downloadJSON(
                    evaluation,
                    "feature-reconstruction-" +
                      evaluation.dictionary_id +
                      ".json",
                  )
                }
              >
                <Download size={14} />
                {t("Export reconstruction")}
              </button>
            </div>
            <p className="inline-note">
              {t(
                evaluation.prompt_split === "train"
                  ? "Training text"
                  : evaluation.prompt_split === "validation"
                    ? "Held out text"
                    : "No matching corpus document",
              )}
            </p>

            <p dir="auto">{evaluation.prompt}</p>
            <div className="metrics-grid">
              <Metric
                label={t("Original cross entropy")}
                value={number(evaluation.baseline_cross_entropy)}
              />
              <Metric
                label={t("Reconstructed cross entropy")}
                value={number(evaluation.reconstructed_cross_entropy)}
              />
              <Metric
                label={t("Mean ablated cross entropy")}
                value={number(evaluation.mean_ablated_cross_entropy)}
              />
              <Metric
                label={t("Loss recovered")}
                value={
                  evaluation.loss_recovered === null
                    ? t("Unavailable")
                    : number(evaluation.loss_recovered * 100, 1) + "%"
                }
              />
            </div>
            <p className="inline-note">
              {t(
                "Recovery can be negative or above 100 percent. A zero baseline gap or a one-token prompt has no defined recovery.",
              )}
            </p>
          </div>
        )}
      </div>
    </Panel>
  );
}
