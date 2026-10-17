"use client";

import "./prompt-set.css";
import { useState } from "react";
import { Download } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Panel } from "@/components/ui";
import { SeriesChart } from "@/components/charts";
import { downloadJSON, number } from "@/lib/api";
import type { MeasuredPair, PromptSetResult } from "@/lib/prompt-sets";

function PairEvidence({ pair }: { pair: MeasuredPair }) {
  const { t } = useI18n();
  return (
    <details className="set-pair-evidence">
      <summary>{t("Inspect captured pair")}</summary>
      <div className="set-prompt-grid">
        <div>
          <strong>{t("Clean prompt")}</strong>
          <pre dir="auto">{pair.result.clean_prompt}</pre>
        </div>
        <div>
          <strong>{t("Corrupted prompt")}</strong>
          <pre dir="auto">{pair.result.corrupted_prompt}</pre>
        </div>
      </div>
      {pair.metric_recovery_warning && (
        <p className="scientific-caveat">{t(pair.metric_recovery_warning)}</p>
      )}
      <div className="set-facts">
        <span>
          {t("Target token")}{" "}
          <bdi dir="auto">{JSON.stringify(pair.result.target_token)}</bdi>{" "}
          <bdi>{pair.result.target_token_id}</bdi>
        </span>
        {pair.result.alternative_token_id != null && (
          <span>
            {t("Alternative token")}{" "}
            <bdi dir="auto">
              {JSON.stringify(pair.result.alternative_token)}
            </bdi>{" "}
            <bdi>{pair.result.alternative_token_id}</bdi>
          </span>
        )}
        <span>
          {t("Patched positions")} <bdi>{pair.positions.join(", ")}</bdi>
        </span>
        <span>
          {t("Baseline score gap")} <bdi>{number(pair.baseline_gap, 6)}</bdi>
        </span>
      </div>
      <div className="set-table-scroll">
        <table className="set-table">
          <caption className="sr-only">
            {t("Captured layer effects for this pair")}
          </caption>
          <thead>
            <tr>
              <th>{t("Layer")}</th>
              <th>{t("Score change")}</th>
              <th>{t("Recovery")}</th>
            </tr>
          </thead>
          <tbody>
            {pair.result.layers.map((layer) => (
              <tr key={layer.layer}>
                <td>{layer.layer}</td>
                <td>{number(layer.metric_delta, 6)}</td>
                <td>
                  {layer.metric_recovery == null
                    ? t("Unavailable")
                    : number(layer.metric_recovery, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!pair.result.complete && (
        <p className="scientific-caveat">
          {t("This pair is incomplete and is excluded from every aggregate.")}
        </p>
      )}
    </details>
  );
}

export function PromptSetResults({ result }: { result: PromptSetResult }) {
  const { t } = useI18n();
  const [layerIndex, setLayerIndex] = useState(
    result.layers[0]?.layer ?? result.selected_layers[0],
  );
  const [measure, setMeasure] = useState("effect");
  const selected =
    result.layers.find((layer) => layer.layer === layerIndex) ??
    result.layers[0];
  const recovery = measure === "recovery";
  const cohort = recovery ? result.recovery_pairs : result.completed_pairs;
  const data = result.layers.map((layer) => ({
    index: layer.layer,
    mean: recovery ? layer.recovery : layer.mean_delta,
    low:
      (recovery ? layer.recovery_interval : layer.mean_interval)?.[0] ?? null,
    high:
      (recovery ? layer.recovery_interval : layer.mean_interval)?.[1] ?? null,
  }));
  const rows = result.pairs
    .filter((pair) => pair.result.complete)
    .map((pair) => ({
      pair,
      effect:
        pair.result.layers.find((layer) => layer.layer === selected?.layer)
          ?.metric_delta ?? null,
    }))
    .sort((a, b) => (a.effect ?? 0) - (b.effect ?? 0));
  return (
    <Panel
      title={t("Prompt-set evidence")}
      help={t(
        "Compare effects across the same completed pairs. Intervals are pointwise bootstrap estimates, not evidence of a general circuit.",
      )}
      action={
        <button
          className="button secondary"
          onClick={() => downloadJSON(result, "prompt-set-evidence.json")}
        >
          <Download size={14} />
          {t("Export prompt set")}
        </button>
      }
    >
      <div className="set-results">
        <div className="set-facts" dir="auto">
          <bdi>{result.model_id}</bdi>
          <bdi>{result.revision.slice(0, 8)}</bdi>
          <bdi>{result.device}</bdi>
          <span>
            {t("{done} of {total} pairs completed", {
              done: result.completed_pairs,
              total: result.total_pairs,
            })}
          </span>
        </div>
        <ErrorNotice message={result.error} />
        {!result.complete && (
          <div className="notice warning" role="status">
            {t(
              "This is a partial prompt set. Only fully completed pairs enter the aggregates.",
            )}
          </div>
        )}
        {result.metric === "logit_difference" && result.completed_pairs > 0 && (
          <p className="control-note">
            {t(
              "The clean prompt preferred the target in {count} of {total} completed pairs.",
              {
                count: result.clean_target_preferred_pairs ?? 0,
                total: result.completed_pairs,
              },
            )}
          </p>
        )}
        {result.layers.length > 0 ? (
          <>
            <div className="set-result-controls">
              <label>
                {t("Result measure")}
                <select
                  aria-label={t("Result measure")}
                  value={measure}
                  onChange={(event) => setMeasure(event.target.value)}
                >
                  <option value="effect">{t("Mean score change")}</option>
                  <option value="recovery">{t("Normalized recovery")}</option>
                </select>
              </label>
              <span className="control-note">
                {result.metric === "logit_difference"
                  ? t("Score is target logit minus alternative logit.")
                  : t("Score is target log probability.")}
              </span>
            </div>
            {recovery && (
              <p className="control-note">
                {t(
                  "Recovery uses {count} positive-gap pairs. Other completed pairs remain in the raw effect view.",
                  { count: result.recovery_pairs },
                )}
              </p>
            )}
            {!recovery && (
              <p className="control-note">
                {t(
                  "Raw effects include all {count} completed pairs, including {excluded} excluded from recovery.",
                  {
                    count: result.completed_pairs,
                    excluded: result.completed_pairs - result.recovery_pairs,
                  },
                )}
              </p>
            )}
            <SeriesChart
              data={data}
              interpolation="linear"
              showPoints
              series={[
                {
                  key: "mean",
                  label: recovery ? "Normalized recovery" : "Mean score change",
                  color: "var(--accent)",
                },
                ...((recovery
                  ? result.recovery_pairs
                  : result.completed_pairs) >= 2
                  ? [
                      {
                        key: "low",
                        label: "95% lower bound",
                        color: "var(--blue)",
                      },
                      {
                        key: "high",
                        label: "95% upper bound",
                        color: "var(--blue)",
                      },
                    ]
                  : []),
              ]}
            />
            <p className="scientific-caveat">
              {t(
                "Intervals resample these pairs only. Correlated templates can make them overconfident. They do not correct for testing multiple layers or establish generalization.",
              )}
            </p>
            {cohort < 10 && (
              <p className="control-note">
                {cohort < 2
                  ? t(
                      recovery
                        ? "At least two positive-gap pairs are needed for a recovery interval."
                        : "At least two completed pairs are needed for an interval.",
                    )
                  : t("Small prompt sets give unstable uncertainty estimates.")}
              </p>
            )}
            <div className="set-result-controls">
              <label>
                {t("Inspect layer")}
                <select
                  aria-label={t("Inspect layer")}
                  value={selected.layer}
                  onChange={(event) =>
                    setLayerIndex(Number(event.target.value))
                  }
                >
                  {result.layers.map((layer) => (
                    <option key={layer.layer} value={layer.layer}>
                      {t("Layer {v0}", { v0: layer.layer })}
                    </option>
                  ))}
                </select>
              </label>
              <span>
                {t("{positive}% help the target and {negative}% hurt it", {
                  positive: number(selected.positive_fraction * 100, 0),
                  negative: number(selected.negative_fraction * 100, 0),
                })}
              </span>
              <span>
                {t("Median change {value}", {
                  value: number(selected.median_delta, 4),
                })}
              </span>
            </div>
            <div className="set-table-scroll">
              <table className="set-table">
                <caption>{t("Per-pair effects at the selected layer")}</caption>
                <thead>
                  <tr>
                    <th>{t("Prompt pair")}</th>
                    <th>{t("Score change")}</th>
                    <th>{t("Direction")}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ pair, effect }) => (
                    <tr key={pair.index}>
                      <td>
                        <bdi>
                          {pair.label ||
                            t("Pair {count}", { count: pair.index + 1 })}
                        </bdi>
                        <PairEvidence pair={pair} />
                      </td>
                      <td>
                        <bdi>
                          {effect == null
                            ? t("Unavailable")
                            : number(effect, 5)}
                        </bdi>
                      </td>
                      <td>
                        {effect != null &&
                        Math.abs(effect) > result.statistics.sign_epsilon ? (
                          <span
                            className={
                              effect > 0 ? "set-positive" : "set-negative"
                            }
                          >
                            {effect > 0 ? t("Helps target") : t("Hurts target")}
                            {Math.abs(selected.mean_delta) >
                              result.statistics.sign_epsilon &&
                              effect * selected.mean_delta < 0 && (
                                <small>{t("Disagrees with mean")}</small>
                              )}
                          </span>
                        ) : (
                          t("Near zero")
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p className="muted">
            {t("No complete pairs are available for an aggregate yet.")}
          </p>
        )}
        {result.pairs.some((pair) => !pair.result.complete) && (
          <details>
            <summary>{t("Retained partial pairs")}</summary>
            {result.pairs
              .filter((pair) => !pair.result.complete)
              .map((pair) => (
                <PairEvidence key={pair.index} pair={pair} />
              ))}
          </details>
        )}
        <details>
          <summary>{t("Method and reproducibility")}</summary>
          <div className="set-facts">
            <span>
              {t("Bootstrap draws")}{" "}
              <bdi>{result.statistics.bootstrap_samples}</bdi>
            </span>
            <span>
              {t("Seed")} <bdi>{result.statistics.seed}</bdi>
            </span>
            <span>
              {t("Near-zero threshold")}{" "}
              <bdi>{result.statistics.sign_epsilon}</bdi>
            </span>
            <span>
              {t("Recovery gap threshold")}{" "}
              <bdi>{result.statistics.minimum_recovery_gap}</bdi>
            </span>
          </div>
          <p className="scientific-caveat">
            {t(
              "Recovery is summed score change divided by summed positive clean-to-corrupted gaps. Partial pairs are excluded from every layer. Block-output patching shows where information is present, not where it was first computed.",
            )}
          </p>
          <code>{result.input_fingerprint}</code>
        </details>
      </div>
    </Panel>
  );
}
