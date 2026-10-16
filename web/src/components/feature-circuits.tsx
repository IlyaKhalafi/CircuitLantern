"use client";
import { useEffect, useState } from "react";
import { Download, Play, Square } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Panel } from "@/components/ui";
import { FeatureCircuitGraph } from "@/components/feature-circuit-graph";
import { api, downloadJSON, number } from "@/lib/api";
import { useBackgroundJob } from "@/lib/jobs";
import type {
  DictionarySummary,
  FeatureCircuit,
  FeatureRunSummary,
} from "@/lib/features";

export function FeatureCircuits({
  dictionaries,
  executionId,
  enabled,
}: {
  dictionaries: DictionarySummary[];
  executionId?: string;
  enabled: boolean;
}) {
  const { t } = useI18n();
  const [selected, setSelected] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("The capital of France is");
  const [count, setCount] = useState(4);
  const [saved, setSaved] = useState<FeatureCircuit | null>(null);
  const [runs, setRuns] = useState<FeatureRunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const task = useBackgroundJob<FeatureCircuit>("lantern-feature-circuit-job");
  const result = saved || task.job?.result;
  useEffect(() => {
    setSelected((current) =>
      current.filter((id) =>
        dictionaries.some((dictionary) => dictionary.id === id),
      ),
    );
  }, [dictionaries]);

  useEffect(() => {
    api<FeatureRunSummary[]>("/features/runs")
      .then(setRuns)
      .catch((reason) => setError(reason.message));
  }, [task.job?.id, task.job?.status]);
  async function open(identifier: string) {
    if (!identifier) {
      setSaved(null);
      return;
    }
    setError(null);
    try {
      setSaved(await api<FeatureCircuit>("/features/runs/" + identifier));
    } catch (reason) {
      setError((reason as Error).message);
    }
  }
  return (
    <>
      <Panel
        title={t("Discover feature effects")}
        subtitle={t(
          "Ablate selected sparse features and measure downstream responses in the original model.",
        )}
      >
        <div className="feature-form">
          <ErrorNotice message={error || task.error} />
          <label>
            {t("Circuit prompt")}
            <textarea
              aria-label={t("Circuit prompt")}
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={task.busy}
              maxLength={8000}
              dir="auto"
            />
          </label>
          <fieldset className="feature-dictionary-choices">
            <legend>{t("Choose one dictionary per layer")}</legend>
            {dictionaries.map((dictionary) => (
              <label className="checkbox-row" key={dictionary.id}>
                <input
                  type="checkbox"
                  checked={selected.includes(dictionary.id)}
                  onChange={(e) =>
                    setSelected(
                      e.target.checked
                        ? [
                            ...selected.filter(
                              (id) =>
                                dictionaries.find((item) => item.id === id)
                                  ?.layer !== dictionary.layer,
                            ),
                            dictionary.id,
                          ]
                        : selected.filter((id) => id !== dictionary.id),
                    )
                  }
                  disabled={
                    task.busy ||
                    (!selected.includes(dictionary.id) &&
                      selected.length >= 8 &&
                      !selected.some(
                        (id) =>
                          dictionaries.find((item) => item.id === id)?.layer ===
                          dictionary.layer,
                      ))
                  }
                />
                <span dir="auto">
                  {dictionary.name}
                  <small className="feature-choice-model">
                    {dictionary.model_id}
                  </small>
                </span>
                <small>{t("Layer {v0}", { v0: dictionary.layer })}</small>
              </label>
            ))}
            {!dictionaries.length && (
              <p>
                {t("Train a dictionary before measuring feature circuits.")}
              </p>
            )}
          </fieldset>
          <div className="feature-toolbar">
            <label>
              {t("Features per layer")}
              <select
                aria-label={t("Features per layer")}
                value={count}
                onChange={(e) => setCount(Number(e.target.value))}
                disabled={task.busy}
              >
                {[1, 2, 3, 4].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="button primary"
              onClick={() => {
                setSaved(null);
                task.start("/features/circuits", {
                  dictionary_ids: selected,
                  prompt,
                  features_per_layer: count,
                  expected_execution_id: executionId,
                });
              }}
              disabled={
                task.busy || !enabled || !selected.length || !prompt.trim()
              }
            >
              <Play size={14} />
              {t("Measure feature circuit")}
            </button>
          </div>
          <p className="inline-note">
            {t(
              "Edges show total downstream effects and may include indirect paths. They are not direct attribution edges from a transcoder model.",
            )}
          </p>
          {task.job && (
            <div className="feature-job" role="status">
              <span>{t(task.job.message)}</span>
              <progress
                max={100}
                value={task.job.progress}
                aria-label={t("Feature circuit progress")}
              />
              {task.canCancel && (
                <button
                  className="button secondary"
                  onClick={task.cancel}
                  disabled={task.cancelling}
                >
                  <Square size={13} />
                  {t("Cancel measurement")}
                </button>
              )}
            </div>
          )}
        </div>
      </Panel>
      <Panel
        title={t("Saved feature circuits")}
        subtitle={t(
          "Completed measurements remain available after the model is unloaded.",
        )}
      >
        <div className="feature-form">
          <label>
            {t("Open saved feature circuit")}
            <select
              aria-label={t("Open saved feature circuit")}
              value={saved?.id || ""}
              onChange={(e) => open(e.target.value)}
            >
              <option value="">{t("Current result")}</option>
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.created_at.slice(0, 16).replace("T", " ")} ·{" "}
                  {run.prompt.slice(0, 60)}
                </option>
              ))}
            </select>
          </label>
        </div>
        {result && (
          <>
            <div className="feature-form">
              <div className="feature-toolbar">
                <p dir="auto">{result.prompt}</p>
                <button
                  className="button secondary"
                  onClick={() =>
                    downloadJSON(
                      result,
                      "feature-circuit-" + result.id + ".json",
                    )
                  }
                >
                  <Download size={14} />
                  {t("Export feature circuit")}
                </button>
              </div>
              {!result.complete && (
                <p className="notice info">
                  {t(
                    "This measurement is incomplete. Only completed ablations are shown.",
                  )}
                </p>
              )}
              <p>
                {t("Target prediction")}{" "}
                <strong className="mono" dir="auto">
                  {result.target_token}
                </strong>
              </p>
            </div>
            <FeatureCircuitGraph result={result} />
            <div className="feature-form">
              <h3>{t("Measured output effects")}</h3>
              <div className="feature-effects">
                {result.measurements.map((row) => {
                  const node = result.nodes.find(
                    (node) => node.id === row.source,
                  );
                  return (
                    <div key={row.source}>
                      <span>
                        {node && t("Layer {v0}", { v0: node.layer })} ·{" "}
                        {node?.label ||
                          t("Feature {v0}", { v0: node?.feature ?? 0 })}
                      </span>
                      <strong className="mono">
                        {number(row.target_log_probability_change, 5)}
                      </strong>
                      <span>
                        {t("KL divergence")}{" "}
                        {number(row.kl_baseline_to_patched, 5)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </Panel>
    </>
  );
}
