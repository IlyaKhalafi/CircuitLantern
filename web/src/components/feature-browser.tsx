"use client";
import { useMemo, useState } from "react";
import { Download, Save } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Metric, Panel } from "@/components/ui";
import { SeriesChart } from "@/components/charts";
import { FeatureDirections } from "@/components/feature-directions";
import { api, downloadJSON, number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import type { DictionaryManifest } from "@/lib/features";

export function FeatureBrowser({
  dictionary,
  feature,
  onSelect,
  onUpdated,
  onDeleted,
}: {
  dictionary: DictionaryManifest;
  feature: number;
  onSelect: (value: number) => void;
  onUpdated: () => void;
  onDeleted: () => void;
}) {
  const { t } = useI18n();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [label, setLabel] = useState(
    dictionary.feature_labels[String(feature)]?.label || "",
  );
  const [notes, setNotes] = useState(
    dictionary.feature_labels[String(feature)]?.notes || "",
  );
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const features = useMemo(
    () =>
      dictionary.feature_statistics
        .filter((row) =>
          (
            String(row.index) +
            " " +
            (dictionary.feature_labels[String(row.index)]?.label || "")
          )
            .toLowerCase()
            .includes(search.toLowerCase()),
        )
        .sort((a, b) => b.frequency - a.frequency || a.index - b.index),
    [dictionary, search],
  );
  const selected = dictionary.feature_statistics[feature];
  function select(index: number) {
    onSelect(index);
    setLabel(dictionary.feature_labels[String(index)]?.label || "");
    setNotes(dictionary.feature_labels[String(index)]?.notes || "");
    setError(null);
  }
  async function save() {
    setBusy(true);
    setError(null);
    try {
      await api(
        "/features/dictionaries/" + dictionary.id + "/features/" + feature,
        { method: "PATCH", body: JSON.stringify({ label, notes }) },
      );
      onUpdated();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    setBusy(true);
    setError(null);
    try {
      await api("/features/dictionaries/" + dictionary.id, {
        method: "DELETE",
      });
      onDeleted();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const metric = dictionary.validation_metrics;
  return (
    <>
      <div className="feature-toolbar feature-dictionary-heading">
        <div>
          <h2 dir="auto">{dictionary.name}</h2>
          {dictionary.training_summary.cancelled && (
            <span className="pill subtle">{t("Partial training")}</span>
          )}

          <p className="mono">
            {dictionary.model_id} · {t("Layer {v0}", { v0: dictionary.layer })}{" "}
            · {dictionary.method === "topk" ? "TopK" : "ReLU"} ·{" "}
            {t(
              executionLabel(
                dictionary.execution_precision,
                dictionary.execution_config,
              ),
            )}{" "}
            · {dictionary.revision.slice(0, 12)}
          </p>
        </div>
        <div className="header-actions">
          <button
            className="button secondary"
            onClick={() =>
              downloadJSON(dictionary, "dictionary-" + dictionary.id + ".json")
            }
          >
            <Download size={14} />
            {t("Export dictionary record")}
          </button>
          <a
            className="button secondary"
            href={"/api/features/dictionaries/" + dictionary.id + "/weights"}
            download
          >
            <Download size={14} />
            {t("Download SAE weights")}
          </a>
        </div>
      </div>
      <div className="metrics-grid">
        <Metric
          label={t("Held out explained variance")}
          value={
            metric.explained_variance === null
              ? t("Unavailable")
              : number(metric.explained_variance * 100) + "%"
          }
        />
        <Metric label={t("Held out MSE")} value={number(metric.mse)} />
        <Metric
          label={t("Mean active features")}
          value={number(metric.mean_active_features, 1)}
        />
        <Metric
          label={t("Inactive features")}
          value={number(metric.inactive_features, 0)}
        />
      </div>
      <div className="feature-browser-layout">
        <Panel
          title={t("Feature dictionary")}
          subtitle={t(
            "Activation frequencies use the complete captured corpus.",
          )}
        >
          <div className="feature-form">
            <label>
              {t("Find a feature")}
              <input
                className="form-input"
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setPage(0);
                }}
                placeholder={t("Feature number or annotation")}
              />
            </label>
          </div>
          <div className="feature-list" aria-label={t("Dictionary features")}>
            {features.slice(page * 40, page * 40 + 40).map((row) => (
              <button
                type="button"
                key={row.index}
                onClick={() => select(row.index)}
                aria-pressed={feature === row.index}
                className={
                  "feature-row " + (feature === row.index ? "selected" : "")
                }
              >
                <span>
                  <strong>
                    {dictionary.feature_labels[String(row.index)]?.label ||
                      t("Feature {v0}", { v0: row.index })}
                  </strong>
                  <small>{t("Feature {v0}", { v0: row.index })}</small>
                </span>
                <span className="mono">{number(row.frequency * 100, 1)}%</span>
              </button>
            ))}
          </div>
          <div className="feature-toolbar feature-pagination">
            <button
              className="button secondary"
              onClick={() => setPage(page - 1)}
              disabled={page === 0}
            >
              {t("Previous")}
            </button>
            <span>
              {page + 1} / {Math.max(1, Math.ceil(features.length / 40))}
            </span>
            <button
              className="button secondary"
              onClick={() => setPage(page + 1)}
              disabled={(page + 1) * 40 >= features.length}
            >
              {t("Next")}
            </button>
          </div>
        </Panel>
        <Panel
          title={t("Feature {v0}", { v0: feature })}
          subtitle={t(
            "Use activating contexts to form a hypothesis, then test it.",
          )}
        >
          {selected && (
            <div className="feature-form">
              <div className="metrics-grid">
                <Metric
                  label={t("Activation frequency")}
                  value={number(selected.frequency * 100, 2) + "%"}
                />
                <Metric
                  label={t("Maximum activation")}
                  value={number(selected.max_activation)}
                />
              </div>
              <div className="feature-split-frequencies">
                <span>
                  {t("Training activation frequency")}{" "}
                  <strong>
                    {selected.train_frequency == null
                      ? t("Unavailable")
                      : number(selected.train_frequency * 100, 2) + "%"}
                  </strong>
                </span>
                <span>
                  {t("Held out activation frequency")}{" "}
                  <strong>
                    {selected.validation_frequency == null
                      ? t("Unavailable")
                      : number(selected.validation_frequency * 100, 2) + "%"}
                  </strong>
                </span>
              </div>
              <ErrorNotice message={error} />
              <label>
                {t("Feature annotation")}
                <input
                  className="form-input"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  maxLength={100}
                  dir="auto"
                />
              </label>
              <label>
                {t("Research notes")}
                <textarea
                  aria-label={t("Research notes")}
                  rows={2}
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  maxLength={10000}
                  dir="auto"
                />
              </label>
              <button
                className="button secondary"
                onClick={save}
                disabled={busy}
              >
                <Save size={14} />
                {t("Save annotation")}
              </button>
              <FeatureDirections
                dictionaryId={dictionary.id}
                feature={feature}
                onSelect={select}
              />
              <h3>{t("Strongest activating contexts")}</h3>
              {!selected.examples.length && (
                <p className="muted">
                  {t("This feature did not activate on the captured corpus.")}
                </p>
              )}
              {selected.examples.map((example, index) => {
                const sample = dictionary.corpus.samples[example.sample],
                  document = dictionary.corpus.documents[sample.document];
                return (
                  <article className="feature-example" key={index}>
                    <div className="feature-toolbar">
                      <span className="pill subtle">
                        {t(
                          sample.split === "validation"
                            ? "Held out"
                            : "Training",
                        )}
                      </span>
                      <span className="mono">{number(example.activation)}</span>
                    </div>
                    <p dir="auto">{document.text}</p>
                    <div className="feature-token-context" dir="ltr">
                      {document.tokens
                        .slice(
                          Math.max(0, sample.position - 7),
                          sample.position + 8,
                        )
                        .map((token, offset) => {
                          const position =
                            Math.max(0, sample.position - 7) + offset;
                          return position === sample.position ? (
                            <mark key={position}>{token || "∅"}</mark>
                          ) : (
                            <span key={position}>{token || "∅"}</span>
                          );
                        })}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </Panel>
      </div>
      <Panel
        title={t("Training and validation")}
        subtitle={t(
          "The training curve uses a fixed sample. Final metrics use every captured token in each split.",
        )}
      >
        <SeriesChart
          data={dictionary.history}
          xKey="step"
          series={[
            {
              key: "train_sample_mse",
              label: "Training sample MSE",
              color: "var(--blue)",
            },
            {
              key: "validation_mse",
              label: "Held out MSE",
              color: "var(--orange)",
            },
          ]}
        />
        <div className="feature-provenance">
          <span>
            {t("Unique training tokens")}{" "}
            <strong>
              {number(dictionary.training_summary.unique_train_tokens, 0)}
            </strong>
          </span>
          <span>
            {t("Optimizer token exposures")}{" "}
            <strong>
              {number(dictionary.training_summary.tokens_seen, 0)}
            </strong>
          </span>
          <span>
            {t("Completed steps")}{" "}
            <strong>
              {dictionary.training_summary.completed_steps} /{" "}
              {dictionary.training_summary.requested_steps}
            </strong>
          </span>
        </div>
      </Panel>
      <div className="feature-delete">
        {deleting ? (
          <>
            <p>
              {t(
                "Deleting this dictionary removes its weights, feature examples and annotations. Saved circuit measurements are retained.",
              )}
            </p>
            <button className="button danger" disabled={busy} onClick={remove}>
              {t("Permanently delete dictionary")}
            </button>
            <button
              className="button secondary"
              onClick={() => setDeleting(false)}
              disabled={busy}
            >
              {t("Keep dictionary")}
            </button>
          </>
        ) : (
          <button
            className="button secondary"
            onClick={() => setDeleting(true)}
          >
            {t("Delete dictionary")}
          </button>
        )}
      </div>
    </>
  );
}
