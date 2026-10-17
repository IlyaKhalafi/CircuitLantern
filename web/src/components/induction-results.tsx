"use client";

import { useState } from "react";
import { Heatmap, SeriesChart } from "@/components/charts";
import { ErrorNotice, Metric, Panel } from "@/components/ui";
import { useI18n } from "@/i18n/provider";
import { number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import type { InductionResult } from "@/lib/induction";
import "./induction-results.css";

const PAGE_SIZE = 12;

export function InductionResults({ result }: { result: InductionResult }) {
  const { t } = useI18n();
  const [selection, setSelection] = useState<string | null>(null);
  const [layer, setLayer] = useState("all");
  const [page, setPage] = useState(0);
  const key = (row: { layer: number; head: number }) =>
    row.layer + "/" + row.head;
  const selected =
    result.heads.find((row) => key(row) === selection) || result.heads[0];
  const visible = result.heads.filter(
    (row) => layer === "all" || row.layer === +layer,
  );
  const pages = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  const currentPage = Math.min(page, pages - 1);
  const rows = visible.slice(
    currentPage * PAGE_SIZE,
    (currentPage + 1) * PAGE_SIZE,
  );
  const causal =
    selected && result.causal_results.find((row) => key(row) === key(selected));
  const example =
    result.representative_probe === null
      ? null
      : result.dataset.probes[result.representative_probe];
  const query = result.representative_query_position;
  const matchingKeys =
    example?.token_ids
      .map((token, index, tokens) =>
        index > 0 && index <= query && tokens[index - 1] === tokens[query]
          ? index
          : null,
      )
      .filter((index): index is number => index !== null) || [];
  const repeatLoss = Array.from(
    { length: result.dataset.config.repeats },
    (_, index) => ({
      index: index + 1,
      loss: result.probe_metrics.length
        ? result.probe_metrics.reduce(
            (sum, row) => sum + row.repeat_nll[index],
            0,
          ) / result.probe_metrics.length
        : null,
    }),
  );
  return (
    <div className="induction-results">
      <div className="trace-meta">
        <span className={`pill ${result.complete ? "green" : "orange"}`}>
          {result.complete ? t("Probe complete") : t("Partial probe")}
        </span>
        <span className="trace-source">{result.model_id}</span>
        <span>{t("revision {v0}", { v0: result.revision.slice(0, 8) })}</span>
        <span>
          {t(
            executionLabel(result.execution_precision, result.execution_config),
          )}
        </span>
        <span>{result.device.toUpperCase()}</span>
        {result.cancelled && <span>{t("Cancelled between forwards")}</span>}
      </div>
      <ErrorNotice message={result.error} />
      <div className="metrics-grid">
        <Metric
          label={t("Measured probes")}
          value={result.measured_examples + " / " + result.requested_examples}
        />
        <Metric
          label={t("First block loss")}
          value={number(result.mean_first_repeat_nll, 4)}
          note={t("Mean negative log probability in nats")}
        />
        <Metric
          label={t("Repeated token loss")}
          value={number(result.mean_repeated_nll, 4)}
          note={t("Mean negative log probability in nats")}
        />
        <Metric
          label={t("Causal candidates")}
          value={
            result.completed_causal_heads +
            " / " +
            result.requested_causal_heads
          }
        />
      </div>
      {result.heads.length > 0 ? (
        <>
          <div className="induction-evidence">
            <Panel
              title={t("Ranked attention patterns")}
              subtitle={t(
                "Prefix matching mass averaged over repeated queries and completed probes.",
              )}
            >
              <div className="induction-ranking">
                <label>
                  {t("Layer")}
                  <select
                    aria-label={t("Probe results layer")}
                    value={layer}
                    onChange={(event) => {
                      const chosenLayer = event.target.value;
                      setLayer(chosenLayer);
                      setPage(0);
                      const first = result.heads.find(
                        (row) =>
                          chosenLayer === "all" || row.layer === +chosenLayer,
                      );
                      setSelection(first ? key(first) : null);
                    }}
                  >
                    <option value="all">{t("All tested layers")}</option>
                    {result.layer_targets.map((item) => (
                      <option key={item.layer} value={item.layer}>
                        {t("Layer {v0}", { v0: item.layer })}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="induction-rank-list">
                  {rows.map((row) => (
                    <button
                      type="button"
                      key={key(row)}
                      aria-pressed={selected && key(row) === key(selected)}
                      onClick={() => setSelection(key(row))}
                    >
                      <span>
                        {t("Layer {v0} · head {v1}", {
                          v0: row.layer,
                          v1: row.head,
                        })}
                      </span>
                      <strong>
                        <bdi>{number(row.prefix_attention * 100, 2)}%</bdi>
                      </strong>
                      <span className="induction-rank-bar" aria-hidden="true">
                        <i
                          style={{ width: row.prefix_attention * 100 + "%" }}
                        />
                      </span>
                    </button>
                  ))}
                </div>
                <div className="induction-pagination">
                  <button
                    type="button"
                    className="button secondary"
                    disabled={currentPage === 0}
                    onClick={() => setPage(currentPage - 1)}
                  >
                    {t("Previous")}
                  </button>
                  <span>
                    {t("Page {v0} of {v1}", { v0: currentPage + 1, v1: pages })}
                  </span>
                  <button
                    type="button"
                    className="button secondary"
                    disabled={currentPage + 1 >= pages}
                    onClick={() => setPage(currentPage + 1)}
                  >
                    {t("Next")}
                  </button>
                </div>
                <p className="inline-note">
                  {t("Uniform full causal attention reference {v0}", {
                    v0:
                      number(
                        (result.uniform_full_causal_baseline || 0) * 100,
                        2,
                      ) + "%",
                  })}
                </p>
              </div>
            </Panel>
            {selected && (
              <Panel
                title={t("Layer {v0} · head {v1}", {
                  v0: selected.layer,
                  v1: selected.head,
                })}
                subtitle={t(
                  "Pattern evidence and a causal test on the same probes.",
                )}
                className="induction-inspector"
              >
                <div className="induction-inspector-body">
                  <dl className="induction-scores">
                    {(
                      [
                        ["Prefix matching", selected.prefix_attention],
                        ["Duplicate token", selected.duplicate_attention],
                        ["Previous token", selected.previous_attention],
                      ] as const
                    ).map(([label, value]) => (
                      <div key={label}>
                        <dt>{t(label)}</dt>
                        <dd>
                          <bdi>{number(value * 100, 3)}%</bdi>
                        </dd>
                      </div>
                    ))}
                  </dl>
                  <p className="inline-note">
                    {t("Prefix score standard deviation across probes {v0}", {
                      v0: number(selected.prefix_std * 100, 3) + "%",
                    })}
                  </p>
                  <div className="induction-causal">
                    <h3>{t("Causal effect on repeated targets")}</h3>
                    {causal ? (
                      <>
                        <dl className="induction-scores">
                          <div>
                            <dt>{t("Delta log probability")}</dt>
                            <dd>
                              <bdi>
                                {number(causal.mean_delta_log_probability, 6)}
                              </bdi>
                            </dd>
                          </div>
                          <div>
                            <dt>{t("KL divergence")}</dt>
                            <dd>
                              <bdi>{number(causal.mean_kl_divergence, 6)}</bdi>
                            </dd>
                          </div>
                        </dl>
                        <p className="inline-note">
                          {t("{v0} of {v1} paired probes completed", {
                            v0: causal.completed_examples,
                            v1: causal.requested_examples,
                          })}
                        </p>
                        {!causal.complete && (
                          <span className="pill orange">
                            {t("Partial causal evidence")}
                          </span>
                        )}
                      </>
                    ) : (
                      <p className="inline-note">
                        {t(
                          "This head has no completed ablation pairs. Attention scores remain pattern evidence.",
                        )}
                      </p>
                    )}
                    <p className="inline-note">
                      {t(
                        "Delta is ablated minus baseline log probability in nats. Negative values mean removing the head reduced the periodic target probability. KL compares the baseline distribution to the ablated distribution over the full vocabulary.",
                      )}
                    </p>
                  </div>
                  {example && (
                    <>
                      <h3>{t("One captured attention row")}</h3>
                      <p className="inline-note">
                        {t(
                          "Probe {v0} · query position {v1}. This row is illustrative, not the aggregate score.",
                          { v0: result.representative_probe, v1: query },
                        )}
                      </p>
                      <Heatmap
                        data={[selected.representative_attention]}
                        sequential
                        maxAbs={1}
                        height={100}
                        label={t("Captured attention to each prompt position")}
                        xLabels={result.representative_tokens.map(
                          (token, index) =>
                            t("Position {v0}", { v0: index }) +
                            " " +
                            JSON.stringify(token),
                        )}
                        yLabels={[t("Query {v0}", { v0: query })]}
                      />
                      <p className="inline-note">{t("Matching prefix keys")}</p>
                      <div className="induction-matches" dir="ltr">
                        {matchingKeys.map((index) => (
                          <span key={index}>
                            <small>{index}</small>
                            <bdi dir="auto">
                              {JSON.stringify(
                                result.representative_tokens[index],
                              )}
                            </bdi>
                            <strong>
                              {number(
                                selected.representative_attention[index] * 100,
                                2,
                              )}
                              %
                            </strong>
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              </Panel>
            )}
          </div>
          <Panel
            title={t("Prediction loss across repeats")}
            subtitle={t(
              "Every scored position predicts the next token in the known periodic sequence, including the last prompt position.",
            )}
            className="section-gap"
          >
            <SeriesChart
              data={repeatLoss}
              series={[
                {
                  key: "loss",
                  label: "Mean negative log probability in nats",
                  color: "var(--blue)",
                },
              ]}
              interpolation="linear"
              showPoints
              height={200}
            />
          </Panel>
        </>
      ) : (
        <div className="notice info">
          {t(
            "No complete probe examples were captured. No head scores are available.",
          )}
        </div>
      )}
      <details className="capture-details">
        <summary>{t("Probe setup and evidence")}</summary>
        <p className="inline-note">
          {t(
            "Uniform distinct nonspecial token IDs are sampled per block. This adapts the paper sampling policy. Prefix, duplicate and previous token scores use the same queries after the first block. Pattern matching alone does not establish copying.",
          )}
        </p>
        <p className="inline-note">
          {t(
            "Prefix matching attention can identify candidate induction heads. It does not establish copying or a mechanism for general in-context learning. Head ablation measures signed effects on this synthetic probe distribution.",
          )}
        </p>
        <p className="inline-note">
          {t("Seed {v0} · vocabulary pool {v1} · completed forwards {v2}", {
            v0: result.dataset.config.seed,
            v1: result.dataset.vocabulary_pool_size,
            v2: result.completed_forwards,
          })}
        </p>
        <p className="mono induction-fingerprint">
          {result.dataset_fingerprint}
        </p>
        <p className="inline-note">
          {t(
            "Export includes exact input and target token IDs, measured query positions, method details and execution provenance.",
          )}
        </p>
        {result.references.map((reference) => (
          <p key={reference.url}>
            <a href={reference.url} target="_blank" rel="noreferrer">
              {reference.title} ({reference.year})
            </a>
          </p>
        ))}
      </details>
    </div>
  );
}
