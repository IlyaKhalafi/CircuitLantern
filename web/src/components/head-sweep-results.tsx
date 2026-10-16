"use client";
import { useI18n } from "@/i18n/provider";
import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { ArrowUpRight } from "lucide-react";
import { Heatmap } from "@/components/charts";
import { ErrorNotice, Metric, Panel } from "@/components/ui";
import { number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import type { HeadSweepResult, HeadSweepRow } from "@/lib/types";
import { prepareHeadInspection } from "@/lib/inspection";
export function HeadSweepResults({ result }: { result: HeadSweepResult }) {
  const { t, rich } = useI18n();
  const router = useRouter();
  const [opening, startOpening] = useTransition();
  const [inspectionError, setInspectionError] = useState<string | null>(null);
  function inspect(head: HeadSweepRow) {
    setInspectionError(null);
    try {
      const path = prepareHeadInspection(result, head);
      startOpening(() => router.push(path));
    } catch (reason) {
      setInspectionError((reason as Error).message);
    }
  }
  const [order, setOrder] = useState("magnitude");
  const [layer, setLayer] = useState("all");
  const byPosition = new Map(
    result.heads.map((head) => [
      `${head.layer}/${head.head}`,
      head.delta_log_probability,
    ]),
  );
  const columns = Math.max(
    1,
    ...result.layer_targets.map((item) => item.heads),
  );
  const rows = result.layer_targets.map((item) =>
    Array.from(
      { length: columns },
      (_, head) => byPosition.get(`${item.layer}/${head}`) ?? null,
    ),
  );
  const sorted = result.heads
    .filter((head) => layer === "all" || head.layer === +layer)
    .sort((a, b) =>
      order === "magnitude"
        ? Math.abs(b.delta_log_probability) - Math.abs(a.delta_log_probability)
        : order === "reduction"
          ? a.delta_log_probability - b.delta_log_probability
          : order === "kl"
            ? b.kl_divergence - a.kl_divergence
            : a.layer - b.layer || a.head - b.head,
    );
  return (
    <>
      <div className="trace-meta">
        <span className={`pill ${!result.complete ? "orange" : "green"}`}>
          {!result.complete
            ? t("Partial head sweep")
            : t("Head sweep complete")}
        </span>
        <span className="trace-source">{result.model_id}</span>
        {result.cancelled && <span>{t("Cancelled between forwards")}</span>}
        <span>
          {rich("revision {v0}", {
            v0: result.revision.slice(0, 8),
          })}
        </span>
        <span>
          {t(
            executionLabel(result.execution_precision, result.execution_config),
          )}
        </span>
        <span>{result.device.toUpperCase()}</span>
        <span>
          {rich("{v0} ms", {
            v0: number(result.duration_ms, 0),
          })}
        </span>
      </div>
      <ErrorNotice message={inspectionError} />
      {result.failed_head && (
        <div className="notice error" role="alert">
          {rich(
            "The sweep stopped at layer {v0}, head {v1}. {v2} Completed heads remain available for inspection and export.",
            {
              v0: result.failed_head.layer,
              v1: result.failed_head.head,
              v2: result.error,
            },
          )}
        </div>
      )}
      <div className="metrics-grid">
        <Metric
          label={t("Target token")}
          value={
            <span className="target-token">{result.target_token || "∅"}</span>
          }
          note={t("Vocabulary ID {v0}", {
            v0: result.target_token_id,
          })}
        />
        <Metric
          label={t("Baseline probability")}
          value={`${number(result.baseline.probability * 100, 3)}%`}
          note={t("Log probability {v0}", {
            v0: number(result.baseline.log_probability, 5),
          })}
        />
        <Metric
          label={t("Heads tested")}
          value={result.heads.length}
          note={t("Out of {v0} selected heads", {
            v0: result.total_heads,
          })}
        />
        <Metric
          label={t("Intervention scope")}
          value="Prompt"
          note={t("Every prompt position, one head per run")}
        />
      </div>
      <Panel
        title={t("Head effects on the target")}
        subtitle={t(
          "Each cell is the target log probability change after one head is zeroed.",
        )}
      >
        <div>
          <Heatmap
            data={rows}
            height={Math.min(300, Math.max(80, rows.length * 18))}
            label={t(
              "Columns are query head indices. Rows are selected transformer layers. Values are ablated minus baseline target log probability in nats.",
            )}
            xLabels={Array.from({ length: columns }, (_, head) =>
              t("Head {v0}", { v0: head }),
            )}
            yLabels={result.layer_targets.map((item) =>
              t("Layer {v0}", { v0: item.layer }),
            )}
          />
        </div>
        <p className="inline-note">
          {t(
            "Negative values mean removal reduced the target probability. Positive values mean removal increased it. Gray cells were not tested or have no head. These effects describe this prompt and intervention.",
          )}
        </p>
      </Panel>
      <Panel
        title={t("Per head ablation results")}
        subtitle={t(
          "Compare target changes and shifts in the full next token distribution.",
        )}
        className="section-gap"
      >
        <div className="head-sweep-controls">
          <label>
            {t("Layer")}
            <select
              aria-label={t("Head results layer")}
              value={layer}
              onChange={(event) => setLayer(event.target.value)}
            >
              <option value="all">{t("All tested layers")}</option>
              {result.layer_targets.map((item) => (
                <option key={item.layer} value={item.layer}>
                  {rich("Layer {v0}", {
                    v0: item.layer,
                  })}
                </option>
              ))}
            </select>
          </label>
          <label>
            {t("Order")}
            <select
              aria-label={t("Head results order")}
              value={order}
              onChange={(event) => setOrder(event.target.value)}
            >
              <option value="magnitude">{t("Largest target effect")}</option>
              <option value="reduction">{t("Most reduced target")}</option>
              <option value="kl">{t("Largest distribution shift")}</option>
              <option value="position">{t("Layer and head")}</option>
            </select>
          </label>
        </div>
        <p className="inline-note">
          {t(
            "Select a head index to prepare its prompt and ablation in the activation lab.",
          )}
        </p>
        <div className="table-wrap head-sweep-table">
          <table>
            <caption className="sr-only">
              {t("Independent attention head ablation scores")}
            </caption>
            <thead>
              <tr>
                <th scope="col">{t("LAYER")}</th>
                <th scope="col">{t("HEAD")}</th>
                <th scope="col">{t("DELTA NATS")}</th>
                <th scope="col">{t("TARGET PROBABILITY")}</th>
                <th scope="col">{t("KL NATS")}</th>
                <th scope="col">{t("TOP PREDICTION")}</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((head) => (
                <tr key={`${head.layer}/${head.head}`}>
                  <td>{head.layer}</td>
                  <td>
                    <button
                      className="head-inspect-button"
                      disabled={opening}
                      onClick={() => inspect(head)}
                      aria-label={t(
                        "Inspect layer {v0} head {v1} in activation lab",
                        {
                          v0: head.layer,
                          v1: head.head,
                        },
                      )}
                    >
                      {head.head} <ArrowUpRight size={12} />
                    </button>
                  </td>
                  <td className="code">
                    {t(number(head.delta_log_probability, 6))}
                  </td>
                  <td className="code">
                    {t(number(head.probability * 100, 4))}%
                  </td>
                  <td className="code">{t(number(head.kl_divergence, 6))}</td>
                  <td
                    className="code"
                    title={t("Vocabulary ID {v0}", {
                      v0: head.top_token_id,
                    })}
                  >
                    {JSON.stringify(head.top_token)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {sorted.length === 0 && (
            <p className="inline-note">
              {t("No completed heads for this selection.")}
            </p>
          )}
        </div>
        <p className="inline-note">
          {t(
            "KL measures divergence from the baseline distribution to the ablated distribution over the complete vocabulary. It is nonnegative and can reveal changes beyond the selected target. Zero ablation alone does not establish that a head is unnecessary across other prompts.",
          )}
        </p>
      </Panel>
      <details className="capture-details">
        <summary>{t("Captured prompt and method")}</summary>
        <p className="trace-output">{result.prompt}</p>
        <p className="inline-note">{t(result.method)}</p>
      </details>
    </>
  );
}
