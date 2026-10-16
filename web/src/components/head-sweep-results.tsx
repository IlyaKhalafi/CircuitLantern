"use client";

import { useState } from "react";
import { Heatmap } from "@/components/charts";
import { Metric, Panel } from "@/components/ui";
import { number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import type { HeadSweepResult } from "@/lib/types";

export function HeadSweepResults({ result }: { result: HeadSweepResult }) {
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
          {!result.complete ? "Partial head sweep" : "Head sweep complete"}
        </span>
        <span className="trace-source">{result.model_id}</span>
        {result.cancelled && <span>Cancelled between forwards</span>}
        <span>revision {result.revision.slice(0, 8)}</span>
        <span>
          {executionLabel(result.execution_precision, result.execution_config)}
        </span>
        <span>{result.device.toUpperCase()}</span>
        <span>{number(result.duration_ms, 0)} ms</span>
      </div>
      {result.failed_head && (
        <div className="notice error" role="alert">
          The sweep stopped at layer {result.failed_head.layer}, head{" "}
          {result.failed_head.head}. {result.error} Completed heads remain
          available for inspection and export.
        </div>
      )}
      <div className="metrics-grid">
        <Metric
          label="Target token"
          value={
            <span className="target-token">{result.target_token || "∅"}</span>
          }
          note={`Vocabulary ID ${result.target_token_id}`}
        />
        <Metric
          label="Baseline probability"
          value={`${number(result.baseline.probability * 100, 3)}%`}
          note={`Log probability ${number(result.baseline.log_probability, 5)}`}
        />
        <Metric
          label="Heads tested"
          value={result.heads.length}
          note={`Out of ${result.total_heads} selected heads`}
        />
        <Metric
          label="Intervention scope"
          value="Prompt"
          note="Every prompt position, one head per run"
        />
      </div>
      <Panel
        title="Head effects on the target"
        subtitle="Each cell is the target log probability change after one head is zeroed."
      >
        <div>
          <Heatmap
            data={rows}
            height={Math.min(300, Math.max(80, rows.length * 18))}
            label="Columns are query head indices. Rows are selected transformer layers. Values are ablated minus baseline target log probability in nats."
            xLabels={Array.from(
              { length: columns },
              (_, head) => `Head ${head}`,
            )}
            yLabels={result.layer_targets.map((item) => `Layer ${item.layer}`)}
          />
        </div>
        <p className="inline-note">
          Negative values mean removal reduced the target probability. Positive
          values mean removal increased it. Gray cells were not tested or have
          no head. These effects describe this prompt and intervention.
        </p>
      </Panel>
      <Panel
        title="Per head ablation results"
        subtitle="Compare target changes and shifts in the full next token distribution."
        className="section-gap"
      >
        <div className="head-sweep-controls">
          <label>
            Layer
            <select
              aria-label="Head results layer"
              value={layer}
              onChange={(event) => setLayer(event.target.value)}
            >
              <option value="all">All tested layers</option>
              {result.layer_targets.map((item) => (
                <option key={item.layer} value={item.layer}>
                  Layer {item.layer}
                </option>
              ))}
            </select>
          </label>
          <label>
            Order
            <select
              aria-label="Head results order"
              value={order}
              onChange={(event) => setOrder(event.target.value)}
            >
              <option value="magnitude">Largest target effect</option>
              <option value="reduction">Most reduced target</option>
              <option value="kl">Largest distribution shift</option>
              <option value="position">Layer and head</option>
            </select>
          </label>
        </div>
        <div className="table-wrap head-sweep-table">
          <table>
            <caption className="sr-only">
              Independent attention head ablation scores
            </caption>
            <thead>
              <tr>
                <th scope="col">LAYER</th>
                <th scope="col">HEAD</th>
                <th scope="col">DELTA NATS</th>
                <th scope="col">TARGET PROBABILITY</th>
                <th scope="col">KL NATS</th>
                <th scope="col">TOP PREDICTION</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((head) => (
                <tr key={`${head.layer}/${head.head}`}>
                  <td>{head.layer}</td>
                  <td>{head.head}</td>
                  <td className="code">
                    {number(head.delta_log_probability, 6)}
                  </td>
                  <td className="code">{number(head.probability * 100, 4)}%</td>
                  <td className="code">{number(head.kl_divergence, 6)}</td>
                  <td
                    className="code"
                    title={`Vocabulary ID ${head.top_token_id}`}
                  >
                    {JSON.stringify(head.top_token)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {sorted.length === 0 && (
            <p className="inline-note">
              No completed heads for this selection.
            </p>
          )}
        </div>
        <p className="inline-note">
          KL measures divergence from the baseline distribution to the ablated
          distribution over the complete vocabulary. It is nonnegative and can
          reveal changes beyond the selected target. Zero ablation alone does
          not establish that a head is unnecessary across other prompts.
        </p>
      </Panel>
      <details className="capture-details">
        <summary>Captured prompt and method</summary>
        <p className="trace-output">{result.prompt}</p>
        <p className="inline-note">{result.method}</p>
      </details>
    </>
  );
}
