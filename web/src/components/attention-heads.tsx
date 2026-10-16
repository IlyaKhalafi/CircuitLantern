"use client";

import { Heatmap } from "@/components/charts";
import { Panel } from "@/components/ui";
import { number } from "@/lib/api";
import type { Attention } from "@/lib/types";

export function AttentionHeads({
  attention,
  tokens,
}: {
  attention: Attention;
  tokens: string[];
}) {
  if (!attention.query_by_head) return null;
  const labels = attention.query_by_head.map((_, index) => `Head ${index}`);
  const query = attention.query_position ?? tokens.length - 1;
  return (
    <Panel
      title="Final token attention by head"
      subtitle={`Layer ${attention.layer} · Query position ${query} · ${tokens[query] || "∅"}`}
    >
      <Heatmap
        data={attention.query_by_head}
        xLabels={tokens.map((token, index) => `${index} · ${token || "∅"}`)}
        yLabels={labels}
        sequential
        label="Rows are individual attention heads. Columns are prompt key positions. Each row shows routing from the final prompt token. Color is normalized across heads within this layer."
      />
      <details className="estimate-details">
        <summary>Inspect head routing values</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>HEAD</th>
                <th>STRONGEST KEY POSITION</th>
                <th>ATTENTION WEIGHT</th>
                <th>FINAL QUERY ENTROPY</th>
              </tr>
            </thead>
            <tbody>
              {attention.query_by_head.map((weights, head) => {
                const strongest = weights.reduce(
                  (best, weight, index) =>
                    weight > weights[best] ? index : best,
                  0,
                );
                return (
                  <tr key={head}>
                    <td className="code">Head {head}</td>
                    <td className="code">
                      {strongest} · {tokens[strongest] || "∅"}
                    </td>
                    <td className="code">
                      {number(weights[strongest] * 100, 3)}%
                    </td>
                    <td className="code">
                      {number(attention.query_head_entropy?.[head] ?? null, 4)}{" "}
                      nats
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
      <p className="inline-note">
        These weights describe attention routing. Use causal tracing to test an
        effect on predictions. Final query entropy differs from entropy averaged
        across all prompt positions.
      </p>
    </Panel>
  );
}
