"use client";
import { useI18n } from "@/i18n/provider";
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
  const { t, rich } = useI18n();
  if (!attention.query_by_head) return null;
  const labels = attention.query_by_head.map((_, index) => `Head ${index}`);
  const query = attention.query_position ?? tokens.length - 1;
  return (
    <Panel
      title={
        query === tokens.length - 1
          ? t("Final token attention by head")
          : t("Prompt token attention by head")
      }
      subtitle={t("Layer {v0} \u00B7 Query position {v1} \u00B7 {v2}", {
        v0: attention.layer,
        v1: query,
        v2: tokens[query] || "∅",
      })}
    >
      <Heatmap
        data={attention.query_by_head}
        xLabels={tokens.map((token, index) => `${index} · ${token || "∅"}`)}
        yLabels={labels}
        sequential
        label={t(
          "Rows are individual attention heads. Columns are prompt key positions. Each row shows routing from prompt query position {v0}. Color is normalized across heads within this layer.",
          {
            v0: query,
          },
        )}
      />
      <details className="estimate-details">
        <summary>{t("Inspect head routing values")}</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t("HEAD")}</th>
                <th>{t("STRONGEST KEY POSITION")}</th>
                <th>{t("ATTENTION WEIGHT")}</th>
                <th>{t("QUERY ENTROPY")}</th>
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
                    <td className="code">
                      {rich("Head {v0}", {
                        v0: head,
                      })}
                    </td>
                    <td className="code">
                      {strongest} · {tokens[strongest] || "∅"}
                    </td>
                    <td className="code">
                      {t(number(weights[strongest] * 100, 3))}%
                    </td>
                    <td className="code">
                      {rich("{v0} nats", {
                        v0: number(
                          attention.query_head_entropy?.[head] ?? null,
                          4,
                        ),
                      })}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
      <p className="inline-note">
        {t(
          "These weights describe attention routing. Use causal tracing to test an effect on predictions. Query entropy differs from entropy averaged across all prompt positions.",
        )}
      </p>
    </Panel>
  );
}
