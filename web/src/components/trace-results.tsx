"use client";
import { useI18n } from "@/i18n/provider";
import { useState } from "react";
import { Heatmap, SeriesChart } from "@/components/charts";
import { Panel } from "@/components/ui";
import { RepresentationGeometry } from "@/components/representation-geometry";
import { GeneratedOutput } from "@/components/generated-output";
import { AttentionHeads } from "@/components/attention-heads";
import { LogitLens } from "@/components/logit-lens";
import { number } from "@/lib/api";
import { interventionLabel } from "@/lib/interventions";
import { executionLabel } from "@/lib/execution";
import type { Trace } from "@/lib/types";
export function TraceResults({
  trace,
  comparison,
  comparisonLabel = "Baseline",
  traceLabel = "Current trace",
}: {
  trace: Trace;
  comparison?: Trace | null;
  comparisonLabel?: string;
  traceLabel?: string;
}) {
  const { t, rich } = useI18n();
  const [tab, setTab] = useState("residual");
  const [attentionLayer, setAttentionLayer] = useState(0);
  const [logResidualColors, setLogResidualColors] = useState(false);
  const [tokenIndex, setTokenIndex] = useState(trace.tokens.length - 1);
  const attention =
    trace.attention[Math.min(attentionLayer, trace.attention.length - 1)];
  const residual = trace.layers.map((layer, i) => ({
    index: layer.layer,
    norm: layer.norm,
    rms: layer.rms,
    entropy: layer.entropy,
    baseline: comparison?.layers[i]?.norm ?? null,
    referenceEntropy: comparison?.layers[i]?.entropy ?? null,
  }));
  return (
    <>
      <div className="trace-meta">
        <span className="pill green">{t("Capture complete")}</span>
        <span>
          {rich("{v0} prompt tokens", {
            v0: trace.tokens.length,
          })}
        </span>
        <span>
          {rich("{v0} ms", {
            v0: number(trace.duration_ms, 0),
          })}
        </span>
        <span>{trace.device.toUpperCase()}</span>
        <span>
          {t(executionLabel(trace.execution_precision, trace.execution_config))}
        </span>
        <span
          className="trace-source"
          title={`${trace.model_id} · ${trace.revision}`}
        >
          {trace.model_id} · {trace.revision.slice(0, 8)}
        </span>
        {(trace.ablate_layer != null || trace.ablate_head) && (
          <span className="pill orange">{t(interventionLabel(trace))}</span>
        )}
      </div>
      {trace.ablate_head && (
        <div className="notice info">
          {rich(
            "This trace zeros one attention head output at every prompt{v0} token before the attention output projection.",
            {
              v0:
                trace.ablate_head.scope === "all_prompt_tokens" ||
                trace.generation_performed === false
                  ? ""
                  : t(" and generated"),
            },
          )}
        </div>
      )}
      <Panel
        title={t("Tokenized prompt")}
        subtitle={t("Hover over a token to see its vocabulary ID.")}
      >
        <div className="token-strip">
          {trace.tokens.map((token, i) => (
            <span
              className="token"
              key={i}
              title={t("Position {v0} \u00B7 token ID {v1}", {
                v0: i,
                v1: trace.token_ids[i],
              })}
            >
              {token.replace(/\n/g, "↵") || "∅"}
            </span>
          ))}
        </div>
        <GeneratedOutput trace={trace} />
      </Panel>
      <div
        className="tabs section-gap"
        role="tablist"
        aria-label={t("Trace analysis")}
      >
        {[
          { id: "residual", name: "Residual stream" },
          { id: "lens", name: "Logit lens" },
          { id: "attention", name: "Attention" },
          { id: "logits", name: "Next token logits" },
          ...(trace.geometry
            ? [{ id: "geometry", name: "Representation geometry" }]
            : []),
        ].map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={tab === t.id ? "active" : ""}
            onClick={() => setTab(t.id)}
          >
            {t.name}
          </button>
        ))}
      </div>
      {tab === "residual" && (
        <>
          <div className="two-column">
            <Panel
              title={t("Residual stream magnitude")}
              subtitle={t(
                "L2 norm of the final prompt token at each captured layer.",
              )}
            >
              <SeriesChart
                data={residual}
                series={[
                  { key: "norm", label: traceLabel, color: "var(--accent)" },
                  ...(comparison
                    ? [
                        {
                          key: "baseline",
                          label: comparisonLabel,
                          color: "var(--blue)",
                        },
                      ]
                    : []),
                ]}
                area={!comparison}
              />
              <p className="inline-note">
                {t(
                  "Residual magnitudes are captured before final normalization when the architecture exposes its transformer blocks.",
                )}
              </p>
            </Panel>
            <Panel
              title={t("Prediction uncertainty")}
              subtitle={t(
                "Entropy of the last token logit lens distribution in natural units.",
              )}
            >
              <SeriesChart
                data={residual}
                series={[
                  {
                    key: "entropy",
                    label: comparison ? traceLabel : "Entropy in nats",
                    color: comparison ? "var(--accent)" : "var(--blue)",
                  },
                  ...(comparison
                    ? [
                        {
                          key: "referenceEntropy",
                          label: comparisonLabel,
                          color: "var(--blue)",
                        },
                      ]
                    : []),
                ]}
                area={!comparison}
              />
            </Panel>
          </div>
          <Panel
            title={t("Token \u00D7 layer activation norms")}
            subtitle={t(
              "Every cell is the residual vector L2 norm at a token position. Hover to inspect.",
            )}
            action={
              <label className="checkbox-row residual-color-toggle">
                <input
                  dir="auto"
                  type="checkbox"
                  aria-label={t("Logarithmic residual colors")}
                  checked={logResidualColors}
                  onChange={(event) =>
                    setLogResidualColors(event.target.checked)
                  }
                />
                {t("Log colors")}
              </label>
            }
          >
            <Heatmap
              data={trace.layers.map((l) => l.token_norms)}
              label={t(
                "Columns are prompt tokens. Rows progress from embeddings through transformer blocks. {v0}",
                {
                  v0: logResidualColors
                    ? t(
                        "Logarithmic colors use log(1 + norm). Hover values are unscaled.",
                      )
                    : t("Linear colors are normalized over this capture."),
                },
              )}
              xLabels={trace.tokens}
              yLabels={trace.layers.map((l) => l.label)}
              logarithmic={logResidualColors}
              sequential
            />
          </Panel>
        </>
      )}
      {tab === "geometry" && trace.geometry && (
        <RepresentationGeometry geometry={trace.geometry} trace={trace} />
      )}
      {tab === "lens" && (
        <LogitLens
          trace={trace}
          position={tokenIndex}
          onPositionChange={setTokenIndex}
        />
      )}
      {tab === "attention" &&
        (attention ? (
          <>
            {trace.ablate_layer === attention.layer && (
              <div className="notice info">
                {t(
                  "This block's output was bypassed. Its captured attention reflects computation before the intervention.",
                )}
              </div>
            )}
            {trace.ablate_head?.layer === attention.layer && (
              <div className="notice info">
                {rich(
                  "Head {v0} output was zeroed after attention routing. The captured patterns still show its routing before the intervention.",
                  {
                    v0: trace.ablate_head.head,
                  },
                )}
              </div>
            )}
            <div className="two-column">
              <Panel
                title={t("Attention pattern")}
                subtitle={t(
                  "Mean across heads. Rows attend to columns. Future positions are causally masked.",
                )}
              >
                <div className="attention-controls">
                  <select
                    aria-label={t("Attention layer")}
                    value={attentionLayer}
                    onChange={(e) => setAttentionLayer(+e.target.value)}
                  >
                    {trace.attention.map((a, i) => (
                      <option key={a.layer} value={i}>
                        {rich("Layer {v0}", {
                          v0: a.layer,
                        })}
                      </option>
                    ))}
                  </select>
                  <span>
                    {rich("{v0} heads \u00B7 {v1} tokens", {
                      v0: attention.heads,
                      v1: trace.tokens.length,
                    })}
                  </span>
                </div>
                <Heatmap
                  data={attention.pattern}
                  label={t(
                    "Hover over a cell to inspect query \u2192 key token attention. Colors are normalized within this layer.",
                  )}
                  xLabels={trace.tokens}
                  yLabels={trace.tokens}
                  sequential
                />
              </Panel>
              <Panel
                title={t("Attention head entropy")}
                subtitle={t(
                  "Mean entropy across query positions in natural units. Lower values indicate more focused attention.",
                )}
              >
                <SeriesChart
                  data={attention.head_entropy.map((v, i) => ({
                    index: i,
                    entropy: v,
                  }))}
                  series={[
                    {
                      key: "entropy",
                      label: "Entropy in nats",
                      color: "var(--green)",
                    },
                  ]}
                  area
                />
                <p className="inline-note">
                  {t(
                    "Causal masking gives earlier query positions fewer available keys. Compare heads within the same capture.",
                  )}
                </p>
              </Panel>
            </div>
            <AttentionHeads attention={attention} tokens={trace.tokens} />
          </>
        ) : (
          <div className="notice info">
            {t(
              "Attention was not captured for this trace. Enable attention capture and run again.",
            )}
          </div>
        ))}
      {tab === "logits" && (
        <div className="two-column">
          <Panel
            title={t("Next token distribution")}
            subtitle={t(
              "The ten most likely next tokens from the model output logits.",
            )}
          >
            <div className="predictions">
              {trace.predictions.map((p) => (
                <div className="prediction-row" key={p.token_id}>
                  <code
                    title={t("Token ID {v0}", {
                      v0: p.token_id,
                    })}
                  >
                    {p.token || "∅"}
                  </code>
                  <div className="bar">
                    <div
                      style={{
                        width: `${(p.probability / trace.predictions[0].probability) * 100}%`,
                      }}
                    />
                  </div>
                  <span>{(p.probability * 100).toFixed(3)}%</span>
                </div>
              ))}
            </div>
          </Panel>
          <Panel
            title={t("Raw logits")}
            subtitle={t(
              "Unnormalized scores for the same next token candidates.",
            )}
          >
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t("TOKEN")}</th>
                    <th>{t("TOKEN ID")}</th>
                    <th>{t("LOGIT")}</th>
                    <th>{t("PROBABILITY")}</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.predictions.map((p) => (
                    <tr key={p.token_id}>
                      <td>
                        <span className="lens-token">{p.token}</span>
                      </td>
                      <td className="code">{p.token_id}</td>
                      <td className="code">{t(number(p.logit, 4))}</td>
                      <td className="code">
                        {(p.probability * 100).toFixed(3)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      )}
    </>
  );
}
