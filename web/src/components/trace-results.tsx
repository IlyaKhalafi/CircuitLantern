"use client";

import { useState } from "react";
import { Heatmap, SeriesChart } from "@/components/charts";
import { Panel } from "@/components/ui";
import { RepresentationGeometry } from "@/components/representation-geometry";
import { AttentionHeads } from "@/components/attention-heads";
import { number } from "@/lib/api";
import { interventionLabel } from "@/lib/interventions";
import { executionLabel } from "@/lib/execution";
import type { Trace } from "@/lib/types";

export function TraceResults({
  trace,
  comparison,
}: {
  trace: Trace;
  comparison?: Trace | null;
}) {
  const [tab, setTab] = useState("residual");
  const [attentionLayer, setAttentionLayer] = useState(0);
  const [tokenIndex, setTokenIndex] = useState(trace.tokens.length - 1);
  const attention =
    trace.attention[Math.min(attentionLayer, trace.attention.length - 1)];
  const residual = trace.layers.map((layer, i) => ({
    index: layer.layer,
    norm: layer.norm,
    rms: layer.rms,
    entropy: layer.entropy,
    baseline: comparison?.layers[i]?.norm ?? null,
  }));
  return (
    <>
      <div className="trace-meta">
        <span className="pill green">Capture complete</span>
        <span>{trace.tokens.length} prompt tokens</span>
        <span>{number(trace.duration_ms, 0)} ms</span>
        <span>{trace.device.toUpperCase()}</span>
        <span>{executionLabel(trace.execution_precision)}</span>
        {(trace.ablate_layer != null || trace.ablate_head) && (
          <span className="pill orange">{interventionLabel(trace)}</span>
        )}
      </div>
      {trace.ablate_head && (
        <div className="notice info">
          This trace zeros one attention head output at every prompt and
          generated token before the attention output projection.
        </div>
      )}
      <Panel
        title="Tokenized prompt"
        subtitle="Hover over a token to see its vocabulary ID."
      >
        <div className="token-strip">
          {trace.tokens.map((token, i) => (
            <span
              className="token"
              key={i}
              title={`Position ${i} · token ID ${trace.token_ids[i]}`}
            >
              {token.replace(/\n/g, "↵") || "∅"}
            </span>
          ))}
        </div>
        <div className="trace-output">
          <span>{trace.prompt}</span>
          <strong>{trace.continuation}</strong>
        </div>
      </Panel>
      <div
        className="tabs section-gap"
        role="tablist"
        aria-label="Trace analysis"
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
              title="Residual stream magnitude"
              subtitle="L2 norm of the final prompt token at each captured layer."
            >
              <SeriesChart
                data={residual}
                series={[
                  { key: "norm", label: "Current trace", color: "#b85b28" },
                  ...(comparison
                    ? [{ key: "baseline", label: "Baseline", color: "#557ca0" }]
                    : []),
                ]}
                area={!comparison}
              />
              <p className="inline-note">
                Residual magnitudes are captured before final normalization when
                the architecture exposes its transformer blocks.
              </p>
            </Panel>
            <Panel
              title="Prediction uncertainty"
              subtitle="Entropy of the last token logit lens distribution in natural units."
            >
              <SeriesChart
                data={residual}
                series={[
                  {
                    key: "entropy",
                    label: "Entropy in nats",
                    color: "#809cbb",
                  },
                ]}
                area
              />
            </Panel>
          </div>
          <Panel
            title="Token × layer activation norms"
            subtitle="Every cell is the residual vector L2 norm at a token position. Hover to inspect."
          >
            <Heatmap
              data={trace.layers.map((l) => l.token_norms)}
              label="Columns are prompt tokens. Rows progress from embeddings through transformer blocks. Color is normalized over this capture."
              xLabels={trace.tokens}
              yLabels={trace.layers.map((l) => l.label)}
              sequential
            />
          </Panel>
        </>
      )}
      {tab === "geometry" && trace.geometry && (
        <RepresentationGeometry geometry={trace.geometry} trace={trace} />
      )}
      {tab === "lens" && (
        <Panel
          title="Watch predictions take shape"
          subtitle="Project intermediate residual states through the final norm and unembedding."
        >
          <div className="attention-controls">
            <label className="checkbox-row">
              Token position{" "}
              <select
                aria-label="Logit lens token position"
                value={Math.min(tokenIndex, trace.tokens.length - 1)}
                onChange={(e) => setTokenIndex(+e.target.value)}
              >
                {trace.tokens.map((t, i) => (
                  <option key={i} value={i}>
                    {i} · {t || "∅"}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>CAPTURE</th>
                  <th>POSITION PREDICTION</th>
                  <th>LAST TOKEN TOP PREDICTION</th>
                  <th>PROBABILITY</th>
                  <th>ENTROPY</th>
                  <th>COSINE TO PREVIOUS</th>
                </tr>
              </thead>
              <tbody>
                {trace.layers.map((l) => (
                  <tr key={l.layer}>
                    <td className="code">{l.label}</td>
                    <td>
                      <span className="lens-token">
                        {
                          l.token_predictions[
                            Math.min(tokenIndex, trace.tokens.length - 1)
                          ]
                        }
                      </span>
                    </td>
                    <td>
                      <span className="lens-token">
                        {l.predictions[0].token}
                      </span>
                    </td>
                    <td className="code">
                      {(l.predictions[0].probability * 100).toFixed(2)}%
                    </td>
                    <td className="code">{number(l.entropy)}</td>
                    <td className="code">{number(l.cosine_similarity, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="inline-note">{trace.method}</p>
        </Panel>
      )}
      {tab === "attention" &&
        (attention ? (
          <>
            {trace.ablate_layer === attention.layer && (
              <div className="notice info">
                This block's output was bypassed. Its captured attention
                reflects computation before the intervention.
              </div>
            )}
            {trace.ablate_head?.layer === attention.layer && (
              <div className="notice info">
                Head {trace.ablate_head.head} output was zeroed after attention
                routing. The captured patterns still show its routing before the
                intervention.
              </div>
            )}
            <div className="two-column">
              <Panel
                title="Attention pattern"
                subtitle="Mean across heads. Rows attend to columns. Future positions are causally masked."
              >
                <div className="attention-controls">
                  <select
                    aria-label="Attention layer"
                    value={attentionLayer}
                    onChange={(e) => setAttentionLayer(+e.target.value)}
                  >
                    {trace.attention.map((a, i) => (
                      <option key={a.layer} value={i}>
                        Layer {a.layer}
                      </option>
                    ))}
                  </select>
                  <span>
                    {attention.heads} heads · {trace.tokens.length} tokens
                  </span>
                </div>
                <Heatmap
                  data={attention.pattern}
                  label="Hover over a cell to inspect query → key token attention. Colors are normalized within this layer."
                  xLabels={trace.tokens}
                  yLabels={trace.tokens}
                  sequential
                />
              </Panel>
              <Panel
                title="Attention head entropy"
                subtitle="Mean entropy across query positions in natural units. Lower values indicate more focused attention."
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
                      color: "#88a68e",
                    },
                  ]}
                  area
                />
                <p className="inline-note">
                  Causal masking gives earlier query positions fewer available
                  keys. Compare heads within the same capture.
                </p>
              </Panel>
            </div>
            <AttentionHeads attention={attention} tokens={trace.tokens} />
          </>
        ) : (
          <div className="notice info">
            Attention was not captured for this trace. Enable attention capture
            and run again.
          </div>
        ))}
      {tab === "logits" && (
        <div className="two-column">
          <Panel
            title="Next token distribution"
            subtitle="The ten most likely next tokens from the model output logits."
          >
            <div className="predictions">
              {trace.predictions.map((p) => (
                <div className="prediction-row" key={p.token_id}>
                  <code title={`Token ID ${p.token_id}`}>{p.token || "∅"}</code>
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
            title="Raw logits"
            subtitle="Unnormalized scores for the same next token candidates."
          >
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>TOKEN</th>
                    <th>TOKEN ID</th>
                    <th>LOGIT</th>
                    <th>PROBABILITY</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.predictions.map((p) => (
                    <tr key={p.token_id}>
                      <td>
                        <span className="lens-token">{p.token}</span>
                      </td>
                      <td className="code">{p.token_id}</td>
                      <td className="code">{number(p.logit, 4)}</td>
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
