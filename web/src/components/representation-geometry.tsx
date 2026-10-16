"use client";

import { Heatmap, SeriesChart } from "@/components/charts";
import { Panel } from "@/components/ui";
import { number } from "@/lib/api";
import type { Trace, TraceGeometry } from "@/lib/types";

export function RepresentationGeometry({
  geometry,
  trace,
}: {
  geometry: TraceGeometry;
  trace: Trace;
}) {
  const tokens = geometry.sampled_positions.map(
    (position) => `${position} · ${trace.tokens[position] || "∅"}`,
  );
  const insufficient =
    geometry.sampled_positions.length < geometry.minimum_samples;
  return (
    <>
      <div className="two-column">
        <Panel
          title="Layer representation similarity"
          subtitle="Linear CKA across prompt token positions. Both axes progress from embeddings through transformer blocks."
        >
          {insufficient ? (
            <div className="panel-body">
              <p className="inline-note">
                Use at least {geometry.minimum_samples} prompt tokens to compare
                layer geometry. A single token has no centered sample variance.
              </p>
            </div>
          ) : (
            <Heatmap
              data={geometry.cka}
              sequential
              maxAbs={1}
              xLabels={geometry.labels}
              yLabels={geometry.labels}
              label="Hover to compare two captures. One means aligned centered geometry. Gray cells have zero centered variance and are undefined."
            />
          )}
          <p className="inline-note">
            This uses the biased linear CKA estimator within one prompt. High
            similarity does not establish equivalent computations or a causal
            relationship.
          </p>
        </Panel>
        <Panel
          title="Token representation similarity"
          subtitle="Cosine similarity between token residual vectors at the final capture, before final normalization when available."
        >
          <Heatmap
            data={geometry.token_cosine}
            maxAbs={1}
            xLabels={tokens}
            yLabels={tokens}
            label="Both axes are sampled prompt positions. Orange means aligned directions, blue means opposite directions. Gray cells contain a zero vector and are undefined."
          />
        </Panel>
      </div>
      {geometry.sampled_positions.length < geometry.token_count && (
        <p className="inline-note">
          Geometry uses {geometry.sampled_positions.length} evenly spaced
          positions from {geometry.token_count} prompt tokens. Residual updates
          use every position.
        </p>
      )}
      <Panel
        title="Residual updates"
        subtitle="L2 magnitude of each block's change to the final prompt token. A bypassed block should contribute zero."
      >
        <SeriesChart
          data={trace.layers
            .filter((layer) => layer.layer >= 0)
            .map((layer) => ({
              index: layer.layer,
              update: layer.update_norm ?? null,
            }))}
          series={[
            { key: "update", label: "Residual update norm", color: "#bd906b" },
          ]}
          area
        />
        <details className="estimate-details">
          <summary>Inspect block update values</summary>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>CAPTURE</th>
                  <th>LAST TOKEN UPDATE</th>
                  <th>ALL TOKEN UPDATE RMS</th>
                  <th>UPDATE / INPUT NORM</th>
                </tr>
              </thead>
              <tbody>
                {trace.layers
                  .filter((layer) => layer.layer >= 0)
                  .map((layer) => (
                    <tr key={layer.layer}>
                      <td className="code">{layer.label}</td>
                      <td className="code">
                        {number(layer.update_norm ?? null, 4)}
                      </td>
                      <td className="code">
                        {number(layer.update_rms ?? null, 4)}
                      </td>
                      <td className="code">
                        {layer.relative_update == null
                          ? "N/A"
                          : `${number(layer.relative_update * 100, 2)}%`}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </details>
        <p className="inline-note">
          Update magnitude measures a vector difference. It does not isolate the
          attention or MLP contribution within the block.
        </p>
      </Panel>
    </>
  );
}
