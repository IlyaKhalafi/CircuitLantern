"use client";

import { Download } from "lucide-react";
import type { ReactNode } from "react";
import { Heatmap, Histogram, SeriesChart } from "@/components/charts";
import { Panel } from "@/components/ui";
import { downloadJSON, number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import type { MatrixMetrics } from "@/lib/types";

export function MatrixResults({
  metrics,
  model,
  footer,
}: {
  metrics: MatrixMetrics;
  model?: { id: string; revision: string };
  footer?: ReactNode;
}) {
  const modelId = metrics.model_id ?? model?.id;
  const revision = metrics.revision ?? model?.revision;
  const device =
    metrics.device ??
    (typeof metrics.execution_config?.device === "string"
      ? metrics.execution_config.device
      : undefined);
  return (
    <>
      <div className="trace-meta">
        <span className="pill subtle">Matrix analysis</span>
        {modelId && (
          <span className="trace-source" title={modelId}>
            {modelId}
          </span>
        )}
        {revision && (
          <span title={revision}>revision {revision.slice(0, 10)}</span>
        )}
        {metrics.execution_precision && (
          <span>
            {executionLabel(
              metrics.execution_precision,
              metrics.execution_config,
            )}
          </span>
        )}
        {device && <span>{device.toUpperCase()}</span>}
      </div>
      <div className="detail-header">
        <div>
          <div className="eyebrow">MATRIX INSPECTOR</div>
          <h2>{metrics.name}</h2>
          <p>
            {metrics.shape.join(" × ")} ·{" "}
            {metrics.source_format || metrics.dtype} source ·{" "}
            {number(metrics.elements, 0)} elements
          </p>
        </div>
        <button
          className="button secondary"
          onClick={() =>
            downloadJSON(
              {
                ...metrics,
                model_id: modelId,
                revision,
              },
              "matrix-analysis.json",
            )
          }
        >
          <Download size={14} /> Export analysis
        </button>
      </div>
      {metrics.quantized && (
        <div className="notice info">
          {metrics.analysis_note} Dimensions describe the logical weight
          operator.
        </div>
      )}
      {!metrics.spectrum_exact && (
        <div className="notice info">
          Spectral norm and stable rank use a randomized SVD on the full matrix.
          The leading {metrics.singular_values.length} singular values are
          approximate. Cumulative energy is measured against the full matrix
          Frobenius norm. Condition numbers, nuclear norms and full rank
          measures are unavailable without a complete spectrum.
        </div>
      )}
      <div className="two-column">
        <Panel
          title="Weight landscape"
          subtitle="Region means reveal structure without listing individual elements."
        >
          <Heatmap data={metrics.heatmap} label={metrics.heatmap_method} />
        </Panel>
        <Panel
          title="Singular value spectrum"
          subtitle={
            metrics.spectrum_exact
              ? `Full SVD in float64 · showing ${metrics.singular_values.length} of ${Math.min(...metrics.shape)} singular values`
              : `Full matrix randomized SVD · ${metrics.singular_values.length} estimated components`
          }
        >
          <SeriesChart
            data={metrics.singular_values.map((v, i) => ({
              index: i + 1,
              value: v,
            }))}
            series={[
              {
                key: "value",
                label: "Singular value",
                color: "#bd906b",
              },
            ]}
            area
          />
        </Panel>
      </div>
      <Panel
        title="Matrix properties"
        subtitle="Spectral radius is defined for square matrices and computed exactly up to 768 × 768."
      >
        <div className="weight-stats">
          {[
            {
              label: "Frobenius norm",
              value: metrics.frobenius_norm,
            },
            { label: "Spectral norm", value: metrics.spectral_norm },
            {
              label: "Spectral radius",
              value: metrics.spectral_radius,
            },
            { label: "Stable rank", value: metrics.stable_rank },
            {
              label: "Condition number",
              value: metrics.condition_number,
            },
            {
              label: "Operator 1 norm",
              value: metrics.operator_1_norm,
            },
            { label: "Entrywise L1 norm", value: metrics.l1_norm },
            { label: "Infinity norm", value: metrics.infinity_norm },
            {
              label: "Numerical rank",
              value: metrics.numerical_rank,
            },
            {
              label: "Effective rank",
              value: metrics.effective_rank,
            },
            { label: "Nuclear norm", value: metrics.nuclear_norm },
            {
              label: "Exact zeros",
              value: `${number(metrics.sparsity * 100)}%`,
            },
            { label: "Mean", value: metrics.mean },
            { label: "Standard deviation", value: metrics.std },
            { label: "Minimum", value: metrics.min },
            { label: "Maximum", value: metrics.max },
          ].map((m) => (
            <div className="weight-stat" key={m.label}>
              <span>{m.label}</span>
              <strong>
                {typeof m.value === "string" ? m.value : number(m.value, 5)}
              </strong>
            </div>
          ))}
        </div>
        <p className="inline-note">
          Operator 1 norm is the maximum absolute column sum. Entrywise L1 sums
          every absolute element. Effective rank uses the entropy of normalized
          singular values.
          {metrics.spectrum_exact && (
            <span>
              {" "}
              Numerical rank uses threshold {number(metrics.rank_tolerance, 3)}.
              Condition numbers are omitted for numerical rank deficiency.
            </span>
          )}
        </p>
      </Panel>
      <div className="two-column section-gap">
        <Panel
          title="Weight distribution"
          subtitle={`${metrics.histogram_sampled ? "Subsampled" : "All"} elements · 48 histogram bins`}
        >
          <Histogram data={metrics.histogram} />
        </Panel>
        <Panel
          title="Cumulative spectral energy"
          subtitle="Fraction of squared singular values captured by the leading components."
        >
          <SeriesChart
            data={metrics.cumulative_energy.map((v, i) => ({
              index: i + 1,
              value: v * 100,
            }))}
            series={[
              {
                key: "value",
                label: "Energy percent",
                color: "#87a38e",
              },
            ]}
            area
          />
        </Panel>
      </div>
      {footer}
    </>
  );
}
