"use client";
import { useI18n } from "@/i18n/provider";
import { Download } from "lucide-react";
import type { ReactNode } from "react";
import { Heatmap, Histogram, SeriesChart } from "@/components/charts";
import { Panel } from "@/components/ui";
import { downloadJSON, number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import { matrixProperties, matrixValue } from "@/lib/matrix-properties";
import type { MatrixMetrics } from "@/lib/types";
export function MatrixResults({
  metrics,
  model,
  footer,
}: {
  metrics: MatrixMetrics;
  model?: {
    id: string;
    revision: string;
  };
  footer?: ReactNode;
}) {
  const { t, rich } = useI18n();
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
        <span className="pill subtle">{t("Matrix analysis")}</span>
        {modelId && (
          <span className="trace-source" title={modelId}>
            {modelId}
          </span>
        )}
        {revision && (
          <span title={revision}>
            {rich("revision {v0}", {
              v0: revision.slice(0, 10),
            })}
          </span>
        )}
        {metrics.execution_precision && (
          <span>
            {t(
              executionLabel(
                metrics.execution_precision,
                metrics.execution_config,
              ),
            )}
          </span>
        )}
        {device && <span>{device.toUpperCase()}</span>}
      </div>
      <div className="detail-header">
        <div>
          <div className="eyebrow">{t("MATRIX INSPECTOR")}</div>
          <h2>{metrics.name}</h2>
          <p>
            {rich("{v0} \u00B7 {v1} source \u00B7 {v2} elements", {
              v0: metrics.shape.join(" × "),
              v1: metrics.source_format || metrics.dtype,
              v2: number(metrics.elements, 0),
            })}
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
          <Download size={14} />
          {t("Export analysis")}
        </button>
      </div>
      {metrics.quantized && (
        <div className="notice info">
          {rich("{v0} Dimensions describe the logical weight operator.", {
            v0: metrics.analysis_note ? t(metrics.analysis_note) : undefined,
          })}
        </div>
      )}
      {!metrics.spectrum_exact && (
        <div className="notice info">
          {rich(
            "Spectral norm and stable rank use a randomized SVD on the full matrix. The leading {v0} singular values are approximate. Cumulative energy is measured against the full matrix Frobenius norm. Condition numbers, nuclear norms and full rank measures are unavailable without a complete spectrum.",
            {
              v0: metrics.singular_values.length,
            },
          )}
        </div>
      )}
      <div className="two-column">
        <Panel
          title={t("Weight landscape")}
          subtitle={t(
            "Region means reveal structure without listing individual elements.",
          )}
        >
          <Heatmap data={metrics.heatmap} label={t(metrics.heatmap_method)} />
        </Panel>
        <Panel
          title={t("Singular value spectrum")}
          subtitle={
            metrics.spectrum_exact
              ? t(
                  "Full SVD in float64 \u00B7 showing {v0} of {v1} singular values",
                  {
                    v0: metrics.singular_values.length,
                    v1: Math.min(...metrics.shape),
                  },
                )
              : t(
                  "Full matrix randomized SVD \u00B7 {v0} estimated components",
                  {
                    v0: metrics.singular_values.length,
                  },
                )
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
                color: "var(--accent)",
              },
            ]}
            area
          />
        </Panel>
      </div>
      <Panel
        title={t("Matrix properties")}
        subtitle={t(
          "Spectral radius is defined for square matrices and computed exactly up to 768 \u00D7 768.",
        )}
      >
        <div className="weight-stats">
          {matrixProperties(metrics).map((property) => (
            <div className="weight-stat" key={property.key}>
              <span>{t(property.label)}</span>
              <strong>{matrixValue(property.value, property.unit)}</strong>
            </div>
          ))}
        </div>
        <p className="inline-note">
          {rich(
            "Operator 1 norm is the maximum absolute column sum. Entrywise L1 sums every absolute element. Effective rank uses the entropy of normalized singular values.{v0}",
            {
              v0: metrics.spectrum_exact && (
                <span>
                  {rich(
                    "Numerical rank uses threshold {v0}. Condition numbers are omitted for numerical rank deficiency.",
                    {
                      v0: number(metrics.rank_tolerance, 3),
                    },
                  )}
                </span>
              ),
            },
          )}
        </p>
      </Panel>
      <div className="two-column section-gap">
        <Panel
          title={t("Weight distribution")}
          subtitle={t("{v0} elements \u00B7 48 histogram bins", {
            v0: metrics.histogram_sampled ? t("Subsampled") : t("All"),
          })}
        >
          <Histogram data={metrics.histogram} />
        </Panel>
        <Panel
          title={t("Cumulative spectral energy")}
          subtitle={t(
            "Fraction of squared singular values captured by the leading components.",
          )}
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
                color: "var(--green)",
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
