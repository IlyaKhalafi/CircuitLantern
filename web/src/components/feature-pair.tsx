"use client";

import { useState } from "react";
import { GitCompare } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { Metric } from "@/components/ui";
import { SeriesChart } from "@/components/charts";
import { number } from "@/lib/api";
import type { FeatureInspection } from "@/lib/features";

export function FeaturePairControls({
  feature,
  featureCount,
  busy,
  enabled,
  onCompare,
}: {
  feature: number;
  featureCount: number;
  busy: boolean;
  enabled: boolean;
  onCompare: (other: number) => void;
}) {
  const { t } = useI18n();
  const [other, setOther] = useState(feature === 0 ? "1" : "0");
  const parsed = Number(other);
  const valid =
    other.trim() !== "" &&
    Number.isInteger(parsed) &&
    parsed >= 0 &&
    parsed < featureCount &&
    parsed !== feature;
  if (featureCount < 2) return null;
  return (
    <details className="feature-pair-controls">
      <summary>{t("Compare two features")}</summary>
      <div className="feature-pair-form">
        <p className="muted">
          {t(
            "Compare the selected feature with another feature across eligible positions in this prompt.",
          )}
        </p>
        <div className="feature-toolbar">
          <span>
            {t("Selected feature")} <strong>{feature}</strong>
          </span>
          <label>
            {t("Compare with feature")}
            <input
              className="form-input"
              aria-label={t("Compare with feature")}
              aria-invalid={!valid}
              type="number"
              dir="ltr"
              min={0}
              max={featureCount - 1}
              step={1}
              value={other}
              disabled={busy}
              onChange={(event) => setOther(event.target.value)}
            />
          </label>
          <button
            className="button secondary"
            disabled={busy || !enabled || !valid}
            onClick={() => onCompare(parsed)}
          >
            <GitCompare size={14} />
            {t("Compare prompt activations")}
          </button>
        </div>
        <p className="muted">
          {t("Choose a different feature from 0 to {v0}.", {
            v0: featureCount - 1,
          })}
        </p>
      </div>
    </details>
  );
}

export function FeaturePairResult({ result }: { result: FeatureInspection }) {
  const { t } = useI18n();
  const comparison = result.feature_comparison;
  if (!comparison) return null;
  const unavailable = (
    <span className="feature-pair-unavailable">{t("Unavailable")}</span>
  );
  const format = (value: number | null) =>
    value === null ? unavailable : number(value, 4);
  const features = comparison.feature_ids.map((index) =>
    result.features.find((row) => row.index === index),
  );
  const values = result.tokens.map((_, position) => ({
    position,
    first: features[0]?.activations[position] ?? null,
    second: features[1]?.activations[position] ?? null,
  }));
  return (
    <section
      className="feature-pair-result"
      aria-label={t("Prompt feature comparison")}
    >
      <h3>{t("Prompt feature comparison")}</h3>
      <p className="muted">
        {t("Compared features {v0} and {v1}", {
          v0: comparison.feature_ids[0],
          v1: comparison.feature_ids[1],
        })}
      </p>
      <p className="muted">
        {t(
          "Decoder cosine measures weight geometry. Activation statistics describe normalized baseline coefficients in this prompt and do not establish shared meaning or causal influence.",
        )}
      </p>
      <div className="metrics-grid">
        <Metric
          label={t("Decoder cosine")}
          value={format(comparison.decoder_cosine)}
        />
        <Metric
          label={t("Pearson correlation")}
          value={format(comparison.pearson_correlation)}
        />
        <Metric
          label={t("Activation cosine")}
          value={format(comparison.activation_cosine)}
        />
        <Metric
          label={t("Activation overlap")}
          value={
            comparison.activation_overlap === null
              ? unavailable
              : number(comparison.activation_overlap * 100, 2) + "%"
          }
          note={t("Active intersection divided by active union")}
        />
      </div>
      <p className="muted">
        {t(
          "Both features activate at {v0} of {v1, plural, one {# eligible position} other {# eligible positions}}.",
          {
            v0: comparison.coactive_count,
            v1: comparison.eligible_position_count,
          },
        )}{" "}
        {t(
          "Correlation is unavailable for fewer than two eligible positions or constant activity.",
        )}
      </p>
      {comparison.eligible_position_count > 0 ? (
        <>
          <SeriesChart
            data={values}
            xKey="position"
            interpolation="linear"
            showPoints
            height={220}
            series={[
              {
                key: "first",
                label: t("Feature {v0}", { v0: comparison.feature_ids[0] }),
                color: "var(--blue)",
              },
              {
                key: "second",
                label: t("Feature {v0}", { v0: comparison.feature_ids[1] }),
                color: "var(--orange)",
              },
            ]}
          />
          <p className="feature-pair-axis">{t("Token position")}</p>
        </>
      ) : (
        <p className="muted">
          {t("No eligible token positions in this prompt.")}
        </p>
      )}
    </section>
  );
}
