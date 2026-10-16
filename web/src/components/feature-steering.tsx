"use client";

import { useId, useState } from "react";
import { useI18n } from "@/i18n/provider";
import { Metric } from "@/components/ui";
import { number } from "@/lib/api";
import type { FeatureIntervention } from "@/lib/features";

const COEFFICIENT_LIMIT = 10000;

export function FeatureSteeringControls({
  busy,
  enabled,
  maximum,
  onApply,
}: {
  busy: boolean;
  enabled: boolean;
  maximum: number | null;
  onApply: (coefficient: number) => void;
}) {
  const { t } = useI18n();
  const [coefficient, setCoefficient] = useState("1");
  const rangeId = useId();
  const value = Number(coefficient);
  const valid =
    coefficient.trim() !== "" &&
    Number.isFinite(value) &&
    Math.abs(value) <= COEFFICIENT_LIMIT;
  return (
    <details className="feature-steering-controls">
      <summary>{t("Set a feature coefficient")}</summary>
      <div className="feature-steering-form">
        <p className="muted">
          {t(
            "A fixed coefficient can activate an inactive feature. Negative values reverse its decoded contribution.",
          )}
        </p>
        <p className="muted">
          {t("Values use normalized SAE coefficient units.")}
        </p>
        <div className="feature-toolbar">
          <label>
            {t("Fixed coefficient")}
            <input
              className="form-input"
              type="number"
              dir="ltr"
              step="any"
              min={-COEFFICIENT_LIMIT}
              max={COEFFICIENT_LIMIT}
              aria-label={t("Fixed coefficient")}
              aria-invalid={!valid}
              aria-describedby={rangeId}
              value={coefficient}
              disabled={busy}
              onChange={(event) => setCoefficient(event.target.value)}
            />
          </label>
          <button
            type="button"
            className="button secondary"
            disabled={busy || !enabled || !valid}
            onClick={() => onApply(value)}
          >
            {t("Apply fixed coefficient")}
          </button>
        </div>
        <div className="feature-steering-guidance">
          <p id={rangeId}>
            {t("Allowed coefficient range")}{" "}
            <bdi dir="ltr">
              [{-COEFFICIENT_LIMIT}, {COEFFICIENT_LIMIT}]
            </bdi>
          </p>
          {maximum !== null && (
            <p>
              {t("Recorded corpus maximum")}{" "}
              <bdi dir="ltr">{number(maximum, 4)}</bdi>
            </p>
          )}
        </div>
        <p className="muted">
          {t("Uses the intervention positions selected above.")}
        </p>
        <p className="muted">
          {t("Re-encoding can produce a different activation.")}
        </p>
      </div>
    </details>
  );
}

export function FeatureInterventionLabel({
  evidence,
}: {
  evidence: FeatureIntervention;
}) {
  const { t } = useI18n();
  return (
    <p className="inline-note feature-intervention-evidence">
      {t("Feature {v0}", { v0: evidence.feature })}
      {" · "}
      {evidence.mode === "set" ? (
        <>
          {t("Fixed coefficient")}{" "}
          <bdi dir="ltr">{String(evidence.coefficient)}</bdi>
        </>
      ) : (
        <>
          ×<bdi dir="ltr">{evidence.multiplier}</bdi>
        </>
      )}
      {" · "}
      {evidence.position === null ? (
        t("All eligible tokens")
      ) : (
        <>
          {t("Token position")} <bdi dir="ltr">{evidence.position}</bdi>
        </>
      )}
    </p>
  );
}

export function FeatureInterventionPrecision({
  evidence,
}: {
  evidence: FeatureIntervention;
}) {
  const { t } = useI18n();
  const rounding = evidence.rounding;
  if (!rounding) return null;
  return (
    <details className="feature-intervention-precision">
      <summary>{t("Intervention precision")}</summary>
      <div className="feature-precision-grid">
        <Metric
          label={t("Intended residual change norm")}
          value={number(rounding.intended_delta_norm, 5)}
        />
        <Metric
          label={t("Realized residual change norm")}
          value={number(rounding.realized_delta_norm, 5)}
        />
        <Metric
          label={t("Relative rounding error")}
          value={
            rounding.relative_rounding_error == null
              ? t("Unavailable")
              : number(rounding.relative_rounding_error, 5)
          }
        />
      </div>
      <p className="mono">{rounding.residual_dtype}</p>
    </details>
  );
}
