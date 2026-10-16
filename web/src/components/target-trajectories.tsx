"use client";
import { useI18n } from "@/i18n/provider";
import { useState } from "react";
import { SeriesChart } from "@/components/charts";
import { Panel } from "@/components/ui";
import { TargetRanks } from "@/components/target-ranks";
import { number } from "@/lib/api";
import {
  targetLabel,
  targetTrajectory,
  trackedTargets,
} from "@/lib/target-trajectories";
import type { Trace } from "@/lib/types";
function probability(score: number | null) {
  return score === null ? "N/A" : number(Math.exp(score) * 100, 4) + "%";
}
export function TargetTrajectories({
  trace,
  position,
}: {
  trace: Trace;
  position: number;
}) {
  const { t, rich } = useI18n();
  const targets = trackedTargets(trace);
  const [selected, setSelected] = useState(0);
  const [reference, setReference] = useState(targets.length > 1 ? 1 : -1);
  if (!targets.length) return null;
  const rows = targetTrajectory(trace, position, selected, reference);
  const hasRanks =
    !!trace.target_rank_method ||
    trace.layers.some((layer) => Array.isArray(layer.token_target_ranks));
  const compared =
    reference >= 0 && reference !== selected && !!targets[reference];
  const unavailable = rows.some(
    (row) => row.selected === null || (compared && row.reference === null),
  );
  return (
    <div className="target-trajectories">
      <h3 className="target-trajectory-title">{t("Chosen target scores")}</h3>
      <div className="comparison-controls">
        <label>
          {t("Tracked target")}
          <select
            aria-label={t("Tracked target")}
            value={selected}
            onChange={(event) => {
              const next = +event.target.value;
              if (next === reference) setReference(selected);
              setSelected(next);
            }}
          >
            {targets.map((target, index) => (
              <option key={target.token_id} value={index}>
                {targetLabel(target)}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t("Reference target")}
          <select
            aria-label={t("Reference target")}
            disabled={targets.length < 2}
            value={reference}
            onChange={(event) => setReference(+event.target.value)}
          >
            <option value={-1}>{t("No reference")}</option>
            {targets.map(
              (target, index) =>
                index !== selected && (
                  <option key={target.token_id} value={index}>
                    {targetLabel(target)}
                  </option>
                ),
            )}
          </select>
        </label>
      </div>
      <p className="inline-note">
        {rich(
          "Scores follow prompt position {v0}. Target text is resolved when capturing. Requested target {v1}{v2}.",
          {
            v0: position,
            v1: JSON.stringify(targets[selected].requested_text),
            v2:
              compared &&
              " · requested reference " +
                JSON.stringify(targets[reference].requested_text),
          },
        )}
      </p>
      {unavailable && (
        <p className="inline-note" role="status">
          {t(
            "Some target scores at this position are unavailable. Missing values remain gaps.",
          )}
        </p>
      )}
      <div className="two-column target-trajectory-charts">
        <Panel
          title={t("Target log probability")}
          subtitle={t(
            "Full vocabulary diagnostic probabilities shown in natural log units.",
          )}
        >
          <SeriesChart
            data={rows}
            series={[
              {
                key: "selected",
                label: targetLabel(targets[selected]),
                color: "var(--accent)",
              },
              ...(compared
                ? [
                    {
                      key: "reference",
                      label: targetLabel(targets[reference]),
                      color: "var(--blue)",
                    },
                  ]
                : []),
            ]}
          />
        </Panel>
        <Panel
          title={t("Target logit difference")}
          subtitle={t(
            "Selected target minus reference at the same layer and prompt position.",
          )}
        >
          {compared ? (
            <>
              <SeriesChart
                data={rows}
                series={[
                  {
                    key: "difference",
                    label: "Logit difference",
                    color: "var(--blue)",
                  },
                ]}
              />
              <p className="inline-note">
                {t(
                  "Positive values favor the selected target. Negative values favor the reference.",
                )}
              </p>
            </>
          ) : (
            <p className="inline-note">
              {targets.length < 2
                ? t(
                    "Track a second token when capturing to compare alternatives, then choose it as the reference.",
                  )
                : t(
                    "Choose a reference target to compare two captured alternatives.",
                  )}
            </p>
          )}
        </Panel>
      </div>
      <TargetRanks
        rows={rows}
        selectedLabel={targetLabel(targets[selected])}
        referenceLabel={compared ? targetLabel(targets[reference]) : undefined}
        recorded={hasRanks}
        method={trace.target_rank_method}
      />
      <details className="comparison-values">
        <summary>{t("Inspect target scores")}</summary>
        <div
          className="table-wrap"
          role="region"
          aria-label={t("Tracked target measurements")}
          tabIndex={0}
        >
          <table className="target-trajectory-table">
            <caption className="sr-only">
              {rich(
                "Captured target log probabilities at prompt position {v0}",
                {
                  v0: position,
                },
              )}
            </caption>
            <thead>
              <tr>
                <th>{t("CAPTURE")}</th>
                <th>{t("SELECTED LOG PROBABILITY")}</th>
                <th>{t("REFERENCE LOG PROBABILITY")}</th>
                <th>{t("LOGIT DIFFERENCE")}</th>
                <th>{t("SELECTED PROBABILITY")}</th>
                <th>{t("REFERENCE PROBABILITY")}</th>
                {hasRanks && (
                  <>
                    <th>{t("SELECTED RANK")}</th>
                    <th>{t("REFERENCE RANK")}</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.index}>
                  <th scope="row">{t(row.label)}</th>
                  <td>{t(number(row.selected, 5))}</td>
                  <td>{t(number(compared ? row.reference : null, 5))}</td>
                  <td>{t(number(compared ? row.difference : null, 5))}</td>
                  <td>{t(probability(row.selected))}</td>
                  <td>{t(probability(compared ? row.reference : null))}</td>
                  {hasRanks && (
                    <>
                      <td>{t(number(row.selectedRank, 0))}</td>
                      <td>
                        {t(number(compared ? row.referenceRank : null, 0))}
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="inline-note">
          {t(
            "Differences subtract stored log probabilities and equal diagnostic logit differences up to floating point error. These intermediate projections are not calibrated confidence.",
          )}
        </p>
        {trace.target_score_method && (
          <p className="inline-note">{t(trace.target_score_method)}</p>
        )}
      </details>
    </div>
  );
}
