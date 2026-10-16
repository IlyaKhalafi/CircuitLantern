"use client";

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
      <h3 className="target-trajectory-title">Chosen target scores</h3>
      <div className="comparison-controls">
        <label>
          Tracked target
          <select
            aria-label="Tracked target"
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
          Reference target
          <select
            aria-label="Reference target"
            disabled={targets.length < 2}
            value={reference}
            onChange={(event) => setReference(+event.target.value)}
          >
            <option value={-1}>No reference</option>
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
        Scores follow prompt position {position}. Target text is resolved when
        capturing. Requested target{" "}
        {JSON.stringify(targets[selected].requested_text)}
        {compared &&
          " · requested reference " +
            JSON.stringify(targets[reference].requested_text)}
        .
      </p>
      {unavailable && (
        <p className="inline-note" role="status">
          Some target scores at this position are unavailable. Missing values
          remain gaps.
        </p>
      )}
      <div className="two-column target-trajectory-charts">
        <Panel
          title="Target log probability"
          subtitle="Full vocabulary diagnostic probabilities shown in natural log units."
        >
          <SeriesChart
            data={rows}
            series={[
              {
                key: "selected",
                label: targetLabel(targets[selected]),
                color: "#d57b43",
              },
              ...(compared
                ? [
                    {
                      key: "reference",
                      label: targetLabel(targets[reference]),
                      color: "#97adc6",
                    },
                  ]
                : []),
            ]}
          />
        </Panel>
        <Panel
          title="Target logit difference"
          subtitle="Selected target minus reference at the same layer and prompt position."
        >
          {compared ? (
            <>
              <SeriesChart
                data={rows}
                series={[
                  {
                    key: "difference",
                    label: "Logit difference",
                    color: "#809cbb",
                  },
                ]}
              />
              <p className="inline-note">
                Positive values favor the selected target. Negative values favor
                the reference.
              </p>
            </>
          ) : (
            <p className="inline-note">
              {targets.length < 2
                ? "Track a second token when capturing to compare alternatives, then choose it as the reference."
                : "Choose a reference target to compare two captured alternatives."}
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
        <summary>Inspect target scores</summary>
        <div
          className="table-wrap"
          role="region"
          aria-label="Tracked target measurements"
          tabIndex={0}
        >
          <table className="target-trajectory-table">
            <caption className="sr-only">
              Captured target log probabilities at prompt position {position}
            </caption>
            <thead>
              <tr>
                <th>CAPTURE</th>
                <th>SELECTED LOG PROBABILITY</th>
                <th>REFERENCE LOG PROBABILITY</th>
                <th>LOGIT DIFFERENCE</th>
                <th>SELECTED PROBABILITY</th>
                <th>REFERENCE PROBABILITY</th>
                {hasRanks && (
                  <>
                    <th>SELECTED RANK</th>
                    <th>REFERENCE RANK</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.index}>
                  <th scope="row">{row.label}</th>
                  <td>{number(row.selected, 5)}</td>
                  <td>{number(compared ? row.reference : null, 5)}</td>
                  <td>{number(compared ? row.difference : null, 5)}</td>
                  <td>{probability(row.selected)}</td>
                  <td>{probability(compared ? row.reference : null)}</td>
                  {hasRanks && (
                    <>
                      <td>{number(row.selectedRank, 0)}</td>
                      <td>{number(compared ? row.referenceRank : null, 0)}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="inline-note">
          Differences subtract stored log probabilities and equal diagnostic
          logit differences up to floating point error. These intermediate
          projections are not calibrated confidence.
        </p>
        {trace.target_score_method && (
          <p className="inline-note">{trace.target_score_method}</p>
        )}
      </details>
    </div>
  );
}
