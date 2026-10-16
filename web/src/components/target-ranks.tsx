"use client";

import { useState } from "react";
import { SeriesChart } from "@/components/charts";
import type { targetTrajectory } from "@/lib/target-trajectories";

export function TargetRanks({
  rows,
  selectedLabel,
  referenceLabel,
  recorded,
  method,
}: {
  rows: ReturnType<typeof targetTrajectory>;
  selectedLabel: string;
  referenceLabel?: string;
  recorded: boolean;
  method?: string;
}) {
  const [logarithmic, setLogarithmic] = useState(true);
  const missing = rows.some(
    (row) =>
      row.selectedRank === null ||
      (!!referenceLabel && row.referenceRank === null),
  );
  const maximum = Math.max(
    2,
    ...rows.flatMap((row) => [
      row.selectedRank ?? 1,
      referenceLabel ? (row.referenceRank ?? 1) : 1,
    ]),
  );
  const ticks = Array.from(
    new Set(
      logarithmic
        ? [
            1,
            ...Array.from(
              { length: Math.floor(Math.log10(maximum)) + 1 },
              (_, index) => 10 ** index,
            ).filter((value) => value < maximum / 1.5),
            maximum,
          ]
        : Array.from({ length: 5 }, (_, index) =>
            Math.round(1 + ((maximum - 1) * index) / 4),
          ),
    ),
  );
  return (
    <details className="comparison-values target-rank-details">
      <summary>Vocabulary rank</summary>
      {recorded ? (
        <>
          <div className="comparison-controls">
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={logarithmic}
                onChange={(event) => setLogarithmic(event.target.checked)}
              />
              Log rank scale
            </label>
          </div>
          <p className="inline-note">
            Rank 1 is most likely and appears at the top. Equal logits share a
            rank. Ranks compare each target against the complete vocabulary at
            the selected prompt position.
          </p>
          <SeriesChart
            data={rows}
            series={[
              { key: "selectedRank", label: selectedLabel, color: "#d57b43" },
              ...(referenceLabel
                ? [
                    {
                      key: "referenceRank",
                      label: referenceLabel,
                      color: "#97adc6",
                    },
                  ]
                : []),
            ]}
            yAxis={{
              scale: logarithmic ? "log" : "linear",
              reversed: true,
              domain: [1, maximum],
              integer: true,
              ticks,
            }}
          />
          {missing && (
            <p className="inline-note" role="status">
              Some target ranks at this position are unavailable. Missing values
              remain gaps.
            </p>
          )}
          {method && <p className="inline-note">{method}</p>}
        </>
      ) : (
        <p className="inline-note">
          This capture does not include vocabulary ranks. Run a new trace with
          tracked targets to record them. Its captured target scores remain
          available.
        </p>
      )}
    </details>
  );
}
