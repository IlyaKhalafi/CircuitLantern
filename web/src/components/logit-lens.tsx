"use client";

import { Panel } from "@/components/ui";
import { TargetTrajectories } from "@/components/target-trajectories";
import { number } from "@/lib/api";
import type { LayerTrace, Trace } from "@/lib/types";

function measurement(values: unknown, position: number, maximum = Infinity) {
  const value = Array.isArray(values) ? values[position] : null;
  return typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= maximum
    ? value
    : null;
}

function positionValues(layer: LayerTrace, position: number) {
  const id = measurement(layer.token_prediction_ids, position);
  return {
    id: id !== null && Number.isSafeInteger(id) ? id : null,
    probability: measurement(layer.token_prediction_probabilities, position, 1),
    entropy: measurement(layer.token_entropies, position),
    norm: measurement(layer.token_norms, position),
  };
}

function probability(value: number | null) {
  return value === null ? "N/A" : number(value * 100, 4) + "%";
}

export function LogitLens({
  trace,
  position,
  onPositionChange,
}: {
  trace: Trace;
  position: number;
  onPositionChange: (position: number) => void;
}) {
  const selected = Math.max(0, Math.min(position, trace.tokens.length - 1));
  const perPosition =
    !!trace.token_prediction_method ||
    trace.layers.some(
      (layer) =>
        Array.isArray(layer.token_prediction_ids) ||
        Array.isArray(layer.token_prediction_probabilities) ||
        Array.isArray(layer.token_entropies),
    );
  const incomplete =
    perPosition &&
    trace.layers.some((layer) => {
      const values = positionValues(layer, selected);
      return (
        values.id === null ||
        values.probability === null ||
        values.entropy === null ||
        values.norm === null
      );
    });
  return (
    <Panel
      title="Watch predictions take shape"
      subtitle="Project intermediate residual states through the final norm and unembedding."
    >
      <div className="attention-controls">
        <label className="checkbox-row">
          Token position{" "}
          <select
            aria-label="Logit lens token position"
            value={selected}
            onChange={(event) => onPositionChange(+event.target.value)}
          >
            {trace.tokens.map((token, index) => (
              <option key={index} value={index}>
                {index} · {token || "∅"}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="inline-note" role="status">
        Prompt position {selected} · token ID {trace.token_ids[selected]}.{" "}
        {selected === trace.tokens.length - 1
          ? "Predictions describe the next token after the prompt."
          : "Predictions use the causal prefix ending at this token."}
      </p>
      <TargetTrajectories trace={trace} position={selected} />
      {!perPosition && (
        <p className="inline-note">
          This older capture stores confidence and entropy for the final prompt
          token. Those columns remain labeled separately from the selected
          position.
        </p>
      )}
      {incomplete && (
        <p className="inline-note">
          Some measurements at this position are unavailable in this capture.
        </p>
      )}
      <div
        className="table-wrap"
        role="region"
        aria-label="Logit lens predictions"
        tabIndex={0}
      >
        <table
          className={perPosition ? "position-lens-table" : "legacy-lens-table"}
        >
          <caption className="sr-only">
            Logit lens measurements at prompt position {selected}
          </caption>
          <thead>
            <tr>
              <th>CAPTURE</th>
              <th>POSITION PREDICTION</th>
              {perPosition ? (
                <>
                  <th>PREDICTED ID</th>
                  <th>PROBABILITY</th>
                  <th>ENTROPY IN NATS</th>
                  <th>RESIDUAL NORM</th>
                </>
              ) : (
                <>
                  <th>LAST TOKEN TOP PREDICTION</th>
                  <th>LAST TOKEN PROBABILITY</th>
                  <th>LAST TOKEN ENTROPY</th>
                  <th>LAST TOKEN COSINE</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {trace.layers.map((layer) => {
              const values = positionValues(layer, selected);
              const prediction = layer.token_predictions[selected];
              return (
                <tr key={layer.layer}>
                  <th scope="row" className="code">
                    {layer.label}
                  </th>
                  <td>
                    <span
                      className="lens-token"
                      title={
                        values.id === null
                          ? undefined
                          : "Vocabulary ID " + values.id
                      }
                    >
                      {prediction === "" ? "∅" : (prediction ?? "Unavailable")}
                    </span>
                  </td>
                  {perPosition ? (
                    <>
                      <td className="code">{values.id ?? "N/A"}</td>
                      <td className="code">
                        {probability(values.probability)}
                      </td>
                      <td className="code">{number(values.entropy, 5)}</td>
                      <td className="code">{number(values.norm, 5)}</td>
                    </>
                  ) : (
                    <>
                      <td>
                        <span className="lens-token">
                          {layer.predictions[0]?.token ?? "Unavailable"}
                        </span>
                      </td>
                      <td className="code">
                        {probability(layer.predictions[0]?.probability ?? null)}
                      </td>
                      <td className="code">{number(layer.entropy)}</td>
                      <td className="code">
                        {number(layer.cosine_similarity, 4)}
                      </td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {perPosition && (
        <>
          <p className="inline-note">
            Prediction, vocabulary ID, probability, entropy and residual norm
            follow the selected prompt position. Residual norms use the recorded
            capture states.
            {trace.final_state_normalized &&
              " The final layer residual was captured after final normalization."}
          </p>
          <details className="comparison-values">
            <summary>Inspect final token transitions</summary>
            <p className="inline-note">
              These transitions always describe the final prompt token,
              including when another position is selected above.
            </p>
            <div
              className="table-wrap"
              role="region"
              aria-label="Final token transitions"
              tabIndex={0}
            >
              <table className="lens-transitions-table">
                <caption className="sr-only">
                  Final prompt token residual transitions between captured
                  layers
                </caption>
                <thead>
                  <tr>
                    <th>CAPTURE</th>
                    <th>COSINE TO PREVIOUS</th>
                    <th>UPDATE NORM</th>
                    <th>UPDATE TO INPUT NORM</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.layers.map((layer) => (
                    <tr key={layer.layer}>
                      <th scope="row" className="code">
                        {layer.label}
                      </th>
                      <td className="code">
                        {number(layer.cosine_similarity, 5)}
                      </td>
                      <td className="code">{number(layer.update_norm, 5)}</td>
                      <td className="code">
                        {number(layer.relative_update, 5)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
      <p className="inline-note">{trace.method}</p>
      {trace.token_prediction_method && (
        <p className="inline-note">{trace.token_prediction_method}</p>
      )}
    </Panel>
  );
}
