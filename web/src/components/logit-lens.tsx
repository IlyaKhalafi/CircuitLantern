"use client";
import { useI18n } from "@/i18n/provider";
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
  const { t, rich } = useI18n();
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
      title={t("Watch predictions take shape")}
      subtitle={t(
        "Project intermediate residual states through the final norm and unembedding.",
      )}
    >
      <div className="attention-controls">
        <label className="checkbox-row">
          {t("Token position")}
          <select
            aria-label={t("Logit lens token position")}
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
        {rich("Prompt position {v0} \u00B7 token ID {v1}. {v2}", {
          v0: selected,
          v1: trace.token_ids[selected],
          v2:
            selected === trace.tokens.length - 1
              ? t("Predictions describe the next token after the prompt.")
              : t("Predictions use the causal prefix ending at this token."),
        })}
      </p>
      <TargetTrajectories trace={trace} position={selected} />
      {!perPosition && (
        <p className="inline-note">
          {t(
            "This older capture stores confidence and entropy for the final prompt token. Those columns remain labeled separately from the selected position.",
          )}
        </p>
      )}
      {incomplete && (
        <p className="inline-note">
          {t(
            "Some measurements at this position are unavailable in this capture.",
          )}
        </p>
      )}
      <div
        className="table-wrap"
        role="region"
        aria-label={t("Logit lens predictions")}
        tabIndex={0}
      >
        <table
          className={perPosition ? "position-lens-table" : "legacy-lens-table"}
        >
          <caption className="sr-only">
            {rich("Logit lens measurements at prompt position {v0}", {
              v0: selected,
            })}
          </caption>
          <thead>
            <tr>
              <th>{t("CAPTURE")}</th>
              <th>{t("POSITION PREDICTION")}</th>
              {perPosition ? (
                <>
                  <th>{t("PREDICTED ID")}</th>
                  <th>{t("PROBABILITY")}</th>
                  <th>{t("ENTROPY IN NATS")}</th>
                  <th>{t("RESIDUAL NORM")}</th>
                </>
              ) : (
                <>
                  <th>{t("LAST TOKEN TOP PREDICTION")}</th>
                  <th>{t("LAST TOKEN PROBABILITY")}</th>
                  <th>{t("LAST TOKEN ENTROPY")}</th>
                  <th>{t("LAST TOKEN COSINE")}</th>
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
                    {t(layer.label)}
                  </th>
                  <td>
                    <span
                      className="lens-token"
                      title={
                        values.id === null
                          ? undefined
                          : t("Vocabulary ID {v0}", {
                              v0: values.id,
                            })
                      }
                    >
                      {prediction === "" ? "∅" : (prediction ?? "Unavailable")}
                    </span>
                  </td>
                  {perPosition ? (
                    <>
                      <td className="code">{values.id ?? "N/A"}</td>
                      <td className="code">
                        {t(probability(values.probability))}
                      </td>
                      <td className="code">{t(number(values.entropy, 5))}</td>
                      <td className="code">{t(number(values.norm, 5))}</td>
                    </>
                  ) : (
                    <>
                      <td>
                        <span className="lens-token">
                          {layer.predictions[0]?.token ?? t("Unavailable")}
                        </span>
                      </td>
                      <td className="code">
                        {t(
                          probability(
                            layer.predictions[0]?.probability ?? null,
                          ),
                        )}
                      </td>
                      <td className="code">{t(number(layer.entropy))}</td>
                      <td className="code">
                        {t(number(layer.cosine_similarity, 4))}
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
            {rich(
              "Prediction, vocabulary ID, probability, entropy and residual norm follow the selected prompt position. Residual norms use the recorded capture states.{v0}",
              {
                v0:
                  trace.final_state_normalized &&
                  t(
                    " The final layer residual was captured after final normalization.",
                  ),
              },
            )}
          </p>
          <details className="comparison-values">
            <summary>{t("Inspect final token transitions")}</summary>
            <p className="inline-note">
              {t(
                "These transitions always describe the final prompt token, including when another position is selected above.",
              )}
            </p>
            <div
              className="table-wrap"
              role="region"
              aria-label={t("Final token transitions")}
              tabIndex={0}
            >
              <table className="lens-transitions-table">
                <caption className="sr-only">
                  {t(
                    "Final prompt token residual transitions between captured layers",
                  )}
                </caption>
                <thead>
                  <tr>
                    <th>{t("CAPTURE")}</th>
                    <th>{t("COSINE TO PREVIOUS")}</th>
                    <th>{t("UPDATE NORM")}</th>
                    <th>{t("UPDATE TO INPUT NORM")}</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.layers.map((layer) => (
                    <tr key={layer.layer}>
                      <th scope="row" className="code">
                        {t(layer.label)}
                      </th>
                      <td className="code">
                        {t(number(layer.cosine_similarity, 5))}
                      </td>
                      <td className="code">
                        {t(number(layer.update_norm, 5))}
                      </td>
                      <td className="code">
                        {t(number(layer.relative_update, 5))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
      <p className="inline-note">{t(trace.method)}</p>
      {trace.token_prediction_method && (
        <p className="inline-note">{t(trace.token_prediction_method)}</p>
      )}
    </Panel>
  );
}
