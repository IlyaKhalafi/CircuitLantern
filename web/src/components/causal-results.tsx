"use client";
import { useI18n } from "@/i18n/provider";
import { SeriesChart } from "@/components/charts";
import { Metric, Panel } from "@/components/ui";
import { number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import type { CausalResult } from "@/lib/types";
export function CausalResults({ result }: { result: CausalResult }) {
  const { t, rich } = useI18n();
  const partial =
    result.complete === false ||
    result.cancelled ||
    !!result.error ||
    result.layers.length !== result.total_layers;
  return (
    <>
      <div className="trace-meta">
        <span className={`pill ${partial ? "orange" : "green"}`}>
          {partial ? t("Partial sweep") : t("Sweep complete")}
        </span>
        <span className="trace-source">{result.model_id}</span>
        {result.cancelled && <span>{t("Cancelled between forwards")}</span>}
        {partial && (
          <span>
            {rich("{v0} of {v1} blocks completed", {
              v0: result.layers.length,
              v1: result.total_layers,
            })}
          </span>
        )}
        <span>
          {rich("{v0} ms", {
            v0: number(result.duration_ms, 0),
          })}
        </span>
        <span>
          {rich("revision {v0}", {
            v0: result.revision.slice(0, 8),
          })}
        </span>
        <span>
          {t(
            executionLabel(result.execution_precision, result.execution_config),
          )}
        </span>
        {result.device && <span>{result.device.toUpperCase()}</span>}
      </div>
      {result.error && (
        <div className="notice error" role="alert">
          {rich(
            "{v0} {v1} Completed blocks and baseline scores remain available for inspection, saving and export.",
            {
              v0:
                result.failed_layer == null
                  ? t("The sweep stopped.")
                  : t("The sweep stopped at block {v0}.", {
                      v0: result.failed_layer,
                    }),
              v1: t(result.error),
            },
          )}
        </div>
      )}
      <div className="metrics-grid">
        <Metric
          label={t("Target token")}
          value={
            <span className="target-token">{result.target_token || "∅"}</span>
          }
          note={t("Vocabulary ID {v0}", {
            v0: result.target_token_id,
          })}
        />
        <Metric
          label={t("Clean probability")}
          value={`${number(result.clean.probability * 100, 3)}%`}
          note={t("Log probability {v0}", {
            v0: number(result.clean.log_probability, 4),
          })}
        />
        <Metric
          label={t("Corrupted probability")}
          value={`${number(result.corrupted.probability * 100, 3)}%`}
          note={t("Log probability {v0}", {
            v0: number(result.corrupted.log_probability, 4),
          })}
        />
        <Metric
          label={t("Blocks tested")}
          value={result.layers.length}
          note={
            result.position === null
              ? t("All token positions restored")
              : result.position === -1
                ? t("Last token position restored")
                : t("Position {v0} restored", {
                    v0: result.position,
                  })
          }
        />
      </div>
      {result.position === null && (
        <div className="notice info">
          {t(
            "All position restoration is a sanity check. Later layers receive the complete clean residual stream. When the clean to corrupted gap permits normalized recovery, values near one are expected. Use a single token position to investigate localization.",
          )}
        </div>
      )}
      {result.recovery_warning && (
        <div className="notice info">{result.recovery_warning}</div>
      )}
      {!result.recovery_defined && (
        <div className="notice info">
          {rich(
            "The corrupted prompt did not lower the target log probability by more than {v0} nats. Normalized recovery is undefined. The raw log probability change remains available.",
            {
              v0: result.recovery_minimum_gap,
            },
          )}
        </div>
      )}
      <div className="two-column">
        <Panel
          title={
            result.position === null
              ? t("All position restoration check")
              : result.recovery_defined
                ? t("Recovery by transformer block")
                : t("Target log probability change")
          }
          subtitle={
            result.recovery_defined
              ? t(
                  "One is clean baseline recovery. Zero is no change from the corrupted baseline.",
                )
              : t(
                  "Change from the corrupted baseline in nats. Zero means no change.",
                )
          }
        >
          {result.layers.length ? (
            <SeriesChart
              data={result.layers.map((l) => ({
                index: l.layer,
                recovery: l.recovery,
                delta: l.delta_log_probability,
              }))}
              series={[
                {
                  key: result.recovery_defined ? "recovery" : "delta",
                  label: result.recovery_defined
                    ? "Normalized recovery"
                    : "Log probability change",
                  color: "var(--accent)",
                },
              ]}
            />
          ) : (
            <p className="inline-note">
              {t(
                "No patch measurements completed. Baseline scores remain available above.",
              )}
            </p>
          )}
          <p className="inline-note">
            {result.recovery_defined
              ? t(result.method)
              : t(
                  "Each point is the patched target log probability minus the corrupted baseline. Negative values mean the patch reduced the target probability. These values are not divided by the clean to corrupted score gap.",
                )}
          </p>
        </Panel>
        <Panel
          title={t("Token alignment")}
          subtitle={t(
            "Columns preserve the positions used in the patching intervention.",
          )}
        >
          <div
            className="alignment-table table-wrap"
            role="region"
            aria-label={t("Token alignment")}
            tabIndex={0}
          >
            <table>
              <thead>
                <tr>
                  <th>{t("POSITION")}</th>
                  <th>{t("CLEAN TOKEN")}</th>
                  <th>{t("CORRUPTED TOKEN")}</th>
                </tr>
              </thead>
              <tbody>
                {result.tokens.map((token, i) => (
                  <tr key={i}>
                    <td className="code">{i}</td>
                    <td>
                      <span className="lens-token">{token || "∅"}</span>
                    </td>
                    <td>
                      <span className="lens-token">
                        {result.corrupted_tokens[i] || "∅"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
      <Panel
        title={t("Per block patching results")}
        subtitle={t(
          "Each row is an independent forward pass with exactly one block patched.",
        )}
      >
        <div
          className="table-wrap"
          role="region"
          aria-label={t("Per block patching results")}
          tabIndex={0}
        >
          <table className="causal-results-table">
            <thead>
              <tr>
                <th>{t("BLOCK")}</th>
                <th>{t("TARGET PROBABILITY")}</th>
                <th>{t("LOG PROBABILITY")}</th>
                <th>{t("CHANGE FROM CORRUPTED")}</th>
                <th>{t("NORMALIZED RECOVERY")}</th>
              </tr>
            </thead>
            <tbody>
              {result.layers.map((layer) => (
                <tr key={layer.layer}>
                  <td className="code">
                    {rich("Layer {v0}", {
                      v0: layer.layer,
                    })}
                  </td>
                  <td className="code">
                    {t(number(layer.probability * 100, 4))}%
                  </td>
                  <td className="code">
                    {t(number(layer.log_probability, 5))}
                  </td>
                  <td className="code">
                    {t(number(layer.delta_log_probability, 5))}
                  </td>
                  <td className="code">{t(number(layer.recovery, 4))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <details className="capture-details">
        <summary>{t("Captured prompts")}</summary>
        <p className="inline-note">{t("Clean prompt")}</p>
        <p className="trace-output">{result.clean_prompt}</p>
        <p className="inline-note">{t("Corrupted prompt")}</p>
        <p className="trace-output">{result.corrupted_prompt}</p>
      </details>
    </>
  );
}
