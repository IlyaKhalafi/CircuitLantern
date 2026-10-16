"use client";

import { SeriesChart } from "@/components/charts";
import { Metric, Panel } from "@/components/ui";
import { number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import type { CausalResult } from "@/lib/types";

export function CausalResults({ result }: { result: CausalResult }) {
  return (
    <>
      <div className="trace-meta">
        <span className={`pill ${result.cancelled ? "orange" : "green"}`}>
          {result.cancelled ? "Partial sweep" : "Sweep complete"}
        </span>
        <span className="trace-source">{result.model_id}</span>
        {result.cancelled && <span>Cancelled between forwards</span>}
        <span>{number(result.duration_ms, 0)} ms</span>
        <span>revision {result.revision.slice(0, 8)}</span>
        <span>
          {executionLabel(result.execution_precision, result.execution_config)}
        </span>
        {result.device && <span>{result.device.toUpperCase()}</span>}
      </div>
      <div className="metrics-grid">
        <Metric
          label="Target token"
          value={
            <span className="target-token">{result.target_token || "∅"}</span>
          }
          note={`Vocabulary ID ${result.target_token_id}`}
        />
        <Metric
          label="Clean probability"
          value={`${number(result.clean.probability * 100, 3)}%`}
          note={`Log probability ${number(result.clean.log_probability, 4)}`}
        />
        <Metric
          label="Corrupted probability"
          value={`${number(result.corrupted.probability * 100, 3)}%`}
          note={`Log probability ${number(result.corrupted.log_probability, 4)}`}
        />
        <Metric
          label="Blocks tested"
          value={result.layers.length}
          note={
            result.position === null
              ? "All token positions restored"
              : result.position === -1
                ? "Last token position restored"
                : `Position ${result.position} restored`
          }
        />
      </div>
      {result.position === null && (
        <div className="notice info">
          All position restoration is a sanity check. Later layers receive the
          complete clean residual stream. When the clean to corrupted gap
          permits normalized recovery, values near one are expected. Use a
          single token position to investigate localization.
        </div>
      )}
      {result.recovery_warning && (
        <div className="notice info">{result.recovery_warning}</div>
      )}
      {!result.recovery_defined && (
        <div className="notice info">
          The corrupted prompt did not lower the target log probability by more
          than {result.recovery_minimum_gap} nats. Normalized recovery is
          undefined. The raw log probability change remains available.
        </div>
      )}
      <div className="two-column">
        <Panel
          title={
            result.position === null
              ? "All position restoration check"
              : result.recovery_defined
                ? "Recovery by transformer block"
                : "Target log probability change"
          }
          subtitle={
            result.recovery_defined
              ? "One is clean baseline recovery. Zero is no change from the corrupted baseline."
              : "Change from the corrupted baseline in nats. Zero means no change."
          }
        >
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
                color: "#b68159",
              },
            ]}
          />
          <p className="inline-note">
            {result.recovery_defined
              ? result.method
              : "Each point is the patched target log probability minus the corrupted baseline. Negative values mean the patch reduced the target probability. These values are not divided by the clean to corrupted score gap."}
          </p>
        </Panel>
        <Panel
          title="Token alignment"
          subtitle="Columns preserve the positions used in the patching intervention."
        >
          <div className="alignment-table table-wrap">
            <table>
              <thead>
                <tr>
                  <th>POSITION</th>
                  <th>CLEAN TOKEN</th>
                  <th>CORRUPTED TOKEN</th>
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
        title="Per block patching results"
        subtitle="Each row is an independent forward pass with exactly one block patched."
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>BLOCK</th>
                <th>TARGET PROBABILITY</th>
                <th>LOG PROBABILITY</th>
                <th>CHANGE FROM CORRUPTED</th>
                <th>NORMALIZED RECOVERY</th>
              </tr>
            </thead>
            <tbody>
              {result.layers.map((layer) => (
                <tr key={layer.layer}>
                  <td className="code">Layer {layer.layer}</td>
                  <td className="code">
                    {number(layer.probability * 100, 4)}%
                  </td>
                  <td className="code">{number(layer.log_probability, 5)}</td>
                  <td className="code">
                    {number(layer.delta_log_probability, 5)}
                  </td>
                  <td className="code">{number(layer.recovery, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <details className="capture-details">
        <summary>Captured prompts</summary>
        <p className="inline-note">Clean prompt</p>
        <p className="trace-output">{result.clean_prompt}</p>
        <p className="inline-note">Corrupted prompt</p>
        <p className="trace-output">{result.corrupted_prompt}</p>
      </details>
    </>
  );
}
