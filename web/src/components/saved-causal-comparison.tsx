"use client";

import { Download } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { SeriesChart } from "@/components/charts";
import { CausalResults } from "@/components/causal-results";
import { SavedReferencePicker } from "@/components/saved-reference-picker";
import { ErrorNotice, Loading, Panel } from "@/components/ui";
import { downloadJSON, number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import { useExperiment } from "@/lib/use-experiment";
import {
  causalComparisonProblem,
  causalDifferences,
  hasCausalComparisonMeasurements,
  patchPositionLabel,
} from "@/lib/causal-comparison";
import type { CausalResult, Experiment } from "@/lib/types";

const series = [
  { key: "selected", label: "Selected sweep", color: "#d57b43" },
  { key: "reference", label: "Reference sweep", color: "#97adc6" },
];

function Source({
  label,
  record,
  result,
}: {
  label: string;
  record: Experiment;
  result: CausalResult;
}) {
  const partial =
    result.complete === false ||
    result.cancelled ||
    !!result.error ||
    result.layers.length !== result.total_layers;
  return (
    <div className="matrix-comparison-source">
      <strong>
        {label} · {record.name}
      </strong>
      <p>{patchPositionLabel(result)}</p>
      <p>
        {partial ? "Partial" : "Complete"} · {result.layers.length} of{" "}
        {result.total_layers} blocks measured
      </p>
      <p>
        {result.model_id} · revision {result.revision}
      </p>
      <p>
        {executionLabel(result.execution_precision, result.execution_config)} ·{" "}
        {result.device?.toUpperCase()}
      </p>
      <p>Model session {result.execution_id ?? "unavailable"}</p>
      {result.error && (
        <p>
          Stopped
          {result.failed_layer == null
            ? ""
            : " at block " + result.failed_layer}{" "}
          · {result.error}
        </p>
      )}
      {result.cancelled && <p>Cancelled between forwards</p>}
    </div>
  );
}

export function SavedCausalComparison({
  experiment,
  result,
  experiments,
  libraryLoading,
  libraryError,
  onRetryLibrary,
}: {
  experiment: Experiment;
  result: CausalResult;
  experiments: Experiment[];
  libraryLoading: boolean;
  libraryError: string | null;
  onRetryLibrary: () => void;
}) {
  const params = useSearchParams();
  const referenceId = params.get("compare");
  const { selected: reference, loading, error } = useExperiment(referenceId);
  const candidates = experiments.filter(
    (item) => item.id !== experiment.id && item.kind === "causal",
  );
  const referenceResult =
    reference?.kind === "causal" ? reference.result : undefined;
  const problem = reference
    ? reference.id === experiment.id
      ? "Choose a different saved residual patch sweep as the reference."
      : !referenceResult
        ? "The reference is not a residual patch sweep. Choose saved residual patching evidence."
        : causalComparisonProblem(result, referenceResult)
    : null;
  const comparable = referenceResult && !problem ? referenceResult : null;
  const differences = comparable ? causalDifferences(result, comparable) : [];
  return (
    <>
      <Panel
        title="Compare residual patches"
        subtitle="Compare saved patch positions using the same prompt pair and baseline scores."
        className="trace-comparison"
      >
        <SavedReferencePicker
          label="Reference patch sweep"
          referenceId={referenceId}
          reference={reference}
          candidates={candidates}
          libraryLoading={libraryLoading}
          libraryError={libraryError}
          onRetryLibrary={onRetryLibrary}
          emptyText="Save another residual patch sweep with the same prompt pair and target to compare positions here."
        >
          {comparable && reference && (
            <button
              className="button secondary"
              onClick={() =>
                downloadJSON(
                  {
                    format: "circuit-lantern-causal-comparison-v1",
                    convention:
                      "Selected minus reference. Raw patch effects use target log probability changes from the shared corrupted baseline in nats. Normalized recovery uses the shared clean to corrupted gap. Unmeasured blocks remain null. Patch positions may differ.",
                    selected: experiment,
                    reference,
                    layers: differences,
                  },
                  "patch-comparison-" +
                    experiment.id.slice(0, 8) +
                    "-" +
                    reference.id.slice(0, 8) +
                    ".json",
                )
              }
            >
              <Download size={14} /> Export comparison
            </button>
          )}
        </SavedReferencePicker>
        {loading && <Loading label="Opening reference patch sweep" />}
        <ErrorNotice message={error || problem} />
        {reference &&
          referenceResult &&
          hasCausalComparisonMeasurements(referenceResult) && (
            <div className="matrix-comparison-sources">
              <Source label="Selected" record={experiment} result={result} />
              <Source
                label="Reference"
                record={reference}
                result={referenceResult}
              />
            </div>
          )}
        {comparable && (
          <>
            <p className="inline-note" role="status">
              Checkpoint, prompt vocabulary IDs, target, execution settings and
              baseline scores match. Patch positions may differ.
              {result.execution_id !== comparable.execution_id &&
                " The captures come from different model sessions."}
            </p>
            <p className="inline-note">
              {
                differences.filter((row) => row.delta_difference !== null)
                  .length
              }{" "}
              of {result.total_layers} blocks have measurements in both
              captures. Missing blocks are unavailable and are not treated as
              zero.
            </p>
            {(result.position === null || comparable.position === null) && (
              <p className="inline-note">
                Restoring all token positions is a sanity check. It does not
                localize a signal to one position.
              </p>
            )}
          </>
        )}
      </Panel>
      {comparable && (
        <>
          <div className="two-column">
            <Panel
              title="Patch effect comparison"
              subtitle="Target log probability change from the shared corrupted baseline in nats."
            >
              <SeriesChart
                data={differences.map((row) => ({
                  index: row.layer,
                  selected: row.selected_delta,
                  reference: row.reference_delta,
                }))}
                series={series}
              />
            </Panel>
            <Panel
              title="Recovery comparison"
              subtitle="One is clean baseline recovery. Zero is no change from the corrupted baseline."
            >
              {result.recovery_defined ? (
                <SeriesChart
                  data={differences.map((row) => ({
                    index: row.layer,
                    selected: row.selected_recovery,
                    reference: row.reference_recovery,
                  }))}
                  series={series}
                />
              ) : (
                <p className="inline-note">
                  Normalized recovery is undefined for this prompt pair. The raw
                  log probability changes remain comparable.
                </p>
              )}
            </Panel>
          </div>
          <Panel
            title="Patch measurement changes"
            subtitle="Selected minus reference, aligned by transformer block."
          >
            <details className="comparison-values">
              <summary>Inspect per block changes</summary>
              <p className="inline-note">
                Raw changes use nats. Normalized recovery changes use the shared
                baseline gap. Missing measurements remain unavailable.
              </p>
              <div
                className="table-wrap"
                role="region"
                aria-label="Patch comparison measurements"
                tabIndex={0}
              >
                <table className="causal-comparison-table">
                  <caption className="sr-only">
                    Selected and reference patch effects for each transformer
                    block
                  </caption>
                  <thead>
                    <tr>
                      <th>BLOCK</th>
                      <th>SELECTED EFFECT</th>
                      <th>REFERENCE EFFECT</th>
                      <th>EFFECT CHANGE</th>
                      <th>SELECTED RECOVERY</th>
                      <th>REFERENCE RECOVERY</th>
                      <th>RECOVERY CHANGE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {differences.map((row) => (
                      <tr key={row.layer}>
                        <th scope="row">{row.layer}</th>
                        <td>{number(row.selected_delta, 5)}</td>
                        <td>{number(row.reference_delta, 5)}</td>
                        <td>{number(row.delta_difference, 5)}</td>
                        <td>{number(row.selected_recovery, 5)}</td>
                        <td>{number(row.reference_recovery, 5)}</td>
                        <td>{number(row.recovery_difference, 5)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </Panel>
        </>
      )}
      <CausalResults result={result} />
    </>
  );
}
