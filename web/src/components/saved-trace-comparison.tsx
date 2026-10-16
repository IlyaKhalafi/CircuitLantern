"use client";

import { Download } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { ErrorNotice, Loading, Panel } from "@/components/ui";
import { TraceResults } from "@/components/trace-results";
import { useExperiment } from "@/lib/use-experiment";
import { notebookHref } from "@/lib/experiments";
import { executionLabel } from "@/lib/execution";
import { interventionLabel } from "@/lib/interventions";
import { downloadJSON, number } from "@/lib/api";
import {
  hasComparisonMeasurements,
  traceComparisonProblem,
  traceDifferences,
} from "@/lib/trace-comparison";
import type { Experiment, Trace } from "@/lib/types";

export function SavedTraceComparison({
  experiment,
  trace,
  experiments,
  libraryLoading,
  libraryError,
  onRetryLibrary,
}: {
  experiment: Experiment;
  trace: Trace;
  experiments: Experiment[];
  libraryLoading: boolean;
  libraryError: string | null;
  onRetryLibrary: () => void;
}) {
  const params = useSearchParams();
  const referenceId = params.get("compare");
  const { selected: reference, loading, error } = useExperiment(referenceId);
  const candidates = experiments.filter(
    (item) =>
      item.id !== experiment.id && (!item.kind || item.kind === "trace"),
  );
  const referenceTrace =
    reference && (!reference.kind || reference.kind === "trace")
      ? reference.trace
      : undefined;
  const referenceUsable =
    referenceTrace && hasComparisonMeasurements(referenceTrace);
  const problem = reference
    ? reference.id === experiment.id
      ? "Choose a different saved trace as the reference."
      : !referenceTrace
        ? "The reference is not an activation trace. Choose a saved activation capture."
        : traceComparisonProblem(trace, referenceTrace)
    : null;
  const comparable = referenceTrace && !problem ? referenceTrace : null;
  const differences = comparable ? traceDifferences(trace, comparable) : [];
  const linkedReference =
    referenceId && !candidates.some((item) => item.id === referenceId);
  function choose(id: string) {
    window.history.pushState(
      null,
      "",
      notebookHref(window.location.search, { compare: id || null }),
    );
  }
  return (
    <>
      <Panel
        title="Compare prompt measurements"
        subtitle="Choose saved evidence to overlay residual magnitudes and prediction uncertainty."
        className="trace-comparison"
      >
        <div className="comparison-controls">
          <label>
            Reference trace
            <select
              aria-label="Reference trace"
              value={referenceId ?? ""}
              onChange={(event) => choose(event.target.value)}
              disabled={libraryLoading && !referenceId}
            >
              <option value="">No reference</option>
              {linkedReference && (
                <option value={referenceId}>
                  {reference?.name ?? "Linked reference"}
                </option>
              )}
              {candidates.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {new Date(item.created_at).toLocaleString()} ·{" "}
                  {item.id.slice(0, 8)}
                </option>
              ))}
            </select>
          </label>
          {comparable && reference && (
            <button
              className="button secondary"
              onClick={() =>
                downloadJSON(
                  {
                    format: "circuit-lantern-trace-comparison-v1",
                    convention:
                      "Selected minus reference. Norm differences compare magnitudes, not residual vector distances. Only prompt measurements are compared.",
                    selected: experiment,
                    reference,
                    layers: differences,
                  },
                  `trace-comparison-${experiment.id.slice(0, 8)}-${reference.id.slice(0, 8)}.json`,
                )
              }
            >
              <Download size={14} /> Export comparison
            </button>
          )}
        </div>
        {libraryLoading && <Loading label="Finding saved traces" />}
        <ErrorNotice message={libraryError} />
        {libraryError && (
          <div className="comparison-controls">
            <button className="button secondary" onClick={onRetryLibrary}>
              Retry saved traces
            </button>
          </div>
        )}
        {!libraryLoading &&
          !libraryError &&
          !candidates.length &&
          !referenceId && (
            <p className="inline-note">
              Save another activation trace to compare its prompt measurements
              here.
            </p>
          )}
        {loading && <Loading label="Opening reference trace" />}
        <ErrorNotice message={error || problem} />
        {referenceUsable && referenceTrace && (
          <div className="comparison-source">
            <strong>Reference · {reference!.name}</strong>
            <p>
              {referenceTrace.model_id} · revision {referenceTrace.revision}
            </p>
            <p>
              {executionLabel(
                referenceTrace.execution_precision,
                referenceTrace.execution_config,
              )}{" "}
              · {referenceTrace.device.toUpperCase()} ·{" "}
              {interventionLabel(referenceTrace)}
            </p>
          </div>
        )}
        {comparable && (
          <>
            <p className="inline-note" role="status">
              Checkpoint, prompt tokens and execution settings match.
              {!trace.execution_id || !comparable.execution_id
                ? " Session identity is unavailable for at least one capture."
                : trace.execution_id !== comparable.execution_id
                  ? " The captures come from different model sessions."
                  : ""}{" "}
              Only prompt measurements are compared. Generated continuations
              belong to the selected trace.
            </p>
            <details className="comparison-values">
              <summary>Inspect per-layer changes</summary>
              <p className="inline-note">
                Differences are selected minus reference. Norm changes compare
                magnitudes and do not give the distance between residual
                vectors.
              </p>
              <div className="table-wrap">
                <table className="trace-comparison-table">
                  <caption className="sr-only">
                    Selected and reference prompt measurements at each captured
                    residual position
                  </caption>
                  <thead>
                    <tr>
                      <th>LAYER</th>
                      <th>SELECTED NORM</th>
                      <th>REFERENCE NORM</th>
                      <th>NORM CHANGE</th>
                      <th>ENTROPY CHANGE</th>
                      <th>SELECTED PREDICTION</th>
                      <th>REFERENCE PREDICTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {differences.map((row) => (
                      <tr key={row.layer}>
                        <th scope="row" title={row.label}>
                          {row.layer}
                        </th>
                        <td>{number(row.selected_norm, 5)}</td>
                        <td>{number(row.reference_norm, 5)}</td>
                        <td>{number(row.norm_difference, 5)}</td>
                        <td>{number(row.entropy_difference, 5)}</td>
                        <td className="code">
                          {row.selected_prediction ?? "Unavailable"}
                        </td>
                        <td className="code">
                          {row.reference_prediction ?? "Unavailable"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="inline-note">
                Norms are L2 magnitudes of the final prompt token. Entropy
                changes use nats. Intermediate predictions use the recorded
                logit lens.
              </p>
            </details>
          </>
        )}
      </Panel>
      <TraceResults
        trace={trace}
        comparison={comparable}
        comparisonLabel="Reference trace"
        traceLabel="Selected trace"
      />
    </>
  );
}
