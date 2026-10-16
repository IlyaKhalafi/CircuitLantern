"use client";
import { useI18n } from "@/i18n/provider";
import { Download } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { ErrorNotice, Loading, Panel } from "@/components/ui";
import { TraceResults } from "@/components/trace-results";
import { useExperiment } from "@/lib/use-experiment";
import { SavedReferencePicker } from "@/components/saved-reference-picker";
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
  const { t, rich } = useI18n();
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
  return (
    <>
      <Panel
        title={t("Compare prompt measurements")}
        subtitle={t(
          "Choose saved evidence to overlay residual magnitudes and prediction uncertainty.",
        )}
        className="trace-comparison"
      >
        <SavedReferencePicker
          label={t("Reference trace")}
          referenceId={referenceId}
          reference={reference}
          candidates={candidates}
          libraryLoading={libraryLoading}
          libraryError={libraryError}
          onRetryLibrary={onRetryLibrary}
          retryLabel="Retry saved traces"
          loadingLabel="Finding saved traces"
          emptyText="Save another activation trace to compare its prompt measurements here."
        >
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
              <Download size={14} />
              {t("Export comparison")}
            </button>
          )}
        </SavedReferencePicker>
        {loading && <Loading label={t("Opening reference trace")} />}
        <ErrorNotice message={error || problem} />
        {referenceUsable && referenceTrace && (
          <div className="comparison-source">
            <strong>
              {rich("Reference \u00B7 {v0}", {
                v0: reference!.name,
              })}
            </strong>
            <p>
              {rich("{v0} \u00B7 revision {v1}", {
                v0: referenceTrace.model_id,
                v1: referenceTrace.revision,
              })}
            </p>
            <p>
              {t(
                executionLabel(
                  referenceTrace.execution_precision,
                  referenceTrace.execution_config,
                ),
              )}{" "}
              · {referenceTrace.device.toUpperCase()} ·{" "}
              {t(interventionLabel(referenceTrace))}
            </p>
          </div>
        )}
        {comparable && (
          <>
            <p className="inline-note" role="status">
              {rich(
                "Checkpoint, prompt tokens and execution settings match.{v0} Only prompt measurements are compared. Generated continuations belong to the selected trace.",
                {
                  v0:
                    !trace.execution_id || !comparable.execution_id
                      ? t(
                          " Session identity is unavailable for at least one capture.",
                        )
                      : trace.execution_id !== comparable.execution_id
                        ? t(" The captures come from different model sessions.")
                        : "",
                },
              )}
            </p>
            <details className="comparison-values">
              <summary>{t("Inspect per-layer changes")}</summary>
              <p className="inline-note">
                {t(
                  "Differences are selected minus reference. Norm changes compare magnitudes and do not give the distance between residual vectors.",
                )}
              </p>
              <div className="table-wrap">
                <table className="trace-comparison-table">
                  <caption className="sr-only">
                    {t(
                      "Selected and reference prompt measurements at each captured residual position",
                    )}
                  </caption>
                  <thead>
                    <tr>
                      <th>{t("LAYER")}</th>
                      <th>{t("SELECTED NORM")}</th>
                      <th>{t("REFERENCE NORM")}</th>
                      <th>{t("NORM CHANGE")}</th>
                      <th>{t("ENTROPY CHANGE")}</th>
                      <th>{t("SELECTED PREDICTION")}</th>
                      <th>{t("REFERENCE PREDICTION")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {differences.map((row) => (
                      <tr key={row.layer}>
                        <th scope="row" title={t(row.label)}>
                          {row.layer}
                        </th>
                        <td>{t(number(row.selected_norm, 5))}</td>
                        <td>{t(number(row.reference_norm, 5))}</td>
                        <td>{t(number(row.norm_difference, 5))}</td>
                        <td>{t(number(row.entropy_difference, 5))}</td>
                        <td className="code">
                          {row.selected_prediction ?? t("Unavailable")}
                        </td>
                        <td className="code">
                          {row.reference_prediction ?? t("Unavailable")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="inline-note">
                {t(
                  "Norms are L2 magnitudes of the final prompt token. Entropy changes use nats. Intermediate predictions use the recorded logit lens.",
                )}
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
