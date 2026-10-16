"use client";
import { useI18n } from "@/i18n/provider";
import Link from "next/link";
import { Download } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { ErrorNotice, Loading, Panel } from "@/components/ui";
import { MatrixResults } from "@/components/matrix-results";
import { SeriesChart } from "@/components/charts";
import { SavedReferencePicker } from "@/components/saved-reference-picker";
import { useExperiment } from "@/lib/use-experiment";
import { downloadJSON, number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import { matrixValue } from "@/lib/matrix-properties";
import {
  hasMatrixComparisonMeasurements,
  matrixComparisonProblem,
  matrixDifferences,
  matrixSpectrumComparison,
} from "@/lib/matrix-comparison";
import type { Experiment, MatrixMetrics } from "@/lib/types";
const series = [
  { key: "selected", label: "Selected matrix", color: "var(--accent)" },
  { key: "reference", label: "Reference matrix", color: "var(--blue)" },
];
function Source({
  label,
  record,
  metrics,
}: {
  label: string;
  record: Experiment;
  metrics: MatrixMetrics;
}) {
  const { t, rich } = useI18n();
  const knownFormat =
    metrics.execution_precision ||
    typeof metrics.execution_config?.precision === "string";
  return (
    <div className="matrix-comparison-source">
      <strong>
        {t(label)} · {record.name}
      </strong>
      <p>
        {rich("{v0} \u00B7 revision {v1}", {
          v0: metrics.model_id,
          v1: metrics.revision,
        })}
      </p>
      <p>
        {rich("{v0}{v1}{v2}{v3}{v4} source", {
          v0: knownFormat
            ? executionLabel(
                metrics.execution_precision,
                metrics.execution_config,
              )
            : t("Execution format unavailable"),
          v1: " · ",
          v2: metrics.device?.toUpperCase() ?? t("Device unavailable"),
          v3: " · ",
          v4: metrics.source_format ?? metrics.dtype,
        })}
      </p>
      <p>
        {rich("{v0} \u00B7 {v1} retained components", {
          v0: metrics.spectrum_method,
          v1: metrics.singular_values.length,
        })}
      </p>
      {metrics.spectrum_exact && (
        <p>
          {rich("Numerical rank threshold {v0}", {
            v0: number(metrics.rank_tolerance, 5),
          })}
        </p>
      )}
      {metrics.quantized && (
        <p>{t("Statistics describe dequantized weight values.")}</p>
      )}
    </div>
  );
}
export function SavedMatrixComparison({
  experiment,
  metrics,
  experiments,
  libraryLoading,
  libraryError,
  onRetryLibrary,
}: {
  experiment: Experiment;
  metrics: MatrixMetrics;
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
    (item) => item.id !== experiment.id && item.kind === "matrix",
  );
  const referenceMetrics =
    reference?.kind === "matrix" ? reference.result : undefined;
  const referenceUsable =
    referenceMetrics && hasMatrixComparisonMeasurements(referenceMetrics);
  const problem = reference
    ? reference.id === experiment.id
      ? "Choose a different saved matrix analysis as the reference."
      : !referenceMetrics
        ? "The reference is not a matrix analysis. Choose saved matrix measurements."
        : matrixComparisonProblem(metrics, referenceMetrics)
    : null;
  const comparable = referenceMetrics && !problem ? referenceMetrics : null;
  const differences = comparable ? matrixDifferences(metrics, comparable) : [];
  const spectrum = comparable
    ? matrixSpectrumComparison(metrics, comparable)
    : [];
  const estimated =
    comparable && (!metrics.spectrum_exact || !comparable.spectrum_exact);
  return (
    <>
      <Panel
        title={t("Compare matrix measurements")}
        subtitle={t(
          "Compare the same weight across saved revisions or execution formats.",
        )}
        className="trace-comparison"
      >
        <SavedReferencePicker
          label={t("Reference matrix")}
          referenceId={referenceId}
          reference={reference}
          candidates={candidates}
          libraryLoading={libraryLoading}
          libraryError={libraryError}
          onRetryLibrary={onRetryLibrary}
          emptyText="Save another analysis of the same weight to compare its measurements here."
        >
          {comparable && reference && (
            <button
              className="button secondary"
              onClick={() =>
                downloadJSON(
                  {
                    format: "circuit-lantern-matrix-comparison-v1",
                    convention:
                      "Selected minus reference. Norm differences compare magnitudes and are not norms of matrix differences. Singular value ranks do not align singular vectors. Energy fractions use each matrix's own Frobenius norm. Fraction changes are stored in fraction units. Estimated values retain method flags.",
                    selected: experiment,
                    reference,
                    properties: differences,
                    spectrum,
                  },
                  "matrix-comparison-" +
                    experiment.id.slice(0, 8) +
                    "-" +
                    reference.id.slice(0, 8) +
                    ".json",
                )
              }
            >
              <Download size={14} />
              {t("Export comparison")}
            </button>
          )}
        </SavedReferencePicker>
        {loading && <Loading label={t("Opening reference matrix")} />}
        <ErrorNotice message={error || problem} />
        {problem && (
          <p className="inline-note">
            <Link className="text-link" href="/workspace?view=compare">
              {t("Compare base and finetuned checkpoints in the explorer")}
            </Link>
          </p>
        )}
        {referenceUsable && reference && referenceMetrics && (
          <div className="matrix-comparison-sources">
            <Source
              label={t("Selected")}
              record={experiment}
              metrics={metrics}
            />
            <Source
              label={t("Reference")}
              record={reference}
              metrics={referenceMetrics}
            />
          </div>
        )}
        {comparable && (
          <>
            <p className="inline-note" role="status">
              {t(
                "Model repository, weight name and logical dimensions match. Revisions and execution settings may differ.",
              )}
            </p>
            <p className="inline-note">
              {rich(
                "Differences can reflect revisions, quantization, execution settings and analysis methods. They do not isolate a single cause.{v0}",
                {
                  v0:
                    estimated &&
                    t(
                      " At least one spectrum is estimated. Its spectral norm, stable rank and spectrum differences are estimates.",
                    ),
                },
              )}
            </p>
          </>
        )}
      </Panel>
      {comparable && (
        <>
          <div className="two-column">
            <Panel
              title={t("Singular value comparison")}
              subtitle={t(
                "Component ranks are aligned. Singular vectors are not compared.",
              )}
            >
              <SeriesChart data={spectrum} series={series} />
            </Panel>
            <Panel
              title={t("Spectral energy comparison")}
              subtitle={t(
                "Percent of each matrix's own squared Frobenius norm.",
              )}
            >
              <SeriesChart
                data={spectrum.map((row) => ({
                  index: row.index,
                  selected:
                    row.selected_energy === null
                      ? null
                      : row.selected_energy * 100,
                  reference:
                    row.reference_energy === null
                      ? null
                      : row.reference_energy * 100,
                }))}
                series={series}
              />
            </Panel>
          </div>
          <Panel
            title={t("Matrix measurement changes")}
            subtitle={t(
              "Selected minus reference. Missing measurements remain unavailable.",
            )}
          >
            <details className="comparison-values">
              <summary>{t("Inspect property changes")}</summary>
              <p className="inline-note">
                {t(
                  "Norm changes compare magnitudes and do not give the norm of the difference between matrices. Exact zero changes use percentage points. A tilde marks estimates.",
                )}
              </p>
              <div className="table-wrap">
                <table className="matrix-comparison-table">
                  <caption className="sr-only">
                    {t(
                      "Saved matrix properties and changes from the reference",
                    )}
                  </caption>
                  <thead>
                    <tr>
                      <th>{t("PROPERTY")}</th>
                      <th>{t("SELECTED")}</th>
                      <th>{t("REFERENCE")}</th>
                      <th>{t("CHANGE")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {differences.map((row) => (
                      <tr key={row.key}>
                        <th scope="row">{t(row.label)}</th>
                        <td>
                          {row.selected_estimated ? "~" : ""}
                          {matrixValue(row.selected, row.unit)}
                        </td>
                        <td>
                          {row.reference_estimated ? "~" : ""}
                          {matrixValue(row.reference, row.unit)}
                        </td>
                        <td>
                          {row.difference_estimated && row.difference !== null
                            ? "~"
                            : ""}
                          {matrixValue(row.difference, row.unit, true)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="inline-note">
                {t(
                  "Numerical rank, nuclear norm, effective rank and condition numbers require complete spectra. Numerical ranks use each capture's recorded threshold. Displayed spectra may retain different numbers of components. Missing tails are not treated as zero.",
                )}
              </p>
            </details>
          </Panel>
        </>
      )}
      <MatrixResults metrics={metrics} />
    </>
  );
}
