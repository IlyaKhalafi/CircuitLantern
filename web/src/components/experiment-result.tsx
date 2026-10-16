import { SavedMatrixComparison } from "@/components/saved-matrix-comparison";
import { CausalResults } from "@/components/causal-results";
import { HeadSweepResults } from "@/components/head-sweep-results";
import { SavedTraceComparison } from "@/components/saved-trace-comparison";
import type { Experiment } from "@/lib/types";

export function ExperimentResult({
  experiment,
  experiments,
  libraryLoading,
  libraryError,
  onRetryLibrary,
}: {
  experiment: Experiment;
  experiments: Experiment[];
  libraryLoading: boolean;
  libraryError: string | null;
  onRetryLibrary: () => void;
}) {
  if (experiment.kind === "matrix")
    return experiment.result ? (
      <SavedMatrixComparison
        experiment={experiment}
        metrics={experiment.result}
        experiments={experiments}
        libraryLoading={libraryLoading}
        libraryError={libraryError}
        onRetryLibrary={onRetryLibrary}
      />
    ) : null;
  if (experiment.kind === "causal")
    return experiment.result ? (
      <CausalResults result={experiment.result} />
    ) : null;
  if (experiment.kind === "head_sweep")
    return experiment.result ? (
      <HeadSweepResults result={experiment.result} />
    ) : null;
  return experiment.trace ? (
    <SavedTraceComparison
      experiment={experiment}
      trace={experiment.trace}
      experiments={experiments}
      libraryLoading={libraryLoading}
      libraryError={libraryError}
      onRetryLibrary={onRetryLibrary}
    />
  ) : null;
}
