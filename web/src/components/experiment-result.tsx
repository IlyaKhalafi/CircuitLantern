import { PromptSetResults } from "@/components/prompt-set-results";
import { SavedMatrixComparison } from "@/components/saved-matrix-comparison";
import { SavedCausalComparison } from "@/components/saved-causal-comparison";
import { HeadSweepResults } from "@/components/head-sweep-results";
import { SavedInductionComparison } from "@/components/saved-induction-comparison";
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
  if (experiment.kind === "causal_batch")
    return experiment.result ? (
      <PromptSetResults result={experiment.result} />
    ) : null;
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
      <SavedCausalComparison
        experiment={experiment}
        result={experiment.result}
        experiments={experiments}
        libraryLoading={libraryLoading}
        libraryError={libraryError}
        onRetryLibrary={onRetryLibrary}
      />
    ) : null;
  if (experiment.kind === "head_sweep")
    return experiment.result ? (
      <HeadSweepResults result={experiment.result} />
    ) : null;
  if (experiment.kind === "induction_probe")
    return experiment.result ? (
      <SavedInductionComparison
        experiment={experiment}
        result={experiment.result}
        experiments={experiments}
        libraryLoading={libraryLoading}
        libraryError={libraryError}
        onRetryLibrary={onRetryLibrary}
      />
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
