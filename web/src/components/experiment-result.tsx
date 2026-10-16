import { CausalResults } from "@/components/causal-results";
import { HeadSweepResults } from "@/components/head-sweep-results";
import { TraceResults } from "@/components/trace-results";
import type { Experiment } from "@/lib/types";

export function ExperimentResult({ experiment }: { experiment: Experiment }) {
  if (experiment.kind === "causal")
    return experiment.result ? (
      <CausalResults result={experiment.result} />
    ) : null;
  if (experiment.kind === "head_sweep")
    return experiment.result ? (
      <HeadSweepResults result={experiment.result} />
    ) : null;
  return experiment.trace ? <TraceResults trace={experiment.trace} /> : null;
}
