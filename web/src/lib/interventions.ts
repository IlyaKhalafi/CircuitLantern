import type { Trace } from "@/lib/types";

export function interventionLabel(
  trace: Pick<Trace, "ablate_layer" | "ablate_head">,
) {
  if (trace.ablate_head)
    return `Layer ${trace.ablate_head.layer} · head ${trace.ablate_head.head} zeroed`;
  if (trace.ablate_layer != null) return `Block ${trace.ablate_layer} bypassed`;
  return "Baseline";
}
