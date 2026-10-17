"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ProbeComparison } from "@/lib/induction-comparison";

export function useProbeComparison(selected: string, reference: string | null) {
  const key = selected + "/" + (reference || "");
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<{
    key: string;
    data: ProbeComparison | null;
    error: string | null;
  } | null>(null);
  useEffect(() => {
    if (!reference || selected === reference) {
      setState(null);
      return;
    }
    const controller = new AbortController();
    setState(null);
    const query = new URLSearchParams({ selected, reference });
    api<ProbeComparison>("/induction/comparison?" + query, {
      signal: controller.signal,
    })
      .then((data) => {
        if (controller.signal.aborted) return;
        if (
          data.selected_record?.id !== selected ||
          data.reference_record?.id !== reference
        )
          throw new Error(
            "The comparison response does not match the selected records.",
          );
        setState({ key, data, error: null });
      })
      .catch((reason: Error) => {
        if (!controller.signal.aborted)
          setState({ key, data: null, error: reason.message });
      });
    return () => controller.abort();
  }, [selected, reference, key, attempt]);
  const current = state?.key === key ? state : null;
  return {
    data: current?.data || null,
    error: current?.error || null,
    loading: !!reference && reference !== selected && !current,
    retry: () => setAttempt((value) => value + 1),
  };
}
