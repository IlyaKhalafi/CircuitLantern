"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Experiment } from "@/lib/types";

interface Selection {
  id: string;
  record: Experiment | null;
  error: string | null;
}

export function useExperiment(id: string | null) {
  const [selection, setSelection] = useState<Selection | null>(null);
  useEffect(() => {
    if (!id) {
      setSelection(null);
      return;
    }
    const controller = new AbortController();
    api<Experiment>(`/experiments/${encodeURIComponent(id)}`, {
      signal: controller.signal,
    })
      .then((record) => {
        if (!controller.signal.aborted)
          setSelection({ id, record, error: null });
      })
      .catch((reason: Error) => {
        if (!controller.signal.aborted)
          setSelection({ id, record: null, error: reason.message });
      });
    return () => controller.abort();
  }, [id]);
  const current = selection?.id === id ? selection : null;
  return {
    selected: current?.record ?? null,
    error: current?.error ?? null,
    loading: !!id && !current,
    replace: (record: Experiment) =>
      setSelection({ id: record.id, record, error: null }),
  };
}
