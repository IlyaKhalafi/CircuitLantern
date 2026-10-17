"use client";

import { useEffect, useRef } from "react";
import type { Job } from "@/lib/types";

export function useCompletedJob<T>(
  task: { job: (Job & { result?: T }) | null },
  key: string,
  onComplete: (result: T) => void,
) {
  const delivered = useRef<string | null>(null);
  useEffect(() => {
    const job = task.job;
    if (
      !job ||
      !["ready", "cancelled", "failed"].includes(job.status) ||
      !job.result ||
      delivered.current === job.id
    )
      return;
    delivered.current = job.id;
    try {
      if (sessionStorage.getItem(key) === job.id) return;
      sessionStorage.setItem(key, job.id);
    } catch {
      /* Optional persistence. */
    }
    onComplete(job.result);
  }, [task.job, key, onComplete]);
}
