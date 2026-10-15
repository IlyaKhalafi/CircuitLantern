"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, api, post } from "@/lib/api";
import type { Job } from "@/lib/types";

const terminalStates = new Set(["ready", "failed", "cancelled"]);

export function useBackgroundJob<T>(storageKey: string) {
  const [job, setJob] = useState<(Job & { result?: T }) | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const cancelPending = useRef(false);
  const startingRef = useRef(false);

  useEffect(() => {
    const id = sessionStorage.getItem(storageKey);
    setJobId(id);
    setJob(null);
    setError(null);
  }, [storageKey]);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await api<Job & { result?: T }>(`/jobs/${jobId}`);
        if (!active) return;
        setJob(result);
        setError(result.status === "failed" ? result.message : null);
        if (!terminalStates.has(result.status)) {
          timer = setTimeout(poll, 900);
        }
      } catch (reason) {
        if (!active) return;
        setError((reason as Error).message);
        if (reason instanceof ApiError && reason.status === 404) {
          sessionStorage.removeItem(storageKey);
          setJobId(null);
          setJob(null);
        } else {
          timer = setTimeout(poll, 2500);
        }
      }
    };
    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [jobId, storageKey]);

  async function start(path: string, payload: unknown) {
    if (
      startingRef.current ||
      (jobId && (!job || !terminalStates.has(job.status)))
    )
      return;
    startingRef.current = true;
    setStarting(true);
    setError(null);
    try {
      const result = await post<Job>(path, payload);
      setJob(result);
      setJobId(result.id);
      sessionStorage.setItem(storageKey, result.id);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      startingRef.current = false;
      setStarting(false);
    }
  }

  async function cancel() {
    if (!jobId || cancelPending.current) return;
    cancelPending.current = true;
    try {
      await post(`/jobs/${jobId}/cancel`, {});
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      cancelPending.current = false;
    }
  }

  return {
    job,
    error,
    start,
    cancel,
    busy: starting || (!!jobId && (!job || !terminalStates.has(job.status))),
  };
}
