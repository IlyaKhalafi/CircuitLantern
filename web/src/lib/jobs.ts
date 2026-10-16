"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, api, post } from "@/lib/api";
import type { Job } from "@/lib/types";

const terminalStates = new Set(["ready", "failed", "cancelled"]);

function readStoredJob(storageKey: string) {
  try {
    return sessionStorage.getItem(storageKey);
  } catch {
    return null;
  }
}

function storeJob(storageKey: string, id: string | null) {
  try {
    if (id === null) sessionStorage.removeItem(storageKey);
    else sessionStorage.setItem(storageKey, id);
  } catch {
    // Persistence is optional when browser storage is unavailable.
  }
}

export function useBackgroundJob<T>(storageKey: string) {
  const [job, setJob] = useState<(Job & { result?: T }) | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [restoredKey, setRestoredKey] = useState<string | null>(null);
  const [cancellationRequested, setCancellationRequested] = useState(false);
  const cancelPending = useRef(false);
  const startingRef = useRef(false);
  const restoring = restoredKey !== storageKey;

  useEffect(() => {
    setJobId(readStoredJob(storageKey));
    setJob(null);
    setError(null);
    setActionError(null);
    setCancellationRequested(false);
    setRestoredKey(storageKey);
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
        } else {
          setCancellationRequested(false);
        }
      } catch (reason) {
        if (!active) return;
        setError((reason as Error).message);
        if (reason instanceof ApiError && reason.status === 404) {
          storeJob(storageKey, null);
          setJobId(null);
          setJob(null);
          setCancellationRequested(false);
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
      restoring ||
      startingRef.current ||
      (jobId && (!job || !terminalStates.has(job.status)))
    )
      return;
    startingRef.current = true;
    setStarting(true);
    setError(null);
    setActionError(null);
    setCancellationRequested(false);
    try {
      const result = await post<Job>(path, payload);
      setJob(result);
      setJobId(result.id);
      storeJob(storageKey, result.id);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      startingRef.current = false;
      setStarting(false);
    }
  }

  const canCancel =
    !!jobId &&
    !starting &&
    !restoring &&
    (!job || !terminalStates.has(job.status));

  async function cancel() {
    if (
      !canCancel ||
      cancelPending.current ||
      cancellationRequested ||
      job?.cancel_requested
    )
      return;
    cancelPending.current = true;
    setCancellationRequested(true);
    setActionError(null);
    try {
      const response = await post<{
        status: string;
        cancel_requested: boolean;
      }>(`/jobs/${jobId}/cancel`, {});
      if (!response.cancel_requested || terminalStates.has(response.status))
        setCancellationRequested(false);
    } catch (reason) {
      setActionError((reason as Error).message);
      setCancellationRequested(false);
    } finally {
      cancelPending.current = false;
    }
  }

  return {
    job,
    error: error || actionError,
    clearError: () => {
      setError(null);
      setActionError(null);
    },
    start,
    cancel,
    canCancel,
    cancelling: canCancel && (cancellationRequested || !!job?.cancel_requested),
    busy:
      restoring ||
      starting ||
      (!!jobId && (!job || !terminalStates.has(job.status))),
  };
}
