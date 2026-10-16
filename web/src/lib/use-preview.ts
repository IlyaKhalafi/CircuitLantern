"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

// The runtime rejects overlapping model operations, including tokenization.
let previewQueue: Promise<unknown> = Promise.resolve();

function requestPreview<T>(path: string, body: string, signal: AbortSignal) {
  const request = previewQueue.then(() => {
    signal.throwIfAborted();
    return api<T>(path, { method: "POST", body, signal });
  });
  previewQueue = request.catch(() => undefined);
  return request;
}

export interface PreviewState<T> {
  data: T | null;
  pending: boolean;
  error: string | null;
  retry: () => void;
}

export function usePreview<T>(
  path: string,
  payload: Record<string, unknown>,
  enabled: boolean,
  failureMessage: string,
): PreviewState<T> {
  const key = JSON.stringify(payload);
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<
    Omit<PreviewState<T>, "retry"> & { key: string }
  >({
    key: "",
    data: null,
    pending: false,
    error: null,
  });
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setState((current) =>
      current.key === key
        ? { ...current, pending: true, error: null }
        : { key, data: null, pending: true, error: null },
    );
    const timer = setTimeout(async () => {
      try {
        const data = await requestPreview<T>(path, key, controller.signal);
        if (!controller.signal.aborted)
          setState({ key, data, pending: false, error: null });
      } catch {
        if (!controller.signal.aborted)
          setState({ key, data: null, pending: false, error: failureMessage });
      }
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key, enabled, path, failureMessage, attempt]);
  return {
    data: state.key === key ? state.data : null,
    pending: enabled && (state.key !== key || state.pending),
    error: state.key === key ? state.error : null,
    retry: () => setAttempt((value) => value + 1),
  };
}
