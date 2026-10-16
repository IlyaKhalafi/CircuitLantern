"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { RuntimeState } from "@/lib/types";

export function executionPath(path: string, executionId?: string | null) {
  if (!executionId) return path;
  const query = new URLSearchParams({ expected_execution_id: executionId });
  return `${path}${path.includes("?") ? "&" : "?"}${query}`;
}

export function useRuntime() {
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);
  const refresh = useCallback(async () => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const result = await api<RuntimeState>("/runtime");
      if (id === requestId.current) setRuntime(result);
    } catch (reason) {
      if (id === requestId.current) setError((reason as Error).message);
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }, []);
  useEffect(() => {
    refresh();
    return () => {
      ++requestId.current;
    };
  }, [refresh]);
  return { runtime, loading, error, refresh };
}
