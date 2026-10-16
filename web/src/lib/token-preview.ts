"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { TokenPreviewResponse } from "@/lib/types";

export interface TokenPreviewState {
  data: TokenPreviewResponse | null;
  pending: boolean;
  error: string | null;
}

export function useTokenPreview(
  prompts: string[],
  reservedTokens: number,
  enabled: boolean,
  executionId?: string | null,
): TokenPreviewState {
  const key = JSON.stringify({
    prompts,
    reserved_tokens: reservedTokens,
    expected_execution_id: executionId ?? undefined,
  });
  const [state, setState] = useState<TokenPreviewState & { key: string }>({
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
        const data = await api<TokenPreviewResponse>("/tokenize", {
          method: "POST",
          body: key,
          signal: controller.signal,
        });
        if (!controller.signal.aborted)
          setState({ key, data, pending: false, error: null });
      } catch {
        if (!controller.signal.aborted)
          setState({
            key,
            data: null,
            pending: false,
            error: "Token preview is temporarily unavailable.",
          });
      }
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key, enabled]);
  return {
    data: state.key === key ? state.data : null,
    pending: enabled && (state.key !== key || state.pending),
    error: state.key === key ? state.error : null,
  };
}
