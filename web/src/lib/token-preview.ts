"use client";

import { usePreview, type PreviewState } from "@/lib/use-preview";
import type { TokenPreviewResponse } from "@/lib/types";

export type TokenPreviewState = PreviewState<TokenPreviewResponse>;

export function useTokenPreview(
  prompts: string[],
  reservedTokens: number,
  enabled: boolean,
  executionId?: string | null,
): TokenPreviewState {
  return usePreview<TokenPreviewResponse>(
    "/tokenize",
    {
      prompts,
      reserved_tokens: reservedTokens,
      expected_execution_id: executionId ?? undefined,
    },
    enabled,
    "Token preview is temporarily unavailable.",
  );
}
