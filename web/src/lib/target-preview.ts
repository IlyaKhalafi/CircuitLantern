"use client";

import { usePreview, type PreviewState } from "@/lib/use-preview";
import type { TargetPreviewResponse } from "@/lib/types";

export type TargetPreviewState = PreviewState<TargetPreviewResponse>;

export function useTargetPreview(
  tokens: string[],
  enabled: boolean,
  executionId?: string | null,
): TargetPreviewState {
  return usePreview<TargetPreviewResponse>(
    "/targets/preview",
    { tracked_tokens: tokens, expected_execution_id: executionId ?? undefined },
    enabled && tokens.some((token) => token.length > 0),
    "Target preview is temporarily unavailable. The server will validate targets when you trace.",
  );
}
