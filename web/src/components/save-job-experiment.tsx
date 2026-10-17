"use client";

import { SaveExperiment } from "@/components/save-experiment";

export function SaveJobExperiment({
  jobId,
  defaultName,
  disabled = false,
  compareId,
}: {
  jobId: string;
  defaultName: string;
  disabled?: boolean;
  compareId?: string;
}) {
  return (
    <SaveExperiment
      source={{ kind: "job", id: jobId }}
      defaultName={defaultName}
      disabled={disabled}
      compareId={compareId}
    />
  );
}
