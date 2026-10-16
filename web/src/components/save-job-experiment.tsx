"use client";

import { SaveExperiment } from "@/components/save-experiment";

export function SaveJobExperiment({
  jobId,
  defaultName,
  disabled = false,
}: {
  jobId: string;
  defaultName: string;
  disabled?: boolean;
}) {
  return (
    <SaveExperiment
      source={{ kind: "job", id: jobId }}
      defaultName={defaultName}
      disabled={disabled}
    />
  );
}
