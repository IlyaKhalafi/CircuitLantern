"use client";
import Link from "next/link";
import { useI18n } from "@/i18n/provider";
import { executionLabel } from "@/lib/execution";
import type { ModelMetadata, RuntimeState } from "@/lib/types";

export function ExecutionContext({
  runtime,
  metadata,
}: {
  runtime?: RuntimeState | null;
  metadata?: ModelMetadata;
}) {
  const { t } = useI18n();
  const model = runtime?.metadata ?? metadata;
  if (!model) return null;
  const precision = runtime?.precision ?? model.execution_precision;
  const device = runtime?.device ?? model.execution_config?.device;
  const context = runtime?.context_length ?? model.execution_context_length;
  return (
    <Link className="execution-context" href="/hardware#active-model">
      <bdi>{model.id}</bdi>
      <span className="execution-values">
        {precision && (
          <span>{t(executionLabel(precision, model.execution_config))}</span>
        )}
        {typeof device === "string" && (
          <bdi dir="ltr">{device.toUpperCase()}</bdi>
        )}
        {context != null && (
          <span>{t("{count} context tokens", { count: context })}</span>
        )}
      </span>
      <span className="sr-only">{t("View active model settings")}</span>
    </Link>
  );
}
