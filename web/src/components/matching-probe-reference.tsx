"use client";

import Link from "next/link";
import { ArrowUpRight, X } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import type { useInduction } from "@/hooks/use-induction";
import { matchingModelLibraryHref } from "@/lib/induction-comparison";

export function MatchingProbeReference({
  lab,
}: {
  lab: ReturnType<typeof useInduction>;
}) {
  const { t } = useI18n();
  if (!lab.reference.identifier && !lab.reference.error) return null;
  return (
    <section
      className="panel matching-probe-reference"
      aria-label={t("Matching probe reference")}
    >
      <div>
        <span className="eyebrow">{t("EXACT REFERENCE INPUTS")}</span>
        <h2 dir="auto">
          {lab.reference.record?.name ||
            lab.reference.identifier ||
            t("Matching probe reference")}
        </h2>
        {lab.reference.result && (
          <p>
            <bdi>{lab.reference.result.model_id}</bdi>
          </p>
        )}
        <p>
          {t(
            "The saved token IDs and source candidate head indices are reused. Load another compatible checkpoint, then run the matching probe.",
          )}
        </p>
      </div>
      <div className="action-row">
        {lab.reference.identifier && (
          <Link
            className="button secondary"
            href={
              lab.reference.result
                ? matchingModelLibraryHref(
                    lab.reference.identifier,
                    lab.reference.result,
                  )
                : "/?return=induction&reference=" + lab.reference.identifier
            }
          >
            <ArrowUpRight size={14} />
            {t("Choose a checkpoint")}
          </Link>
        )}
        <button
          type="button"
          className="button secondary"
          onClick={lab.reference.clear}
        >
          <X size={14} />
          {t("Clear reference")}
        </button>
      </div>
    </section>
  );
}
