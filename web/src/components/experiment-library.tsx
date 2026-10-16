"use client";
import { useI18n } from "@/i18n/provider";
import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowRight, FlaskConical, Search } from "lucide-react";
import { EmptyState } from "@/components/ui";
import { experimentLabel, notebookHref } from "@/lib/experiments";
import { truncateText } from "@/lib/text";
import type { Experiment } from "@/lib/types";
export function ExperimentLibrary({
  experiments,
  query,
  kind,
  params,
  update,
}: {
  experiments: Experiment[];
  query: string;
  kind: string;
  params: string;
  update: (changes: Record<string, string | null>) => void;
}) {
  const { t, dateTime } = useI18n();
  const [search, setSearch] = useState(query);
  useEffect(() => {
    // A router transition can finish after a newer keystroke updates the URL.
    const current = new URLSearchParams(window.location.search).get("q") ?? "";
    if (query === current) setSearch(query);
  }, [query]);
  const terms = search.trim().toLocaleLowerCase().split(/\s+/u).filter(Boolean);
  const visible = experiments.filter((experiment) => {
    if (kind !== "all" && (experiment.kind ?? "trace") !== kind) return false;
    const text = [experiment.name, experiment.model_id, experiment.notes]
      .join(" ")
      .toLocaleLowerCase();
    return terms.every((term) => text.includes(term));
  });
  return (
    <>
      <section
        className="panel notebook-filters"
        aria-label={t("Find experiments")}
      >
        <label className="notebook-search">
          {t("Search notebook")}
          <span>
            <Search size={15} aria-hidden="true" />
            <input
              dir="auto"
              className="form-input"
              type="search"
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                update({ q: event.target.value || null });
              }}
              placeholder={t("Names, models and observations")}
              maxLength={200}
            />
          </span>
        </label>
        <label>
          {t("Experiment type")}
          <select
            value={kind}
            onChange={(event) =>
              update({
                kind: event.target.value === "all" ? null : event.target.value,
              })
            }
          >
            <option value="all">{t("All experiments")}</option>
            <option value="trace">{t("Activation traces")}</option>
            <option value="causal">{t("Residual patching")}</option>
            <option value="head_sweep">{t("Head ablation")}</option>
            <option value="matrix">{t("Matrix analyses")}</option>
          </select>
        </label>
        <p className="notebook-count" role="status">
          {t(
            "{total, plural, one {{visible} of # experiment} other {{visible} of # experiments}}",
            { visible: visible.length, total: experiments.length },
          )}
        </p>
        {(search || kind !== "all") && (
          <button
            className="button secondary"
            onClick={() => update({ q: null, kind: null })}
          >
            {t("Clear filters")}
          </button>
        )}
      </section>
      {visible.length ? (
        <div className="experiment-grid">
          {visible.map((experiment) => {
            const preview = truncateText(experiment.notes, 110);
            return (
              <Link
                key={experiment.id}
                className="panel experiment-card"
                href={notebookHref(params, {
                  experiment: experiment.id,
                  compare: null,
                  q: search || null,
                })}
                prefetch={false}
              >
                <span className="pill subtle">
                  <FlaskConical size={11} />{" "}
                  {t(experimentLabel(experiment.kind))}
                </span>
                <h3 dir="auto">{experiment.name}</h3>
                <span className="mono">{experiment.model_id}</span>
                <p dir="auto" style={{ marginTop: 13 }}>
                  {experiment.notes
                    ? preview + (preview === experiment.notes ? "" : "\u2026")
                    : t(
                        "Open this experiment to explore its captured result and add research notes.",
                      )}
                </p>
                <div className="experiment-meta">
                  <span>{dateTime(experiment.created_at)}</span>
                  <ArrowRight size={14} />
                </div>
              </Link>
            );
          })}
        </div>
      ) : (
        <EmptyState
          icon={<Search size={28} />}
          title={t("No matching experiments.")}
        >
          {t(
            "Try a different name, model or observation, or clear the filters.",
          )}
        </EmptyState>
      )}
    </>
  );
}
