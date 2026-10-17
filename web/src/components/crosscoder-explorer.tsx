"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Save, Sparkles } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { SeriesChart } from "@/components/charts";
import { CaptureCard } from "@/components/crosscoder-shared";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  Metric,
  Panel,
} from "@/components/ui";
import { api, downloadJSON, number } from "@/lib/api";
import type {
  CrosscoderOverview,
  SharedFeature,
  SharedFeatureDetail,
  SharedFeaturePage,
} from "@/lib/crosscoders";

function DecoderBalance({ feature }: { feature: SharedFeature }) {
  const { t } = useI18n();
  const share = feature.relative_decoder_norms?.[0];
  return (
    <div className="crosscoder-balance">
      <div
        className={
          "crosscoder-balance-track" + (share == null ? " unavailable" : "")
        }
        role="img"
        aria-label={
          share == null
            ? t("Unavailable")
            : t("Reference decoder share {v0} percent", {
                v0: number(share * 100, 1),
              })
        }
      >
        {share != null && <span style={{ width: share * 100 + "%" }} />}
      </div>
      <small>
        {share == null
          ? t("Unavailable")
          : number(share * 100, 1) +
            "% / " +
            number((1 - share) * 100, 1) +
            "%"}
      </small>
    </div>
  );
}

function SharedFeatureInspector({
  identifier,
  index,
  draft,
  editDraft,
  saved,
}: {
  identifier: string;
  index: number;
  draft?: { label: string; notes: string };
  editDraft: (
    index: number,
    value: { label: string; notes: string } | null,
  ) => void;
  saved: () => void;
}) {
  const { t } = useI18n();
  const [feature, setFeature] = useState<SharedFeatureDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    const controller = new AbortController();
    api<SharedFeatureDetail>(
      "/crosscoders/" + identifier + "/features/" + index,
      { signal: controller.signal },
    )
      .then((value) => {
        if (!controller.signal.aborted) setFeature(value);
      })
      .catch((reason) => {
        if (!controller.signal.aborted) setError(reason.message);
      });
    return () => {
      active.current = false;
      controller.abort();
    };
  }, [identifier, index]);
  const value = draft || {
    label: feature?.label || "",
    notes: feature?.notes || "",
  };
  const dirty =
    value.label !== (feature?.label || "") ||
    value.notes !== (feature?.notes || "");
  async function save() {
    setSaving(true);
    setError(null);
    try {
      const result = await api<{ label: string; notes: string }>(
        "/crosscoders/" + identifier + "/features/" + index,
        { method: "PATCH", body: JSON.stringify(value) },
      );
      if (active.current) {
        setFeature((current) =>
          current ? { ...current, ...result } : current,
        );
        editDraft(index, null);
        saved();
      }
    } catch (reason) {
      if (active.current) setError((reason as Error).message);
    } finally {
      if (active.current) setSaving(false);
    }
  }
  return (
    <Panel
      title={t("Feature {v0}", { v0: index })}
      className="crosscoder-inspector"
    >
      <ErrorNotice message={error} />
      {!feature && !error && <Loading />}
      {feature && (
        <div className="feature-form">
          <div className="metrics-grid">
            <Metric
              label="Activation frequency"
              value={number(feature.frequency * 100, 2) + "%"}
            />
            <Metric
              label="Held out frequency"
              value={
                feature.validation_frequency == null
                  ? t("Unavailable")
                  : number(feature.validation_frequency * 100, 2) + "%"
              }
            />
            <Metric
              label="Decoder cosine"
              value={
                feature.decoder_cosine == null
                  ? t("Unavailable")
                  : number(feature.decoder_cosine)
              }
            />
          </div>
          <div className="crosscoder-norms">
            <strong>{t("Relative decoder norms")}</strong>
            <DecoderBalance feature={feature} />
            <div className="crosscoder-norm-key">
              <span>{t("Reference")}</span>
              <span>{t("Compared")}</span>
            </div>
            <small className="muted">
              {t(
                "Decoder shares use normalized bank units. They describe direction magnitudes and do not prove model exclusive concepts.",
              )}
            </small>
          </div>
          <details className="feature-advanced">
            <summary>{t("Raw decoder measurements")}</summary>
            <div className="metrics-grid crosscoder-details">
              <Metric
                label="Reference decoder norm"
                value={number(feature.decoder_norms[0], 6)}
              />
              <Metric
                label="Compared decoder norm"
                value={number(feature.decoder_norms[1], 6)}
              />
              <Metric
                label="Mean activation"
                value={number(feature.mean_activation, 6)}
              />
            </div>
          </details>
          <div className="feature-form">
            <label>
              {t("Feature label")}
              <input
                dir="auto"
                maxLength={100}
                value={value.label}
                disabled={saving}
                onChange={(event) =>
                  editDraft(index, { ...value, label: event.target.value })
                }
              />
            </label>
            <label>
              {t("Research notes")}
              <textarea
                dir="auto"
                aria-label={t("Research notes")}
                rows={3}
                maxLength={10000}
                value={value.notes}
                disabled={saving}
                onChange={(event) =>
                  editDraft(index, { ...value, notes: event.target.value })
                }
              />
            </label>
            <button
              className="button secondary"
              disabled={!dirty || saving}
              onClick={save}
            >
              <Save size={14} />
              {t(saving ? "Saving" : "Save annotation")}
            </button>
          </div>
          <h3>{t("Activating contexts")}</h3>
          <p className="muted">
            {t(
              "Activations come from the joint encoder using both checkpoints. Highlighted tokens are the measured positions.",
            )}
          </p>
          {feature.examples.length === 0 && (
            <p className="muted">
              {t("No positive activations were observed for this feature.")}
            </p>
          )}
          {feature.examples.map((example) => (
            <div className="feature-example" key={example.sample}>
              <small>
                {t(example.split === "train" ? "Training" : "Held out")} ·{" "}
                {t("Activation {v0}", { v0: number(example.activation, 4) })}
              </small>
              <p className="feature-token-context" dir="ltr">
                {example.tokens?.map((token, offset) =>
                  offset === example.active_offset ? (
                    <mark key={offset}>{token}</mark>
                  ) : (
                    <span key={offset}>{token}</span>
                  ),
                )}
              </p>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

export function CrosscoderExplorer({
  identifier,
  refreshToken = 0,
}: {
  identifier: string;
  refreshToken?: number;
}) {
  const { t } = useI18n();
  const [overview, setOverview] = useState<CrosscoderOverview | null>(null);
  const [page, setPage] = useState<SharedFeaturePage | null>(null);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("frequency");
  const [direction, setDirection] = useState("highest");
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [revision, setRevision] = useState(0);
  const [reading, setReading] = useState(true);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [rowsError, setRowsError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const error = overviewError || rowsError || exportError;
  const [exporting, setExporting] = useState(false);
  const [drafts, setDrafts] = useState<
    Record<number, { label: string; notes: string }>
  >({});
  const alive = useRef(true);
  const explorer = useRef<HTMLDivElement>(null);
  function chooseFeature(index: number) {
    setSelected(index);
    if (window.matchMedia("(max-width: 1150px)").matches) {
      requestAnimationFrame(() => {
        const heading = explorer.current?.querySelector<HTMLElement>(
          ".crosscoder-inspector h2",
        );
        if (heading) {
          heading.tabIndex = -1;
          heading.scrollIntoView({ block: "start", behavior: "instant" });
          heading.focus({ preventScroll: true });
        }
      });
    }
  }
  const refreshRows = useCallback(() => setRevision((value) => value + 1), []);
  const editDraft = useCallback(
    (index: number, value: { label: string; notes: string } | null) => {
      setDrafts((current) => {
        const next = { ...current };
        if (value) next[index] = value;
        else delete next[index];
        return next;
      });
    },
    [],
  );
  useEffect(() => {
    alive.current = true;
    const controller = new AbortController();
    setOverviewError(null);
    api<CrosscoderOverview>("/crosscoders/" + identifier + "/overview", {
      signal: controller.signal,
    })
      .then((value) => {
        if (!controller.signal.aborted) setOverview(value);
      })
      .catch((reason) => {
        if (!controller.signal.aborted) setOverviewError(reason.message);
      });
    return () => {
      alive.current = false;
      controller.abort();
    };
  }, [identifier, refreshToken]);
  useEffect(() => {
    const controller = new AbortController();
    setReading(true);
    setRowsError(null);
    const timer = setTimeout(
      () => {
        const query = new URLSearchParams({
          q: search,
          sort,
          direction,
          offset: String(offset),
          limit: "20",
        });
        api<SharedFeaturePage>(
          "/crosscoders/" + identifier + "/features?" + query,
          { signal: controller.signal },
        )
          .then((value) => {
            if (!controller.signal.aborted) {
              setPage(value);
              setSelected(
                (current) => current ?? value.features[0]?.index ?? null,
              );
            }
          })
          .catch((reason) => {
            if (!controller.signal.aborted) {
              setRowsError(reason.message);
              setPage(null);
            }
          })
          .finally(() => {
            if (!controller.signal.aborted) setReading(false);
          });
      },
      search ? 200 : 0,
    );
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [identifier, search, sort, direction, offset, revision, refreshToken]);
  async function exportRecord() {
    setExporting(true);
    setExportError(null);
    try {
      const bundle = await api("/crosscoders/" + identifier + "/export");
      if (alive.current)
        downloadJSON(bundle, "crosscoder-" + identifier + ".json");
    } catch (reason) {
      if (alive.current) setExportError((reason as Error).message);
    } finally {
      if (alive.current) setExporting(false);
    }
  }
  return (
    <>
      <ErrorNotice message={error} />
      {!overview && !error && <Loading />}
      {overview && (
        <>
          <Panel
            title={overview.name}
            translateTitle={false}
            subtitle={t(
              "Held out reconstruction and joint feature evidence from the recorded checkpoints.",
            )}
            action={
              <div className="header-actions">
                <button
                  className="button secondary"
                  disabled={exporting}
                  onClick={exportRecord}
                >
                  <Download size={14} />
                  {t("Export evidence")}
                </button>
                <a
                  className="button secondary"
                  href={"/api/crosscoders/" + identifier + "/weights"}
                  download
                >
                  {t("Download weights")}
                </a>
              </div>
            }
          >
            <div className="feature-form">
              <div className="crosscoder-sources">
                {overview.banks.map((bank, index) => (
                  <CaptureCard
                    key={index}
                    capture={bank}
                    role={index === 0 ? "Reference" : "Compared"}
                  />
                ))}
              </div>
              <div className="metrics-grid crosscoder-overview-metrics">
                <Metric
                  label="Reference explained variance"
                  value={
                    overview.validation_metrics.banks[0].explained_variance ==
                    null
                      ? t("Unavailable")
                      : number(
                          overview.validation_metrics.banks[0]
                            .explained_variance,
                        )
                  }
                />
                <Metric
                  label="Compared explained variance"
                  value={
                    overview.validation_metrics.banks[1].explained_variance ==
                    null
                      ? t("Unavailable")
                      : number(
                          overview.validation_metrics.banks[1]
                            .explained_variance,
                        )
                  }
                />
                <Metric
                  label="Mean active features"
                  value={number(
                    overview.validation_metrics.mean_active_features,
                    1,
                  )}
                  note={t("of {v0} features", { v0: overview.features })}
                />
                <Metric
                  label="Held out token positions"
                  value={overview.validation_metrics.samples}
                />
              </div>
              {overview.training_summary.cancelled && (
                <p className="notice">
                  {t(
                    "Training stopped early. These measurements use the retained partial dictionary.",
                  )}
                </p>
              )}
              <details className="feature-advanced">
                <summary>{t("Training diagnostics and provenance")}</summary>
                <div className="crosscoder-details feature-form">
                  <p className="muted">
                    {t("{v0} of {v1} optimizer steps completed", {
                      v0: overview.training_summary.completed_steps,
                      v1: overview.training_summary.requested_steps,
                    })}
                  </p>
                  <SeriesChart
                    data={overview.history.map((row) => ({
                      step: row.step,
                      train: row.train_sample_normalized_l2_error,
                      validation: row.validation_normalized_l2_error,
                    }))}
                    xKey="step"
                    interpolation="linear"
                    series={[
                      {
                        key: "train",
                        label: "Training reconstruction",
                        color: "var(--blue)",
                      },
                      {
                        key: "validation",
                        label: "Held out reconstruction",
                        color: "var(--action)",
                      },
                    ]}
                  />
                  <div className="metrics-grid">
                    <Metric
                      label="Reference raw MSE"
                      value={number(
                        overview.validation_metrics.banks[0].mse,
                        6,
                      )}
                    />
                    <Metric
                      label="Compared raw MSE"
                      value={number(
                        overview.validation_metrics.banks[1].mse,
                        6,
                      )}
                    />
                    <Metric
                      label="Sparsity penalty"
                      value={number(overview.training.l1_coefficient)}
                    />
                  </div>
                  <p className="crosscoder-hash">
                    <strong>{t("Manifest checksum")}</strong>
                    <code dir="ltr">{overview.manifest_sha256}</code>
                  </p>
                  <p className="crosscoder-hash">
                    <strong>{t("Tensor checksum")}</strong>
                    <code dir="ltr">{overview.tensors_sha256}</code>
                  </p>
                  <p className="muted">
                    {t(
                      "The manifest checksum identifies the complete immutable export. Labels and notes are stored separately.",
                    )}
                  </p>
                </div>
              </details>
              <p className="crosscoder-scope">
                {t(
                  "Relative decoder norms do not establish semantic exclusivity or causal importance. Decoder cosine assumes aligned model coordinates. Independently trained comparisons do not share feature indices.",
                )}
              </p>
            </div>
          </Panel>
          <div className="crosscoder-explorer" ref={explorer}>
            <Panel title={t("Joint features")}>
              <div className="feature-form">
                <label>
                  {t("Search features")}
                  <input
                    dir="auto"
                    maxLength={100}
                    value={search}
                    onChange={(event) => {
                      setSearch(event.target.value);
                      setOffset(0);
                    }}
                    placeholder={t("Feature index, label or notes")}
                  />
                </label>
                <div className="crosscoder-ranking">
                  <label>
                    {t("Rank features")}
                    <select
                      aria-label={t("Rank features")}
                      value={sort}
                      onChange={(event) => {
                        setSort(event.target.value);
                        setOffset(0);
                      }}
                    >
                      {[
                        ["frequency", "Activation frequency"],
                        ["reference", "Reference decoder share"],
                        ["compared", "Compared decoder share"],
                        ["cosine", "Decoder cosine"],
                      ].map(([value, label]) => (
                        <option
                          key={value}
                          value={value}
                          disabled={
                            value === "cosine" &&
                            overview.banks[0].input_dim !==
                              overview.banks[1].input_dim
                          }
                        >
                          {t(label)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    {t("Order")}
                    <select
                      aria-label={t("Order")}
                      value={direction}
                      onChange={(event) => {
                        setDirection(event.target.value);
                        setOffset(0);
                      }}
                    >
                      <option value="highest">{t("Highest first")}</option>
                      <option value="lowest">{t("Lowest first")}</option>
                    </select>
                  </label>
                </div>
                {overview.banks[0].input_dim !==
                  overview.banks[1].input_dim && (
                  <p className="muted">
                    {t(
                      "Decoder cosine requires equal bank widths and aligned model coordinates.",
                    )}
                  </p>
                )}
                {reading && <Loading />}
                {!reading && page?.features.length === 0 && (
                  <p className="muted">{t("No matching features")}</p>
                )}
                <div className="crosscoder-feature-list" aria-busy={reading}>
                  {page?.features.map((feature) => (
                    <button
                      key={feature.index}
                      className={
                        "crosscoder-feature-row" +
                        (selected === feature.index ? " selected" : "")
                      }
                      aria-pressed={selected === feature.index}
                      disabled={reading}
                      onClick={() => chooseFeature(feature.index)}
                    >
                      <span>
                        <strong>
                          {t("Feature {v0}", { v0: feature.index })}
                        </strong>
                        <small dir="auto">
                          {feature.label || t("Unlabeled")}
                        </small>
                      </span>
                      <span>
                        <small>{t("Frequency")}</small>
                        <bdi>{number(feature.frequency * 100, 1)}%</bdi>
                      </span>
                      <DecoderBalance feature={feature} />
                    </button>
                  ))}
                </div>
                {page && (
                  <div className="crosscoder-pagination">
                    <button
                      className="button secondary"
                      disabled={offset === 0 || reading}
                      onClick={() =>
                        setOffset((value) => Math.max(0, value - 20))
                      }
                    >
                      {t("Previous")}
                    </button>
                    <span>{t("{v0} results", { v0: page.total })}</span>
                    <button
                      className="button secondary"
                      disabled={offset + 20 >= page.total || reading}
                      onClick={() => setOffset((value) => value + 20)}
                    >
                      {t("Next")}
                    </button>
                  </div>
                )}
              </div>
            </Panel>
            {selected == null ? (
              <EmptyState
                icon={<Sparkles />}
                title={t("Choose a shared feature")}
              >
                {t(
                  "Inspect joint activations, decoder geometry and research notes.",
                )}
              </EmptyState>
            ) : (
              <SharedFeatureInspector
                key={identifier + "-" + selected}
                identifier={identifier}
                index={selected}
                draft={drafts[selected]}
                editDraft={editDraft}
                saved={refreshRows}
              />
            )}
          </div>
        </>
      )}
    </>
  );
}
