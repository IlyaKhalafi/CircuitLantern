"use client";
import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Loading } from "@/components/ui";
import { api, downloadJSON, number } from "@/lib/api";
import type { DecoderDirections } from "@/lib/features";

function DirectionResults({
  dictionaryId,
  feature,
  onSelect,
}: {
  dictionaryId: string;
  feature: number;
  onSelect: (feature: number) => void;
}) {
  const { t } = useI18n();
  const [result, setResult] = useState<DecoderDirections | null>(null);
  const [order, setOrder] = useState<"highest" | "lowest">("highest");
  const [retry, setRetry] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setResult(null);
    api<DecoderDirections>(
      "/features/dictionaries/" +
        dictionaryId +
        "/features/" +
        feature +
        "/directions",
      { signal: controller.signal },
    )
      .then((value) => {
        if (!controller.signal.aborted) setResult(value);
      })
      .catch((reason) => {
        if (!controller.signal.aborted) setError((reason as Error).message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [dictionaryId, feature, retry]);
  return (
    <div className="feature-directions-content">
      <p className="muted">
        {t(
          "Geometric overlap does not establish shared meaning, activation correlation or causal influence.",
        )}
      </p>
      {loading && <Loading label={t("Comparing decoder directions")} />}
      <ErrorNotice message={error} />
      {error && (
        <button
          className="button secondary"
          onClick={() => setRetry((v) => v + 1)}
        >
          {t("Try again")}
        </button>
      )}
      {result && !loading && (
        <>
          <div className="feature-toolbar">
            <label>
              {t("Direction ranking")}
              <select
                aria-label={t("Direction ranking")}
                value={order}
                onChange={(event) =>
                  setOrder(event.target.value as "highest" | "lowest")
                }
              >
                <option value="highest">{t("Highest cosine")}</option>
                <option value="lowest">{t("Lowest cosine")}</option>
              </select>
            </label>
            <button
              className="button secondary"
              onClick={() =>
                downloadJSON(
                  result,
                  "decoder-directions-" +
                    dictionaryId +
                    "-" +
                    feature +
                    ".json",
                )
              }
            >
              <Download size={14} />
              {t("Export directions")}
            </button>
          </div>
          <p className="muted">
            {t("Compared {v0} decoder directions", {
              v0: result.compared_features,
            })}
            {" · "}
            {t(
              "Inactive features are included. Frequencies use the recorded corpus.",
            )}
          </p>
          {result[order].length ? (
            <div
              className="table-wrap"
              role="region"
              aria-label={t("Decoder direction comparisons")}
              tabIndex={0}
            >
              <table>
                <thead>
                  <tr>
                    <th>{t("Feature")}</th>
                    <th>{t("Decoder cosine")}</th>
                    <th>{t("Activation frequency")}</th>
                  </tr>
                </thead>
                <tbody>
                  {result[order].map((row) => (
                    <tr key={row.feature}>
                      <th scope="row">
                        <button
                          className="feature-direction-link"
                          onClick={() => onSelect(row.feature)}
                        >
                          {t("Feature {v0}", { v0: row.feature })}
                          {row.label && <small dir="auto">{row.label}</small>}
                        </button>
                      </th>
                      <td className="mono">{number(row.cosine, 4)}</td>
                      <td className="mono">
                        {row.frequency === null
                          ? t("Unavailable")
                          : number(row.frequency * 100, 2) + "%"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">
              {t("No other nonzero decoder directions are available.")}
            </p>
          )}
        </>
      )}
    </div>
  );
}

export function FeatureDirections({
  dictionaryId,
  feature,
  onSelect,
}: {
  dictionaryId: string;
  feature: number;
  onSelect: (feature: number) => void;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  return (
    <details
      className="feature-directions"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>{t("Related decoder directions")}</summary>
      {open && (
        <DirectionResults
          key={dictionaryId + ":" + feature}
          dictionaryId={dictionaryId}
          feature={feature}
          onSelect={onSelect}
        />
      )}
    </details>
  );
}
