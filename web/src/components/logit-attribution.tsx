"use client";

import "./logit-attribution.css";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, CheckCircle2, AlertTriangle } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { ErrorNotice, Panel } from "@/components/ui";
import {
  attributionUnavailable,
  contributionLabel,
  selectedAttribution,
} from "@/lib/attribution";
import { preparePatchingHypothesis } from "@/lib/patching-hypothesis";
import { executionProfileKnown } from "@/lib/comparison-profile";
import { targetLabel } from "@/lib/target-trajectories";
import { number } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import type { Trace } from "@/lib/types";

export function LogitAttribution({ trace }: { trace: Trace }) {
  const { t } = useI18n();
  const router = useRouter();
  const [targetIndex, setTargetIndex] = useState(0);
  const [alternativeIndex, setAlternativeIndex] = useState(1);
  const [sort, setSort] = useState("layer");
  const [error, setError] = useState<string | null>(null);
  const result = trace.logit_attribution;
  const targets = result?.targets ?? [];
  const target = targets[targetIndex] ? targetIndex : 0;
  const alternative =
    alternativeIndex >= 0 &&
    alternativeIndex !== target &&
    targets[alternativeIndex]
      ? alternativeIndex
      : -1;
  const validPosition =
    result?.position === trace.token_ids.length - 1 &&
    result?.token_id === trace.token_ids.at(-1) &&
    result?.scope === "final_prompt_token";
  const selected =
    result && validPosition
      ? selectedAttribution(result, target, alternative)
      : null;
  const ready = result?.status === "available" && !!selected;
  const rows = selected ? [...selected.rows] : [];
  if (sort === "magnitude")
    rows.sort(
      (a, b) => Math.abs(b.value) - Math.abs(a.value) || a.index - b.index,
    );
  if (sort === "positive")
    rows.sort((a, b) => b.value - a.value || a.index - b.index);
  if (sort === "negative")
    rows.sort((a, b) => a.value - b.value || a.index - b.index);
  const maximum = Math.max(1e-30, ...rows.map((row) => Math.abs(row.value)));
  const canPrepare =
    !!selected?.patchingAllowed &&
    !!targets[target]?.patchable &&
    !!targets[alternative]?.patchable &&
    executionProfileKnown(trace) &&
    !!trace.trace_id;
  const healthLabels: Record<string, string> = {
    ok: "Contribution sum verified",
    rounding: "Native precision rounding",
    near_zero: "Near-zero observed contrast",
    contrast_error: "Large contrast reconstruction error",
    mismatch: "Reconstruction mismatch",
  };
  return (
    <Panel
      title={t("Score contributions")}
      help={
        t(
          "Direct logit attribution projects actual block updates onto the selected answer direction using a frozen final normalization scale.",
        ) +
        " " +
        t(
          "Recorded answers can be compared without another forward pass, including after unloading the model. Track other target tokens before a new capture to add more answers.",
        )
      }
    >
      <div className="attribution-body">
        <p className="scientific-caveat">
          {t(
            "A decomposition of the observed logit with frozen normalization. It is not an ablation effect.",
          )}
        </p>
        {!result ? (
          <p className="notice info">
            {t(
              "Score contributions were not recorded in this older capture. Run a new trace to measure them.",
            )}
          </p>
        ) : !ready ? (
          <p className="notice warning" role="status">
            {t(attributionUnavailable(result.reason))}
          </p>
        ) : (
          selected && (
            <>
              <div className="attribution-controls">
                <label>
                  {t("Answer token")}
                  <select
                    aria-label={t("Attribution answer")}
                    value={target}
                    onChange={(event) => {
                      const next = Number(event.target.value);
                      if (next === alternative) setAlternativeIndex(target);
                      setTargetIndex(next);
                      setError(null);
                    }}
                  >
                    {targets.map((item, index) => (
                      <option key={item.token_id} value={index}>
                        {targetLabel(item)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  {t("Alternative answer")}
                  <select
                    aria-label={t("Attribution alternative")}
                    value={alternative}
                    onChange={(event) => {
                      setAlternativeIndex(Number(event.target.value));
                      setError(null);
                    }}
                  >
                    <option value={-1}>{t("No alternative")}</option>
                    {targets.map(
                      (item, index) =>
                        index !== target && (
                          <option key={item.token_id} value={index}>
                            {targetLabel(item)}
                          </option>
                        ),
                    )}
                  </select>
                </label>
              </div>
              <div className="attribution-context">
                <span>
                  {t("Final prompt position {position}", {
                    position: result.position,
                  })}
                </span>
                <bdi dir="auto">
                  {JSON.stringify(trace.tokens[result.position] ?? "")}
                </bdi>
                <span>
                  {selected.compared
                    ? t("Target logit minus alternative logit")
                    : t("Raw logit, not probability")}
                </span>
              </div>
              <div
                className="attribution-health"
                data-health={selected.health}
                role="status"
              >
                {selected.health === "ok" ? (
                  <CheckCircle2 size={17} />
                ) : (
                  <AlertTriangle size={17} />
                )}
                <strong>{t(healthLabels[selected.health])}</strong>
                <span>
                  <bdi>{t(executionLabel(result.captured_dtype))}</bdi> ·{" "}
                  <bdi>
                    {result.normalization?.kind === "layer_norm"
                      ? "LayerNorm"
                      : result.normalization?.kind === "rms_norm"
                        ? "RMSNorm"
                        : "Identity"}
                  </bdi>
                </span>
              </div>
              <div className="attribution-totals">
                <div>
                  <span>{t("Observed score")}</span>
                  <strong>
                    <bdi>{number(selected.actual, 6)}</bdi>
                  </strong>
                </div>
                <div>
                  <span>{t("Sum of contributions")}</span>
                  <strong>
                    <bdi>{number(selected.reconstructed, 6)}</bdi>
                  </strong>
                </div>
                <div>
                  <span>{t("Reconstruction error")}</span>
                  <strong>
                    <bdi>{number(selected.error, 6)}</bdi>
                  </strong>
                  <small>
                    {selected.relative == null
                      ? t("Relative error is undefined at zero.")
                      : t("{percent}% of the observed score", {
                          percent: number(selected.relative * 100, 3),
                        })}
                  </small>
                </div>
              </div>
              {["contrast_error", "mismatch"].includes(selected.health) && (
                <p className="notice warning">
                  {t(
                    "The reconstruction error is too large to trust this contrast. Try FP32 execution before using it to choose an intervention.",
                  )}
                </p>
              )}
              {result.selection_mode === "top_predictions" && (
                <p className="control-note">
                  {t(
                    "These are the two highest-scoring next tokens from this capture. Add tracked targets before capturing to investigate specific answers.",
                  )}
                </p>
              )}
              <div className="attribution-controls">
                <label>
                  {t("Sort components")}
                  <select
                    aria-label={t("Sort attribution components")}
                    value={sort}
                    onChange={(event) => setSort(event.target.value)}
                  >
                    <option value="layer">{t("Block order")}</option>
                    <option value="magnitude">{t("Largest magnitude")}</option>
                    <option value="positive">{t("Most positive first")}</option>
                    <option value="negative">{t("Most negative first")}</option>
                  </select>
                </label>
                <span className="control-note">
                  {selected.compared
                    ? t(
                        "Positive values point toward the target. Negative values point toward the alternative.",
                      )
                    : t(
                        "Values are additive raw-logit contributions, not changes in probability.",
                      )}
                </span>
              </div>
              <ErrorNotice message={error} />
              <div
                className="attribution-table-scroll"
                role="region"
                aria-label={t("Block score contributions")}
                tabIndex={0}
              >
                <table className="attribution-table">
                  <caption className="sr-only">
                    {t("Final prompt token score decomposition")}
                  </caption>
                  <thead>
                    <tr>
                      <th>{t("Component")}</th>
                      <th>{t("Signed contribution")}</th>
                      <th>{t("Test a hypothesis")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr
                        key={row.index}
                        data-kind={row.kind}
                        data-layer={row.layer ?? ""}
                      >
                        <th scope="row">
                          {t(contributionLabel(row), { layer: row.layer ?? 0 })}
                        </th>
                        <td>
                          <div className="attribution-value">
                            <bdi>
                              {row.value > 0 ? "+" : ""}
                              {number(row.value, 6)}
                            </bdi>
                            <span>
                              {row.value > 0
                                ? t("Positive")
                                : row.value < 0
                                  ? t("Negative")
                                  : t("Zero")}
                            </span>
                          </div>
                          <div
                            className="attribution-track"
                            dir="ltr"
                            aria-hidden="true"
                          >
                            <span
                              className={
                                row.value < 0 ? "negative" : "positive"
                              }
                              style={{
                                width: `${(Math.abs(row.value) / maximum) * 50}%`,
                              }}
                            />
                          </div>
                        </td>
                        <td>
                          {row.kind === "block" && (
                            <button
                              className="button secondary attribution-test"
                              disabled={!canPrepare}
                              aria-label={t(
                                "Prepare patching after block {layer}",
                                { layer: row.layer ?? 0 },
                              )}
                              onClick={() => {
                                if (!canPrepare || row.layer == null) return;
                                try {
                                  router.push(
                                    preparePatchingHypothesis(
                                      trace,
                                      row.layer,
                                      targets[target],
                                      targets[alternative],
                                    ),
                                  );
                                } catch (reason) {
                                  setError((reason as Error).message);
                                }
                              }}
                            >
                              {t("Test with patching")}
                              <ArrowRight size={13} />
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {!canPrepare && (
                <p className="control-note">
                  {!selected.compared
                    ? t(
                        "Choose an alternative answer to prepare a contrastive patching experiment.",
                      )
                    : !targets[target]?.patchable ||
                        !targets[alternative]?.patchable
                      ? t(
                          "These decoded answers do not roundtrip to their captured token IDs. Track explicit answers and capture again before patching.",
                        )
                      : !executionProfileKnown(trace) || !trace.trace_id
                        ? t(
                            "This capture is missing the source profile needed to prepare a verified patching draft.",
                          )
                        : t(
                            "Resolve the reconstruction warning before preparing a patching hypothesis.",
                          )}
                </p>
              )}
              <details className="attribution-method">
                <summary>{t("Normalization and numerical checks")}</summary>
                <dl>
                  <div>
                    <dt>{t("Normalization module")}</dt>
                    <dd>
                      <bdi>{result.normalization?.module}</bdi>
                    </dd>
                  </div>
                  <div>
                    <dt>{t("Frozen denominator")}</dt>
                    <dd>
                      <bdi>{number(result.normalization?.denominator, 8)}</bdi>
                    </dd>
                  </div>
                  <div>
                    <dt>{t("Normalization epsilon")}</dt>
                    <dd>
                      <bdi>{number(result.normalization?.epsilon, 8)}</bdi>
                    </dd>
                  </div>
                  <div>
                    <dt>{t("Gain convention")}</dt>
                    <dd>
                      <bdi>{result.normalization?.weight_mode}</bdi>
                    </dd>
                  </div>
                  <div>
                    <dt>{t("Projection arithmetic")}</dt>
                    <dd>
                      <bdi>{result.compute_dtype}</bdi>
                    </dd>
                  </div>
                  <div>
                    <dt>{t("Strict reconstruction tolerance")}</dt>
                    <dd>
                      <bdi>{number(selected.recordedStrict, 8)}</bdi>
                    </dd>
                  </div>
                  <div>
                    <dt>{t("Selected-score tolerance")}</dt>
                    <dd>
                      <bdi>{number(selected.strict, 8)}</bdi>
                    </dd>
                  </div>
                  <div>
                    <dt>{t("Native rounding allowance")}</dt>
                    <dd>
                      <bdi>{number(selected.allowance, 8)}</bdi>
                    </dd>
                  </div>
                  <div>
                    <dt>{t("Maximum normalization discrepancy")}</dt>
                    <dd>
                      <bdi>
                        {number(result.checks?.maximum_normalization_error, 8)}
                      </bdi>
                    </dd>
                  </div>
                </dl>
                <p>
                  {t(
                    "Rounding allowances are heuristics based on observed readout magnitudes and execution precision. They are not certified error bounds.",
                  )}
                </p>
                <p>
                  {t(
                    "The embedding input includes token and position processing before the first block. Normalization and readout biases are counted once. Saved scalar projections support only the recorded answer IDs.",
                  )}
                </p>
              </details>
            </>
          )
        )}
      </div>
    </Panel>
  );
}
