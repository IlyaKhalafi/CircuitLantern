"use client";
import { useI18n } from "@/i18n/provider";
import { useId } from "react";
import { Plus, X } from "lucide-react";
import type { TargetTokenPreview } from "@/lib/types";
import type { TargetPreviewState } from "@/lib/target-preview";
function Resolution({
  target,
  id,
}: {
  target?: TargetTokenPreview;
  id: string;
}) {
  const { t, rich } = useI18n();
  if (!target || target.skipped) return <div id={id} />;
  return (
    <div
      id={id}
      className={"target-resolution" + (target.valid ? "" : " invalid")}
    >
      {target.token_count === 1 ? (
        <p>
          {rich("Vocabulary ID {v0} \u00B7 {v1}{v2}", {
            v0: target.token_ids[0],
            v1: JSON.stringify(target.tokens[0] ?? ""),
            v2: target.special_tokens[0] && t(" \u00B7 special token"),
          })}
        </p>
      ) : (
        <p>
          {rich("Split into {v0} vocabulary tokens.", {
            v0: target.token_count,
          })}
        </p>
      )}
      {target.error && <p>{target.error}</p>}
      {target.pieces.length > 0 && (
        <details className="target-token-pieces">
          <summary>{t("Native token pieces")}</summary>
          <div>
            {target.pieces.map((piece, index) => (
              <code key={index}>
                {target.token_ids[index]} · {JSON.stringify(piece)}
              </code>
            ))}
            {target.truncated && (
              <span>
                {rich("{v0} more tokens", {
                  v0: target.token_count - target.pieces.length,
                })}
              </span>
            )}
          </div>
        </details>
      )}
    </div>
  );
}
export function TrackedTokenInputs({
  tokens,
  onChange,
  disabled,
  preview,
}: {
  tokens: string[];
  onChange: (tokens: string[]) => void;
  disabled: boolean;
  preview: TargetPreviewState;
}) {
  const { t, rich } = useI18n();
  const prefix = useId();
  const ready =
    preview.data?.targets.filter((target) => !target.skipped && target.valid)
      .length ?? 0;
  const invalid = preview.data?.valid === false;
  return (
    <details className="attention-capture-options">
      <summary>
        <span>{t("Track target tokens")}</span>
        {invalid && <span className="pill orange">{t("Needs attention")}</span>}
      </summary>
      <p>
        {t(
          "Record up to eight chosen vocabulary tokens through the logit lens. Each entry must encode to one token. Leading spaces matter.",
        )}
      </p>
      {preview.pending ? (
        <p>{t("Checking target tokens")}</p>
      ) : preview.data ? (
        <p role="status">
          {invalid
            ? t("Fix the marked targets before tracing.")
            : t(
                "{count, plural, one {# target token ready.} other {# target tokens ready.}}",
                { count: ready },
              )}
        </p>
      ) : null}
      {tokens.map((token, index) => {
        const id = prefix + "-target-" + index;
        const description = id + "-resolution";
        const target = preview.data?.targets[index];
        return (
          <div className="tracked-token-row" key={index}>
            <label htmlFor={id}>
              {rich("Target token {v0}", {
                v0: index + 1,
              })}
            </label>
            <div className="tracked-token-control">
              <input
                dir="auto"
                id={id}
                className="form-input"
                value={token}
                aria-describedby={description}
                aria-invalid={target && !target.valid ? true : undefined}
                onChange={(event) =>
                  onChange(
                    tokens.map((value, i) =>
                      i === index ? event.target.value : value,
                    ),
                  )
                }
                disabled={disabled}
                maxLength={100}
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                className="icon-button"
                aria-label={t("Remove target token {v0}", {
                  v0: index + 1,
                })}
                disabled={disabled}
                onClick={() => onChange(tokens.filter((_, i) => i !== index))}
              >
                <X size={15} />
              </button>
            </div>
            <Resolution target={target} id={description} />
          </div>
        );
      })}
      <button
        type="button"
        className="button secondary"
        disabled={disabled || tokens.length >= 8}
        onClick={() => onChange([...tokens, ""])}
      >
        <Plus size={14} />
        {t("Add target token")}
      </button>
      {preview.error && (
        <>
          <p>{preview.error}</p>
          <button
            type="button"
            className="button secondary"
            disabled={disabled || preview.pending}
            onClick={preview.retry}
          >
            {t("Retry target preview")}
          </button>
        </>
      )}
      <p>
        {t(
          "Empty entries are skipped. Run a new trace after changing targets. Saved scores can be inspected at any captured prompt position.",
        )}
      </p>
    </details>
  );
}
