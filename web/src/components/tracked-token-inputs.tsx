"use client";

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
  if (!target || target.skipped) return <div id={id} />;
  return (
    <div
      id={id}
      className={"target-resolution" + (target.valid ? "" : " invalid")}
    >
      {target.token_count === 1 ? (
        <p>
          Vocabulary ID {target.token_ids[0]} ·{" "}
          {JSON.stringify(target.tokens[0] ?? "")}
          {target.special_tokens[0] && " · special token"}
        </p>
      ) : (
        <p>Split into {target.token_count} vocabulary tokens.</p>
      )}
      {target.error && <p>{target.error}</p>}
      {target.pieces.length > 0 && (
        <details className="target-token-pieces">
          <summary>Native token pieces</summary>
          <div>
            {target.pieces.map((piece, index) => (
              <code key={index}>
                {target.token_ids[index]} · {JSON.stringify(piece)}
              </code>
            ))}
            {target.truncated && (
              <span>
                {target.token_count - target.pieces.length} more tokens
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
  const prefix = useId();
  const ready =
    preview.data?.targets.filter((target) => !target.skipped && target.valid)
      .length ?? 0;
  const invalid = preview.data?.valid === false;
  return (
    <details className="attention-capture-options">
      <summary>
        <span>Track target tokens</span>
        {invalid && <span className="pill orange">Needs attention</span>}
      </summary>
      <p>
        Record up to eight chosen vocabulary tokens through the logit lens. Each
        entry must encode to one token. Leading spaces matter.
      </p>
      {preview.pending ? (
        <p>Checking target tokens</p>
      ) : preview.data ? (
        <p role="status">
          {invalid
            ? "Fix the marked targets before tracing."
            : ready + " target token" + (ready === 1 ? "" : "s") + " ready."}
        </p>
      ) : null}
      {tokens.map((token, index) => {
        const id = prefix + "-target-" + index;
        const description = id + "-resolution";
        const target = preview.data?.targets[index];
        return (
          <div className="tracked-token-row" key={index}>
            <label htmlFor={id}>Target token {index + 1}</label>
            <div className="tracked-token-control">
              <input
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
                aria-label={"Remove target token " + (index + 1)}
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
        <Plus size={14} /> Add target token
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
            Retry target preview
          </button>
        </>
      )}
      <p>
        Empty entries are skipped. Run a new trace after changing targets. Saved
        scores can be inspected at any captured prompt position.
      </p>
    </details>
  );
}
