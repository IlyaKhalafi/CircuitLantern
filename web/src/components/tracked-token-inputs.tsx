"use client";

import { Plus, X } from "lucide-react";

export function TrackedTokenInputs({
  tokens,
  onChange,
  disabled,
}: {
  tokens: string[];
  onChange: (tokens: string[]) => void;
  disabled: boolean;
}) {
  return (
    <details className="attention-capture-options">
      <summary>Track target tokens</summary>
      <p>
        Record up to eight chosen vocabulary tokens through the logit lens. Each
        entry must encode to one token. Leading spaces matter.
      </p>
      {tokens.map((token, index) => (
        <div className="tracked-token-row" key={index}>
          <label>
            Target token {index + 1}
            <input
              className="form-input"
              value={token}
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
          </label>
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
      ))}
      <button
        type="button"
        className="button secondary"
        disabled={disabled || tokens.length >= 8}
        onClick={() => onChange([...tokens, ""])}
      >
        <Plus size={14} /> Add target token
      </button>
      <p>
        Empty entries are skipped. Run a new trace after changing targets. Saved
        scores can be inspected at any captured prompt position.
      </p>
    </details>
  );
}
