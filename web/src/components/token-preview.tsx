"use client";
import { useI18n } from "@/i18n/provider";
import { useState } from "react";
import { ChevronRight } from "lucide-react";
import type { TokenPreviewState } from "@/lib/token-preview";
function visibleToken(token: string) {
  return token.replace(/ /g, "·").replace(/\n/g, "↵") || "∅";
}
export function TokenPreview({
  preview,
  labels = ["Prompt"],
}: {
  preview: TokenPreviewState;
  labels?: string[];
}) {
  const { t, rich } = useI18n();
  const [expanded, setExpanded] = useState(false);
  const data = preview.data;
  if (!data)
    return preview.pending ? (
      <p className="token-preview-status" role="status">
        {t("Counting prompt tokens")}
      </p>
    ) : preview.error ? (
      <p className="token-preview-status">
        {rich("{v0} Running a capture will check its token count.", {
          v0: preview.error,
        })}
      </p>
    ) : null;
  const pair = data.prompts.length === 2;
  const aligned =
    !pair || data.prompts[0].token_count === data.prompts[1].token_count;
  const rows = Math.max(
    ...data.prompts.map((prompt) => prompt.token_ids.length),
  );
  const truncated = data.prompts.some((prompt) => prompt.truncated);
  return (
    <div className="token-preview">
      <details open={expanded}>
        <summary
          onClick={(event) => {
            event.preventDefault();
            setExpanded((value) => !value);
          }}
        >
          <ChevronRight size={12} aria-hidden="true" />
          <span>{t("Inspect tokenization")}</span>
          {data.prompts.map((prompt, i) => (
            <span
              className={`pill ${prompt.fits ? "subtle" : "orange"}`}
              key={i}
            >
              {rich("{v0} {v1} / {v2} tokens", {
                v0: labels[i] || `Prompt ${i + 1}`,
                v1: prompt.token_count,
                v2: data.token_limit,
              })}
            </span>
          ))}
        </summary>
        <p className="token-preview-caption">
          {rich(
            "Counts include any tokenizer-added special tokens. Spaces appear as \u00B7 and line breaks as \u21B5.{v0}",
            {
              v0:
                truncated &&
                t(
                  " Showing the first 256 positions. Full token counts appear above.",
                ),
            },
          )}
        </p>
        <div className="table-wrap">
          <table
            aria-label={
              pair ? t("Prompt token alignment") : t("Prompt tokenization")
            }
          >
            <thead>
              <tr>
                <th>{t("POSITION")}</th>
                {pair ? (
                  <>
                    <th>{labels[0] || t("Clean")}</th>
                    <th>{labels[1] || t("Corrupted")}</th>
                    <th>{t("ALIGNMENT")}</th>
                  </>
                ) : (
                  <>
                    <th>{t("TOKEN")}</th>
                    <th>{t("VOCABULARY ID")}</th>
                    <th>{t("NATIVE PIECE")}</th>
                    <th>{t("TYPE")}</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: rows }, (_, position) => {
                const changed =
                  pair &&
                  data.prompts[0].token_ids[position] !==
                    data.prompts[1].token_ids[position];
                return (
                  <tr
                    key={position}
                    className={changed ? "token-mismatch" : undefined}
                  >
                    <td className="code">{position}</td>
                    {pair ? (
                      <>
                        {data.prompts.map((prompt, i) => (
                          <td key={i}>
                            {position < prompt.token_ids.length ? (
                              <>
                                <code>
                                  {visibleToken(prompt.tokens[position])}
                                </code>
                                <small>
                                  {rich("ID {v0} \u00B7 {v1}{v2}", {
                                    v0: prompt.token_ids[position],
                                    v1: prompt.pieces[position],
                                    v2: prompt.special_tokens[position]
                                      ? t(" \u00B7 special")
                                      : "",
                                  })}
                                </small>
                              </>
                            ) : (
                              <span>{t("Missing token")}</span>
                            )}
                          </td>
                        ))}
                        <td>{changed ? t("Changed") : t("Same")}</td>
                      </>
                    ) : (
                      <>
                        <td className="code">
                          {visibleToken(data.prompts[0].tokens[position])}
                        </td>
                        <td className="code">
                          {data.prompts[0].token_ids[position]}
                        </td>
                        <td className="code">
                          {data.prompts[0].pieces[position]}
                        </td>
                        <td>
                          {data.prompts[0].special_tokens[position]
                            ? t("Special")
                            : t("Standard")}
                        </td>
                      </>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
      {!aligned && data.prompts.every((prompt) => prompt.token_count > 0) && (
        <p className="token-preview-warning" role="status">
          {t(
            "Clean and corrupted prompts have different token counts. Adjust the wording so positions align.",
          )}
        </p>
      )}
      {data.prompts.map(
        (prompt, i) =>
          !prompt.fits && (
            <p className="token-preview-warning" key={i} role="status">
              {prompt.token_count === 0
                ? t("{v0} needs at least one token.", {
                    v0: labels[i] || t("Prompt"),
                  })
                : t("{v0} exceeds the {v1} token capture allowance.", {
                    v0: labels[i] || t("Prompt"),
                    v1: data.token_limit,
                  })}
            </p>
          ),
      )}
    </div>
  );
}
