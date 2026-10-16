"use client";

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
  const [expanded, setExpanded] = useState(false);
  const data = preview.data;
  if (!data)
    return preview.pending ? (
      <p className="token-preview-status" role="status">
        Counting prompt tokens
      </p>
    ) : preview.error ? (
      <p className="token-preview-status">
        {preview.error} Running a capture will check its token count.
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
      <details
        open={expanded}
        onToggle={(event) => setExpanded(event.currentTarget.open)}
      >
        <summary>
          <ChevronRight size={12} aria-hidden="true" />
          <span>Inspect tokenization</span>
          {data.prompts.map((prompt, i) => (
            <span
              className={`pill ${prompt.fits ? "subtle" : "orange"}`}
              key={i}
            >
              {labels[i] || `Prompt ${i + 1}`} {prompt.token_count} /{" "}
              {data.token_limit} tokens
            </span>
          ))}
        </summary>
        <p className="token-preview-caption">
          Counts include any tokenizer-added special tokens. Spaces appear as ·
          and line breaks as ↵.
          {truncated &&
            " Showing the first 256 positions. Full token counts appear above."}
        </p>
        <div className="table-wrap">
          <table
            aria-label={pair ? "Prompt token alignment" : "Prompt tokenization"}
          >
            <thead>
              <tr>
                <th>POSITION</th>
                {pair ? (
                  <>
                    <th>{labels[0] || "Clean"}</th>
                    <th>{labels[1] || "Corrupted"}</th>
                    <th>ALIGNMENT</th>
                  </>
                ) : (
                  <>
                    <th>TOKEN</th>
                    <th>VOCABULARY ID</th>
                    <th>NATIVE PIECE</th>
                    <th>TYPE</th>
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
                                  ID {prompt.token_ids[position]} ·{" "}
                                  {prompt.pieces[position]}
                                  {prompt.special_tokens[position]
                                    ? " · special"
                                    : ""}
                                </small>
                              </>
                            ) : (
                              <span>Missing token</span>
                            )}
                          </td>
                        ))}
                        <td>{changed ? "Changed" : "Same"}</td>
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
                            ? "Special"
                            : "Standard"}
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
          Clean and corrupted prompts have different token counts. Adjust the
          wording so positions align.
        </p>
      )}
      {data.prompts.map(
        (prompt, i) =>
          !prompt.fits && (
            <p className="token-preview-warning" key={i} role="status">
              {prompt.token_count === 0
                ? `${labels[i] || "Prompt"} needs at least one token.`
                : `${labels[i] || "Prompt"} exceeds the ${data.token_limit} token capture allowance.`}
            </p>
          ),
      )}
    </div>
  );
}
