"use client";
import { useI18n } from "@/i18n/provider";
import type { Trace } from "@/lib/types";
export function GeneratedOutput({ trace }: { trace: Trace }) {
  const { t, rich } = useI18n();
  const contextual = typeof trace.continuation_text === "string";
  const captureOnly = trace.generation_performed === false;
  return (
    <>
      <div
        className="trace-output"
        aria-label={
          captureOnly ? t("Captured prompt") : t("Decoded generation")
        }
      >
        {contextual ? (
          <>
            <span>{trace.decoded_prompt}</span>
            <strong>{trace.continuation_text}</strong>
          </>
        ) : trace.generated_text !== undefined ? (
          trace.generated_text
        ) : (
          <>
            <span>{trace.prompt}</span>
            <strong>{trace.continuation}</strong>
          </>
        )}
      </div>
      {captureOnly && (
        <p className="inline-note">
          {t(
            "Prompt capture only. No continuation was generated. Activations, attention and next token predictions describe the prompt forward pass.",
          )}
        </p>
      )}
      {trace.generated_token_ids && trace.generated_token_ids.length > 0 && (
        <details className="generated-token-details">
          <summary>{t("Generated token IDs")}</summary>
          <p className="inline-note">
            {t(
              "These tokens follow the prompt. Layer activations and attention describe the prompt capture. Vocabulary pieces can differ from decoded text and special tokens may be hidden in the output.",
            )}
          </p>
          {trace.generation_settings && (
            <p className="inline-note">
              {rich(
                "Greedy generation requested up to {v0} new tokens and returned {v1}. Exports include these overrides and whether prompt attention was captured. Remaining generation defaults come from the pinned checkpoint.",
                {
                  v0: trace.generation_settings.max_new_tokens,
                  v1: trace.generated_token_ids.length,
                },
              )}
            </p>
          )}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>{t("POSITION")}</th>
                  <th>{t("TOKEN ID")}</th>
                  <th>{t("DECODED TOKEN")}</th>
                  <th>{t("VOCABULARY PIECE")}</th>
                </tr>
              </thead>
              <tbody>
                {trace.generated_token_ids.map((id, i) => (
                  <tr key={i}>
                    <td>{trace.token_ids.length + i}</td>
                    <td className="code">{id}</td>
                    <td className="code">
                      {JSON.stringify(trace.generated_tokens?.[i] ?? "")}
                    </td>
                    <td className="code">{trace.generated_pieces?.[i]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </>
  );
}
