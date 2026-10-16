import type { Trace } from "@/lib/types";

export function GeneratedOutput({ trace }: { trace: Trace }) {
  const contextual = typeof trace.continuation_text === "string";
  return (
    <>
      <div className="trace-output" aria-label="Decoded generation">
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
      {trace.generated_token_ids && (
        <details className="generated-token-details">
          <summary>Generated token IDs</summary>
          <p className="inline-note">
            These tokens follow the prompt. Layer activations and attention
            describe the prompt capture. Vocabulary pieces can differ from
            decoded text and special tokens may be hidden in the output.
          </p>
          {trace.generation_settings && (
            <p className="inline-note">
              Greedy generation requested up to{" "}
              {trace.generation_settings.max_new_tokens} new tokens and returned{" "}
              {trace.generated_token_ids.length}. Exports include these
              overrides and whether prompt attention was captured. Remaining
              generation defaults come from the pinned checkpoint.
            </p>
          )}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>POSITION</th>
                  <th>TOKEN ID</th>
                  <th>DECODED TOKEN</th>
                  <th>VOCABULARY PIECE</th>
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
