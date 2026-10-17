import { expect, test } from "@playwright/test";
import { normalizeProbeDraft } from "../src/lib/induction-settings";
import type { InductionSettings } from "../src/lib/induction";

const fallback: InductionSettings = {
  block_length: 25,
  repeats: 4,
  examples: 10,
  causal_heads: 3,
  seed: 19,
  layers: null,
  include_start_token: true,
};

test("probe drafts restore valid fields independently and discard obsolete or invalid fields", () => {
  const restored = normalizeProbeDraft(
    {
      ...fallback,
      examples: null,
      seed: 23,
      legacy_setting: true,
      layers: [2],
      include_start_token: false,
    },
    fallback,
    [0, 1, 2],
  );
  expect(restored).toEqual({
    ...fallback,
    seed: 23,
    layers: [2],
    include_start_token: false,
  });
  expect(restored).not.toHaveProperty("legacy_setting");
  for (const invalid of [null, [], "invalid"])
    expect(normalizeProbeDraft(invalid, fallback, [0, 1, 2])).toEqual(fallback);
  for (const layers of [[999], [1, 1], [true], []])
    expect(
      normalizeProbeDraft(
        { ...fallback, layers, seed: 23 },
        fallback,
        [0, 1, 2],
      ),
    ).toEqual({ ...fallback, seed: 23 });
  expect(
    normalizeProbeDraft(
      {
        ...fallback,
        block_length: 3,
        repeats: 5,
        examples: 33,
        causal_heads: -1,
        seed: 4294967296,
      },
      fallback,
      [0, 1],
    ),
  ).toEqual(fallback);
});
