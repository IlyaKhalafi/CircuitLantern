import { expect, test as base, type APIRequestContext } from "@playwright/test";
import type { InductionResult } from "../src/lib/induction";
import type { RuntimeState } from "../src/lib/types";

export const parameters = {
  block_length: 4,
  repeats: 2,
  examples: 2,
  causal_heads: 1,
  seed: 19,
};
export async function load(
  request: APIRequestContext,
  model: string,
  settings: Record<string, unknown> = {},
) {
  const response = await request.post("/api/models/load", {
    data: {
      model_id: model,
      device: "cpu",
      precision: "float32",
      context_length: 128,
      ...settings,
    },
  });
  expect(response.status()).toBe(202);
  const job = await response.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + job.id)).json()).status,
    )
    .toBe("ready");
}
export const test = base.extend<{ researchSession: void }>({
  researchSession: [
    async ({ request }, use) => {
      const before: RuntimeState = await (
        await request.get("/api/runtime")
      ).json();
      await load(request, "hf-internal-testing/tiny-random-gpt2");
      try {
        await use();
      } finally {
        if (before.loaded && before.metadata) {
          await load(request, before.metadata.id, {
            revision: before.metadata.revision,
            device: before.device,
            precision:
              before.precision || before.metadata.execution_precision || "auto",
            context_length:
              before.context_length ||
              before.metadata.execution_context_length ||
              128,
          });
          const restored: RuntimeState = await (
            await request.get("/api/runtime")
          ).json();
          expect(restored.context_length).toBe(before.context_length);
          expect(restored.precision).toBe(before.precision);
          expect(restored.metadata?.revision).toBe(before.metadata.revision);
        } else
          expect((await request.delete("/api/runtime")).status()).toBe(204);
      }
    },
    { auto: true, timeout: 60000 },
  ],
});

export async function seed(request: APIRequestContext) {
  const queued = await request.post("/api/induction/probe", {
    data: parameters,
  });
  expect(queued.status()).toBe(202);
  const job = await queued.json();
  let result: InductionResult | undefined;
  await expect
    .poll(async () => {
      const value = await (await request.get("/api/jobs/" + job.id)).json();
      result = value.result;
      return value.status;
    })
    .toBe("ready");
  return { id: job.id, result: result! };
}
