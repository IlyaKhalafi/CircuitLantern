import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

test("decoded generation preserves word boundaries and exports generated token IDs", async ({
  page,
}) => {
  await page.goto("/playground");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France");
  await page.getByLabel("Generated tokens").selectOption("4");
  const response = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace" }).click();
  const trace = await (await response).json();
  await expect(page.getByLabel("Decoded generation")).toHaveText(
    trace.generated_text,
  );
  expect(trace.generated_text).toBe(
    trace.decoded_prompt + trace.continuation_text,
  );
  expect(trace.continuation_text.startsWith(" ")).toBe(true);
  await page.getByText("Generated token IDs", { exact: true }).click();
  const details = page.locator(".generated-token-details");
  await expect(details.getByRole("row")).toHaveCount(
    trace.generated_token_ids.length + 1,
  );
  const first = details.getByRole("row").nth(1);
  await expect(first.getByRole("cell").nth(0)).toHaveText("4");
  await expect(first.getByRole("cell").nth(1)).toHaveText(
    String(trace.generated_token_ids[0]),
  );
  await expect(first.getByRole("cell").nth(3)).toHaveText(
    trace.generated_pieces[0],
  );
  await expect(details).toContainText("requested up to 4 new tokens");
  await capturePage(page, {
    path: "../artifacts/screenshots/generated-token-ids.png",
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/generated-token-ids-mobile.png",
  });
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export trace" }).click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const exported = JSON.parse(Buffer.concat(chunks).toString());
  expect(exported.generated_token_ids).toEqual(trace.generated_token_ids);
  expect(exported.generated_text).toBe(trace.generated_text);
  expect(exported.generation_settings.max_new_tokens).toBe(4);
  expect(exported.capture_attention).toBe(true);
});

test("a decoder that rewrites the prompt boundary displays the complete sequence", async ({
  page,
}) => {
  await page.route("**/api/trace", async (route) => {
    const response = await route.fetch();
    const trace = await response.json();
    await route.fulfill({
      response,
      json: {
        ...trace,
        decoded_prompt: "hello '",
        generated_text: "hello's",
        continuation_text: null,
        continuation: "s",
      },
    });
  });
  await page.goto("/playground");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(page.getByLabel("Decoded generation")).toHaveText("hello's");
});

test("saved traces without generation metadata retain their original rendering", async ({
  page,
}) => {
  await page.route("**/api/trace", async (route) => {
    const response = await route.fetch();
    const trace = await response.json();
    for (const field of [
      "decoded_prompt",
      "generated_text",
      "continuation_text",
      "generated_token_ids",
      "generated_tokens",
      "generated_pieces",
      "generation_settings",
      "capture_attention",
    ])
      delete trace[field];
    await route.fulfill({
      response,
      json: {
        ...trace,
        prompt: "A legacy prompt",
        continuation: " and its suffix",
      },
    });
  });
  await page.goto("/playground");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(page.getByLabel("Decoded generation")).toHaveText(
    "A legacy prompt and its suffix",
  );
  await expect(
    page.getByText("Generated token IDs", { exact: true }),
  ).toHaveCount(0);
});
