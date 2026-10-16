import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

test("head zero ablation compares a baseline, explains routing and persists provenance", async ({
  page,
  request,
}) => {
  const runtime = await (await request.get("/api/runtime")).json();
  expect(runtime.attention_head_targets).toHaveLength(5);
  expect(runtime.attention_head_targets[0].heads).toBe(4);
  await page.goto("/playground");
  await page.getByLabel("Prompt", { exact: true }).fill("the capital");
  await page.getByLabel("Generated tokens").selectOption("4");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Layer intervention").selectOption("head");
  await page.getByLabel("Head ablation layer").selectOption("2");
  await page.getByLabel("Attention head to zero").selectOption("3");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Layer 2 · head 3 zeroed", { exact: true }),
  ).toBeVisible();
  await expect(
    page
      .locator(".recharts-legend-wrapper")
      .getByText("Baseline", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText(
      /every prompt and generated token before the attention output projection/,
    ),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/head-ablation-residual.png",
  });
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export trace" }).click();
  const stream = await (await downloading).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const trace = JSON.parse(Buffer.concat(chunks).toString());
  expect(trace.ablate_layer).toBeNull();
  expect(trace.ablate_head).toMatchObject({
    layer: 2,
    head: 3,
    heads: 4,
    head_dim: 8,
    projection: "attn.c_proj",
    scope: "all_prompt_and_generated_tokens",
  });
  await page.getByRole("tab", { name: "Attention", exact: true }).click();
  await page.getByLabel("Attention layer").selectOption("2");
  await expect(
    page.getByText(/Head 3 output was zeroed after attention routing/),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/head-ablation-attention.png",
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/head-ablation-mobile.png",
  });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page
    .getByLabel("Experiment name")
    .fill("Head ablation browser experiment");
  await page.getByRole("button", { name: "Save experiment" }).click();
  await expect(
    page.getByRole("button", { name: "Saved to experiments" }),
  ).toBeVisible();
  await page.goto("/experiments");
  await page
    .getByRole("link", { name: /Head ablation browser experiment/ })
    .first()
    .click();
  await expect(
    page.getByText("Layer 2 · head 3 zeroed", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText(
      /every prompt and generated token before the attention output projection/,
    ),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/head-ablation-notebook.png",
  });
});

test("unsupported layouts hide head controls and invalid targets return useful validation errors", async ({
  page,
  request,
}) => {
  await page.route("**/api/runtime", async (route) => {
    const response = await route.fetch();
    const runtime = await response.json();
    await route.fulfill({
      response,
      json: { ...runtime, attention_head_targets: [] },
    });
  });
  await page.goto("/playground");
  await expect(page.getByLabel("Layer intervention")).toBeVisible();
  await expect(
    page.getByRole("option", { name: "Zero attention head" }),
  ).toHaveCount(0);
  const invalid = await request.post("/api/trace", {
    data: {
      prompt: "the capital",
      max_new_tokens: 1,
      ablate_head: { layer: 0, head: 99 },
    },
  });
  expect(invalid.status()).toBe(400);
  expect((await invalid.json()).detail).toMatch(
    /Choose an attention head between/,
  );
  const ambiguous = await request.post("/api/trace", {
    data: {
      prompt: "the capital",
      max_new_tokens: 1,
      ablate_layer: 0,
      ablate_head: { layer: 0, head: 0 },
    },
  });
  expect(ambiguous.status()).toBe(422);
});
