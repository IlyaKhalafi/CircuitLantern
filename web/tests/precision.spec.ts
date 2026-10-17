import { capturePage } from "./helpers";
import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("NF4 selection loads packed weights, traces and exports logical statistics", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await page
    .getByLabel("Execution format", { exact: true })
    .selectOption("int4");
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  await page.getByText("How this estimate works").click();
  await expect(page.getByText(/NF4 \/ FP32 compute/)).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/compatibility-nf4.png",
    fullPage: true,
  });
  const loading = page.waitForRequest(
    (request) =>
      request.url().endsWith("/api/models/load") && request.method() === "POST",
  );
  await page.getByRole("button", { name: "Download & explore" }).click();
  const payload = (await loading).postDataJSON();
  expect(payload.precision).toBe("int4");
  expect(payload.context_length).toBe(256);
  await expect(page).toHaveURL(/workspace/);
  await page.getByLabel("Filter matrices").fill("h.0.attn.c_attn");
  await page
    .locator(".weight-list")
    .getByRole("button", {
      name: /transformer.h.0.attn.c_attn.weight/,
    })
    .click();
  await expect(
    page.getByText(/Statistics describe the dequantized float32 matrix/),
  ).toBeVisible();
  await expect(page.getByText(/96 × 32 · nf4 source/)).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export analysis" }).click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const analysis = JSON.parse(Buffer.concat(chunks).toString());
  expect(analysis.execution_precision).toBe("int4");
  expect(analysis.shape).toEqual([96, 32]);
  expect(analysis.quantized).toBe(true);
  expect(analysis.source_format).toBe("nf4");
  expect(analysis.frobenius_norm).toBeGreaterThan(0);
  expect(analysis.nuclear_norm).toBeGreaterThan(0);
  expect(analysis.numerical_rank).toBeGreaterThan(0);
  expect(analysis.operator_1_norm).toBeGreaterThan(0);
  expect(analysis.spectrum_precision).toBe("float64");
  await capturePage(page, {
    path: "../artifacts/screenshots/matrix-nf4.png",
    fullPage: true,
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);

  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("8");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("NF4 · 4 bit", { exact: true })).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/activation-nf4.png",
    fullPage: true,
  });
});
