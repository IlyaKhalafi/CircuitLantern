import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

test("narrow embedding matrices expose a complete spectrum and rank diagnostics", async ({
  page,
}) => {
  await page.goto("/workspace");
  await page.getByLabel("Filter matrices").fill("wte.weight");
  await page.locator(".weight-item").first().click();
  await expect(
    page.getByText("Full SVD in float64 · showing 32 of 32 singular values", {
      exact: true,
    }),
  ).toBeVisible();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export analysis" }).click();
  const stream = await (await download).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  const result = JSON.parse(Buffer.concat(chunks).toString());
  expect(result.shape).toEqual([1000, 32]);
  expect(result.spectrum_exact).toBe(true);
  expect(result.spectrum_precision).toBe("float64");
  expect(result.singular_values).toHaveLength(32);
  expect(result.numerical_rank).toBe(32);
  expect(result.condition_number).toBeGreaterThanOrEqual(1);
  expect(result.nuclear_norm).toBeGreaterThan(0);
  expect(result.effective_rank).toBeGreaterThan(1);
  expect(result.spectral_radius).toBeNull();
  expect(result.cumulative_energy.at(-1)).toBeCloseTo(1, 10);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await capturePage(page, {
    path: "../artifacts/screenshots/narrow-spectrum.png",
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/narrow-spectrum-mobile.png",
  });
});
