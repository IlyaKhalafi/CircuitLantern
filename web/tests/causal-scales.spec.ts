import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

for (const normalized of [true, false]) {
  test(`patching charts identify ${normalized ? "normalized recovery" : "raw log probability changes"}`, async ({
    page,
  }) => {
    await page.route("**/api/jobs/*", async (route) => {
      const response = await route.fetch();
      const job = await response.json();
      if (!job.result) return route.fulfill({ response });
      const result = job.result;
      const gap = normalized ? 1 : 0;
      const cleanLogProbability = result.corrupted.log_probability + gap;
      await route.fulfill({
        response,
        json: {
          ...job,
          result: {
            ...result,
            recovery_defined: normalized,
            recovery_warning: null,
            baseline_log_probability_gap: gap,
            clean: {
              log_probability: cleanLogProbability,
              probability: Math.exp(cleanLogProbability),
            },
            layers: result.layers.map(
              (layer: { delta_log_probability: number }) => ({
                ...layer,
                recovery: normalized ? layer.delta_log_probability : null,
              }),
            ),
          },
        },
      });
    });
    await page.goto("/causal");
    await page
      .getByRole("button", { name: "Run patch sweep", exact: true })
      .click();
    await expect(
      page.getByText("Sweep complete", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", {
        name: normalized
          ? "Recovery by transformer block"
          : "Target log probability change",
        exact: true,
      }),
    ).toBeVisible();
    await expect(
      page.getByText(
        normalized
          ? "One is clean baseline recovery. Zero is no change from the corrupted baseline."
          : "Change from the corrupted baseline in nats. Zero means no change.",
        { exact: true },
      ),
    ).toBeVisible();
    if (!normalized) {
      await expect(
        page.getByText(
          /These values are not divided by the clean to corrupted score gap/,
        ),
      ).toBeVisible();
      await capturePage(page, {
        path: "../artifacts/screenshots/causal-raw-score.png",
      });
      await page.setViewportSize({ width: 390, height: 844 });
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await capturePage(page, {
        path: "../artifacts/screenshots/causal-raw-score-mobile.png",
      });
    }
  });
}
