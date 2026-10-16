import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

test("individual head routing updates with the layer and labels bypassed computation", async ({
  page,
}) => {
  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("8");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Capture complete", { exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Attention", exact: true }).click();
  await page.getByLabel("Attention layer").selectOption("2");
  await expect(
    page.getByText("Final token attention by head", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Layer 2 · Query position 4 · is", { exact: true }),
  ).toBeVisible();
  await page.getByText("Inspect head routing values", { exact: true }).click();
  await expect(page.getByRole("row", { name: /^Head \d/ })).toHaveCount(4);
  await capturePage(page, {
    path: "../artifacts/screenshots/attention-head-routing.png",
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);

  await page.getByLabel("Layer intervention").selectOption("2");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await expect(
    page.getByText("Block 2 bypassed", { exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Attention", exact: true }).click();
  await page.getByLabel("Attention layer").selectOption("2");
  await expect(
    page.getByText(/This block's output was bypassed/),
  ).toBeVisible();
  await page.getByLabel("Attention layer").selectOption("1");
  await expect(page.getByText(/This block's output was bypassed/)).toHaveCount(
    0,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/attention-head-routing-mobile.png",
  });
});
