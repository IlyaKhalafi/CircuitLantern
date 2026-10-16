import { expect, test } from "@playwright/test";

test("missing pages return a descriptive server title", async ({ page }) => {
  const response = await page.goto("/a-missing-research-route");
  expect(response?.status()).toBe(404);
  await expect(page).toHaveTitle("Page not found · Circuit Lantern");
});

test("page titles follow client navigation through the workbench", async ({
  page,
}) => {
  await page.goto("/");
  for (const title of [
    "Model explorer",
    "Activation lab",
    "Causal tracing",
    "Experiments",
    "Server & runtime",
    "Model library",
  ]) {
    await page
      .getByRole("navigation", { name: "Main navigation" })
      .getByRole("link", { name: title, exact: true })
      .click();
    await expect(page).toHaveTitle(title + " · Circuit Lantern");
  }
  await page.getByRole("link", { name: "Documentation", exact: true }).click();
  await expect(page).toHaveTitle("Field guide · Circuit Lantern");
});
