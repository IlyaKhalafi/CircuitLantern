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
  const main = page.getByRole("navigation", { name: "Main navigation" });
  await main.getByRole("link", { name: "Model explorer", exact: true }).click();
  await expect(page).toHaveTitle("Model explorer · Circuit Lantern");
  await main.getByRole("link", { name: "Research lab", exact: true }).click();
  await expect(page).toHaveTitle("Activation lab · Circuit Lantern");
  for (const title of ["Causal tracing", "Feature lab", "Activation lab"]) {
    await page
      .getByRole("navigation", { name: "Research workflow" })
      .getByRole("link", { name: title, exact: true })
      .click();
    await expect(page).toHaveTitle(title + " · Circuit Lantern");
  }
  for (const title of ["Experiments", "Server & runtime", "Model library"]) {
    await main.getByRole("link", { name: title, exact: true }).click();
    await expect(page).toHaveTitle(title + " · Circuit Lantern");
  }
  await page.getByRole("link", { name: "Documentation", exact: true }).click();
  await expect(page).toHaveTitle("Field guide · Circuit Lantern");
});
