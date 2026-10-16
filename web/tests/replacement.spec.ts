import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

test("GPU replacement budget explains released tensors on desktop and mobile", async ({
  page,
}) => {
  await page.route("**/api/models/compatibility", async (route) => {
    const response = await route.fetch();
    const report = await response.json();
    const device = route.request().postDataJSON().device;
    await route.fulfill({
      response,
      json:
        device === "cuda"
          ? {
              ...report,
              compatible: true,
              blockers: [],
              reclaimable_bytes: 512 * 1024 ** 2,
              available_bytes: 2 * 1024 ** 3,
            }
          : report,
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await expect(
    page.getByText("Available budget", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Execution device").selectOption("cuda");
  await expect(
    page.getByText("Replacement budget", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Includes 512 MiB released by replacing the active model."),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Download & explore" }),
  ).toBeEnabled();
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await capturePage(page, {
    path: "../artifacts/screenshots/replacement-budget.png",
    fullPage: false,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/replacement-budget-mobile.png",
    fullPage: false,
  });
});
