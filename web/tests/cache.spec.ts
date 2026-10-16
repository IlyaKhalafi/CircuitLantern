import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

for (const complete of [true, false]) {
  test(`deployment explains a ${complete ? "complete" : "partial"} cache and loads the checked revision`, async ({
    page,
  }) => {
    await page.route("**/api/models/compatibility", async (route) => {
      const response = await route.fetch();
      const report = await response.json();
      await route.fulfill({
        response,
        json: {
          ...report,
          source_bytes: 100 * 1024 ** 2,
          cached_bytes: (complete ? 100 : 75) * 1024 ** 2,
          download_bytes: (complete ? 0 : 25) * 1024 ** 2,
        },
      });
    });
    await page.goto("/");
    await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
    await expect(
      page.getByText("Ready to explore on this server"),
    ).toBeVisible();
    await expect(
      page
        .locator(".metric")
        .filter({ hasText: "Remaining download" })
        .locator("strong"),
    ).toHaveText(complete ? "0 B" : "25 MiB");
    await expect(
      page.getByText(
        complete
          ? "All checkpoint files are cached locally."
          : /Reusing 75 MiB of checkpoint files from the local Hub cache/,
      ),
    ).toBeVisible();
    await capturePage(page, {
      path: `../artifacts/screenshots/deployment-cache-${complete ? "complete" : "partial"}.png`,
      fullPage: false,
    });
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await page
      .getByRole("button", {
        name: complete ? "Load & explore" : "Download & explore",
      })
      .click();
    await expect(page).toHaveURL(/workspace/);
  });
}
