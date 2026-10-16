import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

for (const readable of [true, false]) {
  test(`server explains a process memory cap when usage is ${readable ? "readable" : "unavailable"}`, async ({
    page,
  }) => {
    await page.route("**/api/hardware", async (route) => {
      const response = await route.fetch();
      const hardware = await response.json();
      await route.fulfill({
        response,
        json: {
          ...hardware,
          ram_total_bytes: 8 * 1024 ** 3,
          ram_available_bytes: readable ? 6 * 1024 ** 3 : 0,
          ram_host_total_bytes: 64 * 1024 ** 3,
          ram_scope: "cgroup",
          ram_constraints: [
            {
              source: "cgroup-v2",
              limit_bytes: 8 * 1024 ** 3,
              used_bytes: readable ? 3 * 1024 ** 3 : null,
              reclaimable_bytes: readable ? 1024 ** 3 : 0,
            },
          ],
        },
      });
    });
    await page.goto("/hardware");
    await expect(
      page.getByText("8.0 GiB process memory capacity", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByText(/RAM is constrained by the backend process or container/),
    ).toBeVisible();
    if (!readable) {
      await expect(
        page.getByText(
          /Current usage for a memory constraint could not be read/,
        ),
      ).toBeVisible();
    }
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await capturePage(page, {
      path: `../artifacts/screenshots/hardware-process-limit-${readable ? "available" : "unknown"}.png`,
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
      path: `../artifacts/screenshots/hardware-process-limit-${readable ? "available" : "unknown"}-mobile.png`,
    });
  });
}
