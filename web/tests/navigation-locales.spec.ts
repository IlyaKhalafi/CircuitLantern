import { expect, test } from "@playwright/test";

for (const locale of ["en", "fa", "ar", "ja", "fr"]) {
  test(
    "mobile navigation stays usable in " + locale,
    async ({ page, context }) => {
      await context.addCookies([
        {
          name: "lantern_locale",
          value: locale,
          url: new URL(test.info().project.use.baseURL as string).origin,
        },
      ]);
      await page.setViewportSize({ width: 390, height: 844 });
      await page.goto("/");
      const header = await page.locator(".topbar").boundingBox();
      const status = await page.locator(".runtime-status").boundingBox();
      expect(status!.y).toBeGreaterThanOrEqual(header!.y);
      expect(status!.y + status!.height).toBeLessThanOrEqual(
        header!.y + header!.height,
      );
      const toggle = page.locator(".mobile-toggle");
      await expect(page.locator(".sidebar")).toHaveAttribute("inert", "");
      await toggle.focus();
      await page.keyboard.press("Tab");
      expect(
        await page.evaluate(
          () => document.activeElement?.closest(".sidebar") !== null,
        ),
      ).toBe(false);
      await toggle.click();
      await expect(toggle).toHaveAttribute("aria-expanded", "true");
      const drawer = page.locator(".sidebar");
      await expect
        .poll(async () => {
          const box = await drawer.boundingBox();
          return box ? Math.min(box.x, 390 - box.x - box.width) : -1;
        })
        .toBeGreaterThanOrEqual(0);
      const brand = await page.locator(".brand").boundingBox(),
        close = await toggle.boundingBox();
      expect(
        close!.x + close!.width <= brand!.x ||
          close!.x >= brand!.x + brand!.width ||
          close!.y + close!.height <= brand!.y,
      ).toBe(true);
      await page.keyboard.press("Escape");
      await expect(toggle).toHaveAttribute("aria-expanded", "false");
      await expect(toggle).toBeFocused();
      await toggle.click();
      await page.locator(".sidebar-bottom a[href='/guide']").click();
      await expect(page).toHaveURL(/\/guide$/);
      await expect(toggle).toHaveAttribute("aria-expanded", "false");
      await toggle.click();
      await page.setViewportSize({ width: 1024, height: 768 });
      await expect(page.locator(".sidebar-backdrop")).toHaveCount(0);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
    },
  );
}
