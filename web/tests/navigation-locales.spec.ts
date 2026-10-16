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
      await expect(drawer).not.toHaveAttribute("inert", "");
      await page.locator(".brand").focus();
      await page.locator(".sidebar-backdrop").click({
        position: { x: locale === "fa" || locale === "ar" ? 20 : 370, y: 400 },
      });
      await expect(toggle).toHaveAttribute("aria-expanded", "false");
      await expect(toggle).toBeFocused();
      await toggle.click();
      await expect(drawer).not.toHaveAttribute("inert", "");
      await page.locator(".sidebar-bottom a[href='/guide']").focus();
      await page.keyboard.press("Enter");
      await expect(page).toHaveURL(/\/guide$/);
      await expect(toggle).toHaveAttribute("aria-expanded", "false");
      await expect
        .poll(() =>
          page.evaluate(() => {
            const active = document.activeElement;
            return (
              active === document.querySelector(".mobile-toggle") ||
              Boolean(active?.closest("#main-content"))
            );
          }),
        )
        .toBe(true);
      await toggle.click();
      await page.setViewportSize({ width: 1024, height: 768 });
      await expect(page.locator(".sidebar-backdrop")).toHaveCount(0);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await page.locator(".brand").focus();
      await page.setViewportSize({ width: 390, height: 844 });
      await expect(drawer).toHaveAttribute("inert", "");
      await expect(toggle).toBeFocused();
      await page.setViewportSize({ width: 1024, height: 768 });
      await expect(page.locator(".brand")).toBeFocused();
      await page.locator("#main-content").focus();
      await page.setViewportSize({ width: 390, height: 844 });
      await expect(drawer).toHaveAttribute("inert", "");
      await expect(page.locator("#main-content")).toBeFocused();
      await page.setViewportSize({ width: 1024, height: 768 });
      await expect(drawer).not.toHaveAttribute("inert", "");
      await expect(page.locator("#main-content")).toBeFocused();
    },
  );
}

test("navigation remains reachable at the layout breakpoint", async ({
  page,
}) => {
  await page.setViewportSize({ width: 720, height: 900 });
  await page.goto("/guide");
  const drawer = page.locator(".sidebar");
  const toggle = page.locator(".mobile-toggle");
  await expect(drawer).toHaveAttribute("inert", "");
  await expect(toggle).toBeVisible();
  await toggle.click();
  await expect(drawer).not.toHaveAttribute("inert", "");
  await page.keyboard.press("Escape");
  await expect(toggle).toBeFocused();
  await page.setViewportSize({ width: 721, height: 900 });
  await expect(toggle).toBeHidden();
  await expect(drawer).not.toHaveAttribute("inert", "");
  await expect(page.locator(".brand")).toBeFocused();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
