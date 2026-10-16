import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const routes = [
  "/guide",
  "/",
  "/workspace",
  "/playground",
  "/causal",
  "/causal/heads",
  "/features",
  "/experiments",
  "/hardware",
  "/missing-research-page",
];

for (const locale of ["fr", "en", "fa", "ar", "ja"]) {
  for (const theme of ["light", "dark"]) {
    test(
      "mobile header stays attached in " + locale + " " + theme,
      async ({ page, context, baseURL }) => {
        await context.addCookies([
          { name: "lantern_locale", value: locale, url: baseURL! },
          { name: "lantern_theme", value: theme, url: baseURL! },
        ]);
        await page.setViewportSize({ width: 390, height: 844 });
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        for (const route of routes) {
          await page.goto(route);
          await expect(
            page.locator("#main-content").getByRole("heading").first(),
          ).toBeVisible();
          await page.evaluate(async () => {
            await document.fonts.ready;
            window.scrollTo({ top: 320, behavior: "instant" });
          });
          if (route === "/guide") {
            expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(
              100,
            );
          }
          const header = page.locator(".topbar");
          const toggle = page.locator(".mobile-toggle");
          const bar = await header.boundingBox();
          const button = await toggle.boundingBox();
          expect(bar!.y).toBeGreaterThanOrEqual(0);
          expect(bar!.y).toBeLessThan(1);
          expect(button!.width).toBeGreaterThanOrEqual(44);
          expect(button!.height).toBeGreaterThanOrEqual(44);
          expect(button!.y).toBeGreaterThanOrEqual(bar!.y);
          expect(button!.y + button!.height).toBeLessThanOrEqual(
            bar!.y + bar!.height,
          );
          expect(button!.x).toBeGreaterThanOrEqual(0);
          expect(button!.x + button!.width).toBeLessThanOrEqual(390);
          expect(
            await toggle.evaluate((element) => {
              const box = element.getBoundingClientRect();
              return element.contains(
                document.elementFromPoint(
                  box.x + box.width / 2,
                  box.y + box.height / 2,
                ),
              );
            }),
          ).toBe(true);
          expect(
            await header.evaluate((element) => {
              const box = element.getBoundingClientRect();
              return element.contains(
                document.elementFromPoint(
                  box.x + box.width / 2,
                  box.bottom - 4,
                ),
              );
            }),
          ).toBe(true);
          expect(
            await header.evaluate((element) => {
              const canvas = document.createElement("canvas");
              canvas.width = canvas.height = 1;
              const context = canvas.getContext("2d")!;
              context.fillStyle = getComputedStyle(element).backgroundColor;
              context.fillRect(0, 0, 1, 1);
              return context.getImageData(0, 0, 1, 1).data[3];
            }),
          ).toBe(255);
          await expect(header.locator(".breadcrumb strong")).toHaveText(
            (await page.title()).split(" \u00b7 Circuit Lantern")[0],
          );
          expect(
            await page.evaluate(
              () => document.documentElement.scrollWidth <= innerWidth,
            ),
            route + " overflows at 390px in " + locale,
          ).toBe(true);
          if (locale === "en" || route === "/guide") {
            const result = await new AxeBuilder({ page })
              .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
              .analyze();
            expect(result.violations).toEqual([]);
            await page.screenshot({
              path:
                "../artifacts/screenshots/mobile-header-" +
                (route.replaceAll("/", "-").replace(/^-/, "") || "library") +
                "-" +
                locale +
                "-" +
                theme +
                ".png",
            });
          }
          await page.setViewportSize({ width: 320, height: 700 });
          const label = header.locator(".breadcrumb strong");
          await expect(label).toBeVisible();
          const text = await label.evaluate((element) => ({
            available: element.clientWidth,
            required: element.scrollWidth,
          }));
          expect(text.available).toBeGreaterThanOrEqual(text.required);
          const tools = await page.locator(".topbar-tools").boundingBox();
          expect(tools!.x).toBeGreaterThanOrEqual(0);
          expect(tools!.x + tools!.width).toBeLessThanOrEqual(320);
          expect(
            await page.evaluate(
              () => document.documentElement.scrollWidth <= innerWidth,
            ),
            route + " overflows at 320px in " + locale,
          ).toBe(true);
          if (route === "/hardware") {
            const result = await new AxeBuilder({ page })
              .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
              .analyze();
            expect(result.violations).toEqual([]);
            await page.screenshot({
              path:
                "../artifacts/screenshots/mobile-header-narrow-" +
                locale +
                "-" +
                theme +
                ".png",
            });
          }
          await page.setViewportSize({ width: 390, height: 844 });
        }
        await page.goto("/guide");
        await page.evaluate(() => window.scrollTo(0, 320));
        const scroll = await page.evaluate(() => window.scrollY);
        await page.locator(".mobile-toggle").click();
        await expect(page.locator(".main-shell")).toHaveAttribute("inert", "");
        await expect(page.locator(".sidebar")).not.toHaveAttribute("inert", "");
        await expect
          .poll(async () => {
            const drawer = await page.locator(".sidebar").boundingBox();
            return drawer
              ? Math.min(drawer.x, 390 - drawer.x - drawer.width)
              : -1;
          })
          .toBeGreaterThanOrEqual(0);
        if (locale === "en" || locale === "fa") {
          await page.screenshot({
            path:
              "../artifacts/screenshots/mobile-drawer-" +
              locale +
              "-" +
              theme +
              ".png",
          });
        }
        await page.keyboard.press("Escape");
        await expect(page.locator(".mobile-toggle")).toBeFocused();
        await expect(page.locator(".main-shell")).not.toHaveAttribute(
          "inert",
          "",
        );
        expect(await page.evaluate(() => window.scrollY)).toBe(scroll);
        expect(errors).toEqual([]);
      },
    );
  }
}
