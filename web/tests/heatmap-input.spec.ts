import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { Trace } from "../src/lib/types";

async function capture(page: Page) {
  await page.goto("/playground");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France");
  await page.getByLabel("Generated tokens").selectOption("0");
  const pending = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/trace") &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  const trace = (await (await pending).json()) as Trace;
  const canvas = page.locator("canvas.heatmap").first();
  await expect(canvas).toBeVisible();
  return {
    trace,
    canvas,
    status: page.locator(".heatmap-wrapper [role=status]").first(),
  };
}

const screenshots = join(process.cwd(), "../artifacts/screenshots");

test("keyboard cell inspection follows source coordinates without repainting measurements", async ({
  page,
}) => {
  const { trace, canvas, status } = await capture(page);
  const pixels = await canvas.evaluate((node) =>
    (node as HTMLCanvasElement).toDataURL(),
  );
  const check = async (x: number, y: number) => {
    await expect(status).toContainText(trace.tokens[x]);
    await expect(status).toContainText(
      trace.layers[y].token_norms[x].toPrecision(4),
    );
  };
  await canvas.focus();
  await check(0, 0);
  await canvas.press("ArrowRight");
  await canvas.press("ArrowDown");
  await check(1, 1);
  await canvas.press("End");
  await check(trace.tokens.length - 1, 1);
  await canvas.press("Home");
  await check(0, 1);
  await canvas.press("Control+End");
  await canvas.press("ArrowRight");
  await canvas.press("ArrowDown");
  await check(trace.tokens.length - 1, trace.layers.length - 1);
  await canvas.press("Control+Home");
  await canvas.press("ArrowLeft");
  await canvas.press("ArrowUp");
  await check(0, 0);
  expect(
    await canvas.evaluate((node) => (node as HTMLCanvasElement).toDataURL()),
  ).toBe(pixels);
  await expect(status).toHaveAttribute("aria-live", "polite");
  await expect(page.locator(".heatmap-cursor").first()).toBeVisible();
  mkdirSync(screenshots, { recursive: true });
  for (const theme of ["light", "dark"]) {
    if (theme === "dark")
      await page
        .getByRole("button", { name: "Dark theme", exact: true })
        .click();
    await canvas.focus();
    await canvas.press("ArrowRight");
    await canvas.press("ArrowDown");
    await expect(canvas).toBeFocused();
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await page
      .locator(".heatmap-wrapper")
      .first()
      .evaluate((node) =>
        window.scrollTo({
          top: node.getBoundingClientRect().top + scrollY - 80,
          behavior: "instant",
        }),
      );
    await page.mouse.move(0, 0);
    await page.screenshot({
      path: join(screenshots, "heatmap-keyboard-" + theme + ".png"),
    });
  }
  await canvas.press("Escape");
  await expect(canvas).not.toBeFocused();
  await expect(page.locator(".heatmap-cursor")).toHaveCount(0);
});

test("touch inspection preserves literal tokens and works in all five languages", async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({
    baseURL,
    hasTouch: true,
    viewport: { width: 390, height: 844 },
  });
  const page = await context.newPage();
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    const { trace, canvas, status } = await capture(page);
    await page.getByRole("button", { name: "Dark theme", exact: true }).click();
    mkdirSync(screenshots, { recursive: true });
    for (const locale of ["fa", "ar", "ja", "fr", "en"]) {
      await page.locator(".language-select select").selectOption(locale);
      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      await canvas.scrollIntoViewIfNeeded();
      const bounds = await canvas.boundingBox();
      await canvas.tap({
        position: {
          x: (bounds!.width * 1.5) / trace.tokens.length,
          y: (bounds!.height * 1.5) / trace.layers.length,
        },
      });
      await expect(status).toContainText(trace.tokens[1]);
      await expect(status).toContainText(
        trace.layers[1].token_norms[1].toPrecision(4),
      );
      await expect(status).toHaveAttribute("aria-live", "off");
      await expect(page.locator(".heatmap-cursor").first()).toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      expect(
        (
          await new AxeBuilder({ page })
            .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
            .analyze()
        ).violations,
      ).toEqual([]);
      await page
        .locator(".heatmap-wrapper")
        .first()
        .evaluate((node) =>
          window.scrollTo({
            top: node.getBoundingClientRect().top + scrollY - 80,
            behavior: "instant",
          }),
        );
      await page.mouse.move(0, 0);
      await page.screenshot({
        path: join(screenshots, "heatmap-touch-" + locale + ".png"),
      });
    }
    expect(errors).toEqual([]);
  } finally {
    await context.close();
  }
});
