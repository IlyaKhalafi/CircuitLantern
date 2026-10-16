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

test("signed scientific heatmap values retain character order in RTL", async ({
  page,
}) => {
  await page.route("**/api/trace", async (route) => {
    const response = await route.fetch();
    const trace = await response.json();
    trace.tokens[0] = "\u0627\u0644\u0639\u0631\u0628\u064a\u0629";
    trace.layers[0].label =
      "\u0631\u062f\u06cc\u0641 \u0641\u0627\u0631\u0633\u06cc";
    trace.layers[0].token_norms[0] = -1.234e-8;
    await route.fulfill({ response, json: trace });
  });
  const { canvas, status } = await capture(page);
  await page.setViewportSize({ width: 390, height: 844 });
  mkdirSync(screenshots, { recursive: true });
  for (const locale of ["fa", "ar"]) {
    await page.locator(".language-select select").selectOption(locale);
    await expect(page.locator("html")).toHaveAttribute("lang", locale);
    await canvas.focus();
    await canvas.press("Control+Home");
    const value = (-1.234e-8).toPrecision(4);
    await expect(status).toContainText(value);
    const positions = await status.evaluate((element, text) => {
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        const start = node.textContent?.indexOf(text) ?? -1;
        if (start < 0) continue;
        return [...text].map((_, index) => {
          const range = document.createRange();
          range.setStart(node, start + index);
          range.setEnd(node, start + index + 1);
          return range.getBoundingClientRect().left;
        });
      }
      throw new Error("Missing signed measurement");
    }, value);
    await page.screenshot({
      path: join(screenshots, "heatmap-signed-" + locale + ".png"),
    });
    expect(
      positions.every((x, index) => !index || x > positions[index - 1]),
    ).toBe(true);
    const help = page.locator(".heatmap-help").first();
    expect(
      await help.evaluate((node) => getComputedStyle(node).direction),
    ).toBe("rtl");
  }
});

test("heatmap navigation preserves modified browser shortcuts", async ({
  page,
}) => {
  const { canvas, status } = await capture(page);
  await canvas.focus();
  const initial = await status.innerText();
  for (const modifiers of [
    { altKey: true },
    { ctrlKey: true },
    { metaKey: true },
    { shiftKey: true },
    { shiftKey: true, ctrlKey: true },
  ]) {
    const nativeDefault = await canvas.evaluate(
      (node, flags) =>
        node.dispatchEvent(
          new KeyboardEvent("keydown", {
            key: "ArrowRight",
            bubbles: true,
            cancelable: true,
            ...flags,
          }),
        ),
      modifiers,
    );
    expect(nativeDefault).toBe(true);
    await expect(status).toHaveText(initial);
  }
  const handled = await canvas.evaluate((node) =>
    node.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "ArrowRight",
        bubbles: true,
        cancelable: true,
      }),
    ),
  );
  expect(handled).toBe(false);
  await expect(status).not.toHaveText(initial);
  await canvas.press("Control+Home");
  await expect(status).toHaveText(initial);
});
