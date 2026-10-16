import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("light is the default and theme changes preserve research drafts", async ({
  page,
  context,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/playground");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  const prompt = page.getByLabel("Prompt", { exact: true });
  await prompt.fill("Paris {x} العربية 日本語");
  await page.getByLabel("Generated tokens").selectOption("0");
  await page.getByRole("button", { name: "Dark theme", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(
    page.getByRole("button", { name: "Dark theme", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(prompt).toHaveValue("Paris {x} العربية 日本語");
  await expect(page.getByLabel("Generated tokens")).toHaveValue("0");
  expect(
    (await context.cookies()).find((cookie) => cookie.name === "lantern_theme")
      ?.value,
  ).toBe("dark");
  await page
    .getByRole("link", { name: "Server & runtime", exact: true })
    .click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("button", { name: "Light theme", exact: true }).click();
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  expect(errors).toEqual([]);
});

test("theme changes never interpolate text and background colors", async ({
  page,
}) => {
  await page.goto("/playground");
  for (const theme of ["Dark theme", "Light theme"]) {
    const transitions = await page
      .getByRole("button", { name: theme, exact: true })
      .evaluate((button) => {
        (button as HTMLButtonElement).click();
        return document
          .getAnimations()
          .filter(
            (animation) =>
              animation instanceof CSSTransition &&
              ["color", "background-color"].includes(
                animation.transitionProperty,
              ),
          )
          .map((animation) => (animation as CSSTransition).transitionProperty);
      });
    expect(transitions).toEqual([]);
  }
});

for (const value of ["light", "dark", "invalid"]) {
  test(
    "server renders a valid theme before hydration for " + value,
    async ({ browser, baseURL }) => {
      const context = await browser.newContext({
        baseURL,
        javaScriptEnabled: false,
      });
      await context.addCookies([
        { name: "lantern_theme", value, url: baseURL! },
      ]);
      const page = await context.newPage();
      try {
        await page.goto("/guide", { waitUntil: "domcontentloaded" });
        await expect(page.locator("html")).toHaveAttribute(
          "data-theme",
          value === "dark" ? "dark" : "light",
        );
        expect(
          await page
            .locator("html")
            .evaluate((element) => getComputedStyle(element).colorScheme),
        ).toBe(value === "dark" ? "dark" : "light");
      } finally {
        await context.close();
      }
    },
  );
}

test("theme changes repaint heatmaps without changing scientific values", async ({
  page,
}) => {
  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("0");
  await page.getByRole("button", { name: "Run & trace" }).click();
  const canvas = page.getByRole("img", { name: /Columns are prompt tokens/ });
  await expect(canvas).toBeVisible();
  const pixel = () =>
    canvas.evaluate((element) =>
      Array.from(
        (element as HTMLCanvasElement)
          .getContext("2d")!
          .getImageData(4, 4, 1, 1).data,
      ),
    );
  const light = await pixel();
  await canvas.hover({ position: { x: 10, y: 10 } });
  const scientificValue = await page
    .locator(".heatmap-wrapper p")
    .first()
    .textContent();
  await page.getByRole("button", { name: "Dark theme", exact: true }).click();
  await expect.poll(pixel).not.toEqual(light);
  await canvas.hover({ position: { x: 10, y: 10 } });
  await expect(page.locator(".heatmap-wrapper p").first()).toHaveText(
    scientificValue!,
  );
  await page.getByRole("button", { name: "Light theme", exact: true }).click();
  await expect.poll(pixel).toEqual(light);
});

test("mobile theme controls remain visible in every language", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  for (const locale of ["en", "fa", "ar", "ja", "fr"]) {
    await page.locator(".language-select select").selectOption(locale);
    await expect(page.locator("html")).toHaveAttribute("lang", locale);
    const buttons = page.locator(".theme-select button");
    for (let index = 0; index < 2; index++) {
      await buttons.nth(index).click();
      await expect(page.locator("html")).toHaveAttribute(
        "data-theme",
        index === 0 ? "light" : "dark",
      );
      const box = await buttons.nth(index).boundingBox();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(390);
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
});

for (const path of [
  "/",
  "/workspace",
  "/playground",
  "/causal",
  "/causal/heads",
  "/experiments",
  "/hardware",
  "/guide",
]) {
  test(
    "dark theme has readable contrast on " + path,
    async ({ page, context, baseURL }) => {
      await context.addCookies([
        { name: "lantern_theme", value: "dark", url: baseURL! },
      ]);
      await page.goto(path, { waitUntil: "networkidle" });
      await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
      const result = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      expect(
        result.violations.map((violation) => ({
          id: violation.id,
          nodes: violation.nodes.map((node) => ({
            target: node.target,
            summary: node.failureSummary,
          })),
        })),
      ).toEqual([]);
    },
  );
}
