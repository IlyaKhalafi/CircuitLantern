import { expect, test } from "@playwright/test";
import french from "../src/i18n/messages/fr.json";

for (const empty of [false, true]) {
  test(
    "heatmap handles " +
      (empty
        ? "an empty attention grid"
        : "grids above argument spread limits"),
    async ({ page }) => {
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.route("**/api/trace", async (route) => {
        const response = await route.fetch();
        const trace = await response.json();
        await route.fulfill({
          response,
          json: {
            ...trace,
            attention: empty
              ? trace.attention.map((layer: object) => ({
                  ...layer,
                  pattern: [],
                }))
              : trace.attention,
            layers: empty
              ? trace.layers
              : Array.from({ length: 256 }, (_, index) => ({
                  ...trace.layers[index % trace.layers.length],
                  layer: index,
                  label: "Layer " + index,
                  token_norms: Array(600).fill(index === 0 ? 1 : 0.5),
                })),
          },
        });
      });
      await page.goto("/playground");
      await page.getByLabel("Generated tokens").selectOption("0");
      await page.getByRole("button", { name: "Run & trace" }).click();
      if (empty)
        await page.getByRole("tab", { name: "Attention", exact: true }).click();
      const canvas = page.getByRole("img", {
        name: empty
          ? /Hover over a cell to inspect query/
          : /Columns are prompt tokens/,
      });
      await expect(canvas).toBeVisible();
      await canvas.hover({ position: { x: 10, y: 10 } });
      const dimensions = await canvas.evaluate((element) => {
        const canvas = element as HTMLCanvasElement;
        return {
          width: canvas.width,
          height: canvas.height,
          alpha: canvas.getContext("2d")!.getImageData(0, 0, 1, 1).data[3],
        };
      });
      if (empty) expect(dimensions.alpha).toBe(0);
      else
        expect(dimensions).toEqual({ width: 4800, height: 2048, alpha: 255 });
      expect(errors).toEqual([]);
    },
  );
}

test("localized hover labels preserve raw research tokens", async ({
  page,
}) => {
  await page.route("**/api/trace", async (route) => {
    const response = await route.fetch();
    const trace = await response.json();
    trace.tokens[0] = "Head 2 {x}";
    await route.fulfill({ response, json: trace });
  });
  await page.goto("/playground");
  await page.getByLabel("Generated tokens").selectOption("0");
  await page.getByRole("button", { name: "Run & trace" }).click();
  await page.locator(".language-select select").selectOption("fr");
  await expect(page.locator("html")).toHaveAttribute("lang", "fr");
  const canvas = page.getByRole("img", { name: /Les colonnes/ });
  await expect(canvas).toBeVisible();
  await canvas.hover({ position: { x: 1, y: 1 } });
  await expect(page.locator(".heatmap-wrapper p").first()).toContainText(
    "Head 2 {x}",
  );
  await expect(page.locator(".heatmap-wrapper p").first()).toContainText(
    french["Embedding"],
  );
});

for (const token of ["", " ", "\n"]) {
  test(
    "whitespace token stays visible in heatmap hover " + JSON.stringify(token),
    async ({ page }) => {
      await page.route("**/api/trace", async (route) => {
        const response = await route.fetch();
        const trace = await response.json();
        trace.tokens[0] = token;
        await route.fulfill({ response, json: trace });
      });
      await page.goto("/playground");
      await page.getByLabel("Generated tokens").selectOption("0");
      await page.getByRole("button", { name: "Run & trace" }).click();
      const canvas = page.getByRole("img", {
        name: /Columns are prompt tokens/,
      });
      await expect(canvas).toBeVisible();
      await canvas.hover({ position: { x: 1, y: 1 } });
      await expect(page.locator(".heatmap-wrapper p").first()).toContainText(
        token === "" ? "∅" : JSON.stringify(token),
      );
    },
  );
}
