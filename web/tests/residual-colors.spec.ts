import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

for (const zero of [false, true]) {
  test(`logarithmic residual colors preserve raw ${zero ? "zero" : "outlier"} values`, async ({
    page,
  }) => {
    await page.route("**/api/trace", async (route) => {
      const response = await route.fetch();
      const trace = await response.json();
      await route.fulfill({
        response,
        json: {
          ...trace,
          layers: trace.layers.map((layer: object, i: number) => ({
            ...layer,
            token_norms: zero ? [0, 0] : [10_000, 100 * (i + 1)],
          })),
        },
      });
    });
    await page.goto("/playground");
    await page.getByLabel("Prompt", { exact: true }).fill("the capital");
    await page.getByLabel("Generated tokens").selectOption("4");
    await page.getByRole("button", { name: "Run & trace" }).click();
    const canvas = page.getByRole("img", { name: /Columns are prompt tokens/ });
    await expect(canvas).toBeVisible();
    const pixel = () =>
      canvas.evaluate((element) =>
        Array.from(
          (element as HTMLCanvasElement)
            .getContext("2d")!
            .getImageData(12, 4, 1, 1).data,
        ),
      );
    const linear = await pixel();
    await page.getByLabel("Logarithmic residual colors").check();
    await expect(canvas).toHaveAttribute(
      "aria-label",
      /Logarithmic colors use log\(1 \+ norm\)/,
    );
    if (zero) {
      await expect.poll(pixel).toEqual(linear);
      await expect(page.locator(".heatmap-midpoint")).toHaveText("0");
    } else {
      await expect.poll(pixel).not.toEqual(linear);
      await expect(page.locator(".heatmap-midpoint")).toHaveText("99.005");
    }
    const bounds = await canvas.boundingBox();
    await canvas.hover({
      position: { x: bounds!.width * 0.75, y: bounds!.height / 12 },
    });
    await expect(page.locator(".heatmap-wrapper p")).toContainText(
      zero ? "0.000" : "100.0",
    );
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await capturePage(page, {
      path: `../artifacts/screenshots/residual-log-colors-${zero ? "zero" : "outliers"}.png`,
    });
    const downloading = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export trace" }).click();
    const stream = await (await downloading).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    const exported = JSON.parse(Buffer.concat(chunks).toString());
    expect(exported.layers[0].token_norms).toEqual(
      zero ? [0, 0] : [10_000, 100],
    );
    await page.setViewportSize({ width: 390, height: 844 });
    await expect
      .poll(() =>
        page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      )
      .toBe(true);
    await capturePage(page, {
      path: `../artifacts/screenshots/residual-log-colors-${zero ? "zero" : "outliers"}-mobile.png`,
    });
    await page.getByLabel("Logarithmic residual colors").uncheck();
    await expect.poll(pixel).toEqual(linear);
  });
}
