import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { DecoderDirections } from "../src/lib/features";
import type { RuntimeState } from "../src/lib/types";

test("decoder geometry remains available offline and preserves signed rankings and source evidence", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  const before = (await (
    await request.get("/api/runtime")
  ).json()) as RuntimeState;
  let dictionaryId = "";
  const errors: string[] = [];
  page.on("pageerror", (reason) => errors.push(reason.message));
  try {
    const starting = await request.post("/api/features/train", {
      data: {
        name: "Decoder geometry study",
        layers: [0],
        documents: [
          "the capital of France is",
          "the capital of Poland is",
          "Poland is the capital of Poland",
          "France is the capital of France",
          "the capital is France",
          "the capital is Poland",
          "France of the is capital",
          "Poland of the is capital",
        ],
        steps: 16,
        batch_size: 16,
        active_features: 4,
      },
    });
    expect(starting.status()).toBe(202);
    const job = await starting.json();
    let training: {
      status: string;
      result?: { dictionaries: { id: string }[] };
    };
    await expect
      .poll(
        async () => {
          training = await (await request.get("/api/jobs/" + job.id)).json();
          return training.status;
        },
        { timeout: 60000 },
      )
      .toBe("ready");
    dictionaryId = training!.result!.dictionaries[0].id;
    expect((await request.delete("/api/runtime")).status()).toBe(204);
    await page.goto("/features");
    await page
      .getByRole("tab", { name: "Explore features", exact: true })
      .click();
    await page
      .getByLabel("Choose a dictionary", { exact: true })
      .selectOption(dictionaryId);
    let inference = 0;
    page.on("request", (req) => {
      if (req.method() === "POST") inference++;
    });
    let geometryRequests = 0;
    const directionsPath =
      "**/api/features/dictionaries/" + dictionaryId + "/features/*/directions";
    await page.route(directionsPath, async (route) => {
      geometryRequests++;
      if (geometryRequests === 1)
        await route.fulfill({
          status: 400,
          contentType: "application/json",
          body: JSON.stringify({
            detail:
              "The dictionary weights have changed since they were recorded.",
          }),
        });
      else await route.continue();
    });
    const summary = page.getByText("Related decoder directions", {
      exact: true,
    });
    expect(geometryRequests).toBe(0);
    await summary.click();
    await expect(
      page.getByText(
        "The dictionary weights have changed since they were recorded.",
        { exact: true },
      ),
    ).toBeVisible();
    await page.getByRole("button", { name: "Try again", exact: true }).click();
    const region = page.getByRole("region", {
      name: "Decoder direction comparisons",
      exact: true,
    });
    await expect(region.locator("tbody tr")).toHaveCount(8);
    const pending = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export directions", exact: true })
      .click();
    const stream = await (await pending).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    const result = JSON.parse(
      Buffer.concat(chunks).toString(),
    ) as DecoderDirections;
    expect(result.dictionary_id).toBe(dictionaryId);
    expect(result.weights_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.highest.every((row) => row.feature !== result.feature)).toBe(
      true,
    );
    expect(result.lowest.every((row) => row.feature !== result.feature)).toBe(
      true,
    );
    for (let index = 1; index < result.highest.length; index++)
      expect(result.highest[index].cosine).toBeLessThanOrEqual(
        result.highest[index - 1].cosine,
      );
    for (let index = 1; index < result.lowest.length; index++)
      expect(result.lowest[index].cosine).toBeGreaterThanOrEqual(
        result.lowest[index - 1].cosine,
      );
    await page
      .getByLabel("Direction ranking", { exact: true })
      .selectOption("lowest");
    await expect(region.locator("tbody tr").first()).toContainText(
      "Feature " + result.lowest[0].feature,
    );
    const screenshots = join(process.cwd(), "../artifacts/screenshots");
    mkdirSync(screenshots, { recursive: true });
    for (const theme of ["light", "dark"]) {
      if (theme === "dark")
        await page
          .getByRole("button", { name: "Dark theme", exact: true })
          .click();
      await page
        .locator(".feature-directions")
        .evaluate((node) => node.scrollIntoView());
      const audit = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      expect(audit.violations).toEqual([]);
      await page.screenshot({
        path: join(screenshots, "feature-directions-" + theme + ".png"),
        fullPage: false,
      });
    }
    const next = result.lowest[0].feature;
    await page
      .getByLabel("Research notes", { exact: true })
      .fill("Unsaved source feature notes");
    await region
      .getByRole("button", { name: "Feature " + next, exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Feature " + next, exact: true }),
    ).toBeVisible();
    await expect(region.locator("tbody tr")).toHaveCount(8);
    await expect(
      page.getByLabel("Research notes", { exact: true }),
    ).toHaveValue("");
    expect(inference).toBe(0);
    expect((await (await request.get("/api/runtime")).json()).loaded).toBe(
      false,
    );
    await page.setViewportSize({ width: 390, height: 844 });
    for (const locale of ["fa", "ar", "ja", "fr", "en"]) {
      await page.locator(".language-select select").selectOption(locale);
      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      await page.locator(".feature-tabs button").nth(1).click();
      const details = page.locator(".feature-directions");
      if (
        !(await details.getAttribute("open")) &&
        !(await details.evaluate((node) => (node as HTMLDetailsElement).open))
      )
        await details.locator("summary").click();
      await expect(details.locator("tbody tr")).toHaveCount(8);
      expect(
        await details
          .locator(".table-wrap")
          .evaluate((node) => node.scrollWidth <= node.clientWidth),
      ).toBe(true);
      expect(
        await details
          .locator("tbody th")
          .first()
          .evaluate((node) => parseFloat(getComputedStyle(node).fontSize)),
      ).toBeGreaterThanOrEqual(12);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await details.evaluate((node) => node.scrollIntoView());
      await page.screenshot({
        path: join(screenshots, "feature-directions-" + locale + "-mobile.png"),
        fullPage: false,
      });
    }
    expect(errors).toEqual([]);
  } finally {
    if (dictionaryId)
      await request.delete("/api/features/dictionaries/" + dictionaryId);
    if (before.loaded) {
      const loading = await request.post("/api/models/load", {
        data: {
          model_id: before.metadata!.id,
          revision: before.metadata!.revision,
          device: before.device,
          precision: before.precision,
          context_length: before.context_length,
        },
      });
      expect(loading.status()).toBe(202);
      const job = await loading.json();
      await expect
        .poll(
          async () =>
            (await (await request.get("/api/jobs/" + job.id)).json()).status,
          { timeout: 60000 },
        )
        .toBe("ready");
    }
  }
});
