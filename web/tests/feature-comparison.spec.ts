import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { number } from "../src/lib/api";
import type {
  DictionaryManifest,
  FeatureInspection,
} from "../src/lib/features";

test("compare two prompt feature streams without confusing geometry with activation overlap", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  let identifier = "";
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const runtime = await (await request.get("/api/runtime")).json();
  try {
    const starting = await request.post("/api/features/train", {
      data: {
        name: "Prompt feature comparison",
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
    let trained: {
      status: string;
      result?: { dictionaries: { id: string }[] };
    };
    await expect
      .poll(
        async () => {
          trained = await (await request.get("/api/jobs/" + job.id)).json();
          return trained.status;
        },
        { timeout: 60000 },
      )
      .toBe("ready");
    identifier = trained!.result!.dictionaries[0].id;
    const manifest = (await (
      await request.get("/api/features/dictionaries/" + identifier)
    ).json()) as DictionaryManifest;
    await page.goto("/features");
    await page
      .getByRole("tab", { name: "Explore features", exact: true })
      .click();
    await page
      .getByLabel("Choose a dictionary", { exact: true })
      .selectOption(identifier);
    await expect(page.locator(".feature-browser-layout")).toBeVisible();
    const heading = await page
      .locator(".feature-browser-layout > .panel")
      .nth(1)
      .locator("h2")
      .innerText();
    const first = Number(heading.replace("Feature ", ""));
    expect(Number.isInteger(first)).toBe(true);
    const other = [...manifest.feature_statistics]
      .filter((row) => row.index !== first)
      .sort((a, b) => b.frequency - a.frequency || a.index - b.index)[0].index;
    await page
      .getByLabel("Feature prompt", { exact: true })
      .fill("the capital of France");
    await page.getByText("Compare two features", { exact: true }).click();
    const input = page.getByLabel("Compare with feature", { exact: true });
    const compare = page.getByRole("button", {
      name: "Compare prompt activations",
      exact: true,
    });
    for (const invalid of [String(first), String(manifest.features), "0.5"]) {
      await input.fill(invalid);
      await expect(compare).toBeDisabled();
      await expect(input).toHaveAttribute("aria-invalid", "true");
    }
    await input.fill(String(other));
    await expect(compare).toBeEnabled();
    let forwards = 0;
    page.on("request", (req) => {
      if (req.method() === "POST" && req.url().endsWith("/inspect")) forwards++;
    });
    const pending = page.waitForResponse(
      (response) =>
        response.url().endsWith("/inspect") &&
        response.request().method() === "POST",
    );
    await compare.click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const result = (await response.json()) as FeatureInspection;
    expect(result.feature_comparison!.feature_ids).toEqual([first, other]);
    expect(result.intervention).toBeUndefined();
    expect(result.feature_comparison!.eligible_position_count).toBe(3);
    const region = page.locator(".feature-pair-result");
    await expect(
      page.getByRole("region", {
        name: "Prompt feature comparison",
        exact: true,
      }),
    ).toBeVisible();
    await expect(region.locator(".metric")).toHaveCount(4);
    await expect(region.locator(".recharts-line")).toHaveCount(2);
    await expect(region.locator(".recharts-line-dot")).toHaveCount(
      2 * result.feature_comparison!.eligible_position_count,
    );
    const pair = result.feature_comparison!;
    for (const [label, value] of [
      ["Decoder cosine", pair.decoder_cosine],
      ["Pearson correlation", pair.pearson_correlation],
      ["Activation cosine", pair.activation_cosine],
    ] as const)
      await expect(
        region.locator(".metric").filter({ hasText: label }).locator("strong"),
      ).toHaveText(value === null ? "Unavailable" : number(value, 4));
    await expect(region).toContainText(first + " and " + other);
    await input.fill(String(first));
    await expect(compare).toBeDisabled();
    const downloading = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export feature inspection", exact: true })
      .click();
    const stream = await (await downloading).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    expect(JSON.parse(Buffer.concat(chunks).toString())).toEqual(result);
    const screenshots = join(process.cwd(), "../artifacts/screenshots");
    mkdirSync(screenshots, { recursive: true });
    async function audit(name: string) {
      expect(
        (
          await new AxeBuilder({ page })
            .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
            .analyze()
        ).violations,
      ).toEqual([]);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await region.evaluate((node) =>
        window.scrollTo({
          top: node.getBoundingClientRect().top + scrollY - 80,
          behavior: "instant",
        }),
      );
      await page.mouse.move(0, 0);
      await page.screenshot({ path: join(screenshots, name + ".png") });
    }
    await audit("feature-pair-light");
    await page.getByRole("button", { name: "Dark theme", exact: true }).click();
    await audit("feature-pair-dark");
    await page.setViewportSize({ width: 390, height: 844 });
    for (const locale of ["fa", "ar", "ja", "fr", "en"]) {
      await page.locator(".language-select select").selectOption(locale);
      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      await expect(region.locator(".metric")).toHaveCount(4);
      await audit("feature-pair-" + locale + "-mobile");
    }
    expect(forwards).toBe(1);
    await input.fill(String(other));
    await page.getByLabel("Feature prompt", { exact: true }).fill("France");
    const emptyResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/inspect") &&
        response.request().method() === "POST",
    );
    await compare.click();
    const empty = (await (await emptyResponse).json()) as FeatureInspection;
    expect(empty.feature_comparison!.eligible_position_count).toBe(0);
    expect(empty.feature_comparison!.pearson_correlation).toBeNull();
    await expect(region).toContainText(
      "No eligible token positions in this prompt.",
    );
    await expect(region.locator(".chart")).toHaveCount(0);
    for (const locale of ["fa", "ar", "ja", "fr", "en"]) {
      await page.locator(".language-select select").selectOption(locale);
      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      await expect(region.locator(".feature-pair-unavailable")).toHaveCount(3);
      await audit("feature-pair-empty-" + locale);
    }
    await page
      .getByLabel("Feature prompt", { exact: true })
      .fill("the capital");
    const singleResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/inspect") &&
        response.request().method() === "POST",
    );
    await compare.click();
    const single = (await (await singleResponse).json()) as FeatureInspection;
    expect(single.feature_comparison!.eligible_position_count).toBe(1);
    expect(single.feature_comparison!.pearson_correlation).toBeNull();
    await expect(region.locator(".recharts-line-dot")).toHaveCount(2);
    await expect(region).toContainText(
      "Both features activate at " +
        single.feature_comparison!.coactive_count +
        " of 1 eligible position.",
    );
    await audit("feature-pair-single-position");
    await page
      .getByLabel("Feature prompt", { exact: true })
      .fill("the capital <eos> France is");
    const gapResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/inspect") &&
        response.request().method() === "POST",
    );
    await compare.click();
    const gap = (await (await gapResponse).json()) as FeatureInspection;
    expect(gap.eligible_positions).toEqual([1, 3, 4]);
    expect(gap.feature_comparison!.eligible_position_count).toBe(3);
    await expect(region.locator(".recharts-line-dot")).toHaveCount(6);
    for (const path of await region.locator(".recharts-line-curve").all()) {
      expect(((await path.getAttribute("d")) || "").match(/M/g)).toHaveLength(
        2,
      );
    }
    await audit("feature-pair-interior-gap");
    expect(forwards).toBe(4);
    expect(
      (await (await request.get("/api/runtime")).json()).metadata.execution_id,
    ).toBe(runtime.metadata.execution_id);
    expect(errors).toEqual([]);
  } finally {
    if (identifier)
      expect(
        (
          await request.delete("/api/features/dictionaries/" + identifier)
        ).status(),
      ).toBe(204);
  }
});
