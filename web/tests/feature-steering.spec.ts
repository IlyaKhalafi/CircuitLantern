import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { FeatureInspection } from "../src/lib/features";
import type { RuntimeState } from "../src/lib/types";

test("fixed feature steering retains its coefficient, positions and evidence across edits and unload", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  const before = (await (
    await request.get("/api/runtime")
  ).json()) as RuntimeState;
  let identifier = "";
  try {
    const started = await request.post("/api/features/train", {
      data: {
        name: "Fixed coefficient study",
        layers: [0],
        steps: 16,
        batch_size: 16,
        active_features: 4,
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
      },
    });
    expect(started.status()).toBe(202);
    const job = await started.json();
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
    identifier = training!.result!.dictionaries[0].id;
    await page.goto("/features");
    await page
      .getByRole("tab", { name: "Explore features", exact: true })
      .click();
    await page
      .getByLabel("Choose a dictionary", { exact: true })
      .selectOption(identifier);
    await expect(page.locator(".feature-browser-layout")).toBeVisible();
    const feature = Number(
      (
        await page
          .locator(".feature-browser-layout > .panel")
          .nth(1)
          .locator("h2")
          .innerText()
      ).replace("Feature ", ""),
    );
    let inspections = 0;
    page.on("request", (req) => {
      if (req.method() === "POST" && req.url().endsWith("/inspect"))
        inspections++;
    });
    await page
      .getByLabel("Feature prompt", { exact: true })
      .fill("the capital <eos> France is");
    const baseline = page.waitForResponse(
      (r) => r.url().endsWith("/inspect") && r.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Inspect features", exact: true })
      .click();
    expect((await baseline).status()).toBe(200);
    await page
      .getByLabel("Intervention positions", { exact: true })
      .selectOption("3");
    await page.getByText("Set a feature coefficient", { exact: true }).click();
    const input = page.getByLabel("Fixed coefficient", { exact: true });
    const apply = page.getByRole("button", {
      name: "Apply fixed coefficient",
      exact: true,
    });
    for (const invalid of ["", "10001", "-10001"]) {
      await input.fill(invalid);
      await expect(apply).toBeDisabled();
      await expect(input).toHaveAttribute("aria-invalid", "true");
    }
    await input.fill("-2.5");
    const pending = page.waitForResponse(
      (r) => r.url().endsWith("/inspect") && r.request().method() === "POST",
    );
    await apply.click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const sent = response.request().postDataJSON();
    expect(sent).toMatchObject({
      feature,
      coefficient: -2.5,
      intervention_mode: "set",
      position: 3,
    });
    const result = (await response.json()) as FeatureInspection;
    expect(result.intervention).toMatchObject({
      feature,
      mode: "set",
      coefficient: -2.5,
      coefficient_units: "normalized_sae",
      position: 3,
    });
    expect(result.intervention).not.toHaveProperty("multiplier");
    expect(result.eligible_positions).toEqual([1, 3, 4]);
    const evidence = page.locator(".feature-intervention-evidence");
    await expect(evidence).toContainText("-2.5");
    await expect(
      page.getByRole("heading", {
        name: "Baseline feature activations",
        exact: true,
      }),
    ).toBeVisible();
    await page.getByText("Intervention precision", { exact: true }).click();
    await expect(page.locator(".feature-precision-grid .metric")).toHaveCount(
      3,
    );
    await input.fill("7.25");
    await page
      .getByLabel("Intervention positions", { exact: true })
      .selectOption("all");
    const other = feature === 0 ? 1 : 0;
    await page
      .getByLabel("Find a feature", { exact: true })
      .fill(String(other));
    await page
      .locator(".feature-list .feature-row")
      .filter({ has: page.getByText("Feature " + other, { exact: true }) })
      .click();
    await expect(evidence).toContainText("Feature " + feature);
    await expect(evidence).toContainText("-2.5");
    await expect(evidence).not.toContainText("7.25");
    const screenshots = join(process.cwd(), "../artifacts/screenshots");
    mkdirSync(screenshots, { recursive: true });
    async function audit(name: string, target = ".feature-inspection-result") {
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
      await page.locator(target).evaluate((node) =>
        window.scrollTo({
          top: node.getBoundingClientRect().top + scrollY - 80,
          behavior: "instant",
        }),
      );
      await page.screenshot({ path: join(screenshots, name + ".png") });
    }
    await audit("feature-steering-light");
    await page.getByRole("button", { name: "Dark theme", exact: true }).click();
    await audit("feature-steering-dark");
    await page.setViewportSize({ width: 390, height: 844 });
    for (const locale of ["fa", "ar", "ja", "fr", "en"]) {
      await page.locator(".language-select select").selectOption(locale);
      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      await audit(
        "feature-steering-controls-" + locale,
        ".feature-steering-controls",
      );
      await audit(
        "feature-steering-result-" + locale,
        ".feature-intervention-evidence",
      );
      await expect(evidence).toContainText("-2.5");
    }
    expect(inspections).toBe(2);
    expect((await request.delete("/api/runtime")).status()).toBe(204);
    await page.getByRole("button", { name: "Refresh", exact: true }).click();
    const downloaded = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export feature inspection", exact: true })
      .click();
    const stream = await (await downloaded).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    expect(JSON.parse(Buffer.concat(chunks).toString())).toEqual(result);
    expect(inspections).toBe(2);
  } finally {
    if (identifier)
      await request.delete("/api/features/dictionaries/" + identifier);
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
