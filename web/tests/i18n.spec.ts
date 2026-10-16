import { expect, test } from "@playwright/test";
import french from "../src/i18n/messages/fr.json";

test("language changes preserve prompt and settings and survive reload", async ({
  page,
  context,
}) => {
  const before = await (await page.request.get("/api/runtime")).json();
  if (!before.loaded) {
    const response = await page.request.post("/api/models/load", {
      data: {
        model_id: "hf-internal-testing/tiny-random-gpt2",
        device: "cpu",
        precision: "fp32",
        context_length: 128,
      },
    });
    expect(response.ok()).toBe(true);
    const job = await response.json();
    await expect
      .poll(
        async () =>
          (await (await page.request.get("/api/jobs/" + job.id)).json()).status,
      )
      .toBe("ready");
  }
  try {
    await page.goto("/playground");
    const prompt = page.getByLabel("Prompt", { exact: true });
    await expect(prompt).toBeVisible();
    await prompt.fill("Mary {x} العربية 日本語");
    await page.getByLabel("Generated tokens").selectOption("0");
    for (const locale of ["fa", "ar", "ja", "fr", "en"]) {
      await page.locator(".language-select select").selectOption(locale);
      await expect(page.locator("html")).toHaveAttribute("lang", locale);
      await expect(page.locator("html")).toHaveAttribute(
        "dir",
        locale === "fa" || locale === "ar" ? "rtl" : "ltr",
      );
      await expect(page.locator(".prompt-controls select").first()).toHaveValue(
        "0",
      );
      await expect(page.locator("textarea").first()).toHaveValue(
        "Mary {x} العربية 日本語",
      );
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
    }
    await page.locator(".language-select select").selectOption("fa");
    await expect(page.locator("html")).toHaveAttribute("lang", "fa");
    expect(
      (await context.cookies()).find(
        (cookie) => cookie.name === "lantern_locale",
      )?.value,
    ).toBe("fa");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
    await expect(page.locator(".language-select select")).toHaveValue("fa");
    await page.locator(".language-select select").selectOption("en");
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    if (before.loaded)
      expect(
        (await (await page.request.get("/api/runtime")).json()).metadata
          .execution_id,
      ).toBe(before.metadata.execution_id);
  } finally {
    if (!before.loaded) await page.request.post("/api/models/unload");
  }
});

for (const name of [
  "Layer 3 · Mary {x}",
  "Block 3 bypassed · Mary",
  "constructor",
  "__proto__",
]) {
  test(
    "saved name remains literal in translated headers " + name,
    async ({ page }) => {
      const id = "literal-name-fixture";
      const record = {
        id,
        name,
        kind: "matrix",
        model_id: "research/checkpoint",
        notes: "Mary {x}",
        created_at: "2026-10-16T10:00:00Z",
        result: null,
      };
      await page.route("**/api/experiments", (route) =>
        route.fulfill({ json: [record] }),
      );
      await page.route("**/api/experiments/" + id, (route) =>
        route.fulfill({ json: record }),
      );
      await page.goto("/experiments?experiment=" + id);
      await page.locator(".language-select select").selectOption("fr");
      await expect(page.locator("html")).toHaveAttribute("lang", "fr");
      await expect(page.getByRole("heading", { level: 1 })).toHaveText(name);
      await expect(page.locator(".experiment-detail input")).toHaveValue(name);
    },
  );
}

test("server page titles follow language cookies before hydration", async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({
    baseURL,
    javaScriptEnabled: false,
  });
  await context.addCookies([
    { name: "lantern_locale", value: "fr", url: baseURL! },
  ]);
  const page = await context.newPage();
  try {
    for (const [path, label] of [
      ["/causal", "Causal tracing"],
      ["/causal/heads", "Head ablation"],
      ["/workspace", "Model explorer"],
    ] as const) {
      await page.goto(path, { waitUntil: "domcontentloaded" });
      await expect(page.locator("html")).toHaveAttribute("lang", "fr");
      await expect(page).toHaveTitle(french[label] + " · Circuit Lantern");
    }
  } finally {
    await context.close();
  }
});
