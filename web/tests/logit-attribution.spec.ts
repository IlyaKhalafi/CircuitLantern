import AxeBuilder from "@axe-core/playwright";
import {
  expect,
  test as base,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { mkdirSync, readFileSync } from "node:fs";
import type { RuntimeState, Trace } from "../src/lib/types";
import { number } from "../src/lib/api";
import en from "../src/i18n/messages/en.json";
import fa from "../src/i18n/messages/fa.json";
import ar from "../src/i18n/messages/ar.json";
import ja from "../src/i18n/messages/ja.json";
import fr from "../src/i18n/messages/fr.json";

async function load(
  request: APIRequestContext,
  settings: Record<string, unknown>,
) {
  const response = await request.post("/api/models/load", { data: settings });
  expect(response.status()).toBe(202);
  const job = await response.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + job.id)).json()).status,
    )
    .toBe("ready");
}
const test = base.extend<{ modelSession: void }>({
  modelSession: [
    async ({ request }, use) => {
      const before: RuntimeState = await (
        await request.get("/api/runtime")
      ).json();
      await load(request, {
        model_id: "hf-internal-testing/tiny-random-gpt2",
        device: "cpu",
        precision: "float32",
        context_length: 128,
      });
      try {
        await use();
      } finally {
        if (before.loaded && before.metadata)
          await load(request, {
            model_id: before.metadata.id,
            revision: before.metadata.revision,
            device: before.device,
            precision: before.precision || "auto",
            context_length: before.context_length || 128,
          });
        else await request.delete("/api/runtime");
      }
    },
    { auto: true },
  ],
});
async function capture(page: Page): Promise<Trace> {
  await page.goto("/playground");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France is");
  await page.getByLabel("Generated tokens", { exact: true }).selectOption("0");
  await page.getByText("Track target tokens", { exact: true }).click();
  await page
    .getByRole("button", { name: "Add target token", exact: true })
    .click();
  await page.getByLabel("Target token 1", { exact: true }).fill("France");
  await page
    .getByRole("button", { name: "Add target token", exact: true })
    .click();
  await page.getByLabel("Target token 2", { exact: true }).fill("Poland");
  await expect(
    page.getByRole("status").filter({ hasText: "2 target tokens ready." }),
  ).toBeVisible();
  const response = page.waitForResponse(
    (r) => r.url().endsWith("/api/trace") && r.ok(),
  );
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  const trace = await (await response).json();
  await page
    .getByRole("tab", { name: "Score contributions", exact: true })
    .click();
  await expect(page.locator(".attribution-table tbody tr")).toHaveCount(
    trace.logit_attribution.components.length,
  );
  return trace;
}

test("contributions reconcile with actual logits and saved answers remain explorable after unloading", async ({
  page,
  request,
}) => {
  let saved = "";
  let forwards = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/api/trace") && r.method() === "POST") forwards++;
  });
  try {
    const trace = await capture(page);
    const attribution = trace.logit_attribution!;
    expect(attribution.status).toBe("available");
    const score =
      attribution.checks!.actual_logits[0] -
      attribution.checks!.actual_logits[1];
    await expect(
      page.locator(".attribution-totals > div").first(),
    ).toContainText(number(score, 6));
    await expect(page.locator(".attribution-health")).toContainText(
      "Contribution sum verified",
    );
    await page
      .getByLabel("Sort attribution components", { exact: true })
      .selectOption("magnitude");
    const largest = attribution.components!.reduce((a, b) =>
      Math.abs(a.values[0] - a.values[1]) >= Math.abs(b.values[0] - b.values[1])
        ? a
        : b,
    );
    await expect(
      page.locator(".attribution-table tbody tr").first(),
    ).toHaveAttribute("data-kind", largest.kind);
    await expect(
      page.locator(".attribution-table tbody tr").first(),
    ).toHaveAttribute(
      "data-layer",
      largest.layer == null ? "" : String(largest.layer),
    );
    await page
      .getByLabel("Attribution answer", { exact: true })
      .selectOption("1");
    await expect(
      page.locator(".attribution-totals > div").first(),
    ).toContainText(number(-score, 6));
    await page
      .getByLabel("Attribution alternative", { exact: true })
      .selectOption("-1");
    await expect(
      page.getByText("Raw logit, not probability", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", {
        name: "Prepare patching after block 0",
        exact: true,
      }),
    ).toBeDisabled();
    expect(forwards).toBe(1);
    await page
      .getByLabel("Experiment name", { exact: true })
      .fill("Contributions {literal}");
    const saving = page.waitForResponse(
      (r) =>
        r.url().endsWith("/api/experiments") && r.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    saved = (await (await saving).json()).id;
    expect((await request.delete("/api/runtime")).status()).toBe(204);
    await page.goto("/experiments?experiment=" + saved);
    await expect(
      page.getByRole("heading", {
        name: "Contributions {literal}",
        exact: true,
      }),
    ).toBeVisible();
    await page
      .getByRole("tab", { name: "Score contributions", exact: true })
      .click();
    await expect(
      page.locator(".attribution-totals > div").first(),
    ).toContainText(number(score, 6));
    await page
      .getByLabel("Attribution alternative", { exact: true })
      .selectOption("-1");
    await expect(
      page.locator(".attribution-totals > div").first(),
    ).toContainText(number(attribution.checks!.actual_logits[0], 6));
    const downloading = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export experiment", exact: true })
      .click();
    const download = await downloading;
    const exported = JSON.parse(readFileSync((await download.path())!, "utf8"));
    expect(exported.trace.logit_attribution).toEqual(attribution);
    expect(exported.trace.trace_id).toBe(trace.trace_id);
    expect(forwards).toBe(1);
  } finally {
    if (saved) await request.delete("/api/experiments/" + saved);
  }
});

test("a block hypothesis prepares a source-bound patching experiment without overwriting other drafts", async ({
  page,
  request,
}) => {
  let saved = "";
  let sourceRecord = "";
  try {
    await page.addInitScript(() => {
      if (!sessionStorage.getItem("lantern-prompt-set-draft"))
        sessionStorage.setItem(
          "lantern-prompt-set-draft",
          JSON.stringify({
            pairs: [
              {
                label: "Existing {literal}",
                clean_prompt: "the capital of France",
                corrupted_prompt: "the capital of Poland",
                target: "France",
                alternative: "Poland",
              },
              {
                label: "Second existing",
                clean_prompt: "The capital of France",
                corrupted_prompt: "The capital of Poland",
                target: "France",
                alternative: "Poland",
              },
            ],
            metric: "logit_difference",
            position: "last",
            layers: "0",
            seed: "19",
          }),
        );
    });
    const trace = await capture(page);
    const originalRecord = await request.post("/api/experiments", {
      data: { name: "Original attribution hypothesis", notes: "", trace },
    });
    expect(originalRecord.status()).toBe(201);
    sourceRecord = (await originalRecord.json()).id;
    await page
      .getByRole("button", {
        name: "Prepare patching after block 2",
        exact: true,
      })
      .click();
    await page.waitForURL(/\/causal\?hypothesis=/);
    const sourceURL = page.url();
    await expect(page.locator(".set-hypothesis")).toContainText(
      "Prepared from block 2 attribution",
    );
    await expect(page.locator(".set-pair")).toHaveCount(1);
    await expect(page.getByLabel("Clean prompt", { exact: true })).toHaveValue(
      trace.prompt,
    );
    await expect(
      page.getByLabel("Corrupted prompt", { exact: true }),
    ).toHaveValue("");
    let runs = 0;
    page.on("request", (r) => {
      if (r.url().endsWith("/api/causal/sets/run")) runs++;
    });
    await page
      .getByRole("button", { name: "Check prompt set", exact: true })
      .click();
    await expect(page.getByRole("main").getByRole("alert")).toContainText(
      "needs a clean prompt, corrupted prompt and target",
    );
    expect(runs).toBe(0);
    await page
      .getByLabel("Corrupted prompt", { exact: true })
      .fill("the capital of Poland is");
    await page
      .getByRole("button", { name: "Check prompt set", exact: true })
      .click();
    await expect(
      page.getByRole("button", { name: "Run prompt set", exact: true }),
    ).toBeEnabled();
    const starting = page.waitForResponse((r) =>
      r.url().endsWith("/api/causal/sets/run"),
    );
    await page
      .getByRole("button", { name: "Run prompt set", exact: true })
      .click();
    const response = await starting;
    expect(response.status()).toBe(202);
    const payload = response.request().postDataJSON();
    expect(payload.layers).toEqual([2]);
    expect(payload.hypothesis_source.trace_id).toBe(trace.trace_id);
    expect(payload.hypothesis_source.token_ids).toEqual(trace.token_ids);
    const job = await response.json();
    await expect
      .poll(
        async () =>
          (await (await request.get("/api/jobs/" + job.id)).json()).status,
      )
      .toBe("ready");
    await expect(page.locator(".set-results")).toBeVisible();
    const result = (await (await request.get("/api/jobs/" + job.id)).json())
      .result;
    expect(result.hypothesis_source).toEqual(payload.hypothesis_source);
    const record = await request.post("/api/experiments/from-job", {
      data: { name: "Source-bound patching", job_id: job.id },
    });
    expect(record.status()).toBe(201);
    saved = (await record.json()).id;
    await page
      .getByRole("button", { name: "Return to other draft", exact: true })
      .click();
    await page.waitForURL(/\/causal\?scope=set/);
    await expect(page.locator(".set-pair")).toHaveCount(2);
    await expect(page.locator(".set-pair").first()).toContainText(
      "Existing {literal}",
    );
    await page.goto(sourceURL);
    await expect(
      page.getByLabel("Corrupted prompt", { exact: true }),
    ).toHaveValue("the capital of Poland is");
    await load(request, {
      model_id: "circuit-lantern-test/finetuned",
      device: "cpu",
      precision: "float32",
      context_length: 128,
    });
    await page.reload();
    await expect(page.locator(".set-hypothesis")).toContainText(
      "another checkpoint",
    );
    await expect(
      page.getByRole("button", { name: "Check prompt set", exact: true }),
    ).toBeDisabled();
    await expect(
      page.getByRole("button", { name: "Run prompt set", exact: true }),
    ).toBeDisabled();
    const current = await (await request.get("/api/runtime")).json();
    const rejected = await request.post("/api/causal/sets/run", {
      data: {
        ...payload,
        expected_execution_id: current.metadata.execution_id,
      },
    });
    // Jobs validate source identity before any model forwards, even if a client bypasses the form.
    expect([400, 202]).toContain(rejected.status());
    if (rejected.status() === 202) {
      const blocked = await rejected.json();
      await expect
        .poll(
          async () =>
            (await (await request.get("/api/jobs/" + blocked.id)).json())
              .status,
        )
        .toBe("failed");
    }
    await page
      .getByRole("button", {
        name: "Use as independent experiment",
        exact: true,
      })
      .click();
    await expect(page.locator(".set-hypothesis")).toContainText(
      "Independent prompt-set draft",
    );
    const checking = page.waitForResponse((r) =>
      r.url().endsWith("/api/causal/sets/plan"),
    );
    await page
      .getByRole("button", { name: "Check prompt set", exact: true })
      .click();
    expect(
      (await checking).request().postDataJSON().hypothesis_source,
    ).toBeUndefined();
    const savedRecord = await (
      await request.get("/api/experiments/" + saved)
    ).json();
    expect(savedRecord.result.hypothesis_source.trace_id).toBe(trace.trace_id);
    await page.goto("/experiments?experiment=" + sourceRecord);
    await page
      .getByRole("tab", { name: "Score contributions", exact: true })
      .click();
    await page
      .getByRole("button", {
        name: "Prepare patching after block 2",
        exact: true,
      })
      .click();
    await page.waitForURL(/\/causal\?hypothesis=/);
    expect(page.url()).not.toBe(sourceURL);
    await expect(page.locator(".set-hypothesis")).toContainText(
      "Prepared from block 2 attribution",
    );
    await expect(
      page.getByLabel("Corrupted prompt", { exact: true }),
    ).toHaveValue("");
    await page.goto(sourceURL);
    await expect(page.locator(".set-hypothesis")).toContainText(
      "Independent prompt-set draft",
    );
    await expect(
      page.getByLabel("Corrupted prompt", { exact: true }),
    ).toHaveValue("the capital of Poland is");
  } finally {
    if (saved) await request.delete("/api/experiments/" + saved);
    if (sourceRecord) await request.delete("/api/experiments/" + sourceRecord);
  }
});

test("low precision exposes numerical discrepancy rather than forcing a perfect reconstruction", async ({
  page,
  request,
}) => {
  await load(request, {
    model_id: "hf-internal-testing/tiny-random-gpt2",
    device: "cpu",
    precision: "bfloat16",
    context_length: 128,
  });
  const trace = await capture(page);
  expect(trace.logit_attribution!.captured_dtype).toBe("bfloat16");
  expect(
    trace.logit_attribution!.checks!.absolute_errors.some((error) => error > 0),
  ).toBe(true);
  await page.locator(".attribution-method summary").click();
  await expect(
    page.getByText(
      "Rounding allowances are heuristics based on observed readout magnitudes and execution precision. They are not certified error bounds.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(page.locator(".attribution-totals > div").last()).toContainText(
    "Reconstruction error",
  );
});

test("contribution evidence is accessible across all five locales and both themes", async ({
  browser,
  request,
  baseURL,
}) => {
  test.setTimeout(180000);
  let saved = "";
  try {
    const state = await (await request.get("/api/runtime")).json();
    const response = await request.post("/api/trace", {
      data: {
        prompt: "the capital of France is",
        max_new_tokens: 0,
        capture_attention: false,
        tracked_tokens: ["France", "Poland"],
        expected_execution_id: state.metadata.execution_id,
      },
    });
    expect(response.ok()).toBe(true);
    const trace = await response.json();
    const record = await request.post("/api/experiments", {
      data: { name: "Evidence {literal}", notes: "", trace },
    });
    expect(record.status()).toBe(201);
    saved = (await record.json()).id;
    mkdirSync("../artifacts/screenshots", { recursive: true });
    const locales = { en, fa, ar, ja, fr };
    for (const [locale, messages] of Object.entries(locales))
      for (const theme of ["light", "dark"]) {
        const context = await browser.newContext({
          baseURL,
          viewport: { width: 390, height: 844 },
          hasTouch: true,
        });
        try {
          await context.addCookies([
            { name: "lantern_locale", value: locale, url: baseURL! },
            { name: "lantern_theme", value: theme, url: baseURL! },
          ]);
          const page = await context.newPage();
          await page.goto("/experiments?experiment=" + saved);
          await expect(
            page.getByRole("heading", {
              name: "Evidence {literal}",
              exact: true,
            }),
          ).toBeVisible();
          await page
            .getByRole("tab", {
              name: messages["Score contributions"],
              exact: true,
            })
            .click();
          await expect(page.locator(".attribution-table tbody tr")).toHaveCount(
            trace.logit_attribution.components.length,
          );
          await page.locator(".attribution-method summary").click();
          await page.setViewportSize({ width: 320, height: 740 });
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
          await page.setViewportSize({ width: 390, height: 844 });
          const panel = page
            .locator(".panel")
            .filter({ has: page.locator(".attribution-body") });
          await panel.evaluate((element) =>
            window.scrollTo({
              top: element.getBoundingClientRect().top + scrollY - 100,
              behavior: "instant",
            }),
          );
          await page.evaluate(() => document.fonts.ready);
          await page.mouse.move(4, 4);
          await page.screenshot({
            path: `../artifacts/screenshots/attribution-${locale}-${theme}.png`,
          });
        } finally {
          await context.close();
        }
      }
  } finally {
    if (saved) await request.delete("/api/experiments/" + saved);
  }
});

test("incomplete numerical metadata and wrong-position attributions stay unavailable", async ({
  page,
  request,
}) => {
  const saved: string[] = [];
  try {
    const original = await capture(page);
    for (const problem of ["missing_allowance", "wrong_position"]) {
      const trace = structuredClone(original);
      if (problem === "missing_allowance")
        trace.logit_attribution!.checks!.rounding_allowances = [];
      else trace.logit_attribution!.position = 0;
      const record = await request.post("/api/experiments", {
        data: { name: "Incomplete numerical record", notes: "", trace },
      });
      expect(record.status()).toBe(201);
      const id = (await record.json()).id;
      saved.push(id);
      await page.goto("/experiments?experiment=" + id);
      await page
        .getByRole("tab", { name: "Score contributions", exact: true })
        .click();
      await expect(page.locator(".attribution-body")).toContainText(
        "Score contributions could not be verified",
      );
      await expect(page.locator(".attribution-health")).toHaveCount(0);
      await expect(
        page.getByRole("button", {
          name: "Prepare patching after block 0",
          exact: true,
        }),
      ).toHaveCount(0);
      await page
        .getByRole("tab", { name: "Residual stream", exact: true })
        .click();
      await expect(
        page.getByRole("heading", {
          name: "Residual stream magnitude",
          exact: true,
        }),
      ).toBeVisible();
    }
  } finally {
    for (const id of saved) await request.delete("/api/experiments/" + id);
  }
});
