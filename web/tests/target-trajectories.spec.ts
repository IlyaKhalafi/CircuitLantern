import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { number } from "../src/lib/api";
import type { RuntimeState, Trace } from "../src/lib/types";
import { capturePage } from "./helpers";

async function prepare(page: Page, targets: string[]) {
  await page.goto("/playground");
  await page
    .getByLabel("Prompt", { exact: true })
    .fill("the capital of France is");
  await page.getByLabel("Generated tokens").selectOption("0");
  await page.getByText("Track target tokens", { exact: true }).click();
  for (const [index, target] of targets.entries()) {
    await page
      .getByRole("button", { name: "Add target token", exact: true })
      .click();
    await page
      .getByLabel("Target token " + (index + 1), { exact: true })
      .fill(target);
  }
}

async function capture(page: Page): Promise<Trace> {
  const capturing = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  const trace = (await (await capturing).json()) as Trace;
  await page.getByRole("tab", { name: "Logit lens", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Chosen target scores", exact: true }),
  ).toBeVisible();
  await page.getByText("Inspect target scores", { exact: true }).click();
  return trace;
}

async function exported(page: Page, button = "Export trace") {
  const pending = page.waitForEvent("download");
  await page.getByRole("button", { name: button, exact: true }).click();
  const stream = await (await pending).createReadStream(),
    chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString());
}

async function inspect(
  page: Page,
  trace: Trace,
  position: number,
  selected: number,
  reference: number,
) {
  const rows = page.locator(".target-trajectory-table tbody tr");
  await expect(rows).toHaveCount(trace.layers.length);
  for (const [index, layer] of trace.layers.entries()) {
    const a = layer.token_target_log_probabilities![position][selected]!;
    const b =
      reference < 0
        ? null
        : layer.token_target_log_probabilities![position][reference]!;
    const cells = rows.nth(index).getByRole("cell");
    await expect(cells.nth(0)).toHaveText(number(a, 5));
    await expect(cells.nth(1)).toHaveText(number(b, 5));
    await expect(cells.nth(2)).toHaveText(number(b === null ? null : a - b, 5));
    await expect(cells.nth(3)).toHaveText(number(Math.exp(a) * 100, 4) + "%");
    await expect(cells.nth(4)).toHaveText(
      b === null ? "N/A" : number(Math.exp(b) * 100, 4) + "%",
    );
  }
}

async function restore(request: APIRequestContext, before: RuntimeState) {
  const response = await request.post("/api/models/load", {
    data: {
      model_id: before.metadata!.id,
      revision: before.metadata!.revision,
      precision: before.precision,
      device: before.device,
      context_length: before.context_length,
    },
  });
  expect(response.status()).toBe(202);
  const job = await response.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + job.id)).json()).status,
    )
    .toBe("ready");
}

test("tracked targets follow position and alternatives, then reopen after an actual unload", async ({
  page,
  request,
}) => {
  const before: RuntimeState = await (await request.get("/api/runtime")).json();
  let forwards = 0,
    unloaded = false,
    recordId: string | undefined;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/trace"))
      forwards++;
  });
  try {
    await prepare(page, [" France", "Poland", "<pad>", ""]);
    const trace = await capture(page);
    expect(trace.tracked_targets!.map((target) => target.token_id)).toEqual([
      6, 9, 0,
    ]);
    expect(trace.tracked_targets![0].requested_text).toBe(" France");
    await inspect(page, trace, 4, 0, 1);
    await page
      .getByRole("combobox", { name: "Logit lens token position", exact: true })
      .selectOption("0");
    await inspect(page, trace, 0, 0, 1);
    await page
      .getByRole("combobox", { name: "Tracked target", exact: true })
      .selectOption("1");
    await expect(
      page.getByRole("combobox", { name: "Reference target", exact: true }),
    ).toHaveValue("0");
    await inspect(page, trace, 0, 1, 0);
    await page
      .getByRole("combobox", { name: "Tracked target", exact: true })
      .selectOption("2");
    await inspect(page, trace, 0, 2, 0);
    expect(await exported(page)).toEqual(trace);
    await page
      .getByRole("combobox", { name: "Reference target", exact: true })
      .selectOption("-1");
    await expect(
      page.getByText(
        "Choose a reference target to compare two captured alternatives.",
        { exact: true },
      ),
    ).toBeVisible();
    await inspect(page, trace, 0, 2, -1);
    await page
      .getByRole("combobox", { name: "Tracked target", exact: true })
      .selectOption("0");
    await page
      .getByRole("combobox", { name: "Reference target", exact: true })
      .selectOption("1");
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await capturePage(page, {
      path: "../artifacts/screenshots/target-trajectories.png",
    });
    await page.setViewportSize({ width: 390, height: 844 });
    const scores = page.getByRole("region", {
      name: "Tracked target measurements",
      exact: true,
    });
    await scores.focus();
    await page.keyboard.press("ArrowRight");
    await expect
      .poll(() => scores.evaluate((element) => element.scrollLeft))
      .toBeGreaterThan(0);
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
    await capturePage(page, {
      path: "../artifacts/screenshots/target-trajectories-mobile.png",
    });
    await page.getByLabel("Experiment name").fill("Chosen target trajectories");
    const saving = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/experiments") &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    const saved = await (await saving).json();
    recordId = saved.id;
    expect((await request.delete("/api/runtime")).status()).toBe(204);
    unloaded = true;
    await page
      .getByRole("link", { name: "View saved experiment", exact: true })
      .click();
    await page.getByRole("tab", { name: "Logit lens", exact: true }).click();
    await page.getByText("Inspect target scores", { exact: true }).click();
    await inspect(page, trace, 4, 0, 1);
    expect((await exported(page, "Export experiment")).trace).toEqual(trace);
    await capturePage(page, {
      path: "../artifacts/screenshots/notebook-target-trajectories-mobile.png",
    });
    await page.setViewportSize({ width: 1440, height: 1100 });
    await capturePage(page, {
      path: "../artifacts/screenshots/notebook-target-trajectories.png",
    });
    expect((await (await request.get("/api/runtime")).json()).loaded).toBe(
      false,
    );
    expect(forwards).toBe(1);
  } finally {
    if (recordId) await request.delete("/api/experiments/" + recordId);
    if (unloaded) await restore(request, before);
  }
});

test("one tracked target keeps its scores without a false reference", async ({
  page,
}) => {
  await prepare(page, ["France"]);
  const trace = await capture(page);
  await expect(
    page.getByRole("combobox", { name: "Reference target", exact: true }),
  ).toHaveValue("-1");
  await expect(
    page.getByText(/Track a second token when capturing/),
  ).toBeVisible();
  await inspect(page, trace, 4, 0, -1);
});

test("an invalid target preserves the previous capture and editable draft", async ({
  page,
}) => {
  await prepare(page, ["France", "Poland"]);
  const trace = await capture(page);
  await page.getByLabel("Target token 1", { exact: true }).fill("the capital");
  await page.getByRole("button", { name: "Run & trace", exact: true }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "exactly one token",
  );
  await expect(page.getByLabel("Target token 1", { exact: true })).toHaveValue(
    "the capital",
  );
  await inspect(page, trace, 4, 0, 1);
  expect(await exported(page)).toEqual(trace);
});

test("tracking controls bound target count and preserve entries while removing a target", async ({
  page,
}) => {
  await prepare(
    page,
    Array.from({ length: 8 }, (_, index) => "token_" + (index + 10)),
  );
  await expect(
    page.getByRole("button", { name: "Add target token", exact: true }),
  ).toBeDisabled();
  await page
    .getByRole("button", { name: "Remove target token 3", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Add target token", exact: true }),
  ).toBeEnabled();
  await expect(page.getByLabel("Target token 3", { exact: true })).toHaveValue(
    "token_13",
  );
  await expect(page.getByLabel("Target token 7", { exact: true })).toHaveValue(
    "token_17",
  );
});

for (const mode of [
  "missing score",
  "missing position",
  "zero scores",
] as const) {
  test(
    "a capture with " + mode + " preserves score availability",
    async ({ page, request }) => {
      const response = await request.post("/api/trace", {
        data: {
          prompt: "the capital of France is",
          max_new_tokens: 0,
          tracked_tokens: ["France", "Poland"],
        },
      });
      expect(response.ok()).toBe(true);
      const trace = (await response.json()) as Trace;
      if (mode === "missing score")
        trace.layers[0].token_target_log_probabilities![4][0] = null;
      else if (mode === "missing position")
        trace.layers[0].token_target_log_probabilities!.pop();
      else trace.layers[0].token_target_log_probabilities![4] = [0, -1000];
      await page.route("**/api/trace", (route) =>
        route.fulfill({ json: trace }),
      );
      await prepare(page, ["France", "Poland"]);
      await capture(page);
      const cells = page
        .locator(".target-trajectory-table tbody tr")
        .first()
        .getByRole("cell");
      if (mode === "zero scores") {
        await expect(cells.nth(0)).toHaveText("0");
        await expect(cells.nth(2)).toHaveText("1,000");
        await expect(cells.nth(3)).toHaveText("100%");
        await expect(cells.nth(4)).toHaveText("0%");
      } else {
        await expect(cells.nth(0)).toHaveText("N/A");
        await expect(cells.nth(2)).toHaveText("N/A");
        await expect(
          page.getByText(/Some target scores at this position are unavailable/),
        ).toBeVisible();
      }
      expect(await exported(page)).toEqual(trace);
    },
  );
}
