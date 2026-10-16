import { expect, test, type APIRequestContext } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { HeadSweepResult, RuntimeState } from "../src/lib/types";
import { capturePage } from "./helpers";

async function saveSweep(
  request: APIRequestContext,
  name: string,
  prompt = "the capital",
) {
  const response = await request.post("/api/head-sweep", {
    data: { prompt, layers: [0] },
  });
  expect(response.status()).toBe(202);
  const queued = await response.json();
  let result: HeadSweepResult | undefined;
  await expect
    .poll(async () => {
      const job = await (await request.get(`/api/jobs/${queued.id}`)).json();
      result = job.result;
      return job.status;
    })
    .toBe("ready");
  const saved = await request.post("/api/experiments/from-job", {
    data: { job_id: queued.id, name },
  });
  expect(saved.status()).toBe(201);
  return { record: await saved.json(), result: result! };
}

async function loadRuntime(
  request: APIRequestContext,
  previous: RuntimeState,
  precision = previous.precision ?? "auto",
) {
  const response = await request.post("/api/models/load", {
    data: {
      model_id: previous.metadata!.id,
      revision: previous.metadata!.revision,
      device: previous.device,
      precision,
      context_length: previous.context_length ?? 128,
    },
  });
  expect(response.status()).toBe(202);
  const job = await response.json();
  await expect
    .poll(
      async () =>
        (await (await request.get(`/api/jobs/${job.id}`)).json()).status,
    )
    .toBe("ready");
}

test("a live head selection prepares an explicit prompt-only capture", async ({
  page,
}) => {
  const prompt = "the capital of France";
  let traces = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/trace"))
      traces++;
  });
  await page.goto("/causal/heads");
  await page.getByLabel("Head sweep prompt").fill(prompt);
  await page.getByLabel("Layers to test").selectOption("2");
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", {
      name: "Inspect layer 2 head 3 in activation lab",
      exact: true,
    })
    .click();
  await expect(page).toHaveURL(/\/playground$/);
  await expect(page.getByLabel("Prompt", { exact: true })).toHaveValue(prompt);
  await expect(page.getByLabel("Generated tokens")).toHaveValue("0");
  await expect(page.getByLabel("Layer intervention")).toHaveValue("head");
  await expect(page.getByLabel("Head ablation layer")).toHaveValue("2");
  await expect(page.getByLabel("Attention head to zero")).toHaveValue("3");
  expect(traces).toBe(0);
  await expect(
    page.getByText("Draft from head sweep · layer 2 · head 3", { exact: true }),
  ).toBeVisible();
  await capturePage(page, {
    path: "../artifacts/screenshots/head-inspection-prepared.png",
  });
  const captured = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace" }).click();
  const trace = await (await captured).json();
  expect(trace.prompt).toBe(prompt);
  expect(trace.generation_performed).toBe(false);
  expect(trace.ablate_head).toMatchObject({
    layer: 2,
    head: 3,
    scope: "all_prompt_tokens",
  });
  expect(traces).toBe(1);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await capturePage(page, {
    path: "../artifacts/screenshots/head-inspection-mobile.png",
  });
  await page.goBack();
  await page
    .getByRole("button", {
      name: "Inspect layer 2 head 1 in activation lab",
      exact: true,
    })
    .click();
  await expect(page.getByLabel("Attention head to zero")).toHaveValue("1");
  await expect(page.getByText("Capture complete", { exact: true })).toHaveCount(
    0,
  );
});

test("saved head inspection accepts a new execution and preserves the full prompt budget", async ({
  page,
  request,
}) => {
  const before: RuntimeState = await (await request.get("/api/runtime")).json();
  const limit = Math.min(
    before.max_prompt_tokens ?? 128,
    before.context_length ?? before.metadata!.context_length,
  );
  const prompt = "the ".repeat(limit).trim();
  const { record, result } = await saveSweep(
    request,
    "Inspection from a saved execution",
    prompt,
  );
  const precision = before.precision === "int4" ? "float32" : "int4";
  try {
    await loadRuntime(request, before, precision);
    await page.goto("/experiments");
    await page
      .getByRole("link", { name: /Inspection from a saved execution/ })
      .click();
    await page
      .getByRole("button", {
        name: "Inspect layer 0 head 1 in activation lab",
        exact: true,
      })
      .click();
    await expect(page.getByLabel("Generated tokens")).toHaveValue("0");
    await expect(page.getByLabel("Prompt", { exact: true })).toHaveValue(
      prompt,
    );
    await expect(
      page.getByText(`Prompt ${limit} / ${limit} tokens`, { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Run & trace" }),
    ).toBeEnabled();
    const captured = page.waitForResponse(
      (response) => response.url().endsWith("/api/trace") && response.ok(),
    );
    await page.getByRole("button", { name: "Run & trace" }).click();
    const trace = await (await captured).json();
    expect(trace.execution_id).not.toBe(result.execution_id);
    expect(trace.execution_precision).toBe(precision);
    expect(trace.ablate_head.head).toBe(1);
    expect(trace.generated_token_ids).toEqual([]);
    const unchanged = await (
      await request.get(`/api/experiments/${record.id}`)
    ).json();
    expect(unchanged.result).toEqual(result);
    await capturePage(page, {
      path: "../artifacts/screenshots/head-inspection-new-execution.png",
    });
  } finally {
    await loadRuntime(request, before);
  }
});

for (const unavailable of ["unloaded", "revision"] as const) {
  test(`a saved head waits for its checkpoint when the runtime is ${unavailable}`, async ({
    page,
    request,
  }) => {
    const name = `Inspection checkpoint guard ${unavailable}`;
    await saveSweep(request, name);
    const state = await (await request.get("/api/runtime")).json();
    await page.route("**/api/runtime", (route) =>
      route.fulfill({
        json:
          unavailable === "unloaded"
            ? { loaded: false, metadata: null, device: "cpu" }
            : {
                ...state,
                metadata: {
                  ...state.metadata,
                  revision: "a-different-checkpoint",
                },
              },
      }),
    );
    await page.goto("/experiments");
    await page.getByRole("link", { name: new RegExp(name) }).click();
    await page
      .getByRole("button", {
        name: "Inspect layer 0 head 1 in activation lab",
        exact: true,
      })
      .click();
    await expect(
      page.getByRole("link", { name: "Open model library", exact: true }),
    ).toBeVisible();
    if (unavailable === "revision")
      await expect(
        page.getByRole("button", { name: "Run & trace" }),
      ).toBeDisabled();
    await expect(
      page.getByRole("button", { name: "Discard head inspection draft" }),
    ).toBeVisible();
    await capturePage(page, {
      path: `../artifacts/screenshots/head-inspection-${unavailable}.png`,
    });
    await page
      .getByRole("link", { name: "Open model library", exact: true })
      .click();
    await page.getByRole("link", { name: "Research lab", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Discard head inspection draft" }),
    ).toBeVisible();
    await page.unroute("**/api/runtime");
    await page
      .getByRole("button", { name: "Refresh runtime", exact: true })
      .click();
    await expect(page.getByLabel("Prompt", { exact: true })).toHaveValue(
      "the capital",
    );
    await expect(page.getByLabel("Attention head to zero")).toHaveValue("1");
    await expect(
      page.getByRole("button", { name: "Run & trace" }),
    ).toBeEnabled();
  });
}

test("an unsupported adapter leaves the inspection draft available to discard", async ({
  page,
  request,
}) => {
  await saveSweep(request, "Inspection adapter guard");
  await page.route("**/api/runtime", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      json: { ...(await response.json()), attention_head_targets: [] },
    });
  });
  await page.goto("/experiments");
  await page.getByRole("link", { name: /Inspection adapter guard/ }).click();
  await page
    .getByRole("button", {
      name: "Inspect layer 0 head 0 in activation lab",
      exact: true,
    })
    .click();
  await expect(
    page.getByText(/This head is unavailable in the active architecture/),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run & trace" }),
  ).toBeDisabled();
  await page
    .getByRole("button", { name: "Discard head inspection draft" })
    .click();
  await expect(page.getByRole("button", { name: "Run & trace" })).toBeEnabled();
});

test("client navigation keeps head inspection usable when session storage is blocked", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      get() {
        throw new DOMException("Blocked for test", "SecurityError");
      },
    });
  });
  await page.goto("/causal/heads");
  await page.getByLabel("Head sweep prompt").fill("the capital");
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", {
      name: "Inspect layer 0 head 0 in activation lab",
      exact: true,
    })
    .click();
  await expect(page.getByLabel("Layer intervention")).toHaveValue("head");
  await expect(page.getByLabel("Prompt", { exact: true })).toHaveValue(
    "the capital",
  );
  await expect(page.getByLabel("Generated tokens")).toHaveValue("0");
});

test("a malformed browser draft shows a recoverable error", async ({
  page,
}) => {
  await page.addInitScript(() =>
    sessionStorage.setItem("lantern.head-inspection.v1", "{broken"),
  );
  await page.goto("/playground?inspect=00000000000000000000000000000000");
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "no longer available",
  );
  await page.getByRole("button", { name: "Dismiss", exact: true }).click();
  await expect(page).toHaveURL(/\/playground$/);
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Run & trace" })).toBeEnabled();
});

test("a stored head draft survives a direct visit and is consumed only once", async ({
  page,
  request,
}) => {
  const { result } = await saveSweep(
    request,
    "Direct head inspection",
    String.fromCodePoint(0x1f3ee).repeat(8000),
  );
  const id = "11111111111111111111111111111111";
  await page.addInitScript(
    (draft) => {
      if (location.search.includes("inspect="))
        sessionStorage.setItem(
          "lantern.head-inspection.v1",
          JSON.stringify(draft),
        );
    },
    {
      version: 1,
      id,
      prompt: result.prompt,
      layer: 0,
      head: 2,
      source: {
        model_id: result.model_id,
        revision: result.revision,
        device: result.device,
        execution_precision: result.execution_precision,
        execution_config: result.execution_config,
      },
    },
  );
  await page.goto("/playground?inspect=" + id);
  await expect(page.getByLabel("Prompt", { exact: true })).toHaveValue(
    result.prompt,
  );
  await expect(page.getByLabel("Attention head to zero")).toHaveValue("2");
  await expect(page.getByLabel("Generated tokens")).toHaveValue("0");
  const captured = page.waitForResponse(
    (response) => response.url().endsWith("/api/trace") && response.ok(),
  );
  await page.getByRole("button", { name: "Run & trace" }).click();
  expect((await (await captured).json()).prompt).toBe(result.prompt);
  await expect(page).toHaveURL(/\/playground$/);
  expect(
    await page.evaluate(() =>
      sessionStorage.getItem("lantern.head-inspection.v1"),
    ),
  ).toBeNull();
  await page.reload();
  await expect(page.getByText(/Draft from head sweep/)).toHaveCount(0);
});

test("an expired link offers an explicit resume for another pending head", async ({
  page,
  request,
}) => {
  const { result } = await saveSweep(
    request,
    "Pending head after a stale link",
  );
  await page.addInitScript(
    (draft) => {
      if (location.search.includes("inspect=33333333333333333333333333333333"))
        sessionStorage.setItem(
          "lantern.head-inspection.v1",
          JSON.stringify(draft),
        );
    },
    {
      version: 1,
      id: "22222222222222222222222222222222",
      prompt: result.prompt,
      layer: 0,
      head: 2,
      source: {
        model_id: result.model_id,
        revision: result.revision,
        device: result.device,
        execution_precision: result.execution_precision,
        execution_config: result.execution_config,
      },
    },
  );
  await page.route("**/api/runtime", (route) =>
    route.fulfill({
      json: { loaded: false, metadata: null, device: "cpu" },
    }),
  );
  await page.goto("/playground?inspect=33333333333333333333333333333333");
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "A different head inspection draft is pending",
  );
  await page
    .getByRole("button", { name: "Resume pending draft", exact: true })
    .click();
  await expect(page).toHaveURL(/\/playground$/);
  await expect(
    page.getByRole("button", { name: "Discard head inspection draft" }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Discard head inspection draft" }),
  ).toBeVisible();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  await page.unroute("**/api/runtime");
  await page
    .getByRole("button", { name: "Refresh runtime", exact: true })
    .click();
  await expect(page.getByLabel("Prompt", { exact: true })).toHaveValue(
    result.prompt,
  );
  await expect(page.getByLabel("Attention head to zero")).toHaveValue("2");
  expect(
    await page.evaluate(() =>
      sessionStorage.getItem("lantern.head-inspection.v1"),
    ),
  ).toBeNull();
});
