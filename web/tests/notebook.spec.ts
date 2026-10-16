import { expect, test, type APIRequestContext } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Experiment } from "../src/lib/types";
import { capturePage } from "./helpers";

let records: Experiment[] = [];

async function saveSweep(
  request: APIRequestContext,
  kind: "causal" | "head_sweep",
) {
  const started = await request.post(
    kind === "causal" ? "/api/causal" : "/api/head-sweep",
    {
      data:
        kind === "causal"
          ? {
              clean_prompt: "the capital of France",
              corrupted_prompt: "the capital of Germany",
            }
          : { prompt: "the capital of France", layers: [0] },
    },
  );
  expect(started.status()).toBe(202);
  const job = await started.json();
  await expect
    .poll(
      async () =>
        (await (await request.get(`/api/jobs/${job.id}`)).json()).status,
    )
    .toBe("ready");
  const saved = await request.post("/api/experiments/from-job", {
    data: {
      job_id: job.id,
      name:
        kind === "causal" ? "Notebook Atlas patching" : "Notebook Atlas heads",
      notes: "Compare the answer token on a second prompt.",
    },
  });
  expect(saved.status()).toBe(201);
  return (await (
    await request.get("/api/experiments/" + (await saved.json()).id)
  ).json()) as Experiment;
}

test.beforeAll(async ({ request }) => {
  const response = await request.post("/api/trace", {
    data: {
      prompt: "the capital of France",
      max_new_tokens: 0,
      capture_attention: false,
    },
  });
  expect(response.ok()).toBe(true);
  const trace = await response.json();
  for (const [name, notes] of [
    [
      "Notebook Atlas baseline",
      "Observe the residual direction before intervening.",
    ],
    ["Notebook Boreal control", "A separate prompt control."],
  ]) {
    const saved = await request.post("/api/experiments", {
      data: { name, notes, trace },
    });
    expect(saved.status()).toBe(201);
    records.push(
      await (
        await request.get("/api/experiments/" + (await saved.json()).id)
      ).json(),
    );
  }
  records.push(await saveSweep(request, "causal"));
  records.push(await saveSweep(request, "head_sweep"));
});

test.afterAll(async ({ request }) => {
  for (const record of records)
    await request.delete(`/api/experiments/${record.id}`);
});

test("notebook searches names, models and notes with persistent type filters", async ({
  page,
}) => {
  await page.goto("/experiments");
  const cards = page.locator("a.experiment-card");
  await page
    .getByRole("searchbox", { name: "Search notebook" })
    .fill("NOTEBOOK ATLAS");
  await expect(cards).toHaveCount(3);
  await page.getByLabel("Experiment type").selectOption("head_sweep");
  await expect(cards).toHaveCount(1);
  await expect(cards).toContainText("Notebook Atlas heads");
  await expect(page).toHaveURL(/kind=head_sweep/);
  await page.reload();
  await expect(
    page.getByRole("searchbox", { name: "Search notebook" }),
  ).toHaveValue("NOTEBOOK ATLAS");
  await expect(page.getByLabel("Experiment type")).toHaveValue("head_sweep");
  await expect(cards).toHaveCount(1);
  await page
    .getByRole("button", { name: "Clear filters", exact: true })
    .click();
  await expect(page).toHaveURL(/\/experiments$/);
  await page
    .getByRole("searchbox", { name: "Search notebook" })
    .fill("Notebook residual direction");
  await expect(cards).toHaveCount(1);
  await expect(cards).toContainText("Notebook Atlas baseline");
  await page
    .getByRole("searchbox", { name: "Search notebook" })
    .fill("Notebook " + records[0].model_id);
  await expect(cards).toHaveCount(4);
  await page.getByLabel("Experiment type").selectOption("trace");
  await expect(cards).toHaveCount(2);
  await page
    .getByRole("searchbox", { name: "Search notebook" })
    .fill("No matching observation exists");
  await expect(
    page.getByRole("heading", { name: "No matching experiments." }),
  ).toBeVisible();
  await expect(
    page.getByRole("status").filter({ hasText: "0 of" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Clear filters", exact: true })
    .click();
  await page
    .getByRole("searchbox", { name: "Search notebook" })
    .fill("Notebook");
  await capturePage(page, {
    path: "../artifacts/screenshots/notebook-search.png",
    fullPage: true,
  });
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
    path: "../artifacts/screenshots/notebook-search-mobile.png",
    fullPage: true,
  });
});

test("record links survive reload and browser history while preserving the search", async ({
  page,
  request,
}) => {
  const record = records[3];
  let inference = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      /\/api\/(trace|causal|head-sweep|models\/load)$/.test(request.url())
    )
      inference++;
  });
  await page.goto("/experiments?q=Notebook+Atlas&kind=head_sweep");
  const link = page.getByRole("link", { name: /Notebook Atlas heads/ });
  await expect(link).toHaveAttribute(
    "href",
    `/experiments?q=Notebook+Atlas&kind=head_sweep&experiment=${record.id}`,
  );
  await link.click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(record.name);
  const bookmark = page.url();
  await page.reload();
  await expect(page.getByLabel("Observations")).toHaveValue(record.notes);
  await expect(
    page.getByRole("heading", { name: "Per head ablation results" }),
  ).toBeVisible();
  await page
    .getByRole("link", { name: "All experiments", exact: true })
    .click();
  await expect(
    page.getByRole("searchbox", { name: "Search notebook" }),
  ).toHaveValue("Notebook Atlas");
  await expect(page.locator("a.experiment-card")).toHaveCount(1);
  await page.goBack();
  await expect(page).toHaveURL(bookmark);
  await expect(page.getByLabel("Observations")).toHaveValue(record.notes);
  await page.goBack();
  await expect(
    page.getByRole("searchbox", { name: "Search notebook" }),
  ).toHaveValue("Notebook Atlas");
  await page.goForward();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(record.name);
  expect(inference).toBe(0);
  const unchanged = await (
    await request.get(`/api/experiments/${record.id}`)
  ).json();
  expect(unchanged).toEqual(record);
});

test("an unavailable bookmark explains the error and keeps the filtered library usable", async ({
  page,
}) => {
  await page.goto(
    "/experiments?q=Notebook+Boreal&experiment=" + "0".repeat(32),
  );
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "not found",
  );
  await page
    .getByRole("link", { name: "All experiments", exact: true })
    .click();
  await expect(
    page.getByRole("searchbox", { name: "Search notebook" }),
  ).toHaveValue("Notebook Boreal");
  await expect(page.locator("a.experiment-card")).toHaveCount(1);
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  await page.getByRole("link", { name: /Notebook Boreal control/ }).click();
  await expect(page.getByLabel("Observations")).toHaveValue(records[1].notes);
});

test("a late record response cannot replace the experiment selected afterward", async ({
  page,
}) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let started!: () => void;
  const waiting = new Promise<void>((resolve) => {
    started = resolve;
  });
  await page.route(`**/api/experiments/${records[0].id}`, async (route) => {
    started();
    await gate;
    await route.fulfill({ json: records[0] });
  });
  await page.goto(`/experiments?experiment=${records[0].id}`);
  await waiting;
  await page.evaluate(
    (id) => window.history.pushState(null, "", "/experiments?experiment=" + id),
    records[1].id,
  );
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    records[1].name,
  );
  release();
  await page.unrouteAll({ behavior: "wait" });
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    records[1].name,
  );
  await expect(page.getByLabel("Observations")).toHaveValue(records[1].notes);
});

test("editing and deleting a bookmarked record keep its list and URL consistent", async ({
  page,
  request,
}) => {
  const saved = await request.post("/api/experiments", {
    data: {
      name: "Notebook editable evidence",
      notes: "",
      trace:
        records[0].kind === "trace" || !records[0].kind
          ? records[0].trace
          : undefined,
    },
  });
  expect(saved.status()).toBe(201);
  const record = await saved.json();
  records.push(record);
  await page.goto(
    `/experiments?q=Notebook+editable&kind=trace&experiment=${record.id}`,
  );
  await page
    .getByLabel("Experiment name")
    .fill("Notebook editable observation");
  await page
    .getByLabel("Observations")
    .fill("The source measurements remain fixed.");
  await page.getByRole("button", { name: "Save notes", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Changes saved", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Notebook editable observation",
  );
  await expect(page).toHaveURL(new RegExp(record.id));
  await page
    .getByRole("link", { name: "All experiments", exact: true })
    .click();
  await expect(
    page.getByRole("link", { name: /Notebook editable observation/ }),
  ).toContainText("The source measurements remain fixed.");
  await page
    .getByRole("link", { name: /Notebook editable observation/ })
    .click();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await page
    .getByRole("button", { name: "Confirm delete", exact: true })
    .click();
  await expect(page).toHaveURL(
    /\/experiments\?q=Notebook\+editable&kind=trace$/,
  );
  await expect(
    page.getByRole("heading", { name: "No matching experiments." }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "No matching experiments." }),
  ).toBeVisible();
  expect((await request.get(`/api/experiments/${record.id}`)).status()).toBe(
    404,
  );
});

test("a list failure can be retried without obstructing a bookmarked record", async ({
  page,
}) => {
  await page.route("**/api/experiments", (route) =>
    route.fulfill({
      status: 502,
      json: { detail: "Notebook list is temporarily unavailable." },
    }),
  );
  await page.goto("/experiments?experiment=" + records[0].id);
  await expect(page.getByLabel("Observations")).toHaveValue(records[0].notes);
  await page
    .getByRole("link", { name: "All experiments", exact: true })
    .click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "temporarily unavailable",
  );
  await expect(
    page.getByRole("heading", { name: "Start collecting evidence." }),
  ).toHaveCount(0);
  await page.unroute("**/api/experiments");
  await page
    .getByRole("button", { name: "Retry notebook", exact: true })
    .click();
  await expect(
    page.getByRole("link", { name: /Notebook Atlas baseline/ }),
  ).toBeVisible();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
});

test("rapid typing and caret edits preserve every search character", async ({
  page,
}) => {
  await page.goto("/experiments");
  const input = page.getByRole("searchbox", { name: "Search notebook" });
  for (let attempt = 0; attempt < 5; attempt++) {
    await input.fill("");
    const query = "Notebook Atlas residual direction " + attempt;
    await input.pressSequentially(query, { delay: 0 });
    await expect(input).toHaveValue(query);
    await expect
      .poll(() => new URL(page.url()).searchParams.get("q"))
      .toBe(query);
  }
  await input.fill("Atlas baseline");
  await input.press("Home");
  await input.pressSequentially("Notebook ", { delay: 0 });
  await expect(input).toHaveValue("Notebook Atlas baseline");
  await expect(page.locator("a.experiment-card")).toHaveCount(1);
  await expect(page.locator("a.experiment-card")).toContainText(
    records[0].name,
  );
  await expect(page.locator("a.experiment-card")).toHaveAttribute(
    "href",
    new RegExp("q=Notebook\\+Atlas\\+baseline"),
  );
});

for (const action of ["save", "delete"] as const) {
  test(
    "a delayed list cannot undo a bookmarked " + action,
    async ({ page, request }) => {
      const source = records[0];
      if (source.kind && source.kind !== "trace")
        throw new Error("Expected trace evidence");
      const created = await request.post("/api/experiments", {
        data: { name: "Notebook slow list " + action, trace: source.trace },
      });
      expect(created.status()).toBe(201);
      const record = await created.json();
      records.push(record);
      const originalList = await (await request.get("/api/experiments")).json();
      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      let started!: () => void;
      const waiting = new Promise<void>((resolve) => {
        started = resolve;
      });
      let requests = 0;
      await page.route("**/api/experiments", async (route) => {
        if (++requests > 1) return route.continue();
        started();
        await gate;
        await route.fulfill({ json: originalList });
      });
      try {
        await page.goto(
          "/experiments?q=Notebook+slow+list&experiment=" + record.id,
        );
        await waiting;
        await expect(page.getByLabel("Experiment name")).toHaveValue(
          record.name,
        );
        if (action === "save") {
          await page
            .getByLabel("Experiment name")
            .fill("Notebook slow list revised evidence");
          await page
            .getByRole("button", { name: "Save notes", exact: true })
            .click();
          await expect(
            page.getByRole("button", { name: "Changes saved", exact: true }),
          ).toBeVisible();
          await page
            .getByRole("link", { name: "All experiments", exact: true })
            .click();
          await expect(
            page.getByRole("link", {
              name: /Notebook slow list revised evidence/,
            }),
          ).toBeVisible();
        } else {
          await page
            .getByRole("button", { name: "Delete", exact: true })
            .click();
          await page
            .getByRole("button", { name: "Confirm delete", exact: true })
            .click();
          await expect(
            page.getByRole("searchbox", { name: "Search notebook" }),
          ).toBeVisible();
          await expect(
            page.getByRole("link", { name: new RegExp(record.name) }),
          ).toHaveCount(0);
        }
        release();
        await page.unrouteAll({ behavior: "wait" });
        if (action === "save") {
          await expect(
            page.getByRole("link", {
              name: /Notebook slow list revised evidence/,
            }),
          ).toBeVisible();
          await expect(
            page.getByRole("link", { name: new RegExp(record.name) }),
          ).toHaveCount(0);
        } else {
          await expect(
            page.getByRole("link", { name: new RegExp(record.name) }),
          ).toHaveCount(0);
        }
      } finally {
        release();
        await page.unrouteAll({ behavior: "wait" });
      }
    },
  );
}
