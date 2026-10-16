import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

for (const [path, runLabel, complete, kind, heading] of [
  [
    "/causal",
    "Run patch sweep",
    "Sweep complete",
    "causal",
    "Per block patching results",
  ],
  [
    "/causal/heads",
    "Run head sweep",
    "Head sweep complete",
    "head_sweep",
    "Per head ablation results",
  ],
]) {
  test(`${kind} sweeps remain inspectable and editable in the notebook without a loaded model`, async ({
    page,
  }) => {
    const name = `Notebook ${kind} evidence`;
    await page.goto(path);
    await page.getByRole("button", { name: runLabel, exact: true }).click();
    await expect(
      page.locator(".trace-meta").getByText(complete, { exact: true }),
    ).toBeVisible();
    await page.getByLabel("Experiment name").fill(name);
    const saving = page.waitForResponse((response) =>
      response.url().endsWith("/api/experiments/from-job"),
    );
    await page
      .getByRole("button", { name: "Save experiment", exact: true })
      .click();
    expect((await saving).status()).toBe(201);
    await expect(
      page.getByRole("button", { name: "Saved to experiments", exact: true }),
    ).toBeDisabled();
    await page
      .getByRole("link", { name: "View saved experiment", exact: true })
      .click();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(name);
    await expect(
      page.getByRole("heading", { name: heading, exact: true }),
    ).toBeVisible();
    await page
      .getByLabel("Observations")
      .fill("Check this intervention on a second prompt.");
    await page.getByRole("button", { name: "Save notes", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Changes saved", exact: true }),
    ).toBeVisible();
    const downloading = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export experiment", exact: true })
      .click();
    const stream = await (await downloading).createReadStream();
    const chunks = [];
    for await (const chunk of stream!) chunks.push(chunk);
    const record = JSON.parse(Buffer.concat(chunks).toString());
    expect(record.kind).toBe(kind);
    expect(record.notes).toBe("Check this intervention on a second prompt.");
    expect(record.result.execution_id).toMatch(/^[a-f0-9]{32}$/);
    expect(record.result.complete).toBe(true);
    expect(record.trace).toBeUndefined();
    await page.route("**/api/runtime", (route) =>
      route.fulfill({
        json: { loaded: false, metadata: null, device: "cpu" },
      }),
    );
    await page.reload();
    await expect(page.getByLabel("Observations")).toHaveValue(
      "Check this intervention on a second prompt.",
    );
    await expect(
      page.getByRole("heading", { name: heading, exact: true }),
    ).toBeVisible();
    if (kind === "causal") {
      await page.getByText("Captured prompts", { exact: true }).click();
      await expect(page.locator(".capture-details")).toContainText(
        record.result.clean_prompt,
      );
      await expect(page.locator(".capture-details")).toContainText(
        record.result.corrupted_prompt,
      );
    }
    await capturePage(page, {
      path: `../artifacts/screenshots/notebook-${kind}.png`,
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
      path: `../artifacts/screenshots/notebook-${kind}-mobile.png`,
    });
  });
}

test("an expired job save leaves its visible result available and can be retried", async ({
  page,
}) => {
  await page.goto("/causal/heads");
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  await page.route("**/api/experiments/from-job", (route) =>
    route.fulfill({
      status: 404,
      json: {
        detail: "The sweep job was not found. Its result may have expired.",
      },
    }),
  );
  await page
    .getByRole("button", { name: "Save experiment", exact: true })
    .click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "expired",
  );
  await expect(
    page.getByRole("button", { name: "Export head sweep" }),
  ).toBeEnabled();
  await expect(page.locator(".head-sweep-table tbody tr")).toHaveCount(4);
  await page.unroute("**/api/experiments/from-job");
  await page
    .getByRole("button", { name: "Save experiment", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Saved to experiments", exact: true }),
  ).toBeDisabled();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
});

test("automatically named sweeps preserve Unicode at the truncation boundary", async ({
  page,
}) => {
  const prompt = "a".repeat(59) + "\u{1f9e0}";
  await page.goto("/causal/heads");
  await page.getByLabel("Head sweep prompt").fill(prompt);
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel("Experiment name")).toHaveValue(
    "Head ablation · " + prompt,
  );
  const saving = page.waitForResponse((response) =>
    response.url().endsWith("/api/experiments/from-job"),
  );
  await page
    .getByRole("button", { name: "Save experiment", exact: true })
    .click();
  expect((await saving).status()).toBe(201);
  await expect(
    page.getByRole("button", { name: "Saved to experiments", exact: true }),
  ).toBeDisabled();
});

test("a lost save response can be retried without duplicating or overwriting the notebook record", async ({
  page,
  request,
}) => {
  await page.goto("/causal/heads");
  await page
    .getByRole("button", { name: "Run head sweep", exact: true })
    .click();
  await expect(
    page
      .locator(".trace-meta")
      .getByText("Head sweep complete", { exact: true }),
  ).toBeVisible();
  let sourceJob = "";
  let recordId = "";
  await page.route("**/api/experiments/from-job", async (route) => {
    sourceJob = route.request().postDataJSON().job_id;
    const response = await route.fetch();
    const record = await response.json();
    recordId = record.id;
    await request.patch("/api/experiments/" + recordId, {
      data: { name: "Edited before retry", notes: "Keep this observation" },
    });
    await route.abort("failed");
  });
  await page
    .getByRole("button", { name: "Save experiment", exact: true })
    .click();
  await expect(page.getByRole("main").getByRole("alert")).toBeVisible();
  await page.unroute("**/api/experiments/from-job");
  await page
    .getByRole("button", { name: "Save experiment", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Saved to experiments", exact: true }),
  ).toBeDisabled();
  await expect(page.getByLabel("Experiment name")).toHaveValue(
    "Edited before retry",
  );
  const records = await (await request.get("/api/experiments")).json();
  expect(
    records.filter(
      (record: { source_job_id?: string }) =>
        record.source_job_id === sourceJob,
    ),
  ).toHaveLength(1);
  const restored = await (
    await request.get("/api/experiments/" + recordId)
  ).json();
  expect(restored.notes).toBe("Keep this observation");
});
