import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { capturePage } from "./helpers";

test("a training revision is checked then loaded by its resolved commit", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  await page.getByText("Checkpoint revision", { exact: true }).click();
  await page.getByLabel("Hub revision").fill("step1000");
  await expect(
    page.getByRole("button", { name: "Download & explore" }),
  ).toBeDisabled();
  const checking = page.waitForRequest((request) =>
    request.url().endsWith("/api/models/compatibility"),
  );
  await page.getByRole("button", { name: "Check revision" }).click();
  expect((await checking).postDataJSON().revision).toBe("step1000");
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  await page.getByText("How this estimate works").click();
  const commit = page.getByRole("link", { name: /^fixture-/ });
  await expect(commit).toHaveAttribute("href", /\/tree\/fixture-/);
  await capturePage(page, {
    path: "../artifacts/screenshots/checkpoint-revision.png",
  });
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  const loading = page.waitForRequest((request) =>
    request.url().endsWith("/api/models/load"),
  );
  let releaseLoad!: () => void;
  const waitingLoad = new Promise<void>((resolve) => {
    releaseLoad = resolve;
  });
  await page.route(
    "**/api/models/load",
    async (route) => {
      await waitingLoad;
      await route.continue();
    },
    { times: 1 },
  );
  await page.getByRole("button", { name: "Download & explore" }).click();
  await expect(page.getByLabel("Hub revision")).toBeDisabled();
  await expect(
    page.getByLabel("Execution format", { exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Close compatibility check" }),
  ).toBeDisabled();
  releaseLoad();
  const payload = (await loading).postDataJSON();
  expect(payload.revision).toMatch(/^fixture-/);
  await expect(page).toHaveURL(/workspace/);
  const runtime = await (await request.get("/api/runtime")).json();
  expect(runtime.metadata.revision).toBe(payload.revision);
});

test("editing a revision invalidates in-flight checks and reopening restores main", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  await page.getByText("Checkpoint revision", { exact: true }).click();
  let release!: () => void;
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  let entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  await page.route(
    "**/api/models/compatibility",
    async (route) => {
      const response = await route.fetch();
      entered();
      await pending;
      await route.fulfill({ response });
    },
    { times: 1 },
  );
  await page.getByLabel("Hub revision").fill("step1000");
  await page.getByRole("button", { name: "Check revision" }).click();
  await started;
  await page.getByLabel("Hub revision").fill("missing-revision");
  const stale = page.waitForResponse((response) =>
    response.url().endsWith("/api/models/compatibility"),
  );
  release();
  await stale;
  await expect(
    page.getByText("Ready to explore on this server"),
  ).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "Download & explore" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Check revision" }).click();
  await expect(
    page.getByText("The requested test revision does not exist."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close compatibility check" }).click();
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await page.getByText("Checkpoint revision", { exact: true }).click();
  await expect(page.getByLabel("Hub revision")).toHaveValue("main");
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
});

test("lost download polling releases the dialog with a useful recovery message", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
  await expect(page.getByText("Ready to explore on this server")).toBeVisible();
  await page.route("**/api/jobs/*", (route) =>
    route.fulfill({
      status: 502,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Connection interrupted." }),
    }),
  );
  const loading = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/models/load") && response.status() === 202,
  );
  await page.getByRole("button", { name: "Download & explore" }).click();
  const job = await (await loading).json();
  try {
    await expect(
      page.getByText(/The server may still be preparing the model/),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Close compatibility check" }),
    ).toBeEnabled();
    await page
      .getByRole("button", { name: "Close compatibility check" })
      .click();
    await expect(page.getByRole("dialog")).not.toBeVisible();
  } finally {
    // Browser polling is interrupted, but the real server job still owns the model.
    await expect
      .poll(
        async () =>
          (await (await request.get("/api/jobs/" + job.id)).json()).status,
      )
      .toBe("ready");
  }
  expect((await (await request.get("/api/runtime")).json()).loaded).toBe(true);
});
