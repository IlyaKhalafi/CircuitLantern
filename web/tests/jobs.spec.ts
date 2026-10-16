import { expect, test } from "@playwright/test";
import { capturePage } from "./helpers";

for (const [path, endpoint, runLabel] of [
  ["/causal", "/api/causal", "Run patch sweep"],
  ["/causal/heads", "/api/head-sweep", "Run head sweep"],
]) {
  test(`${path} keeps cancellation failures visible and restores pending cancellation`, async ({
    page,
  }) => {
    let releaseStart!: () => void;
    const starting = new Promise<void>((resolve) => {
      releaseStart = resolve;
    });
    let cancelAttempts = 0;
    let starts = 0;
    let stopped = false;
    const job = {
      id: "controlled-research-job",
      model_id: "hf-internal-testing/tiny-random-gpt2",
      status: "running",
      progress: 25,
      message: "Running research sweep",
    };
    await page.route(`**${endpoint}`, async (route) => {
      starts += 1;
      await starting;
      await route.fulfill({ status: 202, json: job });
    });
    await page.route("**/api/jobs/controlled-research-job", (route) =>
      route.fulfill({
        json: {
          ...job,
          status: stopped ? "cancelled" : "running",
          message: stopped
            ? "Sweep cancelled before capture"
            : "Running research sweep",
          cancel_requested: cancelAttempts >= 2,
        },
      }),
    );
    await page.route("**/api/jobs/controlled-research-job/cancel", (route) => {
      cancelAttempts += 1;
      return route.fulfill(
        cancelAttempts === 1
          ? {
              status: 503,
              json: {
                detail: "Cancellation could not reach the worker. Retry.",
              },
            }
          : {
              json: { status: "running", cancel_requested: true },
            },
      );
    });
    await page.goto(path);
    await page.getByRole("button", { name: runLabel, exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Cancel sweep", exact: true }),
    ).toHaveCount(0);
    releaseStart();
    await page
      .getByRole("button", { name: "Cancel sweep", exact: true })
      .click();
    const alert = page.getByRole("main").getByRole("alert");
    await expect(alert).toContainText(
      "Cancellation could not reach the worker",
    );
    await page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/jobs/controlled-research-job") &&
        response.request().method() === "GET",
    );
    await expect(alert).toContainText(
      "Cancellation could not reach the worker",
    );
    await page
      .getByRole("button", { name: "Cancel sweep", exact: true })
      .click();
    await expect(
      page.getByRole("button", { name: "Cancelling", exact: true }),
    ).toBeDisabled();
    await expect(alert).toHaveCount(0);
    await page.reload();
    await expect(
      page.getByRole("button", { name: "Cancelling", exact: true }),
    ).toBeDisabled();
    expect(starts).toBe(1);
    await capturePage(page, {
      path: `../artifacts/screenshots/${path.endsWith("heads") ? "head" : "patch"}-sweep-cancelling.png`,
    });
    stopped = true;
    await expect(
      page.getByRole("button", { name: runLabel, exact: true }),
    ).toBeEnabled();
    await expect(
      page.getByText("Sweep cancelled before capture", { exact: true }),
    ).toBeVisible();
  });
}

for (const [path, endpoint, runLabel] of [
  ["/causal", "/api/causal", "Run patch sweep"],
  ["/causal/heads", "/api/head-sweep", "Run head sweep"],
]) {
  test(`${path} shows a new submission failure instead of the previous job error`, async ({
    page,
  }) => {
    let attempts = 0;
    const failed = {
      id: "failed-research-job",
      model_id: "checkpoint",
      status: "failed",
      progress: 0,
      message: "Previous capture failed",
    };
    await page.route(`**${endpoint}`, (route) => {
      attempts += 1;
      return route.fulfill(
        attempts === 1
          ? { status: 202, json: failed }
          : {
              status: 409,
              json: {
                detail: "The active model changed. Refresh the runtime.",
              },
            },
      );
    });
    await page.route("**/api/jobs/failed-research-job", (route) =>
      route.fulfill({ json: failed }),
    );
    await page.goto(path);
    await page.getByRole("button", { name: runLabel, exact: true }).click();
    const alert = page.getByRole("main").getByRole("alert");
    await expect(alert).toContainText("Previous capture failed");
    await page.getByRole("button", { name: runLabel, exact: true }).click();
    await expect(alert).toContainText("The active model changed");
    await expect(alert).not.toContainText("Previous capture failed");
  });
}
