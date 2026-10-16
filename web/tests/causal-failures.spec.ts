import { expect, test, type APIRequestContext } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { CausalResult, Job } from "../src/lib/types";
import { capturePage } from "./helpers";

async function completedSweep(request: APIRequestContext) {
  const response = await request.post("/api/causal", {
    data: {
      clean_prompt: "the capital of France is ".repeat(4).trim(),
      corrupted_prompt: "the capital of Poland is ".repeat(4).trim(),
      position: -1,
    },
  });
  expect(response.status()).toBe(202);
  const queued = await response.json();
  let completed: (Job & { result?: CausalResult }) | undefined;
  await expect
    .poll(async () => {
      completed = await (await request.get("/api/jobs/" + queued.id)).json();
      return completed?.status;
    })
    .toBe("ready");
  return { queued, captured: completed!.result! };
}

function stoppedResult(
  captured: CausalResult,
  count: number,
  failed: boolean,
): CausalResult {
  return {
    ...captured,
    complete: false,
    cancelled: !failed,
    error: failed ? "The next patch forward failed" : null,
    failed_layer: failed ? count : null,
    layers: captured.layers.slice(0, count),
  };
}

async function exported(page: import("@playwright/test").Page, button: string) {
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", { name: button, exact: true }).click();
  const stream = await (await downloading).createReadStream();
  const chunks = [];
  for await (const chunk of stream!) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString());
}

for (const [outcome, count] of [
  ["failed", 0],
  ["failed", 2],
  ["cancelled", 2],
] as const) {
  test(
    "a " +
      outcome +
      " patch sweep keeps " +
      count +
      " completed blocks visible and exportable",
    async ({ page, request }) => {
      const { queued, captured } = await completedSweep(request);
      const result = stoppedResult(captured, count, outcome === "failed");
      let stopped = outcome === "failed";
      const jobId = "controlled-causal-job";
      await page.route("**/api/causal", (route) =>
        route.fulfill({
          status: 202,
          json: { ...queued, id: jobId, status: "running", progress: 0 },
        }),
      );
      await page.route("**/api/jobs/" + jobId, (route) =>
        route.fulfill({
          json: {
            ...queued,
            id: jobId,
            status: stopped ? outcome : "running",
            progress: Math.round((count / captured.total_layers) * 100),
            message: stopped
              ? (result.error ??
                "Causal sweep cancelled. Completed layers are retained.")
              : "Testing patches",
            ...(stopped ? { result } : {}),
          },
        }),
      );
      await page.route("**/api/jobs/" + jobId + "/cancel", (route) => {
        stopped = true;
        return route.fulfill({
          json: { status: "running", cancel_requested: true },
        });
      });
      await page.goto("/causal");
      await page
        .getByRole("button", { name: "Run patch sweep", exact: true })
        .click();
      if (outcome === "cancelled")
        await page
          .getByRole("button", { name: "Cancel sweep", exact: true })
          .click();
      await expect(
        page.locator(".trace-meta").getByText("Partial sweep", { exact: true }),
      ).toBeVisible();
      await expect(page.locator(".trace-meta")).toContainText(
        count + " of " + captured.total_layers + " blocks completed",
      );
      await expect(page.locator(".causal-results-table tbody tr")).toHaveCount(
        count,
      );
      await expect(
        page.getByText("Sweep complete", { exact: true }),
      ).toHaveCount(0);
      if (outcome === "failed") {
        await expect(page.getByRole("main").getByRole("alert")).toHaveCount(1);
        await expect(page.getByRole("main").getByRole("alert")).toContainText(
          "stopped at block " + count,
        );
      } else {
        await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
      }
      if (count === 0)
        await expect(
          page.getByText(
            "No patch measurements completed. Baseline scores remain available above.",
            { exact: true },
          ),
        ).toBeVisible();
      expect(await exported(page, "Export sweep")).toEqual(result);
      await expect(
        page.getByRole("button", { name: "Save experiment", exact: true }),
      ).toBeEnabled();
      await expect(
        page.getByRole("button", { name: "Run patch sweep", exact: true }),
      ).toBeEnabled();
      await page.reload();
      await expect(page.locator(".causal-results-table tbody tr")).toHaveCount(
        count,
      );
      await expect(
        page.locator(".trace-meta").getByText("Partial sweep", { exact: true }),
      ).toBeVisible();
      expect(
        (
          await new AxeBuilder({ page })
            .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
            .analyze()
        ).violations,
      ).toEqual([]);
      if (outcome === "failed" && count === 2) {
        const alignment = page.getByRole("region", {
          name: "Token alignment",
          exact: true,
        });
        await alignment.focus();
        await page.keyboard.press("ArrowDown");
        await expect
          .poll(() => alignment.evaluate((element) => element.scrollTop))
          .toBeGreaterThan(0);
        await capturePage(page, {
          path: "../artifacts/screenshots/causal-failed-partial.png",
        });
        await page.setViewportSize({ width: 390, height: 844 });
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBe(true);
        await capturePage(page, {
          path: "../artifacts/screenshots/causal-failed-partial-mobile.png",
        });
        expect(
          (
            await new AxeBuilder({ page })
              .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
              .analyze()
          ).violations,
        ).toEqual([]);
      }
    },
  );
}

test("a saved failed patch capture retains its partial label and failure details without a loaded model", async ({
  page,
  request,
}) => {
  const { queued, captured } = await completedSweep(request);
  const result = stoppedResult(captured, 2, true);
  const saving = await request.post("/api/experiments/from-job", {
    data: {
      job_id: queued.id,
      name: "Partial patch notebook presentation",
    },
  });
  expect(saving.status()).toBe(201);
  const record = await saving.json();
  try {
    const stored = await (
      await request.get("/api/experiments/" + record.id)
    ).json();
    await page.route("**/api/experiments/" + record.id, (route) =>
      route.fulfill({ json: { ...stored, result } }),
    );
    await page.route("**/api/runtime", (route) =>
      route.fulfill({ json: { loaded: false, metadata: null, device: "cpu" } }),
    );
    await page.goto("/experiments?experiment=" + record.id);
    await expect(
      page.locator(".trace-meta").getByText("Partial sweep", { exact: true }),
    ).toBeVisible();
    await expect(page.getByRole("main").getByRole("alert")).toContainText(
      "The sweep stopped at block 2",
    );
    await expect(page.locator(".causal-results-table tbody tr")).toHaveCount(2);
    expect((await exported(page, "Export experiment")).result).toEqual(result);
    await capturePage(page, {
      path: "../artifacts/screenshots/notebook-causal-failed-partial.png",
    });
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await capturePage(page, {
      path: "../artifacts/screenshots/notebook-causal-failed-partial-mobile.png",
    });
  } finally {
    await request.delete("/api/experiments/" + record.id);
  }
});
