import { expect, test } from "@playwright/test";

for (const modern of [true, false]) {
  test(
    "deployment explains buffers and " +
      (modern ? "combined" : "legacy") +
      " workspace",
    async ({ page }) => {
      await page.route("**/api/models/compatibility", async (route) => {
        const response = await route.fetch();
        const report = await response.json();
        const {
          workspace_bytes,
          model_buffer_bytes,
          model_buffer_workspace_bytes,
          ...legacy
        } = report;
        const bufferBytes = modern ? 256 * 1024 ** 2 : 0;
        const temporaryBytes = (modern ? 512 : 64) * 1024 ** 2;
        const required =
          report.required_bytes -
          (workspace_bytes ?? report.analysis_workspace_bytes ?? 0) -
          (model_buffer_bytes ?? 0) +
          bufferBytes +
          temporaryBytes;
        await route.fulfill({
          response,
          json: {
            ...legacy,
            analysis_workspace_bytes: 64 * 1024 ** 2,
            ...(modern
              ? {
                  workspace_bytes: temporaryBytes,
                  model_buffer_bytes: bufferBytes,
                  model_buffer_workspace_bytes: temporaryBytes,
                }
              : {}),
            required_bytes: required,
            available_bytes: Math.max(report.available_bytes, required * 2),
          },
        });
      });
      await page.goto("/");
      await page.getByRole("button", { name: /Tiny random GPT-2/ }).click();
      await expect(
        page.getByText("Ready to explore on this server"),
      ).toBeVisible();
      await page.getByText("How this estimate works").click();
      const detail = page
        .locator("details")
        .filter({
          has: page.getByText("How this estimate works", { exact: true }),
        })
        .locator("p")
        .first();
      await expect(detail).toContainText(
        "Temporary workspace " + (modern ? "512" : "64") + " MiB",
      );
      await expect(detail).toContainText(
        "Model buffers " + (modern ? "256 MiB" : "0 B"),
      );
    },
  );
}
