import type { Page } from "@playwright/test";

export async function capturePage(
  page: Page,
  options: Parameters<Page["screenshot"]>[0],
) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    window.scrollTo(0, 0);
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => resolve()),
    );
  });
  return page.screenshot({ fullPage: true, ...options });
}
