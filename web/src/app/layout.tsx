import type { Metadata } from "next";
import { cookies } from "next/headers";
import { ThemeProvider } from "@/components/theme-provider";
import { resolveTheme, themeCookie } from "@/lib/theme";
import { Shell } from "@/components/shell";
import { I18nProvider } from "@/i18n/provider";
import { requestLanguage } from "@/i18n/server";
import { text } from "@/i18n/format";
import { direction } from "@/i18n/locale";
import "@fontsource-variable/vazirmatn";
import "@fontsource-variable/noto-sans-jp";
import "@fontsource-variable/dm-sans";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const { locale, messages } = await requestLanguage();
  return {
    title: {
      default: text(locale, messages, "Model library") + " · Circuit Lantern",
      template: "%s · Circuit Lantern",
    },
    description: text(
      locale,
      messages,
      "A local mechanistic interpretability workbench for exploring transformer weights, residual streams, attention and predictions.",
    ),
  };
}

export default async function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const { locale, messages } = await requestLanguage();
  const theme = resolveTheme((await cookies()).get(themeCookie)?.value);
  return (
    <html lang={locale} dir={direction(locale)} data-theme={theme}>
      <body>
        <I18nProvider initialLocale={locale} initialMessages={messages}>
          <ThemeProvider initialTheme={theme}>
            <Shell>{children}</Shell>
          </ThemeProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
