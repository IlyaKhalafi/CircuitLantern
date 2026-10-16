export const locales = ["en", "fa", "ar", "ja", "fr"] as const;
export type Locale = (typeof locales)[number];
export const localeNames: Record<Locale, string> = {
  en: "English",
  fa: "فارسی",
  ar: "العربية",
  ja: "日本語",
  fr: "Français",
};
export const localeCookie = "lantern_locale";
export function resolveLocale(value?: string | null): Locale {
  return locales.includes(value as Locale) ? (value as Locale) : "en";
}
export function direction(locale: Locale) {
  return locale === "fa" || locale === "ar" ? "rtl" : "ltr";
}
