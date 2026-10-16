"use client";
import { Languages } from "lucide-react";
import { useI18n } from "@/i18n/provider";
import { locales, localeNames, type Locale } from "@/i18n/locale";
export function LanguageSelect() {
  const { locale, pending, error, setLocale, t } = useI18n();
  return (
    <label className="language-select">
      <Languages size={15} aria-hidden="true" />
      <span className="sr-only">{t("Interface language")}</span>
      <select
        aria-label={t("Interface language")}
        value={locale}
        aria-busy={pending}
        onChange={(event) => {
          void setLocale(event.target.value as Locale);
        }}
      >
        {locales.map((language) => (
          <option key={language} value={language} lang={language}>
            {localeNames[language]}
          </option>
        ))}
      </select>
      {error && (
        <span role="alert" className="language-error">
          {t(error)}
        </span>
      )}
    </label>
  );
}
