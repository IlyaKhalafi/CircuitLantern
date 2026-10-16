"use client";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "@/components/theme-provider";
import { useI18n } from "@/i18n/provider";
export function ThemeSelect() {
  const { theme, setTheme } = useTheme();
  const { t } = useI18n();
  return (
    <div className="theme-select" role="group" aria-label={t("Color theme")}>
      {(
        [
          ["light", "Light theme", Sun],
          ["dark", "Dark theme", Moon],
        ] as const
      ).map(([value, label, Icon]) => (
        <button
          key={value}
          type="button"
          aria-label={t(label)}
          title={t(label)}
          aria-pressed={theme === value}
          onClick={() => setTheme(value)}
        >
          <Icon size={16} aria-hidden="true" />
        </button>
      ))}
    </div>
  );
}
