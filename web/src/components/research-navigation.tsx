"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, GitBranch, Sparkles } from "lucide-react";
import { useI18n } from "@/i18n/provider";
export function ResearchNavigation() {
  const { t } = useI18n();
  const path = usePathname();
  return (
    <nav
      className="method-navigation research-navigation"
      aria-label={t("Research workflow")}
    >
      {[
        { href: "/playground", label: "Activation lab", icon: Activity },
        { href: "/features", label: "Feature lab", icon: Sparkles },
        { href: "/causal", label: "Causal tracing", icon: GitBranch },
      ].map((item) => {
        const active = path === item.href || path.startsWith(item.href + "/");
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={active ? "active" : ""}
          >
            <item.icon size={15} />
            {t(item.label)}
          </Link>
        );
      })}
    </nav>
  );
}
