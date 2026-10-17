"use client";
import { useI18n } from "@/i18n/provider";
import { ResearchNavigation } from "@/components/research-navigation";
import Link from "next/link";
import { usePathname } from "next/navigation";
export function CausalNavigation() {
  const { t } = useI18n();
  const path = usePathname();
  return (
    <>
      <ResearchNavigation />
      <nav className="method-navigation" aria-label={t("Causal methods")}>
        {[
          { href: "/causal", label: "Residual patching" },
          { href: "/causal/heads", label: "Head ablation" },
          { href: "/causal/heads/induction", label: "Induction probe" },
        ].map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            aria-current={path === href ? "page" : undefined}
            className={path === href ? "active" : ""}
          >
            {t(label)}
          </Link>
        ))}
      </nav>
    </>
  );
}
