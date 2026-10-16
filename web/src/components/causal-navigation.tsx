"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function CausalNavigation() {
  const path = usePathname();
  return (
    <nav className="method-navigation" aria-label="Causal methods">
      {[
        { href: "/causal", label: "Residual patching" },
        { href: "/causal/heads", label: "Head ablation" },
      ].map(({ href, label }) => (
        <Link
          key={href}
          href={href}
          aria-current={path === href ? "page" : undefined}
          className={path === href ? "active" : ""}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
