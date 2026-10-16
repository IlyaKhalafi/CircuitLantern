"use client";
import { useI18n } from "@/i18n/provider";
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  Box,
  ChevronRight,
  Cpu,
  FlaskConical,
  GitBranch,
  Layers3,
  Menu,
  Network,
  X,
} from "lucide-react";
import Link from "next/link";
import Image from "next/image";
import { LanguageSelect } from "@/components/language-select";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { RuntimeState } from "@/lib/types";
const navigation = [
  { href: "/", label: "Model library", icon: Box },
  { href: "/workspace", label: "Model explorer", icon: Network },
  { href: "/playground", label: "Activation lab", icon: Activity },
  { href: "/causal", label: "Causal tracing", icon: GitBranch },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/hardware", label: "Server & runtime", icon: Cpu },
];
export function Shell({ children }: { children: React.ReactNode }) {
  const { t } = useI18n();
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const [mobile, setMobile] = useState(false);
  const toggleRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const desktop = window.matchMedia("(min-width: 721px)");
    const closeOnDesktop = () => {
      setMobile(!desktop.matches);
      if (desktop.matches) setOpen(false);
    };
    closeOnDesktop();
    desktop.addEventListener("change", closeOnDesktop);
    return () => desktop.removeEventListener("change", closeOnDesktop);
  }, []);
  useEffect(() => {
    if (!open) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        toggleRef.current?.focus();
      }
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [open]);
  useEffect(() => {
    if (!mobile || !open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [mobile, open]);
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => {
    let active = true;
    const refresh = () =>
      api<RuntimeState>("/runtime")
        .then((data) => {
          if (active) {
            setRuntime(data);
            setOnline(true);
          }
        })
        .catch(() => {
          if (active) setOnline(false);
        });
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [path]);
  const activeRoute = (href: string) =>
    path === href || (href !== "/" && path.startsWith(href + "/"));
  const current = navigation.find((item) => activeRoute(item.href));
  const currentLabel =
    path === "/causal/heads" ? "Head ablation" : current?.label;
  useEffect(() => {
    document.title =
      t(
        currentLabel || (path === "/guide" ? "Field guide" : "Page not found"),
      ) + " · Circuit Lantern";
  }, [currentLabel, path, t]);
  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        {t("Skip to content")}
      </a>
      <button
        ref={toggleRef}
        className={
          "mobile-toggle icon-button" + (open ? " navigation-open" : "")
        }
        aria-expanded={open}
        aria-controls="workbench-navigation"
        onClick={() => setOpen(!open)}
        aria-label={open ? t("Close navigation") : t("Open navigation")}
      >
        {open ? <X /> : <Menu />}
      </button>
      {open && (
        <div className="sidebar-backdrop" onClick={() => setOpen(false)} />
      )}
      <aside
        inert={mobile && !open ? true : undefined}
        id="workbench-navigation"
        className={`sidebar ${open ? "open" : ""}`}
      >
        <Link href="/" className="brand" onClick={() => setOpen(false)}>
          <span className="brand-mark">
            <Image
              src="/brand/lantern.webp"
              alt=""
              width={52}
              height={52}
              preload
            />
          </span>
          <span>
            Circuit<span className="brand-light">Lantern</span>
            <small>{t("MECHANISTIC INTERPRETABILITY")}</small>
          </span>
        </Link>
        <div className="workspace-label">
          <span className="workspace-avatar">L</span>
          <div>
            {t("Local workspace")}
            <small>{t("Personal research environment")}</small>
          </div>
          <ChevronRight size={14} />
        </div>
        <div className="nav-label">{t("WORKBENCH")}</div>
        <nav aria-label={t("Main navigation")}>
          {navigation.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={
                path === item.href
                  ? "page"
                  : activeRoute(item.href)
                    ? "true"
                    : undefined
              }
              onClick={() => setOpen(false)}
              className={`nav-item ${activeRoute(item.href) ? "active" : ""}`}
            >
              <item.icon size={18} />
              {t(item.label)}
              {activeRoute(item.href) && <span className="active-dot" />}
            </Link>
          ))}
        </nav>
        <div className="sidebar-note">
          <div className="orbit-mark">
            <Layers3 size={28} />
          </div>
          <h3>{t("Understand what\u2019s inside.")}</h3>
          <p>
            {t(
              "From individual weights to the circuits that shape a prediction.",
            )}
          </p>
          <Link href="/guide" onClick={() => setOpen(false)}>
            {t("Explore the field guide")}
            <ArrowUpRight size={14} />
          </Link>
        </div>
        <div className="sidebar-bottom">
          <a
            href="https://github.com/IlyaKhalafi/CircuitLantern"
            target="_blank"
            rel="noreferrer"
          >
            <GitBranch size={16} />
            {t("GitHub repository")}
            <ArrowUpRight size={14} />
          </a>
          <Link href="/guide" onClick={() => setOpen(false)}>
            <BookOpen size={16} />
            {t("Documentation")}
            <ChevronRight size={14} />
          </Link>
          <div className="version">
            <span className="status-dot" />
            {t("Circuit Lantern")}
            <span>v0.1.0</span>
          </div>
        </div>
      </aside>
      <div className="main-shell" inert={mobile && open ? true : undefined}>
        <div className="topbar">
          <div className="breadcrumb">
            <span>{t("Workbench")}</span>
            <ChevronRight size={13} />
            <strong>
              {t(
                currentLabel ||
                  (path === "/guide" ? "Field guide" : "Page not found"),
              )}
            </strong>
          </div>
          <div className="topbar-tools">
            <LanguageSelect />
            <div className="runtime-status">
              <span className={`status-dot ${online ? "" : "offline"}`} />
              <span role="status">
                {online === null
                  ? t("Connecting to backend")
                  : online
                    ? t("Backend connected")
                    : t("Backend offline")}
              </span>
              <span className="topbar-divider" />
              <span className="mono">
                {runtime?.loaded
                  ? runtime.metadata?.id.split("/").pop()
                  : t("No model loaded")}
              </span>
            </div>
          </div>
        </div>
        <main id="main-content" tabIndex={-1}>
          {children}
        </main>
        <footer className="main-footer">
          <span>{t("Built for the questions behind the predictions.")}</span>
          <span>
            <span className="status-dot" />
            {t("Local research workspace")}
          </span>
        </footer>
      </div>
    </div>
  );
}
