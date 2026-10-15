"use client";

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
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
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
  const path = usePathname();
  const [open, setOpen] = useState(false);
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
  const current = navigation.find((item) => item.href === path);
  useEffect(() => {
    document.title =
      (current?.label ||
        (path === "/guide" ? "Field guide" : "Page not found")) +
      " · Circuit Lantern";
  }, [path, current]);

  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <button
        className="mobile-toggle icon-button"
        onClick={() => setOpen(!open)}
        aria-label={open ? "Close navigation" : "Open navigation"}
      >
        {open ? <X /> : <Menu />}
      </button>
      {open && (
        <div className="sidebar-backdrop" onClick={() => setOpen(false)} />
      )}
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <Link href="/" className="brand" onClick={() => setOpen(false)}>
          <span className="brand-mark">
            <GitBranch size={23} />
          </span>
          <span>
            Circuit<span className="brand-light">Lantern</span>
            <small>MECHANISTIC INTERPRETABILITY</small>
          </span>
        </Link>
        <div className="workspace-label">
          <span className="workspace-avatar">L</span>
          <div>
            Local workspace<small>Personal research environment</small>
          </div>
          <ChevronRight size={14} />
        </div>
        <div className="nav-label">WORKBENCH</div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={path === item.href ? "page" : undefined}
              onClick={() => setOpen(false)}
              className={`nav-item ${path === item.href ? "active" : ""}`}
            >
              <item.icon size={18} />
              {item.label}
              {path === item.href && <span className="active-dot" />}
            </Link>
          ))}
        </nav>
        <div className="sidebar-note">
          <div className="orbit-mark">
            <Layers3 size={28} />
          </div>
          <h3>Understand what’s inside.</h3>
          <p>
            From individual weights to the circuits that shape a prediction.
          </p>
          <Link href="/guide" onClick={() => setOpen(false)}>
            Explore the field guide <ArrowUpRight size={14} />
          </Link>
        </div>
        <div className="sidebar-bottom">
          <a
            href="https://github.com/IlyaKhalafi/CircuitLantern"
            target="_blank"
            rel="noreferrer"
          >
            <GitBranch size={16} /> GitHub repository <ArrowUpRight size={14} />
          </a>
          <Link href="/guide">
            <BookOpen size={16} /> Documentation <ChevronRight size={14} />
          </Link>
          <div className="version">
            <span className="status-dot" /> Circuit Lantern <span>v0.1.0</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <div className="topbar">
          <div className="breadcrumb">
            <span>Workbench</span>
            <ChevronRight size={13} />
            <strong>
              {current?.label ||
                (path === "/guide" ? "Field guide" : "Page not found")}
            </strong>
          </div>
          <div className="runtime-status">
            <span className={`status-dot ${online ? "" : "offline"}`} />
            <span role="status">
              {online === null
                ? "Connecting to backend"
                : online
                  ? "Backend connected"
                  : "Backend offline"}
            </span>
            <span className="topbar-divider" />
            <span className="mono">
              {runtime?.loaded
                ? runtime.metadata?.id.split("/").pop()
                : "No model loaded"}
            </span>
          </div>
        </div>
        <main id="main-content" tabIndex={-1}>
          {children}
        </main>
        <footer className="main-footer">
          <span>Built for the questions behind the predictions.</span>
          <span>
            <span className="status-dot" /> Local research workspace
          </span>
        </footer>
      </div>
    </div>
  );
}
