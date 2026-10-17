"use client";
import { useI18n } from "@/i18n/provider";
import { AlertCircle, ArrowUpRight, LoaderCircle } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";
export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1 dir="auto">{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="header-actions">{actions}</div>}
    </header>
  );
}
export function ErrorNotice({
  message,
}: {
  message: string | readonly string[] | null;
}) {
  const { t } = useI18n();
  return message ? (
    <div role="alert" className="notice error">
      <AlertCircle size={17} />
      <span>
        {(typeof message === "string" ? [message] : message)
          .map((part) => t(part))
          .join(" ")}
      </span>
    </div>
  ) : null;
}
export function Loading({ label = "Loading" }: { label?: string }) {
  const { t } = useI18n();
  return (
    <div className="loading" role="status">
      <LoaderCircle size={20} className="spin" />
      <span>{t(label)}</span>
    </div>
  );
}
export function EmptyState({
  icon,
  title,
  children,
  href,
  action,
}: {
  icon: ReactNode;
  title: string;
  children: ReactNode;
  href?: string;
  action?: string;
}) {
  const { t, localize } = useI18n();
  return (
    <div className="empty-state">
      <div className="empty-icon">{icon}</div>
      <h2>{t(title)}</h2>
      <p>{localize(children)}</p>
      {href && (
        <Link className="button primary" href={href}>
          {localize(action)}
          <ArrowUpRight size={16} />
        </Link>
      )}
    </div>
  );
}
export function Metric({
  label,
  value,
  note,
}: {
  label: string;
  value: ReactNode;
  note?: string;
}) {
  const { t } = useI18n();
  return (
    <div className="metric">
      <span>{t(label)}</span>
      <strong>
        {typeof value === "number" ||
        (typeof value === "string" && /^[+−-]?\d/.test(value)) ? (
          <bdi dir="ltr">{value}</bdi>
        ) : (
          value
        )}
      </strong>
      {note && <small>{t(note)}</small>}
    </div>
  );
}
export function Panel({
  title,
  subtitle,
  children,
  action,
  className = "",
  translateTitle = true,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  action?: ReactNode;
  className?: string;
  translateTitle?: boolean;
}) {
  const { t, localize } = useI18n();
  return (
    <section className={`panel ${className}`}>
      <div className="panel-header">
        <div>
          <h2>{translateTitle ? t(title) : title}</h2>
          {subtitle && <p>{t(subtitle)}</p>}
        </div>
        {localize(action)}
      </div>
      {children}
    </section>
  );
}
