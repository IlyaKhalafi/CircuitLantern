"use client";
import { useI18n } from "@/i18n/provider";
import { AlertCircle, ArrowUpRight, LoaderCircle } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";
import { HelpTooltip } from "@/components/help-tooltip";
export function PageHeader({
  title,
  description,
  help,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  help?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <div className="heading-with-help">
          <h1 dir="auto">{title}</h1>
          {help && <HelpTooltip label={title}>{help}</HelpTooltip>}
        </div>
        {description && <p>{description}</p>}
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
  const parts = (
    typeof message === "string" ? [message] : message || []
  ).filter((part) => part.trim());
  return parts.length ? (
    <div role="alert" className="notice error">
      <AlertCircle size={17} />
      <span>{parts.map((part) => t(part)).join(" ")}</span>
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
  help,
}: {
  label: string;
  value: ReactNode;
  note?: string;
  help?: string;
}) {
  const { t } = useI18n();
  return (
    <div className="metric">
      <span className="metric-label">
        <span>{t(label)}</span>
        {help && <HelpTooltip label={label}>{help}</HelpTooltip>}
      </span>
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
  id,
  title,
  subtitle,
  help,
  children,
  action,
  className = "",
  translateTitle = true,
}: {
  id?: string;
  title: string;
  subtitle?: string;
  help?: string;
  children: ReactNode;
  action?: ReactNode;
  className?: string;
  translateTitle?: boolean;
}) {
  const { t, localize } = useI18n();
  return (
    <section id={id} className={`panel ${className}`}>
      <div className="panel-header">
        <div>
          <div className="heading-with-help">
            <h2>{translateTitle ? t(title) : title}</h2>
            {help && (
              <HelpTooltip label={title} translateLabel={translateTitle}>
                {help}
              </HelpTooltip>
            )}
          </div>
          {subtitle && <p>{t(subtitle)}</p>}
        </div>
        {localize(action)}
      </div>
      {children}
    </section>
  );
}
