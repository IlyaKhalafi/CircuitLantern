"use client";

import {
  autoUpdate,
  flip,
  FloatingPortal,
  offset,
  safePolygon,
  shift,
  useClick,
  useDismiss,
  useFloating,
  useFocus,
  useHover,
  useInteractions,
  useRole,
} from "@floating-ui/react";
import { CircleHelp } from "lucide-react";
import { usePathname } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import { useI18n } from "@/i18n/provider";
import "./help-tooltip.css";

const HelpContext = createContext<{
  active: string | null;
  setActive: Dispatch<SetStateAction<string | null>>;
} | null>(null);

export function HelpProvider({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<string | null>(null);
  const path = usePathname();
  const { locale } = useI18n();
  useEffect(() => setActive(null), [path, locale]);
  return (
    <HelpContext.Provider value={{ active, setActive }}>
      {children}
    </HelpContext.Provider>
  );
}

/** Plain text help, available by hover, keyboard focus or a deliberate tap. */
export function HelpTooltip({
  label,
  children,
  translateLabel = true,
}: {
  translateLabel?: boolean;
  label: string;
  children: string;
}) {
  const { t } = useI18n();
  const help = useContext(HelpContext);
  if (!help) throw new Error("HelpTooltip requires HelpProvider");
  const id = useId();
  const descriptionId = id + "-description";
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);
  const open = help.active === id;
  const { refs, floatingStyles, context } = useFloating({
    open,
    onOpenChange: (next, event, reason) => {
      if (reason === "escape-key") event?.preventDefault();
      help.setActive((current) =>
        next ? id : current === id ? null : current,
      );
    },
    placement: "bottom-start",
    strategy: "fixed",
    middleware: [offset(8), flip({ padding: 12 }), shift({ padding: 12 })],
    whileElementsMounted: autoUpdate,
  });
  const reference = refs.setReference;
  const setReference = useCallback(
    (node: HTMLButtonElement | null) => {
      reference(node);
      // A body portal would be inert and behind a native modal's top layer.
      setPortalRoot(node?.closest("dialog") ?? null);
    },
    [reference],
  );
  const hover = useHover(context, {
    mouseOnly: true,
    move: false,
    delay: { open: 250, close: 100 },
    handleClose: safePolygon(),
  });
  const focus = useFocus(context);
  const click = useClick(context);
  const dismiss = useDismiss(context, { ancestorScroll: true });
  const role = useRole(context, { role: "tooltip" });
  const { getReferenceProps, getFloatingProps } = useInteractions([
    hover,
    focus,
    click,
    dismiss,
    role,
  ]);
  return (
    <>
      <button
        ref={setReference}
        type="button"
        className="help-trigger"
        aria-label={t("About {label}", {
          label: translateLabel ? t(label) : label,
        })}
        aria-expanded={open}
        {...getReferenceProps()}
        aria-describedby={descriptionId}
      >
        <CircleHelp size={15} aria-hidden="true" />
      </button>
      <span className="help-description" id={descriptionId} hidden>
        {t(children)}
      </span>
      {open && (
        <FloatingPortal root={portalRoot ?? undefined}>
          <div
            ref={refs.setFloating}
            className="help-tooltip"
            style={floatingStyles}
            {...getFloatingProps()}
          >
            {t(children)}
          </div>
        </FloatingPortal>
      )}
    </>
  );
}

export function FieldLabel({
  id,
  label,
  help,
}: {
  id: string;
  label: string;
  help: string;
}) {
  const { t } = useI18n();
  return (
    <div className="control-label">
      <label htmlFor={id}>{t(label)}</label>
      <HelpTooltip label={label}>{help}</HelpTooltip>
    </div>
  );
}
