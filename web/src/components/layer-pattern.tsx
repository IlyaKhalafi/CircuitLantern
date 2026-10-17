"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useRef } from "react";
import { useI18n } from "@/i18n/provider";
import type { BlockFlow } from "@/lib/types";

export function LayerPattern({
  flows,
  selected,
  onSelect,
}: {
  flows: BlockFlow[];
  selected: number;
  onSelect: (layer: number) => void;
}) {
  const { t, locale } = useI18n();
  const strip = useRef<HTMLDivElement>(null);
  const buttons = useRef(new Map<number, HTMLButtonElement>());
  const index = flows.findIndex((flow) => flow.layer === selected);
  const layerOrder = flows.map((flow) => flow.layer).join(",");
  useEffect(() => {
    const container = strip.current;
    const active = buttons.current.get(selected);
    if (!container || !active) return;
    function revealActive() {
      const outer = container!.getBoundingClientRect(),
        inner = active!.getBoundingClientRect();
      const shift =
        inner.left < outer.left + 4
          ? inner.left - outer.left - 4
          : inner.right > outer.right - 4
            ? inner.right - outer.right + 4
            : 0;
      // Scroll only this strip. Changing layers must not move the whole page.
      if (shift) container!.scrollBy({ left: shift, behavior: "instant" });
    }
    revealActive();
    const observer = new ResizeObserver(revealActive);
    observer.observe(container);
    return () => observer.disconnect();
  }, [selected, layerOrder, locale]);
  function selectAt(next: number, focus = false) {
    const flow = flows[next];
    if (!flow) return;
    if (flow.layer !== selected) onSelect(flow.layer);
    if (focus) buttons.current.get(flow.layer)?.focus({ preventScroll: true });
  }
  return (
    <div
      className="flow-pattern-navigation"
      role="group"
      aria-label={t("Layer pattern navigation")}
    >
      <button
        className="flow-pattern-step"
        aria-label={t("Previous block")}
        disabled={index <= 0}
        onClick={() => selectAt(index - 1, index === 1)}
      >
        <ChevronLeft size={16} />
      </button>
      <div
        ref={strip}
        className="flow-layer-pattern"
        role="group"
        aria-label={t("Layer pattern")}
      >
        {flows.map((flow, position) => (
          <button
            key={flow.layer}
            ref={(button) => {
              if (button) buttons.current.set(flow.layer, button);
              else buttons.current.delete(flow.layer);
            }}
            className={
              (flow.layer === selected ? "active " : "") +
              (flow.mixer_type === "linear_attention"
                ? "linear"
                : flow.mixer_type === "full_attention"
                  ? "full"
                  : "neutral")
            }
            tabIndex={
              flow.layer === selected || (index < 0 && position === 0) ? 0 : -1
            }
            aria-pressed={flow.layer === selected}
            aria-label={t("Block {layer} · {kind}", {
              layer: flow.layer,
              kind: t(
                flow.mixer_type === "linear_attention"
                  ? "Gated DeltaNet"
                  : flow.mixer_type === "full_attention"
                    ? "Full attention"
                    : "Other mixer",
              ),
            })}
            onClick={() => selectAt(position)}
            onKeyDown={(event) => {
              if (
                event.altKey ||
                event.ctrlKey ||
                event.metaKey ||
                event.shiftKey
              )
                return;
              const rtl = document.documentElement.dir === "rtl";
              let next: number | undefined;
              if (event.key === "Home") next = 0;
              if (event.key === "End") next = flows.length - 1;
              if (event.key === "ArrowLeft")
                next = Math.max(
                  0,
                  Math.min(flows.length - 1, position + (rtl ? 1 : -1)),
                );
              if (event.key === "ArrowRight")
                next = Math.max(
                  0,
                  Math.min(flows.length - 1, position + (rtl ? -1 : 1)),
                );
              if (next !== undefined) {
                event.preventDefault();
                selectAt(next, true);
              }
            }}
          >
            {flow.layer}
            <small>
              {t(
                flow.mixer_type === "linear_attention"
                  ? "Linear"
                  : flow.mixer_type === "full_attention"
                    ? "Full"
                    : "Other",
              )}
            </small>
          </button>
        ))}
      </div>
      <button
        className="flow-pattern-step"
        aria-label={t("Next block")}
        disabled={index < 0 || index === flows.length - 1}
        onClick={() => selectAt(index + 1, index === flows.length - 2)}
      >
        <ChevronRight size={16} />
      </button>
    </div>
  );
}
