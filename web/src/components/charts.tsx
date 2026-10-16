"use client";
import { useTheme } from "@/components/theme-provider";
import { useI18n } from "@/i18n/provider";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { number } from "@/lib/api";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
function hasIsolatedValue(
  data: Record<string, number | string | null>[],
  key: string,
) {
  const known = (index: number) => {
    const value = data[index]?.[key];
    return typeof value === "number" && Number.isFinite(value);
  };
  return data.some(
    (_, index) => known(index) && !known(index - 1) && !known(index + 1),
  );
}
export function SeriesChart({
  data,
  series,
  area = false,
  height = 235,
  xKey = "index",
  yAxis,
}: {
  data: Record<string, number | string | null>[];
  series: {
    key: string;
    label: string;
    color: string;
  }[];
  area?: boolean;
  height?: number;
  xKey?: string;
  yAxis?: {
    scale?: "linear" | "log";
    reversed?: boolean;
    domain?: [number, number];
    integer?: boolean;
    ticks?: number[];
  };
}) {
  const { t, localize } = useI18n();
  const Chart = area ? AreaChart : LineChart;
  return (
    <div className="chart" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <Chart data={data} margin={{ top: 5, right: 12, left: -12, bottom: 5 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 4" />
          <XAxis
            dataKey={xKey}
            tickLine={false}
            axisLine={false}
            minTickGap={25}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            scale={yAxis?.scale}
            reversed={yAxis?.reversed}
            domain={yAxis?.domain}
            allowDecimals={!yAxis?.integer}
            ticks={yAxis?.ticks}
            interval={yAxis?.integer ? "preserveStartEnd" : undefined}
            tickFormatter={(v) =>
              typeof v === "number"
                ? yAxis?.integer
                  ? v.toLocaleString("en")
                  : v.toPrecision(2)
                : v
            }
          />
          {series.length > 1 && (
            <Legend
              iconType="line"
              formatter={(value) => (
                <span style={{ color: "var(--muted)" }}>{localize(value)}</span>
              )}
            />
          )}
          <Tooltip
            itemStyle={{ color: "var(--ink)" }}
            formatter={(value) =>
              typeof value === "number"
                ? yAxis?.integer
                  ? value.toLocaleString("en")
                  : value.toPrecision(5)
                : String(value)
            }
          />
          {series.map((item) =>
            area ? (
              <Area
                isAnimationActive={false}
                key={item.key}
                type="monotone"
                dataKey={item.key}
                name={t(item.label)}
                stroke={item.color}
                fill={item.color}
                fillOpacity={0.09}
                dot={hasIsolatedValue(data, item.key) ? { r: 3 } : false}
                strokeWidth={1.7}
              />
            ) : (
              <Line
                isAnimationActive={false}
                key={item.key}
                type="monotone"
                dataKey={item.key}
                name={t(item.label)}
                stroke={item.color}
                dot={hasIsolatedValue(data, item.key) ? { r: 3 } : false}
                strokeWidth={1.7}
              />
            ),
          )}
        </Chart>
      </ResponsiveContainer>
    </div>
  );
}
export function Histogram({
  data,
}: {
  data: {
    value: number;
    count: number;
  }[];
}) {
  const { t } = useI18n();
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 5, right: 10, left: -10, bottom: 5 }}
        >
          <CartesianGrid vertical={false} strokeDasharray="3 4" />
          <XAxis
            dataKey="value"
            tickFormatter={(v) => number(v, 2)}
            tickLine={false}
            axisLine={false}
            minTickGap={30}
          />
          <YAxis tickLine={false} axisLine={false} />
          <Tooltip
            itemStyle={{ color: "var(--ink)" }}
            labelFormatter={(v) =>
              t("Weight {value}", { value: Number(v).toPrecision(4) })
            }
          />
          <Bar
            isAnimationActive={false}
            dataKey="count"
            name={t("Elements")}
            fill="var(--blue)"
            radius={[2, 2, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
function axisLabel(value: string | undefined, fallback: string) {
  if (value === undefined) return fallback;
  if (value === "") return "∅";
  return value.trim() ? value : JSON.stringify(value);
}
export function Heatmap({
  data,
  sequential = false,
  label,
  xLabels,
  yLabels,
  maxAbs,
  logarithmic = false,
  height,
}: {
  data: (number | null)[][];
  sequential?: boolean;
  label: string;
  xLabels?: string[];
  yLabels?: string[];
  maxAbs?: number;
  logarithmic?: boolean;
  height?: number;
}) {
  const { t } = useI18n();
  const { theme } = useTheme();
  const canvas = useRef<HTMLCanvasElement>(null);
  const maximum = useMemo(
    () =>
      maxAbs ??
      data.reduce<number>(
        (maximum, row) =>
          row.reduce<number>(
            (maximum, value) =>
              value === null ? maximum : Math.max(maximum, Math.abs(value)),
            maximum,
          ),
        0,
      ),
    [data, maxAbs],
  );
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    if (!data[0]?.length) {
      context.clearRect(0, 0, context.canvas.width, context.canvas.height);
      return;
    }
    const palette = getComputedStyle(context.canvas);
    const rgb = (name: string) => {
      const hex = palette.getPropertyValue(name).trim();
      return [1, 3, 5].map((offset) =>
        parseInt(hex.slice(offset, offset + 2), 16),
      );
    };
    const low = rgb("--heatmap-zero");
    const positive = rgb("--heatmap-positive");
    const negative = rgb("--heatmap-negative");
    const missing = palette.getPropertyValue("--heatmap-missing").trim();
    const rows = data.length,
      columns = data[0].length;
    const cellWidth = 8,
      cellHeight = 8;
    context.canvas.width = columns * cellWidth;
    context.canvas.height = rows * cellHeight;
    data.forEach((row, y) =>
      row.forEach((value, x) => {
        if (value === null) {
          context.fillStyle = missing;
          context.fillRect(
            x * cellWidth,
            y * cellHeight,
            cellWidth,
            cellHeight,
          );
          return;
        }
        const magnitude = Math.abs(value);
        const strength =
          maximum > 0
            ? Math.min(
                1,
                logarithmic
                  ? Math.log1p(magnitude) / Math.log1p(maximum)
                  : magnitude / maximum,
              )
            : 0;
        const high = value >= 0 ? positive : negative;
        context.fillStyle = `rgb(${low.map((v, i) => Math.round(v + (high[i] - v) * strength)).join(",")})`;
        context.fillRect(
          x * cellWidth,
          y * cellHeight,
          cellWidth - (sequential ? 0.5 : 0),
          cellHeight - (sequential ? 0.5 : 0),
        );
      }),
    );
  }, [data, maximum, sequential, logarithmic, theme]);
  const [cell, setHover] = useState<{ x: number; y: number } | null>(null);
  const [keyboard, setKeyboard] = useState(false);
  const helpId = useId();
  const rows = data.length;
  const columns = data[0]?.length || 0;
  const hover =
    cell && rows && columns
      ? { x: Math.min(cell.x, columns - 1), y: Math.min(cell.y, rows - 1) }
      : null;
  function inspectAt(
    clientX: number,
    clientY: number,
    target: HTMLCanvasElement,
  ) {
    if (!rows || !columns) return;
    const bounds = target.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    setKeyboard(false);
    const next = {
      x: Math.max(
        0,
        Math.min(
          columns - 1,
          Math.floor(((clientX - bounds.left) / bounds.width) * columns),
        ),
      ),
      y: Math.max(
        0,
        Math.min(
          rows - 1,
          Math.floor(((clientY - bounds.top) / bounds.height) * rows),
        ),
      ),
    };
    setHover((current) =>
      current?.x === next.x && current.y === next.y ? current : next,
    );
  }
  const hovered = hover && data[hover.y]?.[hover.x];
  const hoverText =
    hover && hovered !== undefined
      ? `${axisLabel(yLabels?.[hover.y], t("Row {v0}", { v0: hover.y }))} → ${axisLabel(xLabels?.[hover.x], t("Column {v0}", { v0: hover.x }))}  ${hovered?.toPrecision(4) ?? t("Unavailable")}`
      : "";
  return (
    <div className="heatmap-wrapper">
      <div className="heatmap-surface">
        <canvas
          ref={canvas}
          className="heatmap"
          style={height === undefined ? undefined : { height }}
          role="img"
          aria-label={t(label)}
          aria-describedby={helpId}
          aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight Home End"
          tabIndex={columns && rows ? 0 : -1}
          onFocus={() => {
            if (columns && rows) {
              setKeyboard(true);
              setHover((current) => current || { x: 0, y: 0 });
            }
          }}
          onBlur={() => {
            setHover(null);
            setKeyboard(false);
          }}
          onKeyDown={(event) => {
            if (!columns || !rows) return;
            const next = { ...(hover || { x: 0, y: 0 }) };
            switch (event.key) {
              case "ArrowLeft":
                next.x--;
                break;
              case "ArrowRight":
                next.x++;
                break;
              case "ArrowUp":
                next.y--;
                break;
              case "ArrowDown":
                next.y++;
                break;
              case "Home":
                next.x = 0;
                if (event.ctrlKey || event.metaKey) next.y = 0;
                break;
              case "End":
                next.x = columns - 1;
                if (event.ctrlKey || event.metaKey) next.y = rows - 1;
                break;
              case "Escape":
                event.currentTarget.blur();
                return;
              default:
                return;
            }
            event.preventDefault();
            setKeyboard(true);
            setHover({
              x: Math.max(0, Math.min(columns - 1, next.x)),
              y: Math.max(0, Math.min(rows - 1, next.y)),
            });
          }}
          onPointerDown={(event) => {
            event.currentTarget.focus({ preventScroll: true });
            inspectAt(event.clientX, event.clientY, event.currentTarget);
          }}
          onPointerMove={(event) => {
            if (event.pointerType !== "touch")
              inspectAt(event.clientX, event.clientY, event.currentTarget);
          }}
          onPointerLeave={() => {
            if (document.activeElement !== canvas.current) setHover(null);
          }}
        />
        {hover && hovered !== undefined && columns > 0 && rows > 0 && (
          <span
            className="heatmap-cursor"
            aria-hidden="true"
            style={{
              left: (hover.x / columns) * 100 + "%",
              top: (hover.y / rows) * 100 + "%",
              width: 100 / columns + "%",
              height: 100 / rows + "%",
            }}
          />
        )}
      </div>
      <div
        className={`heatmap-key ${logarithmic && sequential ? "logarithmic" : ""}`}
      >
        <span>
          {sequential || maximum === 0 ? "0" : `−${maximum.toPrecision(2)}`}
        </span>
        <div
          style={
            sequential
              ? {
                  background:
                    "linear-gradient(90deg,var(--heatmap-zero),var(--heatmap-positive))",
                }
              : undefined
          }
        >
          {logarithmic && sequential && (
            <span className="heatmap-midpoint">
              {t(number(Math.expm1(Math.log1p(maximum) / 2), 3))}
            </span>
          )}
        </div>
        <span>{maximum.toPrecision(2)}</span>
      </div>
      <p role="status" aria-live={keyboard ? "polite" : "off"}>
        {hoverText || t(label)}
      </p>
      <small className="heatmap-help" id={helpId}>
        {t("Use arrow keys to inspect cells. Tap a cell on touch screens.")}
      </small>
    </div>
  );
}
