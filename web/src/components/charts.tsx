"use client";
import { useTheme } from "@/components/theme-provider";
import { useI18n } from "@/i18n/provider";
import { useEffect, useRef, useState } from "react";
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
  const maximum =
    maxAbs ??
    data.reduce<number>(
      (maximum, row) =>
        row.reduce<number>(
          (maximum, value) =>
            value === null ? maximum : Math.max(maximum, Math.abs(value)),
          maximum,
        ),
      0,
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
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const hovered = hover && data[hover.y]?.[hover.x];
  const hoverText =
    hover && hovered !== undefined
      ? `${axisLabel(yLabels?.[hover.y], t("Row {v0}", { v0: hover.y }))} → ${axisLabel(xLabels?.[hover.x], t("Column {v0}", { v0: hover.x }))}  ${hovered?.toPrecision(4) ?? t("Unavailable")}`
      : "";
  return (
    <div className="heatmap-wrapper">
      <canvas
        ref={canvas}
        className="heatmap"
        style={height === undefined ? undefined : { height }}
        role="img"
        aria-label={t(label)}
        onMouseMove={(event) => {
          if (!data[0]?.length) return;
          const bounds = event.currentTarget.getBoundingClientRect();
          const x = Math.min(
            data[0].length - 1,
            Math.floor(
              ((event.clientX - bounds.left) / bounds.width) * data[0].length,
            ),
          );
          const y = Math.min(
            data.length - 1,
            Math.floor(
              ((event.clientY - bounds.top) / bounds.height) * data.length,
            ),
          );
          setHover({ x, y });
        }}
        onMouseLeave={() => setHover(null)}
      />
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
      <p>{hoverText || t(label)}</p>
    </div>
  );
}
