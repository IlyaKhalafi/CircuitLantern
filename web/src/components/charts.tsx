"use client";
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
  const canvas = useRef<HTMLCanvasElement>(null);
  const maximum =
    maxAbs ??
    Math.max(
      ...data
        .flat()
        .filter((value): value is number => value !== null)
        .map(Math.abs),
      0,
    );
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context || !data.length || !data[0].length) return;
    const rows = data.length,
      columns = data[0].length;
    const cellWidth = 8,
      cellHeight = 8;
    context.canvas.width = columns * cellWidth;
    context.canvas.height = rows * cellHeight;
    data.forEach((row, y) =>
      row.forEach((value, x) => {
        if (value === null) {
          context.fillStyle = "#283954";
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
        const low = [16, 27, 48];
        const high = value >= 0 ? [255, 195, 84] : [131, 171, 244];
        context.fillStyle = `rgb(${low.map((v, i) => Math.round(v + (high[i] - v) * strength)).join(",")})`;
        context.fillRect(
          x * cellWidth,
          y * cellHeight,
          cellWidth - (sequential ? 0.5 : 0),
          cellHeight - (sequential ? 0.5 : 0),
        );
      }),
    );
  }, [data, maximum, sequential, logarithmic]);
  const [hover, setHover] = useState("");
  return (
    <div className="heatmap-wrapper">
      <canvas
        ref={canvas}
        className="heatmap"
        style={height === undefined ? undefined : { height }}
        role="img"
        aria-label={t(label)}
        onMouseMove={(event) => {
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
          setHover(
            `${yLabels?.[y] || `Row ${y}`} → ${xLabels?.[x] || `Column ${x}`}  ${data[y][x]?.toPrecision(4) ?? "Undefined"}`,
          );
        }}
        onMouseLeave={() => setHover("")}
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
              ? { background: "linear-gradient(90deg,#101b30,#ffc354)" }
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
      <p>{hover ? t(hover) : t(label)}</p>
    </div>
  );
}
