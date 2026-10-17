"use client";
import { useI18n } from "@/i18n/provider";
import { Check, Cpu, Info, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import {
  ErrorNotice,
  Loading,
  Metric,
  PageHeader,
  Panel,
} from "@/components/ui";
import { api, bytes } from "@/lib/api";
import { executionLabel } from "@/lib/execution";
import { executionPath } from "@/lib/runtime";
import type { Hardware, RuntimeState } from "@/lib/types";
export default function HardwarePage() {
  const { t, rich } = useI18n();
  const [hardware, setHardware] = useState<Hardware | null>(null);
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  async function refresh() {
    setBusy(true);
    setError(null);
    const results = await Promise.allSettled([
      api<Hardware>("/hardware"),
      api<RuntimeState>("/runtime"),
    ]);
    if (results[0].status === "fulfilled") setHardware(results[0].value);
    else setError(results[0].reason.message);
    if (results[1].status === "fulfilled") setRuntime(results[1].value);
    setBusy(false);
  }
  useEffect(() => {
    refresh();
  }, []);
  async function unload() {
    setBusy(true);
    setError(null);
    try {
      await api(executionPath("/runtime", runtime?.metadata?.execution_id), {
        method: "DELETE",
      });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  const capability =
    (hardware?.execution_gpu || hardware?.gpus[0])?.compute_capability || 0;
  return (
    <>
      <PageHeader
        eyebrow={t("KNOW YOUR ENVIRONMENT")}
        title={t("Server & runtime")}
        help={t(
          "Live resource availability and the runtime behind every compatibility check.",
        )}
        actions={
          <button
            className="button secondary"
            onClick={refresh}
            disabled={busy}
          >
            <RefreshCw size={14} className={busy ? "spin" : ""} />
            {t("Refresh snapshot")}
          </button>
        }
      />
      <ErrorNotice message={error} />
      {!hardware ? (
        busy && <Loading label={t("Reading server hardware")} />
      ) : (
        <>
          <div className="metrics-grid">
            <Metric
              label={t("Physical CPU cores")}
              value={hardware.cpu}
              note={t("{v0} logical threads", {
                v0: hardware.cpu_threads,
              })}
            />
            <Metric
              label={t("Available system memory")}
              value={bytes(hardware.ram_available_bytes)}
              note={t("{value} {scope}", {
                value: bytes(hardware.ram_total_bytes),
                scope: t(
                  hardware.ram_scope === "cgroup"
                    ? "process memory capacity"
                    : "total RAM",
                ),
              })}
            />
            <Metric
              label={t("Free checkpoint storage")}
              value={bytes(hardware.disk_free_bytes)}
              note={t("{v0} total capacity", {
                v0: bytes(hardware.disk_total_bytes),
              })}
            />
            <Metric
              label={t("Detected NVIDIA GPUs")}
              value={hardware.gpus.length}
              note={
                hardware.cuda_available
                  ? t("CUDA runtime available")
                  : t("CPU analysis runtime")
              }
            />
          </div>
          <div className="two-column">
            <Panel
              title={t("GPU inventory")}
              subtitle={t(
                "Execution uses the first GPU visible to PyTorch, with a 15 percent memory reserve.",
              )}
            >
              {hardware.gpus.length ? (
                hardware.gpus.map((gpu) => (
                  <div className="hardware-gpu" key={gpu.index}>
                    <div className="model-icon green">
                      <Cpu size={27} />
                    </div>
                    <div>
                      <h3>{gpu.name}</h3>
                      <p>
                        {rich("GPU {v0} \u00B7 SM {v1} \u00B7 Driver {v2}", {
                          v0: gpu.index,
                          v1: gpu.compute_capability.toFixed(1),
                          v2: gpu.driver,
                        })}
                      </p>
                      <div className="gpu-memory">
                        <div className="memory-bar">
                          <div
                            style={{
                              width: `${(1 - gpu.free_bytes / gpu.total_bytes) * 100}%`,
                              background: "var(--muted)",
                            }}
                          />
                        </div>
                        <div className="memory-legend">
                          <span>
                            {rich("{v0} in use", {
                              v0: bytes(gpu.total_bytes - gpu.free_bytes),
                            })}
                          </span>
                          <span>
                            {rich("{v0} free", {
                              v0: bytes(gpu.free_bytes),
                            })}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="panel-body">
                  <div className="notice info">
                    <Info size={16} />
                    {t(
                      "No NVIDIA GPU detected. Supported models and quantization formats can still run using system memory.",
                    )}
                  </div>
                </div>
              )}
              {hardware.gpus.length > 0 && !hardware.cuda_available && (
                <p className="inline-note">
                  {rich(
                    "The driver detects a GPU, but the Python runtime has no usable CUDA device. Check the PyTorch build and device visibility.{v0}",
                    {
                      v0: hardware.cuda_error && (
                        <span> {hardware.cuda_error}</span>
                      ),
                    },
                  )}
                </p>
              )}
            </Panel>
            <Panel
              title={t("Python runtime")}
              subtitle={t(
                "Installed versions reported by the backend environment.",
              )}
            >
              <div className="runtime-table">
                {Object.entries(hardware.packages).map(([name, version]) => (
                  <div key={name}>
                    <span>{name}</span>
                    <strong>{version || t("Not installed")}</strong>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
          <div className="two-column">
            <Panel
              title={t("Precision requirements")}
              subtitle={t(
                "Hardware eligibility and analysis support are separate checks.",
              )}
            >
              <div className="capability-list">
                {[
                  {
                    title: "FP32 / FP16 / BF16 execution",
                    text: "Choose execution format. Matrix values are inspected in float32.",
                    supported: true,
                    badge: "Supported",
                  },
                  ...(["int4", "int8"] as const).map((precision) => ({
                    title:
                      precision === "int4"
                        ? "Bitsandbytes NF4 and FP4"
                        : "Bitsandbytes LLM.int8",
                    text:
                      precision === "int4"
                        ? "Packed execution and dequantized matrix inspection. Different from native NVFP4."
                        : "LLM.int8 weights with float32 residuals. Full small spectra use float64.",
                    supported:
                      !!hardware.quantization?.cpu[precision]?.supported ||
                      !!hardware.quantization?.cuda[precision]?.supported,
                    badge:
                      [
                        hardware.quantization?.cpu[precision]?.supported &&
                          "CPU",
                        hardware.quantization?.cuda[precision]?.supported &&
                          "CUDA",
                      ]
                        .filter(Boolean)
                        .join(" + ") || "Kernel unavailable",
                  })),
                  {
                    title: "Native BF16 acceleration",
                    text: "Ampere or newer GPU. Compute capability 8.0 or above.",
                    supported: capability >= 8 && hardware.cuda_available,
                    badge: "Hardware eligible",
                  },
                  {
                    title: "Fine grained FP8",
                    text: "Hopper or newer. Packed analysis is not implemented.",
                    supported: false,
                    badge:
                      capability >= 9
                        ? "GPU eligible · runtime pending"
                        : "Requires Hopper or newer",
                  },
                  {
                    title: "Native NVFP4",
                    text: "Blackwell only. Hopper H100 and H200 lack native FP4 kernels.",
                    supported: false,
                    badge:
                      capability >= 10
                        ? "GPU eligible · runtime pending"
                        : "Requires Blackwell",
                  },
                ].map((item) => (
                  <div key={item.title}>
                    <div>
                      <h3>{t(item.title)}</h3>
                      <p>{item.text}</p>
                    </div>
                    <span
                      className={`pill ${item.supported ? "green" : "subtle"}`}
                    >
                      {item.supported && <Check size={11} />}
                      {item.badge}
                    </span>
                  </div>
                ))}
              </div>
            </Panel>
            <div className="stack">
              <Panel
                title={t("Active model")}
                id="active-model"
                subtitle={t("One checkpoint is kept in memory at a time.")}
              >
                <div className="panel-body">
                  <div className="runtime-table">
                    <div>
                      <span>{t("Checkpoint")}</span>
                      <strong
                        style={{
                          overflowWrap: "anywhere",
                          maxWidth: "65%",
                          textAlign: "right",
                        }}
                      >
                        {runtime?.metadata?.id || t("No model loaded")}
                      </strong>
                    </div>
                    <div>
                      <span>{t("Execution device")}</span>
                      <strong>{runtime?.device.toUpperCase() || "CPU"}</strong>
                    </div>
                    <div>
                      <span>{t("Context allowance")}</span>
                      <strong>
                        {runtime?.loaded
                          ? t("{v0} tokens", {
                              v0:
                                runtime.context_length ??
                                runtime.metadata?.context_length,
                            })
                          : t("No model loaded")}
                      </strong>
                    </div>
                    <div>
                      <span>{t("Execution format")}</span>
                      <strong>
                        {runtime?.loaded
                          ? t(
                              executionLabel(
                                runtime.precision,
                                runtime.metadata?.execution_config,
                              ),
                            )
                          : t("No model loaded")}
                      </strong>
                    </div>
                  </div>
                </div>
                {runtime?.loaded && (
                  <div className="danger-zone">
                    <p>
                      {t(
                        "Release model memory. Downloaded weights remain cached.",
                      )}
                    </p>
                    <button
                      className="button secondary"
                      disabled={busy}
                      onClick={unload}
                    >
                      {t("Unload model")}
                    </button>
                  </div>
                )}
              </Panel>
              <Panel
                title={t("Resource estimates")}
                subtitle={t("Why the analysis budget exceeds checkpoint size.")}
              >
                <div className="panel-body">
                  <p
                    style={{
                      fontSize: 11,
                      color: "var(--muted)",
                      lineHeight: 1.9,
                    }}
                  >
                    {rich(
                      "{v0}{v1}Selected matrices are converted to float32. The budget includes temporary analysis space, execution weights, residual states, bounded attention capture, a conservative KV cache allowance, and runtime overhead. Available memory changes as other workloads run.",
                      {
                        v0: hardware.ram_scope === "cgroup" && (
                          <span>
                            {rich(
                              "RAM is constrained by the backend process or container. The host has {v0}. Available memory includes reclaimable inactive file cache.",
                              {
                                v0: bytes(
                                  hardware.ram_host_total_bytes ||
                                    hardware.ram_total_bytes,
                                ),
                              },
                            )}
                          </span>
                        ),
                        v1: hardware.ram_constraints?.some(
                          (limit) => limit.used_bytes === null,
                        ) && (
                          <span>
                            {t(
                              "Current usage for a memory constraint could not be read. The available RAM budget is zero until it can be measured.",
                            )}
                          </span>
                        ),
                      },
                    )}
                  </p>
                </div>
              </Panel>
            </div>
          </div>
        </>
      )}
    </>
  );
}
