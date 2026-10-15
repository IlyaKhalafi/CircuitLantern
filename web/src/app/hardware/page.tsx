"use client";

import {
  Check,
  Cpu,
  HardDrive,
  Info,
  MemoryStick,
  RefreshCw,
  Server,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  ErrorNotice,
  Loading,
  Metric,
  PageHeader,
  Panel,
} from "@/components/ui";
import { api, bytes } from "@/lib/api";
import type { Hardware, RuntimeState } from "@/lib/types";

export default function HardwarePage() {
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
      await api("/runtime", { method: "DELETE" });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  const capability = hardware?.gpus[0]?.compute_capability || 0;
  return (
    <>
      <PageHeader
        eyebrow="KNOW YOUR ENVIRONMENT"
        title="A clear view of your server."
        description="Live resource availability and the runtime behind every compatibility check."
        actions={
          <button
            className="button secondary"
            onClick={refresh}
            disabled={busy}
          >
            <RefreshCw size={14} className={busy ? "spin" : ""} /> Refresh
            snapshot
          </button>
        }
      />
      <ErrorNotice message={error} />
      {!hardware ? (
        busy && <Loading label="Reading server hardware" />
      ) : (
        <>
          <div className="metrics-grid">
            <Metric
              label="Physical CPU cores"
              value={hardware.cpu}
              note={`${hardware.cpu_threads} logical threads`}
            />
            <Metric
              label="Available system memory"
              value={bytes(hardware.ram_available_bytes)}
              note={`${bytes(hardware.ram_total_bytes)} total RAM`}
            />
            <Metric
              label="Free checkpoint storage"
              value={bytes(hardware.disk_free_bytes)}
              note={`${bytes(hardware.disk_total_bytes)} total capacity`}
            />
            <Metric
              label="Detected NVIDIA GPUs"
              value={hardware.gpus.length}
              note={
                hardware.cuda_available
                  ? "CUDA runtime available"
                  : "CPU analysis runtime"
              }
            />
          </div>
          <div className="two-column">
            <Panel
              title="GPU inventory"
              subtitle="Compatibility uses free memory on the first GPU, with a 15 percent reserve."
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
                        GPU {gpu.index} · SM {gpu.compute_capability.toFixed(1)}{" "}
                        · Driver {gpu.driver}
                      </p>
                      <div className="gpu-memory">
                        <div className="memory-bar">
                          <div
                            style={{
                              width: `${(1 - gpu.free_bytes / gpu.total_bytes) * 100}%`,
                              background: "#a8b8cc",
                            }}
                          />
                        </div>
                        <div className="memory-legend">
                          <span>
                            {bytes(gpu.total_bytes - gpu.free_bytes)} in use
                          </span>
                          <span>{bytes(gpu.free_bytes)} free</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="panel-body">
                  <div className="notice info">
                    <Info size={16} />
                    No NVIDIA GPU detected. Unquantized models can still be
                    explored using system memory.
                  </div>
                </div>
              )}
              {hardware.gpus.length > 0 && !hardware.cuda_available && (
                <p className="inline-note">
                  The driver detects a GPU, but this PyTorch build is CPU only.
                  Install a CUDA enabled build to execute models on the GPU.
                </p>
              )}
            </Panel>
            <Panel
              title="Python runtime"
              subtitle="Installed versions reported by the backend environment."
            >
              <div className="runtime-table">
                {Object.entries(hardware.packages).map(([name, version]) => (
                  <div key={name}>
                    <span>{name}</span>
                    <strong>{version || "Not installed"}</strong>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
          <div className="two-column">
            <Panel
              title="Precision requirements"
              subtitle="Hardware eligibility and analysis support are separate checks."
            >
              <div className="capability-list">
                {[
                  {
                    title: "FP32 / FP16 / BF16 checkpoints",
                    text: "Loaded as float32 for stable weight inspection.",
                    supported: true,
                    badge: "Supported",
                  },
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
                      <h3>{item.title}</h3>
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
                title="Active model"
                subtitle="One checkpoint is kept in memory at a time."
              >
                <div className="panel-body">
                  <div className="runtime-table">
                    <div>
                      <span>Checkpoint</span>
                      <strong
                        style={{
                          overflowWrap: "anywhere",
                          maxWidth: "65%",
                          textAlign: "right",
                        }}
                      >
                        {runtime?.metadata?.id || "No model loaded"}
                      </strong>
                    </div>
                    <div>
                      <span>Execution device</span>
                      <strong>{runtime?.device.toUpperCase() || "CPU"}</strong>
                    </div>
                    <div>
                      <span>Analysis precision</span>
                      <strong>float32</strong>
                    </div>
                  </div>
                </div>
                {runtime?.loaded && (
                  <div className="danger-zone">
                    <p>
                      Release model memory. Downloaded weights remain cached.
                    </p>
                    <button
                      className="button secondary"
                      disabled={busy}
                      onClick={unload}
                    >
                      Unload model
                    </button>
                  </div>
                )}
              </Panel>
              <Panel
                title="Resource estimates"
                subtitle="Why the analysis budget exceeds checkpoint size."
              >
                <div className="panel-body">
                  <p
                    style={{ fontSize: 11, color: "#96a0ad", lineHeight: 1.9 }}
                  >
                    Weights are promoted to float32. The budget also includes
                    residual states, bounded attention capture, a conservative
                    KV cache allowance, and runtime overhead. Available memory
                    changes as other workloads run.
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
