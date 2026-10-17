import type { ExecutionProvenance } from "@/lib/types";
import type { FeatureCapabilities, FeatureStatistic } from "@/lib/features";

export interface CaptureEvidence {
  id: string;
  name: string;
  created_at: string;
  source: ExecutionProvenance & {
    architecture: string;
    model_id: string;
    revision: string;
  };
  layer: number;
  input_dim: number;
  capture_point: string;
  sample_count: number;
  tensors_sha256: string;
  manifest_sha256: string;
  corpus_fingerprint: string;
}
export interface CaptureSummary extends CaptureEvidence {
  matched_to: CaptureEvidence | null;
  train_samples: number;
  validation_samples: number;
}
export interface CrosscoderCapabilities extends FeatureCapabilities {
  devices: ("cpu" | "cuda")[];
  max_capture_tokens: number;
}
export interface CrosscoderPlan {
  compatible: boolean;
  blockers: string[];
  estimated_host_bytes: number;
  estimated_gpu_bytes: number;
  estimated_disk_bytes: number;
  device: string;
  features?: number;
  train_samples?: number;
  validation_samples?: number;
  teacher_forward_passes?: number;
}
export interface CrosscoderMetrics {
  samples: number;
  banks: {
    input_dim: number;
    mse: number;
    normalized_mse: number | null;
    explained_variance: number | null;
  }[];
  normalized_l2_error: number;
  weighted_l1: number;
  loss: number;
  mean_active_features: number;
  inactive_features: number;
}
export interface CrosscoderSummary {
  id: string;
  name: string;
  created_at: string;
  features: number;
  banks: (CaptureEvidence & { input_scale: number })[];
  train_metrics: CrosscoderMetrics;
  validation_metrics: CrosscoderMetrics;
  training_summary: {
    completed_steps: number;
    requested_steps: number;
    tokens_seen: number;
    unique_train_tokens: number;
    unique_validation_tokens: number;
    cancelled: boolean;
  };
  tensors_sha256: string;
  manifest_sha256: string;
}
export interface CrosscoderOverview extends CrosscoderSummary {
  history: {
    step: number;
    train_sample_normalized_l2_error: number;
    validation_normalized_l2_error: number;
    validation_mean_active_features: number;
  }[];
  training: {
    l1_coefficient: number;
    seed: number;
    batch_size: number;
    learning_rate: number;
  };
  corpus: {
    fingerprint: string;
    documents: number;
    samples: number;
    train_samples: number;
    validation_samples: number;
  };
}
export interface SharedFeature extends Omit<FeatureStatistic, "examples"> {
  decoder_norms: number[];
  relative_decoder_norms: number[] | null;
  decoder_cosine: number | null;
  label?: string;
  notes?: string;
  examples: {
    sample: number;
    activation: number;
    split: string;
    tokens?: string[];
    active_offset?: number;
    document?: number;
    position?: number;
  }[];
}
export interface SharedFeatureDetail extends SharedFeature {
  crosscoder_id: string;
}
export interface SharedFeaturePage {
  crosscoder_id: string;
  total: number;
  offset: number;
  limit: number;
  features: SharedFeature[];
}
export interface CaptureJobResult {
  captures: CaptureSummary[];
  complete: boolean;
}
export interface CrosscoderJobResult {
  crosscoders: CrosscoderSummary[];
  complete: boolean;
}

export function validInteger(value: string, minimum: number, maximum: number) {
  return (
    value.trim() !== "" &&
    Number.isInteger(Number(value)) &&
    Number(value) >= minimum &&
    Number(value) <= maximum
  );
}
export function validNumber(value: string, minimum: number, maximum: number) {
  return (
    value.trim() !== "" &&
    Number.isFinite(Number(value)) &&
    Number(value) >= minimum &&
    Number(value) <= maximum
  );
}
