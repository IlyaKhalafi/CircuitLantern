export interface FeatureMetrics {
  samples: number;
  mse: number;
  normalized_mse: number | null;
  explained_variance: number | null;
  mean_active_features: number;
  inactive_features: number;
}
export interface DictionarySummary {
  id: string;
  name: string;
  model_id: string;
  revision: string;
  execution_precision: string;
  layer: number;
  features: number;
  input_dim: number;
  method: "relu" | "topk";
  created_at: string;
  train_metrics: FeatureMetrics;
  validation_metrics: FeatureMetrics;
  training_summary: {
    completed_steps: number;
    requested_steps: number;
    effective_batch_size: number;
    tokens_seen: number;
    unique_train_tokens: number;
    unique_validation_tokens: number;
    cancelled: boolean;
  };
}
export interface FeatureStatistic {
  index: number;
  frequency: number;
  mean_activation: number;
  max_activation: number;
  activation_count: number;
  train_frequency?: number | null;
  validation_frequency?: number | null;
  examples: { sample: number; activation: number }[];
}
export interface DictionaryManifest extends DictionarySummary {
  execution_id: string;
  execution_config: Record<string, unknown>;
  weights_sha256: string;
  history: Record<string, number | string | null>[];
  feature_statistics: FeatureStatistic[];
  feature_labels: Record<string, { label: string; notes: string }>;
  corpus: {
    fingerprint: string;
    special_tokens_excluded: boolean;
    first_position_excluded: boolean;
    documents: {
      text: string;
      tokens: string[];
      split: string;
      truncated: boolean;
    }[];
    samples: { document: number; position: number; split: string }[];
  };
}
export interface FeatureCapabilities {
  loaded: boolean;
  supported: boolean;
  layers: number[];
  input_dim: number | null;
  execution: {
    model_id: string;
    revision: string;
    execution_id: string;
  } | null;
  example_documents: string[];
  references: { title: string; url: string; year: number }[];
}
export interface DictionaryPlan {
  compatible: boolean;
  blockers: string[];
  warnings: string[];
  features: number;
  selected_layers: number[];
  estimated_host_bytes: number;
  estimated_gpu_bytes: number;
  estimated_disk_bytes: number;
  unique_train_tokens: number;
  unique_validation_tokens: number;
  document_count: number;
  truncated_documents: number;
  duplicate_documents: number;
  optimizer_steps_total: number;
  teacher_forward_passes: number;
  train_document_groups: number;
  validation_document_groups: number;
  corpus_fingerprint: string;
}
export interface TrainingJobResult {
  dictionaries: DictionarySummary[];
  complete: boolean;
}
export interface FeaturePrediction {
  token: string;
  token_id: number;
  probability: number;
}
export interface FeatureInspection {
  dictionary_id: string;
  prompt: string;
  tokens: string[];
  token_ids: number[];
  features: { index: number; activations: (number | null)[] }[];
  active_features_per_token: number[];
  eligible_positions: number[];
  prompt_split: "train" | "validation" | "unseen";
  dictionary_training: DictionarySummary["training_summary"];
  baseline_predictions: FeaturePrediction[];
  intervention?: {
    feature: number;
    multiplier: number;
    position: number | null;
    target_token: string;
    target_token_id: number;
    target_log_probability_change: number;
    kl_baseline_to_patched: number;
    predictions: FeaturePrediction[];
  };
}
export interface DictionaryEvaluation {
  prompt: string;
  scored_tokens: number;
  prompt_split: "train" | "validation" | "unseen";
  baseline_cross_entropy: number | null;
  reconstructed_cross_entropy: number | null;
  mean_ablated_cross_entropy: number | null;
  loss_recovered: number | null;
}
export interface FeatureCircuit {
  id: string;
  kind: string;
  created_at: string;
  prompt: string;
  complete: boolean;
  cancelled?: boolean;
  target_token: string;
  dictionary_ids: string[];
  nodes: {
    id: string;
    dictionary_id: string;
    layer: number;
    feature: number;
    mean_activation: number;
    label: string;
  }[];
  edges: {
    source: string;
    target: string;
    activation_change?: number;
    log_probability_change?: number;
  }[];
  measurements: {
    source: string;
    target_log_probability_change: number;
    kl_baseline_to_patched: number;
  }[];
}
export type FeatureRunSummary = Pick<
  FeatureCircuit,
  "id" | "kind" | "created_at" | "prompt" | "complete" | "dictionary_ids"
>;

export interface DecoderDirections {
  format: string;
  dictionary_id: string;
  model_id: string;
  revision: string;
  execution_id: string;
  execution_precision: string;
  execution_config: Record<string, unknown>;
  dictionary_method: DictionarySummary["method"];
  input_dim: number;
  layer: number;
  capture_point: string;
  weights_sha256: string;
  manifest_sha256: string;
  feature: number;
  compared_features: number;
  highest: DecoderDirection[];
  lowest: DecoderDirection[];
  convention: string;
}
export interface DecoderDirection {
  feature: number;
  cosine: number;
  frequency: number | null;
  label: string;
}
