export interface ModelCard {
  id: string;
  name: string;
  family: string;
  parameters: string;
  description: string;
  tag: string;
  color: string;
}
export interface ModelMetadata {
  id: string;
  revision: string;
  architecture: string;
  parameters: number;
  parameter_estimated: boolean;
  weight_bytes: number;
  hidden_size: number;
  num_layers: number;
  num_heads: number;
  num_kv_heads: number;
  context_length: number;
  vocab_size: number;
  dtype: string;
  quantization: Record<string, unknown> | null;
  execution_precision?: string;
  quantizable_parameters?: number;
  largest_matrix_elements?: number;
  download_bytes?: number;
}
export interface GPU {
  index: number;
  name: string;
  total_bytes: number;
  free_bytes: number;
  compute_capability: number;
  driver: string;
}
export interface Hardware {
  cpu: number;
  cpu_threads: number;
  ram_total_bytes: number;
  ram_available_bytes: number;
  disk_total_bytes: number;
  disk_free_bytes: number;
  gpus: GPU[];
  packages: Record<string, string | null>;
  cuda_available: boolean;
  runtime_device: string;
  execution_gpu?: Omit<GPU, "driver"> | null;
  cuda_error?: string | null;
  quantization?: {
    version: string | null;
    cpu: Record<string, { supported: boolean; reason?: string }>;
    cuda: Record<string, { supported: boolean; reason?: string }>;
  };
}
export interface Compatibility {
  compatible: boolean;
  status: string;
  precision: string;
  runtime_precision: string;
  device: string;
  weight_bytes: number;
  analysis_weight_bytes: number;
  analysis_workspace_bytes?: number;
  kv_cache_bytes: number;
  overhead_bytes: number;
  required_bytes: number;
  available_bytes: number;
  download_bytes: number;
  blockers: string[];
  warnings: string[];
  metadata: ModelMetadata;
}
export interface Job {
  id: string;
  model_id: string;
  status: string;
  progress: number;
  message: string;
}
export interface RuntimeState {
  precision?: string;
  loaded: boolean;
  metadata: ModelMetadata | null;
  device: string;
}
export interface Weight {
  name: string;
  shape: number[];
  elements: number;
  dtype: string;
}
export interface ModelGraph {
  metadata: ModelMetadata;
  matrices: Weight[];
  total_parameters: number;
}
export interface MatrixMetrics {
  source_format?: string;
  quantized?: boolean;
  execution_precision?: string;
  analysis_note?: string;
  name: string;
  shape: number[];
  elements: number;
  dtype: string;
  frobenius_norm: number;
  l1_norm: number;
  infinity_norm: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  sparsity: number;
  spectral_norm: number;
  spectral_radius: number | null;
  stable_rank: number;
  condition_number: number | null;
  singular_values: number[];
  cumulative_energy: number[];
  spectrum_exact: boolean;
  spectrum_shape: number[];
  spectrum_method: string;
  histogram: { value: number; count: number }[];
  histogram_sampled: boolean;
  heatmap: number[][];
  heatmap_method: string;
}
export interface Prediction {
  token: string;
  probability: number;
  token_id: number;
  logit?: number;
}
export interface LayerTrace {
  layer: number;
  label: string;
  rms: number;
  norm: number;
  token_norms: number[];
  cosine_similarity: number | null;
  entropy: number;
  predictions: Prediction[];
  token_predictions: string[];
}
export interface Attention {
  layer: number;
  pattern: number[][];
  head_entropy: number[];
  heads: number;
}
export interface Trace {
  execution_precision?: string;
  model_id: string;
  revision: string;
  prompt: string;
  tokens: string[];
  token_ids: number[];
  continuation: string;
  layers: LayerTrace[];
  attention: Attention[];
  predictions: Prediction[];
  duration_ms: number;
  device: string;
  ablate_layer: number | null;
  method: string;
}
export interface Experiment {
  id: string;
  name: string;
  notes: string;
  created_at: string;
  model_id: string;
  trace?: Trace;
}

export interface CausalLayer {
  layer: number;
  probability: number;
  log_probability: number;
  delta_log_probability: number;
  recovery: number | null;
}
export interface CausalResult {
  execution_precision?: string;
  requested_target: string;
  total_layers: number;
  cancelled: boolean;
  baseline_log_probability_gap: number;
  recovery_minimum_gap: number;
  recovery_warning: string | null;
  model_id: string;
  revision: string;
  clean_prompt: string;
  corrupted_prompt: string;
  tokens: string[];
  corrupted_tokens: string[];
  target_token: string;
  target_token_id: number;
  position: number | null;
  clean: { probability: number; log_probability: number };
  corrupted: { probability: number; log_probability: number };
  layers: CausalLayer[];
  duration_ms: number;
  recovery_defined: boolean;
  method: string;
}
