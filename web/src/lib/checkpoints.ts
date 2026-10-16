import type { ExecutionProvenance, Prediction } from "@/lib/types";
export interface CheckpointReference extends ExecutionProvenance {
  id: string;
  name: string;
  created_at: string;
  model_id: string;
  revision: string;
  prompt: string;
  tokens: string[];
  matrices: { name: string; shape: number[]; tensor: string }[];
  profile: { architecture: string; hidden_size: number; blocks: string[] };
  layers: { layer: number; entropy: number; predictions: Prediction[] }[];
  predictions: Prediction[];
}
export interface VectorChange {
  selected_norm: number;
  reference_norm: number;
  distance: number;
  relative_distance: number | null;
  cosine_similarity: number | null;
}
export interface CheckpointComparison {
  id: string;
  created_at: string;
  reference: CheckpointReference;
  selected: Omit<CheckpointReference, "id" | "created_at">;
  weights: (VectorChange & {
    name: string;
    shape: number[];
    rms_change: number;
    max_absolute_change: number;
  })[];
  layers: (VectorChange & {
    layer: number;
    selected_prediction: Prediction;
    reference_prediction: Prediction;
    entropy_change: number;
  })[];
  kl_reference_to_selected: number;
  kl_selected_to_reference: number;
  total_variation: number;
  predictions: {
    token_id: number;
    token: string;
    reference_probability: number;
    selected_probability: number;
    log_probability_change: number;
  }[];
  warnings: string[];
  convention: string;
}

export type CheckpointComparisonSummary = Pick<
  CheckpointComparison,
  "id" | "created_at"
> & {
  reference: Pick<CheckpointReference, "model_id" | "revision">;
  selected: Pick<CheckpointReference, "model_id" | "revision">;
};
export interface CheckpointLibrary<T> {
  items: T[];
  unreadable: string[];
}
