export type AlignmentStatus = 'verified' | 'unverified' | 'failed' | 'not_applicable';

export type AssessmentStatus = 'ready_for_review' | 'human_review' | 'insufficient_evidence' | 'unavailable';

export type RecommendationType =
  | 'thermal_baseline'
  | 'rgb_fusion'
  | 'reliability_fusion'
  | 'rerun_evaluation'
  | 'human_review'
  | 'no_recommendation';

export interface QAMetadataInput {
  sample_id: string;
  experiment_id?: string;
  experiment_type?: string;
  scale_factor: number;
  thermal_hr_shape?: [number, number];
  thermal_lr_shape: [number, number];
  rgb_shape?: [number, number] | [number, number, number];
  alignment_status: AlignmentStatus;
  ground_truth_available: boolean;
  metrics?: Record<string, any>;
  deterministic_warnings?: string[];
}

export interface QAAssessmentResponse {
  addon: string;
  status: 'success' | 'insufficient_evidence' | 'unavailable' | 'invalid_request';
  assessment: AssessmentStatus;
  warnings: string[];
  missing_evidence: string[];
  deterministic_checks: Record<string, boolean>;
  probabilities?: Record<string, number> | null;
  model_latency_ms?: number | null;
  model_version?: string | null;
  reconstruction_modified: boolean;
  disclaimer: string;
}

export interface ExperimentRecommendationRequest {
  current_experiment_id?: string;
  current_architecture?: string;
  available_experiments: string[];
  scale_factor: number;
  alignment_status: AlignmentStatus;
  ground_truth_available: boolean;
  metrics?: Record<string, any>;
  deterministic_warnings?: string[];
}

export interface ExperimentRecommendationResponse {
  recommendation: RecommendationType;
  reason_codes: string[];
  explanation: string;
  evidence_used: string[];
  missing_evidence: string[];
  probabilities?: Record<string, number> | null;
  model_latency_ms?: number | null;
  executed: boolean;
  disclaimer: string;
}

export interface QAHealthResponse {
  status: string;
  laya_installed: boolean;
  laya_model_loaded: boolean;
  fallback_mode: boolean;
  supported_statuses: string[];
  supported_recommendations: string[];
}
