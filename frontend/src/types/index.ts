export interface HealthResponse {
  status: string;
  version: string;
  project: string;
  database: string;
  device: string;
  torch_version: string;
  cuda_available: boolean;
}

export interface DeviceInfo {
  device: string;
  device_name: string;
  cuda_available: boolean;
  cuda_version?: string;
  vram_total_mb?: number;
  vram_used_mb?: number;
  vram_free_mb?: number;
  torch_version: string;
  cpu_cores: number;
  recommended_batch_size: number;
  notes: string[];
}

export interface Dataset {
  id: string;
  name: string;
  root_dir: string;
  dataset_type: string;
  split_config: Record<string, number>;
  total_pairs: number;
  valid_pairs: number;
  unmatched_count: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface DatasetPair {
  id: string;
  dataset_id: string;
  split: string;
  thermal_path: string;
  rgb_path?: string;
  reference_path?: string;
  dimensions: {
    thermal: [number, number];
    rgb?: [number, number];
  };
  validation_status: string;
  alignment_status: string;
  error_message?: string;
}

export interface ModelVersion {
  id: string;
  name: string;
  architecture: 'thermal_cnn_baseline' | 'rgb_fusion' | 'reliability_aware_fusion';
  version: string;
  checkpoint_path?: string;
  configuration: Record<string, any>;
  training_run_id?: string;
  supported_scales: number[];
  validation_metrics: Record<string, any>;
  is_available: boolean;
  is_trained: boolean;
  created_at: string;
  updated_at: string;
}

export interface TrainingRun {
  id: string;
  model_architecture: string;
  dataset_id?: string;
  scale_factor: number;
  hyperparameters: Record<string, any>;
  random_seed: number;
  device_info: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  start_time?: string;
  end_time?: string;
  elapsed_seconds: number;
  current_epoch: number;
  total_epochs: number;
  best_checkpoint_path?: string;
  latest_metrics: Record<string, any>;
  metrics_history: Array<{
    epoch: number;
    train_loss: number;
    val_loss: number;
    psnr: number;
    ssim: number;
    lr: number;
    elapsed_seconds: number;
  }>;
  error_details?: string;
  created_at: string;
}

export interface ReconstructionRun {
  id: string;
  task_mode: 'rgb_guided' | 'evaluation' | 'downsample_eval';
  model_version_id?: string;
  model_architecture?: string;
  scale_factor: number;
  input_artifacts: {
    thermal: string;
    rgb?: string;
    reference?: string;
  };
  output_artifacts: {
    reconstructed?: string;
    false_color?: string;
    reliability_map?: string;
    error_map?: string;
  };
  configuration: Record<string, any>;
  status: 'queued' | 'validating' | 'preprocessing' | 'running_inference' | 'calculating_metrics' | 'completed' | 'failed' | 'cancelled';
  progress_pct: number;
  metrics: {
    psnr?: number;
    ssim?: number;
    lpips?: number;
    mae?: number;
    max_error?: number;
    thermal_consistency_l1?: number;
    inference_time_ms: number;
    input_dimensions: [number, number];
    output_dimensions: [number, number];
    has_ground_truth: boolean;
    is_trained_model: boolean;
    device_used: string;
  };
  inference_time_ms: number;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  experiment_type: string;
  related_training_run_ids: string[];
  comparison_results: Record<string, any>;
  created_at: string;
}
