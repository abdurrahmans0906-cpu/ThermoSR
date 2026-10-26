import axios from 'axios';
import {
  HealthResponse,
  DeviceInfo,
  Dataset,
  DatasetPair,
  ModelVersion,
  TrainingRun,
  ReconstructionRun,
  Experiment,
} from '../types';

const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health & System
  getHealth: async (): Promise<HealthResponse> => {
    const res = await apiClient.get('/health');
    return res.data;
  },
  getDeviceInfo: async (): Promise<DeviceInfo> => {
    const res = await apiClient.get('/system/device');
    return res.data;
  },

  // Datasets
  scanDataset: async (rootPath?: string, name?: string): Promise<Dataset> => {
    const res = await apiClient.post('/datasets/scan', { root_path: rootPath, name });
    return res.data;
  },
  getDatasets: async (): Promise<Dataset[]> => {
    const res = await apiClient.get('/datasets');
    return res.data;
  },
  getDataset: async (id: string): Promise<Dataset> => {
    const res = await apiClient.get(`/datasets/${id}`);
    return res.data;
  },
  getDatasetSamples: async (id: string, split?: string, limit = 20): Promise<DatasetPair[]> => {
    const res = await apiClient.get(`/datasets/${id}/samples`, { params: { split, limit } });
    return res.data;
  },
  validateDataset: async (id: string): Promise<any> => {
    const res = await apiClient.post(`/datasets/${id}/validate`);
    return res.data;
  },

  // Reconstructions
  createReconstruction: async (formData: FormData): Promise<ReconstructionRun> => {
    const res = await apiClient.post('/reconstructions', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  getReconstructions: async (limit = 20): Promise<ReconstructionRun[]> => {
    const res = await apiClient.get('/reconstructions', { params: { limit } });
    return res.data;
  },
  getReconstruction: async (id: string): Promise<ReconstructionRun> => {
    const res = await apiClient.get(`/reconstructions/${id}`);
    return res.data;
  },
  cancelReconstruction: async (id: string): Promise<any> => {
    const res = await apiClient.post(`/reconstructions/${id}/cancel`);
    return res.data;
  },

  // Training
  createTrainingRun: async (data: any): Promise<TrainingRun> => {
    const res = await apiClient.post('/training-runs', data);
    return res.data;
  },
  getTrainingRuns: async (): Promise<TrainingRun[]> => {
    const res = await apiClient.get('/training-runs');
    return res.data;
  },
  getTrainingRun: async (id: string): Promise<TrainingRun> => {
    const res = await apiClient.get(`/training-runs/${id}`);
    return res.data;
  },
  getTrainingMetrics: async (id: string): Promise<any> => {
    const res = await apiClient.get(`/training-runs/${id}/metrics`);
    return res.data;
  },
  cancelTrainingRun: async (id: string): Promise<any> => {
    const res = await apiClient.post(`/training-runs/${id}/cancel`);
    return res.data;
  },

  // Model Registry
  getModelRegistry: async (): Promise<ModelVersion[]> => {
    const res = await apiClient.get('/model-registry');
    return res.data;
  },
  registerModel: async (data: any): Promise<ModelVersion> => {
    const res = await apiClient.post('/model-registry/register', data);
    return res.data;
  },
  getModelVersion: async (id: string): Promise<ModelVersion> => {
    const res = await apiClient.get(`/model-registry/${id}`);
    return res.data;
  },

  // Experiments
  getExperiments: async (): Promise<Experiment[]> => {
    const res = await apiClient.get('/experiments');
    return res.data;
  },
  getAblationSummary: async (): Promise<any> => {
    const res = await apiClient.get('/experiments/ablation-summary');
    return res.data;
  },
  createExperiment: async (data: any): Promise<Experiment> => {
    const res = await apiClient.post('/experiments', data);
    return res.data;
  },

  // Artifact URL helper
  getArtifactUrl: (path?: string) => {
    if (!path) return '';
    return `/api/artifacts/${path}`;
  },
};

export function formatApiError(err: any, fallback: string = 'An unexpected error occurred'): string {
  if (!err) return fallback;
  const detail = err.response?.data?.detail;
  if (typeof detail === 'string') {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((d: any) => {
        if (typeof d === 'string') return d;
        const loc = Array.isArray(d.loc) ? d.loc.filter((x: any) => x !== 'body').join('.') : '';
        return loc ? `${loc}: ${d.msg || JSON.stringify(d)}` : d.msg || JSON.stringify(d);
      })
      .join('; ');
  }
  if (detail && typeof detail === 'object') {
    return detail.msg || JSON.stringify(detail);
  }
  return err.message || fallback;
}

