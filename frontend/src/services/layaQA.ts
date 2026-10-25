import axios from 'axios';
import {
  QAMetadataInput,
  QAAssessmentResponse,
  ExperimentRecommendationRequest,
  ExperimentRecommendationResponse,
  QAHealthResponse,
} from '../types/layaQA';

const qaClient = axios.create({
  baseURL: '/api/qa',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

export const layaQAService = {
  getHealth: async (): Promise<QAHealthResponse> => {
    const res = await qaClient.get<QAHealthResponse>('/health');
    return res.data;
  },

  assessMetadata: async (data: QAMetadataInput): Promise<QAAssessmentResponse> => {
    const res = await qaClient.post<QAAssessmentResponse>('/laya-assess', data);
    return res.data;
  },

  getRecommendation: async (
    data: ExperimentRecommendationRequest
  ): Promise<ExperimentRecommendationResponse> => {
    const res = await qaClient.post<ExperimentRecommendationResponse>('/experiment-recommendation', data);
    return res.data;
  },
};
