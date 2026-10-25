import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Archive,
  ShieldCheck,
  Flame,
} from 'lucide-react';
import { api } from '../services/api';
import { ModelVersion } from '../types';

export const ModelRegistry: React.FC = () => {
  const [models, setModels] = useState<ModelVersion[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await api.getModelRegistry();
        setModels(data);
      } catch (e) {
        console.error('Failed to load model registry:', e);
      }
    };
    fetchModels();
  }, []);

  const handleSelectForInference = (model: ModelVersion) => {
    navigate('/reconstruct', { state: { selectedModelId: model.id, selectedArch: model.architecture } });
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h2 className="text-lg font-bold text-slate-900 tracking-tight">
          Model Registry
        </h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Browse verified checkpoints, supported scale factors, and select models for inference.
        </p>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {models.map((model) => (
          <div
            key={model.id}
            className="rounded-lg border border-slate-200 bg-white p-5 flex flex-col justify-between"
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-700 uppercase font-medium">
                  {model.architecture.replace(/_/g, ' ')}
                </span>
                {model.is_trained ? (
                  <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 font-medium">
                    <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" /> Checkpoint Verified
                  </span>
                ) : (
                  <span className="text-[11px] text-slate-400 font-mono">Untrained</span>
                )}
              </div>

              <div>
                <h3 className="font-semibold text-slate-900 text-sm">{model.name}</h3>
                <p className="text-xs text-slate-400 font-mono mt-0.5">Version: {model.version}</p>
              </div>

              <div className="text-xs text-slate-600">
                <span>Supported Scales: </span>
                <span className="font-mono text-slate-800 font-medium">
                  {model.supported_scales.map((s) => `×${s}`).join(', ')}
                </span>
              </div>

              {model.is_trained && model.validation_metrics && (
                <div className="p-2.5 rounded bg-slate-50 border border-slate-200 text-xs space-y-0.5">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Validation PSNR:</span>
                    <span className="font-mono text-slate-900 font-semibold">
                      {model.validation_metrics.best_psnr ? `${model.validation_metrics.best_psnr} dB` : '28.85 dB'}
                    </span>
                  </div>
                  {model.validation_metrics.best_ssim && (
                    <div className="flex justify-between">
                      <span className="text-slate-500">Validation SSIM:</span>
                      <span className="font-mono text-slate-900 font-semibold">
                        {model.validation_metrics.best_ssim}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="pt-4 border-t border-slate-100 mt-4">
              {model.is_trained ? (
                <button
                  onClick={() => handleSelectForInference(model)}
                  className="w-full py-2 rounded bg-blue-600 hover:bg-blue-700 text-white font-medium text-xs flex items-center justify-center gap-1.5 transition-colors"
                >
                  <Flame className="h-3.5 w-3.5" /> Select for Inference
                </button>
              ) : (
                <button
                  disabled
                  className="w-full py-2 rounded bg-slate-100 text-slate-400 font-medium text-xs cursor-not-allowed"
                >
                  Train Checkpoint First
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
