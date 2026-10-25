import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Flame,
  Layers,
  Cpu,
  BarChart3,
  Archive,
  ArrowRight,
  TrendingUp,
  Clock,
} from 'lucide-react';
import { api } from '../services/api';
import { Dataset, ModelVersion, TrainingRun, ReconstructionRun, DeviceInfo } from '../types';

export const Dashboard: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [reconstructions, setReconstructions] = useState<ReconstructionRun[]>([]);
  const [device, setDevice] = useState<DeviceInfo | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [ds, mv, tr, rc, dev] = await Promise.all([
          api.getDatasets(),
          api.getModelRegistry(),
          api.getTrainingRuns(),
          api.getReconstructions(5),
          api.getDeviceInfo(),
        ]);
        setDatasets(ds);
        setModels(mv);
        setTrainingRuns(tr);
        setReconstructions(rc);
        setDevice(dev);
      } catch (err) {
        console.error('Failed to load dashboard metrics:', err);
      }
    };
    fetchData();
  }, []);

  const totalPairs = datasets.reduce((acc, d) => acc + d.total_pairs, 0);
  const validPairs = datasets.reduce((acc, d) => acc + d.valid_pairs, 0);
  const trainedModels = models.filter((m) => m.is_trained).length;
  const completedRuns = reconstructions.filter((r) => r.status === 'completed').length;
  const latestRun = reconstructions.find((r) => r.status === 'completed' && r.metrics?.psnr);

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-white border border-slate-200 rounded-lg p-6">
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">
          Thermal Image Super-Resolution Dashboard
        </h2>
        <p className="mt-1 text-sm text-slate-600 leading-relaxed max-w-3xl">
          A computer vision research environment for evaluating thermal image super-resolution ($\times 8$)
          using RGB guidance. Compare the <strong>Thermal-Only Baseline</strong>, <strong>Standard RGB Fusion</strong>,
          and the proposed <strong>Reliability-Aware RGB Guidance</strong> model.
        </p>
        <div className="mt-4 flex flex-wrap gap-2.5">
          <Link
            to="/reconstruct"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-700 text-white font-medium text-xs transition-colors"
          >
            <Flame className="h-4 w-4" /> Go to Reconstruction
          </Link>
          <Link
            to="/dataset"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-white hover:bg-slate-50 text-slate-700 font-medium text-xs border border-slate-300 transition-colors"
          >
            <Layers className="h-4 w-4" /> Manage Datasets
          </Link>
          <Link
            to="/experiments"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-white hover:bg-slate-50 text-slate-700 font-medium text-xs border border-slate-300 transition-colors"
          >
            <BarChart3 className="h-4 w-4" /> View Experiments
          </Link>
        </div>
      </div>

      {/* Summary Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Dataset Pairs */}
        <div className="p-4 rounded-lg border border-slate-200 bg-white">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs font-medium uppercase tracking-wider">Dataset Pairs</span>
            <Layers className="h-4 w-4 text-slate-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-slate-900 font-mono">
            {totalPairs.toLocaleString()}
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {validPairs > 0 ? `${validPairs.toLocaleString()} verified pairs` : 'No dataset scanned'}
          </p>
        </div>

        {/* Available Models */}
        <div className="p-4 rounded-lg border border-slate-200 bg-white">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs font-medium uppercase tracking-wider">Trained Checkpoints</span>
            <Archive className="h-4 w-4 text-slate-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-slate-900 font-mono">
            {trainedModels}
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {models.length} model architectures
          </p>
        </div>

        {/* Reconstruction Runs */}
        <div className="p-4 rounded-lg border border-slate-200 bg-white">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs font-medium uppercase tracking-wider">Reconstructions</span>
            <Flame className="h-4 w-4 text-slate-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-slate-900 font-mono">
            {completedRuns}
          </div>
          <p className="text-xs text-slate-500 mt-1">Completed inference jobs</p>
        </div>

        {/* Latest PSNR */}
        <div className="p-4 rounded-lg border border-slate-200 bg-white">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-xs font-medium uppercase tracking-wider">Latest Reference PSNR</span>
            <TrendingUp className="h-4 w-4 text-slate-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-blue-600 font-mono">
            {latestRun ? `${latestRun.metrics.psnr} dB` : '—'}
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {latestRun ? `SSIM: ${latestRun.metrics.ssim}` : 'Awaiting evaluation'}
          </p>
        </div>
      </div>

      {/* Main Grid: Recent Runs & Device Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Runs (2 cols) */}
        <div className="lg:col-span-2 rounded-lg border border-slate-200 bg-white p-5">
          <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-100">
            <h3 className="font-semibold text-slate-900 text-sm">Recent Super-Resolution Runs</h3>
            <Link to="/reconstruct" className="text-xs text-blue-600 hover:underline flex items-center gap-1 font-medium">
              Open Reconstruct <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          {reconstructions.length === 0 ? (
            <div className="py-10 text-center text-xs text-slate-500">
              No recent reconstruction jobs. Upload an image pair on the Reconstruct page to begin.
            </div>
          ) : (
            <div className="divide-y divide-slate-100">
              {reconstructions.map((run) => (
                <div key={run.id} className="py-3 flex items-center justify-between text-xs">
                  <div>
                    <div className="font-semibold text-slate-800 capitalize">
                      {run.model_architecture?.replace(/_/g, ' ')}
                    </div>
                    <div className="text-[11px] text-slate-500">
                      Scale ×{run.scale_factor} • Mode: {run.task_mode}
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="font-mono font-medium text-slate-900">
                      {run.metrics?.psnr ? `${run.metrics.psnr} dB` : 'Completed'}
                    </span>
                    <div className="text-[11px] text-slate-400">
                      {run.inference_time_ms ? `${run.inference_time_ms} ms` : '—'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Hardware Status (1 col) */}
        <div className="rounded-lg border border-slate-200 bg-white p-5 space-y-3">
          <h3 className="font-semibold text-slate-900 text-sm flex items-center gap-2">
            <Cpu className="h-4 w-4 text-slate-500" /> Compute Environment
          </h3>

          <div className="divide-y divide-slate-100 text-xs">
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">Compute Device:</span>
              <span className="font-medium text-slate-900">{device?.device_name || 'CPU'}</span>
            </div>
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">CUDA Acceleration:</span>
              <span className="font-medium text-slate-900">
                {device?.cuda_available ? `Yes (${device.cuda_version})` : 'No (CPU Mode)'}
              </span>
            </div>
            {device?.vram_total_mb && (
              <div className="py-2 flex justify-between">
                <span className="text-slate-500">VRAM:</span>
                <span className="font-medium text-slate-900">
                  {device.vram_free_mb} MB / {device.vram_total_mb} MB
                </span>
              </div>
            )}
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">PyTorch Version:</span>
              <span className="font-mono text-slate-800">{device?.torch_version || '2.x'}</span>
            </div>
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">CPU Threads:</span>
              <span className="font-mono text-slate-800">{device?.cpu_cores || 4}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
