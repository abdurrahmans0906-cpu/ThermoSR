import React, { useState, useEffect } from 'react';
import {
  FlaskConical,
  Play,
  Square,
  TrendingUp,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { api } from '../services/api';
import { TrainingRun, Dataset, DeviceInfo } from '../types';

export const TrainingLab: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [activeRun, setActiveRun] = useState<TrainingRun | null>(null);

  // Form Config
  const [architecture, setArchitecture] = useState('reliability_aware_fusion');
  const [scaleFactor, setScaleFactor] = useState(8);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(8);
  const [patchSize, setPatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [optimizer, setOptimizer] = useState('adamw');
  const [useGradLoss, setUseGradLoss] = useState(true);
  const [useConsistLoss, setUseConsistLoss] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fetchRuns = async () => {
    try {
      const data = await api.getTrainingRuns();
      setRuns(data);
      const running = data.find((r) => r.status === 'running');
      if (running) {
        setActiveRun(running);
      } else if (!activeRun && data.length > 0) {
        setActiveRun(data[0]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    const init = async () => {
      try {
        const [ds, dev] = await Promise.all([api.getDatasets(), api.getDeviceInfo()]);
        setDatasets(ds);
        setDevice(dev);
        if (dev.recommended_batch_size) {
          setBatchSize(dev.recommended_batch_size);
        }
      } catch (e) {
        console.error(e);
      }
      fetchRuns();
    };
    init();

    const interval = setInterval(fetchRuns, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async () => {
    setIsStarting(true);
    setErrorMsg(null);
    try {
      const payload = {
        model_architecture: architecture,
        scale_factor: scaleFactor,
        total_epochs: epochs,
        hyperparameters: {
          batch_size: batchSize,
          patch_size: patchSize,
          learning_rate: learningRate,
          optimizer: optimizer,
          use_gradient_loss: useGradLoss,
          use_thermal_consistency_loss: useConsistLoss,
        },
      };
      const created = await api.createTrainingRun(payload);
      setActiveRun(created);
      await fetchRuns();
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || 'Failed to start training run.');
    } finally {
      setIsStarting(false);
    }
  };

  const handleCancel = async (runId: string) => {
    try {
      await api.cancelTrainingRun(runId);
      await fetchRuns();
    } catch (e) {
      console.error(e);
    }
  };

  const chartData = activeRun?.metrics_history?.map((m) => ({
    epoch: m.epoch,
    'Train Loss': m.train_loss,
    'Val Loss': m.val_loss,
    'Val PSNR (dB)': m.psnr,
    'Val SSIM': m.ssim,
  })) || [];

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h2 className="text-lg font-bold text-slate-900 tracking-tight">
          Training Lab
        </h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Train and validate super-resolution models on paired training sets.
        </p>
      </div>

      {/* Main Form & Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Form (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3.5">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-700">
              Training Hyperparameters
            </h3>

            {/* Architecture */}
            <div className="space-y-1 text-xs">
              <label className="text-slate-700 font-medium">Model Architecture</label>
              <select
                value={architecture}
                onChange={(e) => setArchitecture(e.target.value)}
                className="w-full px-3 py-2 rounded border border-slate-300 text-slate-800 text-xs bg-white"
              >
                <option value="reliability_aware_fusion">Model C: Reliability-Aware RGB Guidance (Proposed)</option>
                <option value="rgb_fusion">Model B: Standard RGB-Guided Fusion</option>
                <option value="thermal_cnn_baseline">Model A: Thermal-Only CNN Baseline</option>
              </select>
            </div>

            {/* Scale Factor */}
            <div className="space-y-1 text-xs">
              <label className="text-slate-700 font-medium">Scale Factor</label>
              <div className="grid grid-cols-3 gap-2">
                {[2, 4, 8].map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setScaleFactor(s)}
                    className={`py-1.5 rounded border text-xs font-mono font-medium ${
                      scaleFactor === s
                        ? 'border-blue-600 bg-blue-50 text-blue-700'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    ×{s}
                  </button>
                ))}
              </div>
            </div>

            {/* Epochs & Batch Size */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="space-y-1">
                <label className="text-slate-700 font-medium">Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded border border-slate-300 text-slate-800"
                />
              </div>
              <div className="space-y-1">
                <label className="text-slate-700 font-medium">Batch Size</label>
                <input
                  type="number"
                  min="1"
                  max="64"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded border border-slate-300 text-slate-800"
                />
              </div>
            </div>

            {/* Patch Size & Learning Rate */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="space-y-1">
                <label className="text-slate-700 font-medium">Patch Size</label>
                <select
                  value={patchSize}
                  onChange={(e) => setPatchSize(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded border border-slate-300 text-slate-800 bg-white"
                >
                  <option value={32}>32×32</option>
                  <option value={64}>64×64 (Optimal)</option>
                  <option value={128}>128×128</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-slate-700 font-medium">Learning Rate</label>
                <select
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded border border-slate-300 text-slate-800 bg-white"
                >
                  <option value={0.0002}>2e-4</option>
                  <option value={0.0001}>1e-4</option>
                  <option value={0.00005}>5e-5</option>
                </select>
              </div>
            </div>

            {/* Loss Options */}
            <div className="space-y-1.5 pt-2 border-t border-slate-100 text-xs">
              <label className="text-slate-700 font-medium block">Auxiliary Loss Terms</label>
              <label className="flex items-center gap-2 cursor-pointer text-slate-600">
                <input
                  type="checkbox"
                  checked={useGradLoss}
                  onChange={(e) => setUseGradLoss(e.target.checked)}
                  className="rounded border-slate-300 accent-blue-600"
                />
                Thermal Gradient Sobel Loss
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-slate-600">
                <input
                  type="checkbox"
                  checked={useConsistLoss}
                  onChange={(e) => setUseConsistLoss(e.target.checked)}
                  className="rounded border-slate-300 accent-blue-600"
                />
                Thermal Downsampling Consistency Loss
              </label>
            </div>

            {/* Start Button */}
            <button
              onClick={handleStartTraining}
              disabled={isStarting || activeRun?.status === 'running'}
              className="w-full py-2.5 rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center gap-1.5 transition-colors mt-3"
            >
              {isStarting ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" /> Starting...
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 fill-white" /> Start Training Run
                </>
              )}
            </button>

            {errorMsg && (
              <div className="p-2.5 rounded border border-rose-200 bg-rose-50 text-rose-700 text-xs flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}
          </div>
        </div>

        {/* Live Curves (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {activeRun ? (
            <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200">
                <div>
                  <h3 className="font-semibold text-slate-900 text-sm">
                    {activeRun.model_architecture}
                  </h3>
                  <div className="text-xs text-slate-500 font-mono mt-0.5">
                    Epoch {activeRun.current_epoch} / {activeRun.total_epochs} • Scale ×{activeRun.scale_factor}
                  </div>
                </div>

                {activeRun.status === 'running' && (
                  <button
                    onClick={() => handleCancel(activeRun.id)}
                    className="inline-flex items-center gap-1 px-3 py-1 rounded border border-rose-300 bg-rose-50 text-rose-700 text-xs font-medium hover:bg-rose-100 transition-colors"
                  >
                    <Square className="h-3 w-3 fill-rose-600" /> Cancel Run
                  </button>
                )}
              </div>

              {/* Loss Curve */}
              <div className="space-y-1.5">
                <h4 className="text-xs font-semibold text-slate-700">Loss Trajectory</h4>
                <div className="h-48 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#cbd5e1' }} />
                      <Legend />
                      <Line type="monotone" dataKey="Train Loss" stroke="#2563eb" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="Val Loss" stroke="#d97706" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Validation PSNR Curve */}
              <div className="space-y-1.5 pt-3 border-t border-slate-100">
                <h4 className="text-xs font-semibold text-slate-700">Validation PSNR (dB)</h4>
                <div className="h-44 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', borderColor: '#cbd5e1' }} />
                      <Legend />
                      <Line type="monotone" dataKey="Val PSNR (dB)" stroke="#059669" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Checkpoint Status */}
              {activeRun.best_checkpoint_path && (
                <div className="p-3 rounded border border-emerald-200 bg-emerald-50 text-xs text-emerald-800 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    <span>Best Checkpoint Saved ({activeRun.latest_metrics?.psnr} dB)</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white border border-dashed border-slate-300 rounded-lg p-10 text-center min-h-[340px] flex flex-col items-center justify-center">
              <FlaskConical className="h-6 w-6 text-slate-400 mb-2" />
              <p className="text-xs text-slate-500">
                Configure hyperparameters and click "Start Training Run" to view curves.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
        <h3 className="font-semibold text-slate-800 text-xs uppercase tracking-wider">
          Completed Training Runs
        </h3>
        {runs.length === 0 ? (
          <div className="text-xs text-slate-400 text-center py-4">No completed training runs recorded.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-200 text-slate-500">
                <tr>
                  <th className="pb-2">Architecture</th>
                  <th className="pb-2">Scale</th>
                  <th className="pb-2">Status</th>
                  <th className="pb-2">Best PSNR</th>
                  <th className="pb-2">Elapsed</th>
                  <th className="pb-2">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {runs.map((r) => (
                  <tr key={r.id} className="hover:bg-slate-50">
                    <td className="py-2.5 font-medium text-slate-800">{r.model_architecture}</td>
                    <td className="py-2.5 font-mono text-slate-700">×{r.scale_factor}</td>
                    <td className="py-2.5">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-slate-100 font-medium text-slate-700">
                        {r.status}
                      </span>
                    </td>
                    <td className="py-2.5 font-mono font-medium text-slate-900">
                      {r.latest_metrics?.psnr ? `${r.latest_metrics.psnr} dB` : '—'}
                    </td>
                    <td className="py-2.5 text-slate-500">{r.elapsed_seconds}s</td>
                    <td className="py-2.5">
                      <button
                        onClick={() => setActiveRun(r)}
                        className="text-blue-600 hover:underline font-medium text-xs"
                      >
                        View Curves
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
