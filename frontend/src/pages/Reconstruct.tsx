import React, { useState, useEffect } from 'react';
import {
  Flame,
  Upload,
  CheckCircle2,
  AlertTriangle,
  Info,
  Download,
  RefreshCw,
  Sparkles,
  Columns,
  Image as ImageIcon,
} from 'lucide-react';
import { api } from '../services/api';
import { ModelVersion, ReconstructionRun } from '../types';

export const Reconstruct: React.FC = () => {
  // Step State (1: Mode, 2: Upload, 3: Settings, 4: Results)
  const [taskMode, setTaskMode] = useState<'rgb_guided' | 'evaluation' | 'downsample_eval'>('rgb_guided');
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [selectedModelArch, setSelectedModelArch] = useState<string>('reliability_aware_fusion');
  const [selectedScale, setSelectedScale] = useState<number>(8);
  const [selectedColormap, setSelectedColormap] = useState<string>('inferno');
  const [selectedDevice, setSelectedDevice] = useState<string>('auto');

  // File Upload States
  const [thermalFile, setThermalFile] = useState<File | null>(null);
  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);

  // Previews
  const [thermalPreview, setThermalPreview] = useState<string | null>(null);
  const [rgbPreview, setRgbPreview] = useState<string | null>(null);
  const [referencePreview, setReferencePreview] = useState<string | null>(null);

  // Dimensions & metadata
  const [thermalDims, setThermalDims] = useState<[number, number] | null>(null);
  const [rgbDims, setRgbDims] = useState<[number, number] | null>(null);

  // Execution & Job Tracking
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobStatus, setJobStatus] = useState<string>('idle');
  const [progressPct, setProgressPct] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<ReconstructionRun | null>(null);

  // Viewer Options: 'after' | 'before' | 'side_by_side'
  const [displayView, setDisplayView] = useState<'after' | 'before' | 'side_by_side'>('after');
  const [viewMode, setViewMode] = useState<'reconstructed' | 'false_color' | 'reliability' | 'error'>('false_color');

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const mv = await api.getModelRegistry();
        setModels(mv);
      } catch (err) {
        console.error('Failed to load models:', err);
      }
    };
    fetchModels();
  }, []);

  const handleFileChange = (file: File, type: 'thermal' | 'rgb' | 'reference') => {
    const previewUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      if (type === 'thermal') {
        setThermalFile(file);
        setThermalPreview(previewUrl);
        setThermalDims([img.width, img.height]);
      } else if (type === 'rgb') {
        setRgbFile(file);
        setRgbPreview(previewUrl);
        setRgbDims([img.width, img.height]);
      } else if (type === 'reference') {
        setReferenceFile(file);
        setReferencePreview(previewUrl);
      }
    };
    img.src = previewUrl;
  };

  const loadDemoPair = async () => {
    try {
      const tRes = await fetch('/api/artifacts/../data/sample_pair/sample_thermal_lr_x8.png');
      const rRes = await fetch('/api/artifacts/../data/sample_pair/sample_rgb_guidance.png');
      const gtRes = await fetch('/api/artifacts/../data/sample_pair/sample_thermal_hr.jpg');

      if (tRes.ok && rRes.ok) {
        const tBlob = await tRes.blob();
        const rBlob = await rRes.blob();
        const tFile = new File([tBlob], 'sample_thermal_lr_80x60.png', { type: 'image/png' });
        const rFile = new File([rBlob], 'sample_rgb_guidance_640x480.png', { type: 'image/png' });
        handleFileChange(tFile, 'thermal');
        handleFileChange(rFile, 'rgb');

        if (gtRes.ok) {
          const gtBlob = await gtRes.blob();
          const gtFile = new File([gtBlob], 'sample_thermal_hr_reference.jpg', { type: 'image/jpeg' });
          handleFileChange(gtFile, 'reference');
          setTaskMode('evaluation');
        }
      }
    } catch (e) {
      console.warn('Demo pair error:', e);
    }
  };

  const handleReconstruct = async () => {
    if (!thermalFile) {
      setErrorMsg('Please upload a thermal input image.');
      return;
    }
    if ((selectedModelArch === 'rgb_fusion' || selectedModelArch === 'reliability_aware_fusion') && !rgbFile) {
      setErrorMsg('RGB guidance image is required for RGB-guided super-resolution.');
      return;
    }

    setErrorMsg(null);
    setIsProcessing(true);
    setJobStatus('Submitting...');
    setProgressPct(20);

    try {
      const formData = new FormData();
      formData.append('task_mode', taskMode);
      formData.append('model_architecture', selectedModelArch);
      formData.append('scale_factor', selectedScale.toString());
      formData.append('device', selectedDevice);
      formData.append('colormap', selectedColormap);
      formData.append('thermal_file', thermalFile);
      if (rgbFile) formData.append('rgb_file', rgbFile);
      if (referenceFile) formData.append('reference_file', referenceFile);

      setJobStatus('Running inference...');
      setProgressPct(60);

      const runData = await api.createReconstruction(formData);
      setResult(runData);
      setDisplayView('after');
      setProgressPct(100);
      setJobStatus('Completed');
    } catch (err: any) {
      setErrorMsg(err.response?.data?.detail || 'Reconstruction failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const getActiveOutputUrl = () => {
    if (!result?.output_artifacts) return '';
    if (viewMode === 'false_color') return api.getArtifactUrl(result.output_artifacts.false_color);
    if (viewMode === 'reliability') return api.getArtifactUrl(result.output_artifacts.reliability_map);
    if (viewMode === 'error') return api.getArtifactUrl(result.output_artifacts.error_map);
    return api.getArtifactUrl(result.output_artifacts.reconstructed);
  };

  return (
    <div className="space-y-6">
      {/* Title Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white border border-slate-200 rounded-lg p-5">
        <div>
          <h2 className="text-lg font-bold text-slate-900 tracking-tight">
            Thermal Super-Resolution Pipeline
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Select inputs, configure model architecture, and run reconstruction.
          </p>
        </div>

        <button
          onClick={loadDemoPair}
          type="button"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-blue-200 bg-blue-50 text-blue-700 text-xs font-medium hover:bg-blue-100 transition-colors self-start sm:self-auto"
        >
          <Sparkles className="h-3.5 w-3.5 text-blue-600" /> Load Demo Pair
        </button>
      </div>

      {/* Main 2-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Form & Inputs (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Step 1: Mode Selection */}
          <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 block">
              1. Task Mode
            </span>
            <div className="space-y-2">
              {[
                {
                  id: 'rgb_guided',
                  title: 'A. RGB-Guided Reconstruction',
                  desc: 'Input LR Thermal + HR RGB guidance to reconstruct thermal image.',
                },
                {
                  id: 'evaluation',
                  title: 'B. Evaluation with Reference',
                  desc: 'Reconstruct and calculate PSNR, SSIM & LPIPS against ground-truth.',
                },
                {
                  id: 'downsample_eval',
                  title: 'C. Downsample-and-Eval',
                  desc: 'Downsample HR Thermal to LR, reconstruct, and verify metrics.',
                },
              ].map((m) => (
                <label
                  key={m.id}
                  className={`flex items-start gap-2.5 p-2.5 rounded-md border text-xs cursor-pointer transition-colors ${
                    taskMode === m.id
                      ? 'border-blue-500 bg-blue-50/50 text-slate-900'
                      : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <input
                    type="radio"
                    name="task_mode"
                    value={m.id}
                    checked={taskMode === m.id}
                    onChange={() => setTaskMode(m.id as any)}
                    className="mt-0.5 accent-blue-600"
                  />
                  <div>
                    <div className="font-medium text-slate-800">{m.title}</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">{m.desc}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Step 2: Upload Inputs */}
          <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 block">
              2. Upload Images
            </span>

            {/* Thermal Upload */}
            <div className="border border-slate-300 rounded-md p-3 text-center relative hover:bg-slate-50 transition-colors">
              <input
                type="file"
                accept=".jpg,.jpeg,.png,.tif,.tiff"
                onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0], 'thermal')}
                className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
              />
              {thermalPreview ? (
                <div className="flex items-center gap-3 text-left">
                  <img src={thermalPreview} alt="Thermal" className="h-12 w-12 object-cover rounded border border-slate-200" />
                  <div className="flex-1 min-w-0 text-xs">
                    <div className="font-medium text-slate-800 truncate">{thermalFile?.name}</div>
                    <div className="text-[11px] text-slate-500 font-mono">
                      {thermalDims ? `${thermalDims[0]}×${thermalDims[1]} px` : 'Uploaded'}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-2">
                  <Upload className="h-4 w-4 text-slate-400 mx-auto mb-1" />
                  <div className="text-xs font-medium text-slate-700">
                    {taskMode === 'downsample_eval' ? 'Select HR Thermal Image' : 'Select Low-Res Thermal Image'}
                  </div>
                  <div className="text-[10px] text-slate-400">PNG, JPEG, or 16-bit TIFF</div>
                </div>
              )}
            </div>

            {/* RGB Upload */}
            {selectedModelArch !== 'thermal_cnn_baseline' && (
              <div className="border border-slate-300 rounded-md p-3 text-center relative hover:bg-slate-50 transition-colors">
                <input
                  type="file"
                  accept=".jpg,.jpeg,.png"
                  onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0], 'rgb')}
                  className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                />
                {rgbPreview ? (
                  <div className="flex items-center gap-3 text-left">
                    <img src={rgbPreview} alt="RGB" className="h-12 w-12 object-cover rounded border border-slate-200" />
                    <div className="flex-1 min-w-0 text-xs">
                      <div className="font-medium text-slate-800 truncate">{rgbFile?.name}</div>
                      <div className="text-[11px] text-slate-500 font-mono">
                        {rgbDims ? `${rgbDims[0]}×${rgbDims[1]} px` : 'Uploaded'}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-2">
                    <Upload className="h-4 w-4 text-slate-400 mx-auto mb-1" />
                    <div className="text-xs font-medium text-slate-700">Select High-Res RGB Guidance Image</div>
                    <div className="text-[10px] text-slate-400">Optical RGB image of the scene</div>
                  </div>
                )}
              </div>
            )}

            {/* Reference Upload */}
            {taskMode === 'evaluation' && (
              <div className="border border-slate-300 rounded-md p-3 text-center relative hover:bg-slate-50 transition-colors">
                <input
                  type="file"
                  accept=".jpg,.jpeg,.png,.tif,.tiff"
                  onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0], 'reference')}
                  className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                />
                {referencePreview ? (
                  <div className="flex items-center gap-3 text-left">
                    <img src={referencePreview} alt="Ref" className="h-12 w-12 object-cover rounded border border-slate-200" />
                    <div className="flex-1 min-w-0 text-xs font-medium text-slate-800 truncate">
                      {referenceFile?.name}
                    </div>
                  </div>
                ) : (
                  <div className="py-2">
                    <Upload className="h-4 w-4 text-slate-400 mx-auto mb-1" />
                    <div className="text-xs font-medium text-slate-700">Select Ground Truth Reference (Optional)</div>
                    <div className="text-[10px] text-slate-400">Used for PSNR / SSIM / LPIPS</div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Step 3: Model & Settings */}
          <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 block">
              3. Model Settings
            </span>

            {/* Architecture Selector */}
            <div className="space-y-1 text-xs">
              <label className="text-slate-700 font-medium">Model Architecture</label>
              <select
                value={selectedModelArch}
                onChange={(e) => setSelectedModelArch(e.target.value)}
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
                    onClick={() => setSelectedScale(s)}
                    className={`py-1.5 rounded border text-xs font-mono font-medium ${
                      selectedScale === s
                        ? 'border-blue-600 bg-blue-50 text-blue-700'
                        : 'border-slate-300 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    ×{s}
                  </button>
                ))}
              </div>
            </div>

            {/* Colormap */}
            <div className="space-y-1 text-xs">
              <label className="text-slate-700 font-medium">Thermal Visualization Palette</label>
              <div className="grid grid-cols-4 gap-1.5">
                {['inferno', 'viridis', 'magma', 'jet'].map((cm) => (
                  <button
                    key={cm}
                    type="button"
                    onClick={() => setSelectedColormap(cm)}
                    className={`py-1 rounded text-xs capitalize border ${
                      selectedColormap === cm
                        ? 'border-blue-600 bg-blue-50 text-blue-700 font-medium'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {cm}
                  </button>
                ))}
              </div>
            </div>

            {/* Submit Button */}
            <button
              onClick={handleReconstruct}
              disabled={isProcessing || !thermalFile}
              className="w-full mt-2 py-2.5 rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center gap-2 transition-colors"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" /> {jobStatus}
                </>
              ) : (
                <>
                  <Flame className="h-4 w-4" /> Reconstruct Thermal Image
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

        {/* Right Column: Results & Image Viewer (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {result ? (
            <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
              {/* Output Header */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200">
                <div>
                  <h3 className="font-semibold text-slate-900 text-sm">Super-Resolution Output</h3>
                  <p className="text-xs text-slate-500 font-mono">
                    Scale: ×{result.scale_factor} • Time: {result.inference_time_ms} ms
                  </p>
                </div>

                {/* Colormap / Channel Modes */}
                <div className="flex items-center gap-1 text-xs">
                  <button
                    onClick={() => setViewMode('false_color')}
                    className={`px-2.5 py-1 rounded border text-xs ${
                      viewMode === 'false_color'
                        ? 'border-blue-600 bg-blue-50 text-blue-700 font-medium'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    False Color
                  </button>
                  <button
                    onClick={() => setViewMode('reconstructed')}
                    className={`px-2.5 py-1 rounded border text-xs ${
                      viewMode === 'reconstructed'
                        ? 'border-blue-600 bg-blue-50 text-blue-700 font-medium'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    Grayscale
                  </button>
                  {result.output_artifacts?.reliability_map && (
                    <button
                      onClick={() => setViewMode('reliability')}
                      className={`px-2.5 py-1 rounded border text-xs ${
                        viewMode === 'reliability'
                          ? 'border-indigo-600 bg-indigo-50 text-indigo-700 font-medium'
                          : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      Reliability Map
                    </button>
                  )}
                  {result.output_artifacts?.error_map && (
                    <button
                      onClick={() => setViewMode('error')}
                      className={`px-2.5 py-1 rounded border text-xs ${
                        viewMode === 'error'
                          ? 'border-amber-600 bg-amber-50 text-amber-700 font-medium'
                          : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      Error Heatmap
                    </button>
                  )}
                </div>
              </div>

              {/* View Control Buttons: Before / After / Side-by-Side */}
              <div className="flex items-center justify-between bg-slate-50 p-2 rounded border border-slate-200 text-xs">
                <span className="font-medium text-slate-700">Display View:</span>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setDisplayView('before')}
                    className={`px-3 py-1 rounded border text-xs font-medium transition-colors ${
                      displayView === 'before'
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    Before (LR Input)
                  </button>
                  <button
                    type="button"
                    onClick={() => setDisplayView('after')}
                    className={`px-3 py-1 rounded border text-xs font-medium transition-colors ${
                      displayView === 'after'
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    After (Reconstructed)
                  </button>
                  <button
                    type="button"
                    onClick={() => setDisplayView('side_by_side')}
                    className={`px-3 py-1 rounded border text-xs font-medium transition-colors flex items-center gap-1 ${
                      displayView === 'side_by_side'
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    <Columns className="h-3 w-3" /> Side-by-Side
                  </button>
                  <button
                    type="button"
                    onClick={() => setDisplayView(displayView === 'before' ? 'after' : 'before')}
                    className="ml-2 px-2.5 py-1 rounded border border-slate-300 bg-white hover:bg-slate-100 text-slate-600 text-xs"
                    title="Quick Flip"
                  >
                    Flip ⇄
                  </button>
                </div>
              </div>

              {/* Image Display Area */}
              {displayView === 'side_by_side' ? (
                /* Side-by-Side View */
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <div className="text-[11px] font-medium text-slate-600">
                      Before (Low-Res Input: {result.metrics?.input_dimensions?.join('×')} px)
                    </div>
                    <div className="aspect-[4/3] bg-slate-900 rounded border border-slate-200 overflow-hidden flex items-center justify-center">
                      {thermalPreview && (
                        <img
                          src={thermalPreview}
                          alt="LR Input"
                          className="w-full h-full object-contain filter blur-[0.5px]"
                        />
                      )}
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="text-[11px] font-medium text-slate-600">
                      After (Super-Resolved: {result.metrics?.output_dimensions?.join('×')} px)
                    </div>
                    <div className="aspect-[4/3] bg-slate-900 rounded border border-slate-200 overflow-hidden flex items-center justify-center">
                      <img
                        src={getActiveOutputUrl()}
                        alt="Reconstructed Output"
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                /* Single Image View (Before OR After) */
                <div className="space-y-1">
                  <div className="text-[11px] font-medium text-slate-600 flex justify-between">
                    <span>
                      {displayView === 'before'
                        ? `Before: Low-Resolution Thermal Input (${result.metrics?.input_dimensions?.join('×')} px)`
                        : `After: Super-Resolved Thermal Image (${result.metrics?.output_dimensions?.join('×')} px)`}
                    </span>
                    <span className="text-slate-400">
                      {displayView === 'before' ? 'Bicubic interpolated' : 'Model reconstruction'}
                    </span>
                  </div>
                  <div className="aspect-[4/3] bg-slate-900 rounded border border-slate-200 overflow-hidden flex items-center justify-center">
                    {displayView === 'before' ? (
                      thermalPreview && (
                        <img
                          src={thermalPreview}
                          alt="Before"
                          className="w-full h-full object-contain filter blur-[0.5px]"
                        />
                      )
                    ) : (
                      <img
                        src={getActiveOutputUrl()}
                        alt="After"
                        className="w-full h-full object-contain"
                      />
                    )}
                  </div>
                </div>
              )}

              {/* Quantitative Metrics Table */}
              <div className="pt-2 border-t border-slate-200 space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-600 block">
                  Quantitative Metrics
                </span>

                {result.metrics?.has_ground_truth ? (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
                    <div className="p-2.5 rounded border border-slate-200 bg-slate-50 text-center">
                      <div className="text-slate-500 text-[11px]">PSNR</div>
                      <div className="text-base font-bold font-mono text-slate-900 mt-0.5">
                        {result.metrics.psnr} dB
                      </div>
                      <div className="text-[10px] text-slate-400">Higher is better ↑</div>
                    </div>

                    <div className="p-2.5 rounded border border-slate-200 bg-slate-50 text-center">
                      <div className="text-slate-500 text-[11px]">SSIM</div>
                      <div className="text-base font-bold font-mono text-slate-900 mt-0.5">
                        {result.metrics.ssim}
                      </div>
                      <div className="text-[10px] text-slate-400">Higher is better ↑</div>
                    </div>

                    <div className="p-2.5 rounded border border-slate-200 bg-slate-50 text-center">
                      <div className="text-slate-500 text-[11px]">LPIPS</div>
                      <div className="text-base font-bold font-mono text-slate-900 mt-0.5">
                        {result.metrics.lpips !== null ? result.metrics.lpips : '—'}
                      </div>
                      <div className="text-[10px] text-slate-400">Lower is better ↓</div>
                    </div>

                    <div className="p-2.5 rounded border border-slate-200 bg-slate-50 text-center">
                      <div className="text-slate-500 text-[11px]">Flux Error L1</div>
                      <div className="text-base font-bold font-mono text-slate-900 mt-0.5">
                        {result.metrics.thermal_consistency_l1 || '—'}
                      </div>
                      <div className="text-[10px] text-slate-400">Cycle consistency</div>
                    </div>
                  </div>
                ) : (
                  <div className="p-3 rounded border border-slate-200 bg-slate-50 text-xs text-slate-600 flex items-center gap-2">
                    <Info className="h-4 w-4 text-slate-400 flex-shrink-0" />
                    <span>
                      Inference completed. Quantitative reference metrics (PSNR, SSIM, LPIPS) require a ground truth reference image.
                    </span>
                  </div>
                )}
              </div>

              {/* Download Bar */}
              <div className="pt-2 border-t border-slate-200 flex flex-wrap gap-2">
                {result.output_artifacts?.reconstructed && (
                  <a
                    href={api.getArtifactUrl(result.output_artifacts.reconstructed)}
                    download="reconstructed_thermal.png"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium transition-colors"
                  >
                    <Download className="h-3.5 w-3.5" /> Download Thermal PNG
                  </a>
                )}
                {result.output_artifacts?.false_color && (
                  <a
                    href={api.getArtifactUrl(result.output_artifacts.false_color)}
                    download="reconstructed_false_color.png"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium transition-colors"
                  >
                    <Download className="h-3.5 w-3.5" /> Download False Color
                  </a>
                )}
                {result.output_artifacts?.reliability_map && (
                  <a
                    href={api.getArtifactUrl(result.output_artifacts.reliability_map)}
                    download="reliability_heatmap.png"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-indigo-200 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-medium transition-colors"
                  >
                    <Download className="h-3.5 w-3.5" /> Reliability Heatmap
                  </a>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-white border border-dashed border-slate-300 rounded-lg p-12 text-center min-h-[380px] flex flex-col items-center justify-center">
              <ImageIcon className="h-8 w-8 text-slate-400 mb-2" />
              <h3 className="font-semibold text-slate-700 text-sm">No Active Reconstruction</h3>
              <p className="text-xs text-slate-500 max-w-sm mt-1">
                Upload thermal and RGB images or click "Load Demo Pair" above to view results here.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
