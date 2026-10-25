import React, { useState, useEffect } from 'react';
import {
  Settings,
  Cpu,
  HardDrive,
  CheckCircle2,
  Save,
} from 'lucide-react';
import { api } from '../services/api';
import { DeviceInfo } from '../types';

export const SettingsPage: React.FC = () => {
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [datasetRoot, setDatasetRoot] = useState(
    'c:\\Users\\abdur\\Downloads\\challengedataset - Copy\\challengedataset'
  );
  const [maxUploadMb, setMaxUploadMb] = useState(50);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const fetchDev = async () => {
      try {
        const d = await api.getDeviceInfo();
        setDevice(d);
      } catch (e) {
        console.error(e);
      }
    };
    fetchDev();
  }, []);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="space-y-6 max-w-3xl">
      {/* Title */}
      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h2 className="text-lg font-bold text-slate-900 tracking-tight">
          Settings & Environment
        </h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Hardware configuration, dataset paths, and upload storage settings.
        </p>
      </div>

      {/* Hardware Inspector */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
        <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex items-center gap-2">
          <Cpu className="h-4 w-4 text-slate-500" /> Compute Hardware
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="p-3 rounded border border-slate-200 bg-slate-50">
            <span className="text-slate-500 block text-[11px]">Active Device</span>
            <span className="text-slate-900 font-semibold mt-0.5 block">
              {device?.device_name || 'CPU'}
            </span>
          </div>

          <div className="p-3 rounded border border-slate-200 bg-slate-50">
            <span className="text-slate-500 block text-[11px]">PyTorch Version</span>
            <span className="font-mono text-slate-900 font-semibold mt-0.5 block">
              {device?.torch_version || '2.x'}
            </span>
          </div>

          {device?.vram_total_mb && (
            <div className="p-3 rounded border border-slate-200 bg-slate-50">
              <span className="text-slate-500 block text-[11px]">VRAM Capacity</span>
              <span className="font-mono text-slate-900 font-semibold mt-0.5 block">
                {device.vram_free_mb} MB free / {device.vram_total_mb} MB total
              </span>
            </div>
          )}

          <div className="p-3 rounded border border-slate-200 bg-slate-50">
            <span className="text-slate-500 block text-[11px]">Recommended Batch Size</span>
            <span className="font-mono text-blue-600 font-semibold mt-0.5 block">
              {device?.recommended_batch_size || 8}
            </span>
          </div>
        </div>
      </div>

      {/* Storage & Paths */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3.5">
        <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex items-center gap-2">
          <HardDrive className="h-4 w-4 text-slate-500" /> Paths & Limits
        </h3>

        <div className="space-y-3 text-xs">
          <div className="space-y-1">
            <label className="text-slate-700 font-medium">Dataset Root Directory</label>
            <input
              type="text"
              value={datasetRoot}
              onChange={(e) => setDatasetRoot(e.target.value)}
              className="w-full px-3 py-2 rounded border border-slate-300 text-slate-800 font-mono text-xs bg-white"
            />
          </div>

          <div className="space-y-1">
            <label className="text-slate-700 font-medium">Max Upload Size (MB)</label>
            <input
              type="number"
              value={maxUploadMb}
              onChange={(e) => setMaxUploadMb(Number(e.target.value))}
              className="w-36 px-3 py-2 rounded border border-slate-300 text-slate-800 text-xs bg-white"
            />
          </div>

          <button
            onClick={handleSave}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 text-white font-medium text-xs flex items-center gap-1.5 transition-colors mt-2"
          >
            {saved ? (
              <>
                <CheckCircle2 className="h-4 w-4" /> Saved!
              </>
            ) : (
              <>
                <Save className="h-4 w-4" /> Save Preferences
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
