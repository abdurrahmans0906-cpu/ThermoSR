import React, { useState, useEffect } from 'react';
import {
  Layers,
  Search,
  FolderOpen,
  FileCheck,
  RefreshCw,
  Image as ImageIcon,
} from 'lucide-react';
import { api } from '../services/api';
import { Dataset, DatasetPair } from '../types';

export const DatasetPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [samples, setSamples] = useState<DatasetPair[]>([]);
  const [rootPath, setRootPath] = useState<string>(
    'c:\\Users\\abdur\\Downloads\\challengedataset - Copy\\challengedataset'
  );
  const [scanning, setScanning] = useState(false);
  const [report, setReport] = useState<any>(null);
  const [selectedSplit, setSelectedSplit] = useState<string>('train');

  const fetchDatasets = async () => {
    try {
      const data = await api.getDatasets();
      setDatasets(data);
      if (data.length > 0 && !selectedDataset) {
        setSelectedDataset(data[0]);
        loadSamples(data[0].id, 'train');
      }
    } catch (e) {
      console.error('Failed to load datasets:', e);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const loadSamples = async (datasetId: string, split: string) => {
    try {
      const s = await api.getDatasetSamples(datasetId, split, 12);
      setSamples(s);
    } catch (e) {
      console.error('Failed to load samples:', e);
    }
  };

  const handleScan = async () => {
    setScanning(true);
    try {
      const ds = await api.scanDataset(rootPath);
      await fetchDatasets();
      setSelectedDataset(ds);
      loadSamples(ds.id, 'train');
      const valReport = await api.validateDataset(ds.id);
      setReport(valReport);
    } catch (err: any) {
      alert(`Scan failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h2 className="text-lg font-bold text-slate-900 tracking-tight">
          Dataset Management
        </h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Scan local directories to index and validate paired thermal and RGB datasets.
        </p>
      </div>

      {/* Scanner Card */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-3">
        <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex items-center gap-2">
          <FolderOpen className="h-4 w-4 text-slate-500" /> Dataset Directory Scanner
        </h3>
        <div className="flex flex-col sm:flex-row gap-2.5">
          <input
            type="text"
            value={rootPath}
            onChange={(e) => setRootPath(e.target.value)}
            placeholder="Directory path containing dataset..."
            className="flex-1 px-3 py-2 rounded border border-slate-300 text-slate-800 text-xs font-mono bg-white"
          />
          <button
            onClick={handleScan}
            disabled={scanning || !rootPath}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center gap-1.5 transition-colors flex-shrink-0"
          >
            {scanning ? (
              <>
                <RefreshCw className="h-3.5 w-3.5 animate-spin" /> Scanning...
              </>
            ) : (
              <>
                <Search className="h-3.5 w-3.5" /> Scan & Validate
              </>
            )}
          </button>
        </div>
      </div>

      {/* Dataset Summary */}
      {selectedDataset ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-lg border border-slate-200 bg-white">
              <span className="text-[11px] font-medium text-slate-500 uppercase">Dataset</span>
              <div className="text-base font-bold text-slate-900 mt-1 truncate">{selectedDataset.name}</div>
              <span className="text-[10px] text-slate-500 font-mono capitalize">
                Type: {selectedDataset.dataset_type}
              </span>
            </div>

            <div className="p-4 rounded-lg border border-slate-200 bg-white">
              <span className="text-[11px] font-medium text-slate-500 uppercase">Total Files</span>
              <div className="text-xl font-bold text-slate-900 font-mono mt-1">
                {selectedDataset.total_pairs.toLocaleString()}
              </div>
              <span className="text-[10px] text-slate-500">Discovered</span>
            </div>

            <div className="p-4 rounded-lg border border-slate-200 bg-white">
              <span className="text-[11px] font-medium text-slate-500 uppercase">Valid Pairs</span>
              <div className="text-xl font-bold text-emerald-600 font-mono mt-1">
                {selectedDataset.valid_pairs.toLocaleString()}
              </div>
              <span className="text-[10px] text-emerald-600 font-medium">Verified Readable</span>
            </div>

            <div className="p-4 rounded-lg border border-slate-200 bg-white">
              <span className="text-[11px] font-medium text-slate-500 uppercase">Split Breakdown</span>
              <div className="text-xs font-mono text-slate-700 mt-1 space-y-0.5">
                {Object.entries(selectedDataset.split_config || {}).map(([s, count]) => (
                  <div key={s} className="flex justify-between">
                    <span className="capitalize text-slate-500">{s}:</span>
                    <span className="font-semibold text-slate-900">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Validation Notice */}
          {report && (
            <div className="p-3.5 rounded-lg border border-blue-200 bg-blue-50 text-xs text-blue-900 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <FileCheck className="h-4 w-4 text-blue-600 flex-shrink-0" />
                <span>{report.summary}</span>
              </div>
              <span className="font-medium text-[11px] text-blue-700">Validated</span>
            </div>
          )}

          {/* Discovered Images Gallery */}
          <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-800 text-xs uppercase tracking-wider">
                Sample Image Pairs
              </h3>
              <div className="flex gap-1 text-xs">
                {['train', 'validation', 'test'].map((split) => (
                  <button
                    key={split}
                    onClick={() => {
                      setSelectedSplit(split);
                      loadSamples(selectedDataset.id, split);
                    }}
                    className={`px-2.5 py-1 rounded capitalize text-xs font-medium border ${
                      selectedSplit === split
                        ? 'border-blue-600 bg-blue-50 text-blue-700'
                        : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {split}
                  </button>
                ))}
              </div>
            </div>

            {samples.length === 0 ? (
              <div className="text-center py-8 text-xs text-slate-400">
                No samples discovered in this split.
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {samples.map((pair) => (
                  <div key={pair.id} className="p-2 rounded border border-slate-200 bg-slate-50 text-[11px] space-y-1">
                    <div className="w-full aspect-[4/3] rounded bg-slate-200 flex items-center justify-center text-slate-500 font-mono text-[10px]">
                      {pair.dimensions?.thermal?.join('×') || 'Thermal'}
                    </div>
                    <div className="font-mono text-slate-700 truncate text-[10px]">
                      {pair.thermal_path.split('\\').pop() || pair.thermal_path}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-white border border-dashed border-slate-300 rounded-lg p-10 text-center">
          <Layers className="h-6 w-6 text-slate-400 mx-auto mb-2" />
          <p className="text-xs text-slate-500">
            Click "Scan & Validate" above to discover images from the dataset folder.
          </p>
        </div>
      )}
    </div>
  );
};
