import React, { useState } from 'react';
import {
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Info,
  RefreshCw,
  Cpu,
  Clock,
  ArrowRight,
  ShieldAlert,
  HelpCircle,
} from 'lucide-react';
import { layaQAService } from '../services/layaQA';
import {
  QAAssessmentResponse,
  ExperimentRecommendationResponse,
  AlignmentStatus,
} from '../types/layaQA';

interface LayaQAPanelProps {
  runId: string;
  taskMode: string;
  modelArchitecture?: string;
  scaleFactor: number;
  inputDimensions?: [number, number];
  outputDimensions?: [number, number];
  rgbDimensions?: [number, number] | null;
  hasGroundTruth: boolean;
  metrics?: Record<string, any>;
  alignmentStatus?: AlignmentStatus;
}

export const LayaQAPanel: React.FC<LayaQAPanelProps> = ({
  runId,
  taskMode,
  modelArchitecture = 'reliability_aware_fusion',
  scaleFactor,
  inputDimensions = [80, 60],
  outputDimensions = [640, 480],
  rgbDimensions = null,
  hasGroundTruth,
  metrics = {},
  alignmentStatus = 'unverified',
}) => {
  const [loading, setLoading] = useState(false);
  const [assessment, setAssessment] = useState<QAAssessmentResponse | null>(null);
  const [recommendation, setRecommendation] = useState<ExperimentRecommendationResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const runAnalysis = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      // 1. Run Metadata Assessment
      const assessRes = await layaQAService.assessMetadata({
        sample_id: runId,
        experiment_type: taskMode,
        scale_factor: scaleFactor,
        thermal_lr_shape: inputDimensions,
        thermal_hr_shape: outputDimensions,
        rgb_shape: rgbDimensions || undefined,
        alignment_status: alignmentStatus,
        ground_truth_available: hasGroundTruth,
        metrics: metrics,
      });
      setAssessment(assessRes);

      // 2. Run Experiment Advisor
      const recRes = await layaQAService.getRecommendation({
        current_architecture: modelArchitecture,
        available_experiments: ['thermal_baseline', 'rgb_fusion', 'reliability_fusion'],
        scale_factor: scaleFactor,
        alignment_status: alignmentStatus,
        ground_truth_available: hasGroundTruth,
        metrics: metrics,
      });
      setRecommendation(recRes);
    } catch (err: any) {
      console.error('Laya QA analysis failed:', err);
      setErrorMsg(err.response?.data?.detail || err.message || 'Unable to connect to Laya QA Service.');
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'ready_for_review':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
            <CheckCircle2 className="h-3.5 w-3.5" /> Ready for Review
          </span>
        );
      case 'human_review':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200">
            <AlertTriangle className="h-3.5 w-3.5" /> Human Review Required
          </span>
        );
      case 'insufficient_evidence':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-800 border border-slate-300">
            <HelpCircle className="h-3.5 w-3.5" /> Insufficient Evidence
          </span>
        );
      case 'unavailable':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-200">
            <ShieldAlert className="h-3.5 w-3.5" /> Service Unavailable
          </span>
        );
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-4 text-slate-800">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-md bg-blue-50 text-blue-600 border border-blue-200">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-slate-900 tracking-tight">
                ThermoSR · Intelligent QA
              </h3>
              <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase bg-slate-100 text-slate-600 rounded border border-slate-200">
                Experimental
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Optional Laya System 1 add-on for metadata quality assurance & experiment guidance
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={runAnalysis}
          disabled={loading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-blue-600 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white text-xs font-medium transition-colors shadow-sm"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          {assessment ? 'Re-run QA Review' : 'Run Intelligent QA Review'}
        </button>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="p-3 bg-rose-50 border border-rose-200 rounded-md text-xs text-rose-800 flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 text-rose-500 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">QA Service Notice</div>
            <div className="mt-0.5 text-rose-700">{errorMsg}</div>
          </div>
        </div>
      )}

      {/* Content Area */}
      {!assessment && !loading && (
        <div className="py-6 text-center bg-slate-50 border border-dashed border-slate-200 rounded-md">
          <Info className="h-5 w-5 text-slate-400 mx-auto mb-1.5" />
          <p className="text-xs font-medium text-slate-600">
            No QA assessment executed for this reconstruction run yet.
          </p>
          <p className="text-[11px] text-slate-400 mt-0.5 max-w-md mx-auto">
            Click <strong>"Run Intelligent QA Review"</strong> above to inspect metadata completeness, check camera alignment integrity, and review evidence readiness via Laya System 1.
          </p>
        </div>
      )}

      {loading && (
        <div className="py-8 text-center space-y-2">
          <RefreshCw className="h-5 w-5 text-blue-600 animate-spin mx-auto" />
          <p className="text-xs text-slate-600 font-medium">
            Evaluating metadata completeness and running decision checks...
          </p>
        </div>
      )}

      {assessment && !loading && (
        <div className="space-y-4">
          {/* Top Review Status & Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Status Card */}
            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-md space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Evaluation Review Status
              </div>
              <div className="flex items-center justify-between">
                <div>{getStatusBadge(assessment.assessment)}</div>
                {assessment.model_latency_ms !== null && (
                  <div className="flex items-center gap-1 text-[11px] text-slate-500 font-mono">
                    <Clock className="h-3 w-3" />
                    {assessment.model_latency_ms} ms
                  </div>
                )}
              </div>
              <div className="text-[10px] text-slate-500 border-t border-slate-200 pt-1.5">
                Note: Status evaluates metadata reviewability; does not alter reconstruction outputs.
              </div>
            </div>

            {/* Experiment Advisor Card */}
            <div className="p-3.5 bg-blue-50/50 border border-blue-100 rounded-md space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-blue-900 flex items-center justify-between">
                <span>Suggested Next Step</span>
                {recommendation?.executed === false && (
                  <span className="text-[10px] text-blue-600 font-normal">Advisory only</span>
                )}
              </div>
              <div className="font-semibold text-xs text-slate-900 flex items-center gap-1.5">
                <ArrowRight className="h-3.5 w-3.5 text-blue-600 flex-shrink-0" />
                <span className="capitalize">{recommendation?.recommendation.replace('_', ' ')}</span>
              </div>
              <div className="text-[11px] text-slate-600 leading-snug">
                {recommendation?.explanation}
              </div>
            </div>
          </div>

          {/* Metadata Checks Breakdown */}
          <div className="border border-slate-200 rounded-md p-3.5 space-y-2.5">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-600">
              Deterministic Metadata Checks
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              <div className="flex items-center gap-1.5 text-slate-700">
                {assessment.deterministic_checks?.positive_dimensions ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-3.5 w-3.5 text-rose-500" />
                )}
                <span>Required dimensions present ({inputDimensions.join('×')} px)</span>
              </div>

              <div className="flex items-center gap-1.5 text-slate-700">
                {assessment.deterministic_checks?.dimension_scale_consistent ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                )}
                <span>Dimensions match declared scale (×{scaleFactor})</span>
              </div>

              <div className="flex items-center gap-1.5 text-slate-700">
                {assessment.deterministic_checks?.alignment_verified ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                )}
                <span>RGB/thermal alignment: <strong>{alignmentStatus}</strong></span>
              </div>

              <div className="flex items-center gap-1.5 text-slate-700">
                {assessment.deterministic_checks?.ground_truth_available ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                ) : (
                  <Info className="h-3.5 w-3.5 text-slate-400" />
                )}
                <span>Ground-truth thermal reference {hasGroundTruth ? 'available' : 'not uploaded'}</span>
              </div>
            </div>

            {/* Warnings List */}
            {assessment.warnings.length > 0 && (
              <div className="mt-2 pt-2 border-t border-slate-200 space-y-1">
                <div className="text-[11px] font-medium text-amber-700 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  Warnings & Attention Items:
                </div>
                <ul className="list-disc list-inside text-[11px] text-slate-600 space-y-0.5">
                  {assessment.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Missing Evidence List */}
            {assessment.missing_evidence.length > 0 && (
              <div className="mt-1.5 space-y-1">
                <div className="text-[11px] font-medium text-slate-500 flex items-center gap-1">
                  <Info className="h-3 w-3" />
                  Missing Evidence for Review:
                </div>
                <ul className="list-disc list-inside text-[11px] text-slate-500 space-y-0.5">
                  {assessment.missing_evidence.map((me, idx) => (
                    <li key={idx}>{me}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Laya Probabilities (If returned by model) */}
          {assessment.probabilities && Object.keys(assessment.probabilities).length > 0 && (
            <div className="border border-slate-200 rounded-md p-3.5 space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-600 flex items-center justify-between">
                <span>Laya System 1 Decision Probabilities</span>
                <span className="text-[10px] text-slate-400 font-mono">Calibrated Log-Score</span>
              </div>
              <div className="space-y-1.5">
                {Object.entries(assessment.probabilities).map(([label, prob]) => {
                  const pct = Math.round(prob * 100);
                  return (
                    <div key={label} className="space-y-0.5">
                      <div className="flex justify-between text-[11px] text-slate-700">
                        <span className="capitalize">{label.replace(/_/g, ' ')}</span>
                        <span className="font-mono font-medium">{pct}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-600 rounded-full transition-all duration-300"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Disclaimer Footer */}
          <div className="text-[10px] text-slate-400 leading-normal border-t border-slate-100 pt-2 flex items-start gap-1.5">
            <Info className="h-3 w-3 flex-shrink-0 mt-0.5 text-slate-400" />
            <span>
              {assessment.disclaimer} · Recommendation only · Does not start an experiment · Reconstruction remains unchanged.
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
