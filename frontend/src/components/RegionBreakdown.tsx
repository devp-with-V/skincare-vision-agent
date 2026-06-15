'use client';

import React from 'react';
import { RegionAnalysis } from '../lib/api';

interface RegionBreakdownProps {
  regions: Record<string, RegionAnalysis>;
  faceDetected: boolean;
  activeRegionName: string | null;
  selectedRegion: string | null;
  onHoverRegion: (regionName: string | null) => void;
  onSelectRegion: (regionName: string | null) => void;
}

export default function RegionBreakdown({
  regions,
  faceDetected,
  activeRegionName,
  selectedRegion,
  onHoverRegion,
  onSelectRegion
}: RegionBreakdownProps) {
  
  const getSeverityLabel = (score: number) => {
    if (score < 0.2) return { text: 'Clear / Healthy', color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' };
    if (score < 0.5) return { text: 'Mild Concerns', color: 'text-amber-400 border-amber-500/30 bg-amber-500/10' };
    return { text: 'Needs Attention', color: 'text-red-400 border-red-500/30 bg-red-500/10' };
  };

  const activeRegionData = activeRegionName ? regions[activeRegionName] : null;

  return (
    <div className="flex flex-col h-full">
      <div className="grid grid-cols-2 gap-3 mb-6">
        {['forehead', 'left_cheek', 'right_cheek', 'nose', 'chin'].map((reg) => {
          const isHovered = activeRegionName === reg;
          const score = regions[reg]?.severity_score ?? 0;
          
          return (
            <button
              key={reg}
              onMouseEnter={() => onHoverRegion(reg)}
              onMouseLeave={() => onHoverRegion(null)}
              onClick={() => onSelectRegion(selectedRegion === reg ? null : reg)}
              className={`flex flex-col items-start p-3 rounded-2xl border text-left transition-all ${
                isHovered || selectedRegion === reg
                  ? 'bg-slate-800/80 border-cyan-500/50 shadow-md shadow-cyan-950/20' 
                  : 'bg-slate-950/40 border-slate-800 hover:border-slate-700'
              }`}
            >
              <span className="text-xs font-semibold text-slate-400 capitalize">{reg.replace('_', ' ')}</span>
              <div className="flex items-baseline gap-1.5 mt-1">
                <span className="text-lg font-bold text-slate-100">
                  {faceDetected ? Math.round(score * 100) : 0}%
                </span>
                <span className="text-[10px] text-slate-500 font-semibold">severity</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Region details */}
      <div className="border-t border-slate-800/80 pt-6 flex-1">
        {activeRegionData ? (
          <div className="animate-[fadeIn_0.2s_ease-out]">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-bold capitalize text-cyan-400">
                {activeRegionName?.replace('_', ' ')} Analysis
              </h3>
              <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${getSeverityLabel(activeRegionData.severity_score).color}`}>
                {getSeverityLabel(activeRegionData.severity_score).text}
              </span>
            </div>

            <div className="space-y-4">
              <div>
                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">Dominant Concern</span>
                <p className="text-sm font-semibold text-slate-200 mt-0.5 capitalize">
                  {activeRegionData.dominant_concern?.replace('_', ' ') || 'None Detected'}
                </p>
              </div>

              <div>
                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-1.5 block">
                  Detections ({activeRegionData.detections.length})
                </span>
                {activeRegionData.detections.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {activeRegionData.detections.map((det, i) => (
                      <div key={i} className="text-xs bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl px-2.5 py-1.5 font-medium flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 bg-rose-400 rounded-full" />
                        <span className="capitalize">{det.class_name.replace('_', ' ')}</span>
                        <span className="opacity-60">{Math.round(det.confidence * 100)}%</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 italic">No localized lesions or blemishes detected.</p>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 bg-slate-950/20 border border-dashed border-slate-800 rounded-2xl">
            <p className="text-xs text-slate-500 leading-relaxed max-w-[240px]">
              No face region selected. Hover over a region on the mesh overlay or click a card above to inspect findings.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
