'use client';

import React from 'react';

interface SeverityGaugeProps {
  severity: number;
  faceDetected: boolean;
}

export default function SeverityGauge({ severity, faceDetected }: SeverityGaugeProps) {
  const getSeverityLabel = (score: number) => {
    if (score < 0.2) return { text: 'Clear / Healthy', color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' };
    if (score < 0.5) return { text: 'Mild Concerns', color: 'text-amber-400 border-amber-500/30 bg-amber-500/10' };
    return { text: 'Needs Attention', color: 'text-red-400 border-red-500/30 bg-red-500/10' };
  };

  const status = faceDetected ? getSeverityLabel(severity) : { text: 'No face detected', color: 'text-slate-400 border-slate-800 bg-slate-900/30' };

  return (
    <div className="flex items-center gap-6">
      {/* Severity Ring */}
      <div className="relative w-24 h-24 flex items-center justify-center rounded-full border border-slate-800 bg-slate-950/50">
        <svg className="absolute w-full h-full -rotate-90">
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke="rgba(30, 41, 59, 0.5)"
            strokeWidth="8"
            fill="transparent"
          />
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke="url(#severityGrad)"
            strokeWidth="8"
            strokeDasharray={2 * Math.PI * 40}
            strokeDashoffset={2 * Math.PI * 40 * (1 - (faceDetected ? severity : 0))}
            strokeLinecap="round"
            fill="transparent"
          />
          <defs>
            <linearGradient id="severityGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#22d3ee" />
              <stop offset="50%" stopColor="#10b981" />
              <stop offset="100%" stopColor="#f59e0b" />
            </linearGradient>
          </defs>
        </svg>
        <div className="flex flex-col items-center justify-center">
          <span className="text-2xl font-black text-white">
            {faceDetected ? Math.round(severity * 100) : 0}%
          </span>
          <span className="text-[9px] uppercase tracking-widest text-slate-500 font-semibold">Severity</span>
        </div>
      </div>

      {/* Quick Details */}
      <div className="flex-1 flex flex-col gap-2">
        <div>
          <span className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">Health Status</span>
          <div className={`mt-1 text-xs font-semibold px-2.5 py-1 rounded-md border inline-block ${status.color}`}>
            {status.text}
          </div>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          Hover or select specific face zones on the scanner canvas to view regional analysis reports.
        </p>
      </div>
    </div>
  );
}
