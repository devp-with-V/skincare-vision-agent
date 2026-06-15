'use client';

import React from 'react';
import { AlertOctagon, Heart, ListChecks, HelpCircle, FileText, Stethoscope } from 'lucide-react';
import RoutineTimeline from './RoutineTimeline';

export interface SkincareStep {
  step_number: number;
  time_of_day: string;
  step_name: string;
  ingredients: string[];
  instructions: string;
}

export interface SkincareRecommendations {
  condition_name: string;
  condition_desc: string;
  overall_summary: string;
  dermatologist_flag: boolean;
  dermatologist_reason: string | null;
  disclaimer: string;
  routine: SkincareStep[];
  lifestyle_tips: string[];
}

interface AnalysisReportProps {
  recommendations: SkincareRecommendations;
}

export default function AnalysisReport({ recommendations }: AnalysisReportProps) {
  return (
    <div className="w-full bg-slate-900/40 border border-slate-800/80 rounded-3xl p-6 md:p-8 backdrop-blur-md flex flex-col gap-8 animate-[fadeIn_0.3s_ease-out]">
      
      {/* Suggested Diagnosis Header */}
      <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 pb-6 border-b border-slate-800/80">
        <div className="flex flex-col gap-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold tracking-widest uppercase bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 self-start">
            <FileText className="w-3.5 h-3.5" />
            AI Advisor Diagnosis
          </div>
          <h2 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
            {recommendations.condition_name}
          </h2>
          <p className="text-slate-400 text-sm leading-relaxed max-w-2xl mt-1">
            {recommendations.condition_desc}
          </p>
        </div>
      </div>

      {/* Dermatologist Alert Banner */}
      {recommendations.dermatologist_flag && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-5 rounded-2xl flex items-start gap-4 shadow-lg shadow-red-950/10 animate-pulse">
          <Stethoscope className="w-6 h-6 shrink-0 mt-0.5 text-red-400" />
          <div className="flex flex-col gap-1">
            <h4 className="text-sm font-bold text-red-300">Dermatologist Referral Advised</h4>
            <p className="text-xs text-red-400/90 leading-relaxed">
              {recommendations.dermatologist_reason}
            </p>
          </div>
        </div>
      )}

      {/* Summary Card */}
      <div className="bg-slate-950/40 border border-slate-800/60 p-5 rounded-2xl flex flex-col gap-2">
        <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-cyan-400" />
          Analysis Summary
        </h3>
        <p className="text-xs text-slate-400 leading-relaxed">
          {recommendations.overall_summary}
        </p>
      </div>

      {/* Skincare Routine Section */}
      <div>
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Heart className="w-5 h-5 text-rose-500" />
          Recommended Routine
        </h3>
        <RoutineTimeline routine={recommendations.routine} />
      </div>

      {/* Lifestyle Tips */}
      {recommendations.lifestyle_tips.length > 0 && (
        <div className="border-t border-slate-800/80 pt-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <ListChecks className="w-5 h-5 text-emerald-400" />
            Healthy Lifestyle Tips
          </h3>
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {recommendations.lifestyle_tips.map((tip, idx) => (
              <li key={idx} className="bg-slate-950/30 border border-slate-900/60 p-3.5 rounded-xl text-xs text-slate-300 leading-relaxed flex items-start gap-2.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0 mt-1.5" />
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Medical Disclaimer Footer */}
      <div className="border-t border-slate-800/80 pt-6 flex items-start gap-3 bg-slate-950/10 p-4 rounded-2xl border border-dashed border-slate-800/60">
        <AlertOctagon className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase font-bold tracking-wider text-amber-500">Medical Waiver</span>
          <p className="text-[10px] text-slate-500 leading-relaxed">
            {recommendations.disclaimer}
          </p>
        </div>
      </div>

    </div>
  );
}
