'use client';

import React, { useState } from 'react';
import { Sun, Moon } from 'lucide-react';
import ProductCard from './ProductCard';

interface SkincareStep {
  step_number: int;
  time_of_day: string;
  step_name: string;
  ingredients: string[];
  instructions: string;
}

interface RoutineTimelineProps {
  routine: SkincareStep[];
}

export default function RoutineTimeline({ routine }: RoutineTimelineProps) {
  const [activeTab, setActiveTab] = useState<'both' | 'AM' | 'PM'>('both');

  const amSteps = routine.filter(s => s.time_of_day === 'AM').sort((a, b) => a.step_number - b.step_number);
  const pmSteps = routine.filter(s => s.time_of_day === 'PM').sort((a, b) => a.step_number - b.step_number);

  return (
    <div className="flex flex-col gap-6">
      {/* Tab Selectors */}
      <div className="flex bg-slate-900/60 border border-slate-800 p-1.5 rounded-2xl max-w-sm self-center md:self-start w-full">
        {(['both', 'AM', 'PM'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-2 rounded-xl text-xs font-bold transition-all uppercase tracking-wider ${
              activeTab === tab
                ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/50'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {tab === 'both' ? 'Full Day' : tab}
          </button>
        ))}
      </div>

      {/* Timeline Columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
        
        {/* AM Column */}
        {(activeTab === 'both' || activeTab === 'AM') && (
          <div className="flex flex-col gap-4 bg-slate-950/20 border border-slate-900 p-5 rounded-3xl backdrop-blur-md">
            <h3 className="text-base font-black text-amber-400 flex items-center gap-2 pb-3 border-b border-slate-900">
              <Sun className="w-5 h-5 animate-spin-slow" />
              Morning (AM) Routine
            </h3>
            
            {amSteps.length > 0 ? (
              <div className="flex flex-col gap-4 mt-2">
                {amSteps.map((step, idx) => (
                  <ProductCard
                    key={idx}
                    stepName={`${step.step_number}. ${step.step_name}`}
                    ingredient={step.ingredients.join(', ')}
                    instructions={step.instructions}
                  />
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic p-4 text-center">No AM routine steps recommended.</p>
            )}
          </div>
        )}

        {/* PM Column */}
        {(activeTab === 'both' || activeTab === 'PM') && (
          <div className="flex flex-col gap-4 bg-slate-950/20 border border-slate-900 p-5 rounded-3xl backdrop-blur-md">
            <h3 className="text-base font-black text-indigo-400 flex items-center gap-2 pb-3 border-b border-slate-900">
              <Moon className="w-5 h-5 animate-pulse" />
              Night (PM) Routine
            </h3>

            {pmSteps.length > 0 ? (
              <div className="flex flex-col gap-4 mt-2">
                {pmSteps.map((step, idx) => (
                  <ProductCard
                    key={idx}
                    stepName={`${step.step_number}. ${step.step_name}`}
                    ingredient={step.ingredients.join(', ')}
                    instructions={step.instructions}
                  />
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic p-4 text-center">No PM routine steps recommended.</p>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
