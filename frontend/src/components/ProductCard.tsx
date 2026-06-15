'use client';

import React from 'react';
import { Sparkles, Shield, Droplet, Sun, Pill } from 'lucide-react';

interface ProductCardProps {
  ingredient: string;
  stepName: string;
  instructions: string;
}

export default function ProductCard({ ingredient, stepName, instructions }: ProductCardProps) {
  // Determine icon & color based on step category
  const getStepStyle = (name: string) => {
    const lower = name.toLowerCase();
    if (lower.includes('cleanse')) {
      return {
        icon: <Droplet className="w-4 h-4 text-cyan-400" />,
        borderColor: 'border-l-cyan-500/50',
        bgColor: 'bg-cyan-500/5'
      };
    }
    if (lower.includes('treat') || lower.includes('brighten')) {
      return {
        icon: <Pill className="w-4 h-4 text-indigo-400" />,
        borderColor: 'border-l-indigo-500/50',
        bgColor: 'bg-indigo-500/5'
      };
    }
    if (lower.includes('protect') || lower.includes('sun')) {
      return {
        icon: <Sun className="w-4 h-4 text-amber-400" />,
        borderColor: 'border-l-amber-500/50',
        bgColor: 'bg-amber-500/5'
      };
    }
    // Default (Moisturize, etc.)
    return {
      icon: <Shield className="w-4 h-4 text-emerald-400" />,
      borderColor: 'border-l-emerald-500/50',
      bgColor: 'bg-emerald-500/5'
    };
  };

  const style = getStepStyle(stepName);

  return (
    <div className={`p-4 rounded-2xl border border-slate-800/80 border-l-4 ${style.borderColor} ${style.bgColor} backdrop-blur-sm transition-all hover:scale-[1.01] hover:border-slate-700/80 flex flex-col gap-2`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-slate-900 border border-slate-800">
            {style.icon}
          </div>
          <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">{stepName}</span>
        </div>
      </div>
      
      <div>
        <h4 className="text-sm font-bold text-slate-100 flex items-center gap-1.5">
          {ingredient}
        </h4>
        <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
          {instructions}
        </p>
      </div>
    </div>
  );
}
