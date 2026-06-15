'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Camera, Sparkles, RefreshCw, AlertTriangle, CheckCircle, Zap } from 'lucide-react';
import FaceMeshOverlay from '../../components/FaceMeshOverlay';
import { SkinWebSocketClient, scanImage, AnalysisResult, RegionAnalysis } from '../../lib/api';

const WEBCAM_WIDTH = 640;
const WEBCAM_HEIGHT = 480;

export default function ScanPage() {
  const webcamRef = useRef<Webcam>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const wsClientRef = useRef<SkinWebSocketClient | null>(null);

  // Connection & Scanner state
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState(0);

  // Analysis results state
  const [analysis, setAnalysis] = useState<AnalysisResult>({
    face_detected: false,
    landmarks: [],
    regions: {},
    overall_severity: 0.0
  });
  
  // Interactive UI state
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);

  // WebSocket frame sending loop
  useEffect(() => {
    let frameInterval: NodeJS.Timeout;

    if (isScanning && isWsConnected && wsClientRef.current) {
      frameInterval = setInterval(() => {
        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();
          if (imageSrc) {
            wsClientRef.current?.sendFrame(imageSrc);
          }
        }
      }, 150); // ~6-7 FPS is perfect for tracking without flooding the server
    }

    return () => {
      if (frameInterval) clearInterval(frameInterval);
    };
  }, [isScanning, isWsConnected]);

  // Connect WebSocket on mount
  useEffect(() => {
    const ws = new SkinWebSocketClient(
      (data) => {
        // Handle incoming frame analysis results
        if (data.error) {
          setErrorMsg(data.error);
          return;
        }
        setAnalysis({
          face_detected: data.face_detected,
          landmarks: data.landmarks || [],
          regions: data.regions || {},
          overall_severity: data.overall_severity || 0
        });
        setErrorMsg(null);
      },
      {
        onConnect: () => {
          setIsWsConnected(true);
          setErrorMsg(null);
        },
        onClose: () => {
          setIsWsConnected(false);
        },
        onError: (err) => {
          setErrorMsg('WebSocket connection error. Check backend server.');
        }
      }
    );

    ws.connect();
    wsClientRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  // Run a high-quality single scan (REST request)
  const handlePerformAnalysis = async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      setErrorMsg('Could not capture webcam frame.');
      return;
    }

    setIsScanning(false); // Stop real-time ws scan
    setErrorMsg(null);
    setScanProgress(10);

    const interval = setInterval(() => {
      setScanProgress(p => (p >= 90 ? 90 : p + 15));
    }, 100);

    try {
      const response = await scanImage(imageSrc);
      clearInterval(interval);
      setScanProgress(100);
      
      setAnalysis(response.analysis);

      setTimeout(() => {
        setScanProgress(0);
      }, 600);
    } catch (err: any) {
      clearInterval(interval);
      setScanProgress(0);
      setErrorMsg(err.message || 'Scan failed.');
    }
  };

  const activeRegionName = hoveredRegion || selectedRegion;
  const activeRegionData = activeRegionName ? analysis.regions[activeRegionName] : null;

  const getSeverityLabel = (score: number) => {
    if (score < 0.2) return { text: 'Clear / Healthy', color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' };
    if (score < 0.5) return { text: 'Mild Concerns', color: 'text-amber-400 border-amber-500/30 bg-amber-500/10' };
    return { text: 'Needs Attention', color: 'text-red-400 border-red-500/30 bg-red-500/10' };
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center p-6 md:p-12 selection:bg-cyan-500/30">
      {/* Header */}
      <header className="w-full max-w-6xl mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 via-emerald-400 to-indigo-500 bg-clip-text text-transparent flex items-center gap-2">
            <Sparkles className="w-8 h-8 text-cyan-400 animate-pulse" />
            SkinCare Vision Agent
          </h1>
          <p className="text-slate-400 text-sm mt-1">Real-time face mesh analysis & skin condition detection</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${
            isWsConnected 
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' 
              : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isWsConnected ? 'bg-emerald-400 animate-ping' : 'bg-rose-400'}`} />
            {isWsConnected ? 'Backend Connected' : 'Backend Disconnected'}
          </div>
        </div>
      </header>

      {/* Main Scan Area */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column - Camera Stream */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div 
            ref={containerRef}
            className="relative w-full aspect-[4/3] rounded-3xl overflow-hidden border border-slate-800 bg-slate-900/40 shadow-2xl shadow-cyan-900/10 flex items-center justify-center group"
          >
            {/* Webcam Feed */}
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={{
                width: WEBCAM_WIDTH,
                height: WEBCAM_HEIGHT,
                facingMode: 'user'
              }}
              className="absolute top-0 left-0 w-full h-full object-cover scale-x-[-1]"
            />

            {/* Mesh Overlay Canvas */}
            <FaceMeshOverlay
              landmarks={analysis.landmarks}
              regions={analysis.regions}
              width={WEBCAM_WIDTH}
              height={WEBCAM_HEIGHT}
              activeRegion={activeRegionName}
              onHoverRegion={setHoveredRegion}
            />

            {/* Scanning Line Indicator */}
            {isScanning && analysis.face_detected && (
              <div className="absolute left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_10px_rgba(34,211,238,0.5)] animate-[scan_2s_ease-in-out_infinite] pointer-events-none z-20" />
            )}

            {/* Loading / Scan Overlay progress */}
            {scanProgress > 0 && (
              <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm z-30 flex flex-col items-center justify-center gap-4">
                <RefreshCw className="w-10 h-10 text-cyan-400 animate-spin" />
                <div className="w-48 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-gradient-to-r from-cyan-400 to-emerald-400 h-full transition-all duration-150" style={{ width: `${scanProgress}%` }} />
                </div>
                <span className="text-xs text-cyan-400 font-mono tracking-widest uppercase">Analyzing Face ROI...</span>
              </div>
            )}

            {/* Face Unidentified Alert */}
            {!analysis.face_detected && isScanning && (
              <div className="absolute top-4 left-4 bg-slate-950/80 backdrop-blur-md px-3 py-2 rounded-xl border border-amber-500/30 flex items-center gap-2 text-amber-400 text-xs animate-pulse">
                <AlertTriangle className="w-4 h-4" />
                Align face within the frame
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap gap-4">
            <button
              onClick={() => setIsScanning(!isScanning)}
              disabled={!isWsConnected}
              className={`flex-1 flex items-center justify-center gap-2 px-6 py-3.5 rounded-2xl font-bold tracking-wide transition-all shadow-lg ${
                isScanning
                  ? 'bg-rose-600 hover:bg-rose-700 shadow-rose-900/20 text-white'
                  : 'bg-cyan-500 hover:bg-cyan-600 shadow-cyan-900/30 text-slate-950 disabled:opacity-50'
              }`}
            >
              <Zap className={`w-5 h-5 ${isScanning ? 'fill-white animate-bounce' : 'fill-slate-950'}`} />
              {isScanning ? 'Stop Live Scan' : 'Start Live Tracking'}
            </button>

            <button
              onClick={handlePerformAnalysis}
              className="flex-1 flex items-center justify-center gap-2 px-6 py-3.5 rounded-2xl bg-slate-800 hover:bg-slate-700 font-bold tracking-wide text-slate-100 border border-slate-700 hover:border-slate-600 transition-all shadow-md"
            >
              <Camera className="w-5 h-5" />
              Capture & Analyze
            </button>
          </div>
        </div>

        {/* Right Column - Analysis Dashboard */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {errorMsg && (
            <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-2xl text-sm flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Score & General Stats Card */}
          <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-6 backdrop-blur-md">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-cyan-400" />
              General Analysis
            </h2>
            
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
                    strokeDashoffset={2 * Math.PI * 40 * (1 - (analysis.face_detected ? analysis.overall_severity : 0))}
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
                    {analysis.face_detected ? Math.round(analysis.overall_severity * 100) : 0}%
                  </span>
                  <span className="text-[9px] uppercase tracking-widest text-slate-500 font-semibold">Severity</span>
                </div>
              </div>

              {/* Quick Details */}
              <div className="flex-1 flex flex-col gap-2">
                <div>
                  <span className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">Health Status</span>
                  <div className={`mt-1 text-xs font-semibold px-2.5 py-1 rounded-md border inline-block ${
                    analysis.face_detected 
                      ? getSeverityLabel(analysis.overall_severity).color 
                      : 'text-slate-400 border-slate-800 bg-slate-900/30'
                  }`}>
                    {analysis.face_detected ? getSeverityLabel(analysis.overall_severity).text : 'No face detected'}
                  </div>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Hover or select specific face zones on the scanner canvas to view regional analysis reports.
                </p>
              </div>
            </div>
          </div>

          {/* Regional Breakdown Dashboard */}
          <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-6 backdrop-blur-md flex-1">
            <h2 className="text-xl font-bold mb-4">Face Regions</h2>
            
            <div className="grid grid-cols-2 gap-3 mb-6">
              {['forehead', 'left_cheek', 'right_cheek', 'nose', 'chin'].map((reg) => {
                const isHovered = activeRegionName === reg;
                const score = analysis.regions[reg]?.severity_score ?? 0;
                
                return (
                  <button
                    key={reg}
                    onMouseEnter={() => setHoveredRegion(reg)}
                    onMouseLeave={() => setHoveredRegion(null)}
                    onClick={() => setSelectedRegion(selectedRegion === reg ? null : reg)}
                    className={`flex flex-col items-start p-3 rounded-2xl border text-left transition-all ${
                      isHovered || selectedRegion === reg
                        ? 'bg-slate-800/80 border-cyan-500/50 shadow-md shadow-cyan-950/20' 
                        : 'bg-slate-950/40 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <span className="text-xs font-semibold text-slate-400 capitalize">{reg.replace('_', ' ')}</span>
                    <div className="flex items-baseline gap-1.5 mt-1">
                      <span className="text-lg font-bold text-slate-100">
                        {analysis.face_detected ? Math.round(score * 100) : 0}%
                      </span>
                      <span className="text-[10px] text-slate-500 font-semibold">severity</span>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Selected Region details */}
            <div className="border-t border-slate-800/80 pt-6">
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
        </div>

      </div>

      {/* CSS Animation Keyframes for scanner effect */}
      <style jsx global>{`
        @keyframes scan {
          0%, 100% {
            top: 0%;
          }
          50% {
            top: 100%;
          }
        }
      `}</style>
    </main>
  );
}
