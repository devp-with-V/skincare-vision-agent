'use client';

import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Sparkles, RefreshCw, AlertTriangle, CheckCircle, Zap } from 'lucide-react';
import FaceMeshOverlay from '../../components/FaceMeshOverlay';
import SeverityGauge from '../../components/SeverityGauge';
import RegionBreakdown from '../../components/RegionBreakdown';
import AnalysisReport, { SkincareRecommendations } from '../../components/AnalysisReport';
import { SkinWebSocketClient, scanImage, scanImageVLM, AnalysisResult } from '../../lib/api';

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

  // Recommendations state
  const [recommendations, setRecommendations] = useState<SkincareRecommendations | null>(null);

  // VLM Toggle State
  const [isVlmMode, setIsVlmMode] = useState(false);

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
      }, 150);
    }

    return () => {
      if (frameInterval) clearInterval(frameInterval);
    };
  }, [isScanning, isWsConnected]);

  const severityHistoryRef = useRef<number[]>([]);

  // Connect WebSocket on mount
  useEffect(() => {
    const ws = new SkinWebSocketClient(
      (data) => {
        if (data.error) {
          setErrorMsg(data.error);
          return;
        }

        // Apply moving average temporal filter over 5 frames
        const rawSeverity = data.overall_severity || 0;
        const history = [...severityHistoryRef.current, rawSeverity].slice(-5);
        severityHistoryRef.current = history;
        const smoothedSeverity = history.reduce((sum, val) => sum + val, 0) / history.length;

        setAnalysis({
          face_detected: data.face_detected,
          landmarks: data.landmarks || [],
          regions: data.regions || {},
          overall_severity: smoothedSeverity
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
    setRecommendations(null); // Clear previous recommendations
    setScanProgress(10);

    const interval = setInterval(() => {
      setScanProgress(p => (p >= 90 ? 90 : p + 15));
    }, 100);

    try {
      const response = isVlmMode ? await scanImageVLM(imageSrc) : await scanImage(imageSrc);
      clearInterval(interval);
      setScanProgress(100);
      
      setAnalysis(response.analysis);
      setRecommendations(response.recommendations);

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
          {/* VLM Mode Toggle Switch */}
          <button
            onClick={() => setIsVlmMode(!isVlmMode)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold border transition-all ${
              isVlmMode
                ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300 hover:bg-indigo-500/30 shadow-md shadow-indigo-900/15'
                : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>VLM Agent: {isVlmMode ? 'ON' : 'OFF'}</span>
          </button>

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
            
            <SeverityGauge 
              severity={analysis.overall_severity} 
              faceDetected={analysis.face_detected} 
            />
          </div>

          {/* Regional Breakdown Dashboard */}
          <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-6 backdrop-blur-md flex-1">
            <h2 className="text-xl font-bold mb-4">Face Regions</h2>
            
            <RegionBreakdown
              regions={analysis.regions}
              faceDetected={analysis.face_detected}
              activeRegionName={activeRegionName}
              selectedRegion={selectedRegion}
              onHoverRegion={setHoveredRegion}
              onSelectRegion={setSelectedRegion}
            />
          </div>
        </div>

      </div>

      {recommendations && (
        <div className="w-full max-w-6xl mt-8">
          <AnalysisReport recommendations={recommendations} />
        </div>
      )}

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
