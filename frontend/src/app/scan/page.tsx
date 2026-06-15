'use client';

import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Sparkles, RefreshCw, AlertTriangle, CheckCircle, Zap, X } from 'lucide-react';
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

  // Analysis results state (contains real-time face tracking landmarks + local CV engine regions)
  const [analysis, setAnalysis] = useState<AnalysisResult>({
    face_detected: false,
    landmarks: [],
    regions: {},
    overall_severity: 0.0
  });

  // VLM Static Analysis State
  const [vlmAnalysis, setVlmAnalysis] = useState<AnalysisResult | null>(null);

  // Interactive UI state for main scanner
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);

  // recommendations state
  const [recommendations, setRecommendations] = useState<SkincareRecommendations | null>(null);

  // VLM Toggle State
  const [isVlmMode, setIsVlmMode] = useState(false);

  // Focused Report Modal states
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [modalAnalysis, setModalAnalysis] = useState<AnalysisResult | null>(null);
  const [modalRecommendations, setModalRecommendations] = useState<SkincareRecommendations | null>(null);
  const [hoveredModalRegion, setHoveredModalRegion] = useState<string | null>(null);
  const [selectedModalRegion, setSelectedModalRegion] = useState<string | null>(null);

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

        if (data.type === 'landmarks') {
          setAnalysis((prev) => ({
            ...prev,
            face_detected: data.face_detected,
            landmarks: data.landmarks || []
          }));
        } else if (data.type === 'blemishes') {
          // Apply moving average temporal filter over 5 frames
          const rawSeverity = data.overall_severity || 0;
          const history = [...severityHistoryRef.current, rawSeverity].slice(-5);
          severityHistoryRef.current = history;
          const smoothedSeverity = history.reduce((sum, val) => sum + val, 0) / history.length;

          setAnalysis((prev) => ({
            ...prev,
            regions: data.regions || {},
            overall_severity: smoothedSeverity
          }));
        } else {
          // Fallback if message type is not specified
          const rawSeverity = data.overall_severity || 0;
          const history = [...severityHistoryRef.current, rawSeverity].slice(-5);
          severityHistoryRef.current = history;
          const smoothedSeverity = history.reduce((sum, val) => sum + val, 0) / history.length;

          setAnalysis({
            face_detected: data.face_detected !== undefined ? data.face_detected : true,
            landmarks: data.landmarks || [],
            regions: data.regions || {},
            overall_severity: smoothedSeverity
          });
        }
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

  // Run a high-quality single scan (REST request) and open Focused Report Modal
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
      const response = isVlmMode ? await scanImageVLM(imageSrc) : await scanImage(imageSrc);
      clearInterval(interval);
      setScanProgress(100);
      
      // Store static modal report states
      setCapturedImage(imageSrc);
      setModalAnalysis(response.analysis);
      setModalRecommendations(response.recommendations);
      
      // If VLM was run, save it in vlmAnalysis to enable compound inheritance on main overlay
      if (isVlmMode) {
        setVlmAnalysis(response.analysis);
        setRecommendations(response.recommendations);
      } else {
        // Otherwise, update the active local analysis results
        setAnalysis(prev => ({
          ...prev,
          regions: response.analysis.regions,
          overall_severity: response.analysis.overall_severity
        }));
        setRecommendations(response.recommendations);
      }
      
      setIsModalOpen(true);

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

  // Compound Inheritance Logic:
  // Landmarks (mesh coordinates) always inherit from the real-time WebSocket FaceMesh data (analysis.landmarks).
  // Detections & Severities inherit from the static VLM analysis (if VLM is ON and loaded),
  // otherwise falling back to the local CV engine results.
  const displayRegions = (isVlmMode && vlmAnalysis) ? vlmAnalysis.regions : analysis.regions;
  const displaySeverity = (isVlmMode && vlmAnalysis) ? vlmAnalysis.overall_severity : analysis.overall_severity;

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
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-bold border transition-all cursor-pointer ${
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
              regions={displayRegions}
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
              className={`flex-1 flex items-center justify-center gap-2 px-6 py-3.5 rounded-2xl font-bold tracking-wide transition-all shadow-lg cursor-pointer ${
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
              className="flex-1 flex items-center justify-center gap-2 px-6 py-3.5 rounded-2xl bg-slate-800 hover:bg-slate-700 font-bold tracking-wide text-slate-100 border border-slate-700 hover:border-slate-600 transition-all shadow-md cursor-pointer"
            >
              <Camera className="w-5 h-5" />
              Capture Focused Report
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
              General Analysis {isVlmMode && vlmAnalysis && '(Cloud VLM)'}
            </h2>
            
            <SeverityGauge 
              severity={displaySeverity} 
              faceDetected={analysis.face_detected} 
            />
          </div>

          {/* Regional Breakdown Dashboard */}
          <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-6 backdrop-blur-md flex-1">
            <h2 className="text-xl font-bold mb-4">Face Regions</h2>
            
            <RegionBreakdown
              regions={displayRegions}
              faceDetected={analysis.face_detected}
              activeRegionName={activeRegionName}
              selectedRegion={selectedRegion}
              onHoverRegion={setHoveredRegion}
              onSelectRegion={setSelectedRegion}
            />
          </div>
        </div>

      </div>

      {/* Skincare Recommendations (Only shown below camera if not in modal context) */}
      {recommendations && !isModalOpen && (
        <div className="w-full max-w-6xl mt-8">
          <AnalysisReport recommendations={recommendations} />
        </div>
      )}

      {/* Focused Report Modal Overlay */}
      {isModalOpen && capturedImage && modalAnalysis && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-6 bg-slate-950/80 backdrop-blur-md overflow-y-auto">
          <div className="relative bg-slate-900 border border-slate-800 rounded-3xl w-full max-w-6xl max-h-[92vh] overflow-y-auto shadow-2xl flex flex-col p-6 md:p-8 text-slate-100 animate-[fadeIn_0.2s_ease-out]">
            
            {/* Close Button */}
            <button
              onClick={() => setIsModalOpen(false)}
              className="absolute top-6 right-6 p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-100 transition-all cursor-pointer border border-slate-750"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Modal Header */}
            <div className="mb-6">
              <h2 className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent flex items-center gap-2">
                <Sparkles className="w-6 h-6 text-indigo-400 animate-pulse" />
                Focused Skin Analysis Report
              </h2>
              <p className="text-slate-400 text-xs mt-1">
                Static snapshot scan processed with {isVlmMode ? 'Cloud VLM Agent (OpenRouter)' : 'Local CV Engine'}
              </p>
            </div>

            {/* Modal Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              
              {/* Left Column: Image Overlay & Score */}
              <div className="lg:col-span-5 flex flex-col gap-4">
                <div className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden border border-slate-800 bg-slate-950/40 flex items-center justify-center">
                  <img
                    src={capturedImage}
                    alt="Captured Scan Face"
                    className="absolute top-0 left-0 w-full h-full object-cover scale-x-[-1]"
                  />
                  <FaceMeshOverlay
                    landmarks={modalAnalysis.landmarks}
                    regions={modalAnalysis.regions}
                    width={WEBCAM_WIDTH}
                    height={WEBCAM_HEIGHT}
                    activeRegion={hoveredModalRegion || selectedModalRegion}
                    onHoverRegion={setHoveredModalRegion}
                  />
                </div>

                {/* ScoreGauge in Modal */}
                <div className="bg-slate-950/40 border border-slate-800/80 rounded-2xl p-5">
                  <SeverityGauge
                    severity={modalAnalysis.overall_severity}
                    faceDetected={modalAnalysis.face_detected}
                  />
                </div>
              </div>

              {/* Right Column: Breakdown & recommendations */}
              <div className="lg:col-span-7 flex flex-col gap-6 max-h-[72vh] overflow-y-auto pr-2 select-none">
                {/* RegionBreakdown in Modal */}
                <div className="bg-slate-950/30 border border-slate-850 rounded-2xl p-5">
                  <h3 className="text-sm font-semibold text-indigo-300 mb-3 uppercase tracking-wider">Region Analysis Breakdown</h3>
                  <RegionBreakdown
                    regions={modalAnalysis.regions}
                    faceDetected={modalAnalysis.face_detected}
                    activeRegionName={hoveredModalRegion || selectedModalRegion}
                    selectedRegion={selectedModalRegion}
                    onHoverRegion={setHoveredModalRegion}
                    onSelectRegion={setSelectedModalRegion}
                  />
                </div>

                {/* recommendations in Modal */}
                {modalRecommendations && (
                  <div className="w-full">
                    <AnalysisReport recommendations={modalRecommendations} />
                  </div>
                )}
              </div>
            </div>

          </div>
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
        @keyframes fadeIn {
          from { opacity: 0; transform: scale(0.98); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </main>
  );
}
