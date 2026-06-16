import Link from 'next/link';
import { Sparkles, ShieldCheck, Heart, History, Camera, UserCheck } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500/30 font-sans">
      {/* Header / Nav */}
      <header className="max-w-6xl w-full mx-auto px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-cyan-400" />
          <span className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-cyan-400 to-indigo-500 bg-clip-text text-transparent">
            SkinCare Vision
          </span>
        </div>
        <Link 
          href="/scan" 
          className="bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 px-4 py-2 rounded-xl text-xs font-semibold tracking-wide transition-all"
        >
          Access Scanner
        </Link>
      </header>

      {/* Hero Section */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-6 flex flex-col justify-center py-12 md:py-20 gap-16">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Hero Copy */}
          <div className="lg:col-span-7 flex flex-col items-start text-left gap-6">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold tracking-widest uppercase bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <Sparkles className="w-3 h-3 text-cyan-400" />
              Agentic AI + Computer Vision
            </div>
            
            <h1 className="text-4xl md:text-6xl font-black tracking-tight leading-[1.1] text-white">
              Smart Skin Analysis <br />
              <span className="bg-gradient-to-r from-cyan-400 via-emerald-400 to-indigo-500 bg-clip-text text-transparent">
                Powered by AI
              </span>
            </h1>
            
            <p className="text-slate-400 text-base md:text-lg max-w-xl leading-relaxed">
              Capture or stream real-time camera feeds to map 468 facial landmarks. Extract, segment, and detect skin conditions with high-precision visual heatmaps and get tailored dermatological recommendations.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto mt-2">
              <Link
                href="/scan"
                className="flex items-center justify-center gap-2 px-8 py-4 rounded-2xl bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-600 hover:to-emerald-600 text-slate-950 font-bold tracking-wide transition-all shadow-lg shadow-cyan-500/10 hover:shadow-cyan-500/20"
              >
                <Camera className="w-5 h-5 fill-slate-950" />
                Launch Live Scanner
              </Link>
              
              <Link
                href="https://github.com/devp-with-V/skincare-vision-agent"
                target="_blank"
                className="flex items-center justify-center gap-2 px-8 py-4 rounded-2xl bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-100 font-semibold tracking-wide transition-all"
              >
                View Repository
              </Link>
            </div>
          </div>

          {/* Right Hero Graphic */}
          <div className="lg:col-span-5 relative flex justify-center">
            <div className="relative w-80 h-80 md:w-96 md:h-96 rounded-3xl overflow-hidden border border-slate-800/80 bg-gradient-to-tr from-cyan-950/20 via-slate-900/50 to-indigo-950/20 flex items-center justify-center shadow-2xl">
              {/* Scan simulation element */}
              <div className="absolute w-[90%] h-[90%] border border-dashed border-cyan-500/20 rounded-2xl animate-pulse flex items-center justify-center">
                <div className="absolute w-[80%] h-[80%] border border-dashed border-emerald-500/20 rounded-xl flex items-center justify-center">
                  <div className="absolute w-[70%] h-[70%] border border-dashed border-indigo-500/10 rounded-lg" />
                </div>
              </div>
              
              <div className="text-center p-6 z-10">
                <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mx-auto mb-4">
                  <Sparkles className="w-8 h-8 text-cyan-400 animate-bounce" />
                </div>
                <h3 className="text-lg font-bold text-white mb-1">Face Landmarks Lock-On</h3>
                <p className="text-xs text-slate-400 max-w-[200px] mx-auto leading-relaxed">
                  Real-time face detection dynamically constructs a digital mesh layout.
                </p>
              </div>

              {/* Decorative side lights */}
              <div className="absolute -top-12 -right-12 w-36 h-36 bg-cyan-500/10 rounded-full blur-3xl" />
              <div className="absolute -bottom-12 -left-12 w-36 h-36 bg-indigo-500/10 rounded-full blur-3xl" />
            </div>
          </div>

        </div>

        {/* Feature Cards Section */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-12 border-t border-slate-900">
          
          <div className="bg-slate-950 border border-slate-900 hover:border-slate-800/80 p-6 rounded-2xl transition-all flex flex-col gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
              <Camera className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-100">Live Webcam Tracking</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Streams high-FPS webcam captures to detect coordinates, aligning face contour overlays seamlessly.
            </p>
          </div>

          <div className="bg-slate-950 border border-slate-900 hover:border-slate-800/80 p-6 rounded-2xl transition-all flex flex-col gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-100">Convex Hull Segmentation</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Splits face geometry into discrete skin zones (forehead, nose, chin, cheeks) for targeted region scanning.
            </p>
          </div>

          <div className="bg-slate-950 border border-slate-900 hover:border-slate-800/80 p-6 rounded-2xl transition-all flex flex-col gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
              <Heart className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-100">Intelligent Analysis</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Pipes aggregated region scans through fine-tuned YOLO classifiers and LLM agents to deliver ingredient suggestions.
            </p>
          </div>

        </section>
      </main>

      {/* Footer / Disclaimer */}
      <footer className="w-full bg-slate-950/40 border-t border-slate-900 py-8 px-6 mt-12">
        <div className="max-w-6xl mx-auto flex flex-col gap-4 text-center items-center justify-center">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-md border bg-slate-950 border-amber-500/20 text-amber-400 text-[10px] font-semibold tracking-wide max-w-xl">
            <UserCheck className="w-3.5 h-3.5 text-amber-500 shrink-0" />
            <span>DISCLAIMER: This application is for demonstration purposes only. It is not intended to provide professional medical advice, diagnosis, or treatment.</span>
          </div>
          <p className="text-slate-600 text-[10px] mt-2">© 2026 SkinCare Vision Agent. Developed for AI Portfolio showcase.</p>
        </div>
      </footer>
    </div>
  );
}
