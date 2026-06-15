'use client';

import React, { useRef, useEffect } from 'react';
import { LandmarkPoint, RegionAnalysis } from '../lib/api';

interface FaceMeshOverlayProps {
  landmarks: LandmarkPoint[];
  regions: Record<string, RegionAnalysis>;
  width: number;
  height: number;
  activeRegion: string | null;
  onHoverRegion: (regionName: string | null) => void;
}

// Landmark lists corresponding to backend's face detector regions
const REGION_INDICES: Record<string, number[]> = {
  forehead: [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 297, 332, 284, 251, 389, 356, 454, 323, 361],
  left_cheek: [111, 116, 117, 118, 101, 50, 187, 205, 207, 206, 203, 98, 36, 142, 228, 229, 230, 231, 232, 233],
  right_cheek: [340, 345, 346, 347, 330, 280, 411, 425, 427, 426, 423, 327, 266, 371, 448, 449, 450, 451, 452, 453],
  nose: [168, 6, 197, 195, 5, 4, 122, 196, 3, 51, 45, 275, 274, 351, 419, 420, 294, 327, 98, 197],
  chin: [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 58, 172, 136, 150, 149, 176, 148, 18, 200, 199]
};

// Helper to convert normalized landmarks to pixel space (applying horizontal mirroring)
const getPixelLandmarks = (landmarks: LandmarkPoint[], width: number, height: number) => {
  return landmarks.map(lm => ({
    x: (1.0 - lm.x) * width,  // Mirror X axis to match mirrored webcam scale-x-[-1]
    y: lm.y * height,
    z: lm.z
  }));
};

export default function FaceMeshOverlay({
  landmarks,
  regions,
  width,
  height,
  activeRegion,
  onHoverRegion
}: FaceMeshOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const currentLandmarksRef = useRef<LandmarkPoint[]>([]);
  const targetLandmarksRef = useRef<LandmarkPoint[]>([]);
  const regionsRef = useRef(regions);
  const activeRegionRef = useRef(activeRegion);
  const widthRef = useRef(width);
  const heightRef = useRef(height);

  useEffect(() => {
    targetLandmarksRef.current = landmarks;
    if (currentLandmarksRef.current.length === 0 && landmarks && landmarks.length > 0) {
      currentLandmarksRef.current = JSON.parse(JSON.stringify(landmarks));
    }
  }, [landmarks]);

  useEffect(() => {
    regionsRef.current = regions;
    activeRegionRef.current = activeRegion;
    widthRef.current = width;
    heightRef.current = height;
  }, [regions, activeRegion, width, height]);

  useEffect(() => {
    let animId: number;

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        animId = requestAnimationFrame(render);
        return;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        animId = requestAnimationFrame(render);
        return;
      }

      const w = widthRef.current;
      const h = heightRef.current;
      const activeReg = activeRegionRef.current;
      const regs = regionsRef.current;

      // Clear canvas
      ctx.clearRect(0, 0, w, h);

      const target = targetLandmarksRef.current;
      let current = currentLandmarksRef.current;

      if (!target || target.length === 0) {
        currentLandmarksRef.current = [];
        animId = requestAnimationFrame(render);
        return;
      }

      // If length mismatch, copy target directly
      if (current.length !== target.length) {
        current = JSON.parse(JSON.stringify(target));
        currentLandmarksRef.current = current;
      } else {
        // Interpolate current landmarks towards target (0.25 lerp coefficient for smooth motion)
        for (let i = 0; i < current.length; i++) {
          current[i].x += (target[i].x - current[i].x) * 0.25;
          current[i].y += (target[i].y - current[i].y) * 0.25;
          current[i].z += (target[i].z - current[i].z) * 0.25;
        }
      }

      // Pre-calculate pixel landmarks
      const pixelLandmarks = getPixelLandmarks(current, w, h);

      // Helper to get region color based on severity
      const getRegionColors = (regionName: string, isHovered: boolean) => {
        const score = regs[regionName]?.severity_score ?? 0;
        let r = 16, g = 185, b = 129; // Healthy Emerald Green

        if (score >= 0.2 && score < 0.5) {
          r = 245; g = 158; b = 11; // Amber
        } else if (score >= 0.5) {
          r = 239; g = 68; b = 68; // Red
        }

        const alpha = isHovered ? 0.35 : 0.12;
        return {
          fill: `rgba(${r}, ${g}, ${b}, ${alpha})`,
          stroke: `rgba(${r}, ${g}, ${b}, 0.85)`,
          glow: `rgba(${r}, ${g}, ${b}, 0.4)`
        };
      };

      // 1. Draw connecting mesh network (background)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 0.5;
      
      // Draw simplified grid connections between points to look like a mesh
      for (let i = 0; i < pixelLandmarks.length; i += 4) {
        if (i + 4 < pixelLandmarks.length) {
          ctx.beginPath();
          ctx.moveTo(pixelLandmarks[i].x, pixelLandmarks[i].y);
          ctx.lineTo(pixelLandmarks[i+4].x, pixelLandmarks[i+4].y);
          ctx.stroke();
        }
      }

      // 2. Draw skin region overlays
      Object.entries(REGION_INDICES).forEach(([regionName, indices]) => {
        const pts = indices
          .map(idx => pixelLandmarks[idx])
          .filter(Boolean)
          .map(lm => ({ x: lm.x, y: lm.y }));

        if (pts.length < 3) return;

        const isHovered = activeReg === regionName;
        const colors = getRegionColors(regionName, isHovered);

        // Draw filled region polygon
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();

        // Style and stroke
        ctx.fillStyle = colors.fill;
        ctx.fill();
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = isHovered ? 2.5 : 1.2;
        ctx.shadowColor = colors.glow;
        ctx.shadowBlur = isHovered ? 12 : 0;
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset shadow

        // Draw label near region center if hovered
        if (isHovered) {
          const cX = pts.reduce((sum, p) => sum + p.x, 0) / pts.length;
          const cY = pts.reduce((sum, p) => sum + p.y, 0) / pts.length;

          ctx.font = 'bold 11px sans-serif';
          const labelText = `${regionName.replace('_', ' ').toUpperCase()} (${Math.round((regs[regionName]?.severity_score ?? 0) * 100)}%)`;
          const textWidth = ctx.measureText(labelText).width;

          // Label Background
          ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
          ctx.fillRect(cX - textWidth / 2 - 6, cY - 10, textWidth + 12, 18);
          ctx.strokeStyle = colors.stroke;
          ctx.lineWidth = 1;
          ctx.strokeRect(cX - textWidth / 2 - 6, cY - 10, textWidth + 12, 18);

          // Label Text
          ctx.fillStyle = '#ffffff';
          ctx.textBaseline = 'middle';
          ctx.fillText(labelText, cX - textWidth / 2, cY);
        }
      });

      // 3. Draw Detections Bounding Boxes (if any)
      Object.entries(regs).forEach(([regionName, analysis]) => {
        const indices = REGION_INDICES[regionName];
        if (!indices || !analysis.detections.length) return;

        // Find bounding box limits of this region in canvas coordinates
        const regionLandmarks = indices.map(idx => pixelLandmarks[idx]).filter(Boolean);
        const xs = regionLandmarks.map(lm => lm.x);
        const ys = regionLandmarks.map(lm => lm.y);
        
        const regionMinX = Math.min(...xs);
        const regionMaxX = Math.max(...xs);
        const regionMinY = Math.min(...ys);
        const regionMaxY = Math.max(...ys);

        const regionW = regionMaxX - regionMinX;
        const regionH = regionMaxY - regionMinY;

        analysis.detections.forEach(det => {
          // Map region relative bbox back to canvas
          const [dxMin, dyMin, dxMax, dyMax] = det.bbox;
          
          let boxX = regionMinX + dxMin * regionW;
          const boxY = regionMinY + dyMin * regionH;
          const boxW = (dxMax - dxMin) * regionW;
          const boxH = (dyMax - dyMin) * regionH;

          // Since the region image inside the backend is cropped from the UNMIRRORED frame,
          // and we are drawing on a MIRRORED canvas, we also need to mirror the horizontal 
          // offset of the bounding boxes *inside* the region bounds!
          // Mirroring inside region box: boxX_mirrored = regionMaxX - (boxX - regionMinX) - boxW
          boxX = regionMaxX - (boxX - regionMinX) - boxW;

          // Draw bbox
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.9)'; // Crimson detection box
          ctx.lineWidth = 1.5;
          ctx.strokeRect(boxX, boxY, boxW, boxH);

          // Bounding box neon corners
          ctx.fillStyle = 'rgba(239, 68, 68, 1)';
          const size = 4;
          ctx.fillRect(boxX, boxY, size, size); // top-left
          ctx.fillRect(boxX + boxW - size, boxY, size, size); // top-right
          ctx.fillRect(boxX, boxY + boxH - size, size, size); // bottom-left
          ctx.fillRect(boxX + boxW - size, boxY + boxH - size, size, size); // bottom-right

          // Write label
          ctx.font = '10px sans-serif';
          const label = `${det.class_name} ${Math.round(det.confidence * 100)}%`;
          ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
          const textW = ctx.measureText(label).width;
          ctx.fillRect(boxX - 0.75, boxY - 12, textW + 4, 12);
          
          ctx.fillStyle = '#ffffff';
          ctx.fillText(label, boxX + 1.5, boxY - 3);
        });
      });

      // 4. Draw face landmarks as points (foreground)
      pixelLandmarks.forEach((lm, index) => {
        // Don't draw all 468 to prevent visual clutter; draw a subset
        if (index % 3 !== 0) return;

        ctx.beginPath();
        ctx.arc(lm.x, lm.y, 1.2, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.7)'; // Cyan
        ctx.fill();
      });

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
    };
  }, []);

  // Handle canvas mouse move to identify hover region
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !landmarks || landmarks.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const mX = e.clientX - rect.left;
    const mY = e.clientY - rect.top;

    let hoveredRegion: string | null = null;

    const pixelLandmarks = getPixelLandmarks(landmarks, width, height);

    // Check collision for each region polygon
    const entries = Object.entries(REGION_INDICES);
    for (let i = 0; i < entries.length; i++) {
      const [regionName, indices] = entries[i];
      const pts = indices
        .map(idx => pixelLandmarks[idx])
        .filter(Boolean)
        .map(lm => ({ x: lm.x, y: lm.y }));

      if (pts.length < 3) continue;

      // Point in polygon (ray-casting algorithm)
      let inside = false;
      for (let j = 0, k = pts.length - 1; j < pts.length; k = j++) {
        const xi = pts[j].x, yi = pts[j].y;
        const xj = pts[k].x, yj = pts[k].y;

        const intersect = ((yi > mY) !== (yj > mY)) && 
          (mX < (xj - xi) * (mY - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
      }

      if (inside) {
        hoveredRegion = regionName;
        break; // Stop at first matched region
      }
    }

    if (hoveredRegion !== activeRegion) {
      onHoverRegion(hoveredRegion);
    }
  };

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => onHoverRegion(null)}
      className="absolute top-0 left-0 w-full h-full cursor-crosshair z-10 pointer-events-auto"
    />
  );
}
