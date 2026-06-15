export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: number[]; // [x_min, y_min, x_max, y_max]
}

export interface RegionAnalysis {
  region: string;
  severity_score: number;
  dominant_concern: string | null;
  detections: Detection[];
}

export interface AnalysisResult {
  face_detected: boolean;
  landmarks: LandmarkPoint[];
  regions: Record<string, RegionAnalysis>;
  overall_severity: number;
}

export interface ScanResponse {
  session_id: string | null;
  analysis: AnalysisResult;
  recommendations: any | null;
}

// Dynamically resolve backend URLs based on the browser's current hostname.
// This allows local network devices (like mobile phones) to connect to the backend server automatically.
const getBackendUrls = () => {
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    return {
      api: `http://${hostname}:8000`,
      ws: `ws://${hostname}:8000`
    };
  }
  return {
    api: 'http://localhost:8000',
    ws: 'ws://localhost:8000'
  };
};

const urls = getBackendUrls();
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || urls.api;
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || urls.ws;

export async function scanImage(base64Image: string, userId?: string): Promise<ScanResponse> {
  const response = await fetch(`${API_BASE_URL}/api/scan`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_base64: base64Image,
      user_id: userId || null,
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Failed to scan image: ${errText || response.statusText}`);
  }

  return response.json();
}

export class SkinWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private onMessageCallback: (data: any) => void;
  private onConnectCallback?: () => void;
  private onCloseCallback?: () => void;
  private onErrorCallback?: (err: any) => void;

  constructor(
    onMessage: (data: any) => void,
    options?: {
      onConnect?: () => void;
      onClose?: () => void;
      onError?: (err: any) => void;
    }
  ) {
    this.url = `${WS_BASE_URL}/api/ws`;
    this.onMessageCallback = onMessage;
    this.onConnectCallback = options?.onConnect;
    this.onCloseCallback = options?.onClose;
    this.onErrorCallback = options?.onError;
  }

  connect() {
    if (this.ws) {
      this.close();
    }

    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('SkinCare WS Connected');
      if (this.onConnectCallback) this.onConnectCallback();
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.onMessageCallback(data);
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    this.ws.onerror = (error) => {
      console.error('SkinCare WS Error:', error);
      if (this.onErrorCallback) this.onErrorCallback(error);
    };

    this.ws.onclose = () => {
      console.log('SkinCare WS Closed');
      if (this.onCloseCallback) this.onCloseCallback();
    };
  }

  sendFrame(base64Image: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ image: base64Image }));
    }
  }

  sendBinaryFrame(blob: Blob) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(blob);
    }
  }

  close() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  isOpen(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}
