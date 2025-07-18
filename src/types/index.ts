export interface PineappleDetection {
  id: string;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  ripeness: 'ripe' | 'unripe';
  defects: number;
  grade: 'A' | 'B' | 'C';
}

export interface AnalysisResult {
  totalPineapples: number;
  ripeCount: number;
  unripeCount: number;
  gradeACounts: number;
  gradeBCounts: number;
  gradeCCounts: number;
  detections: PineappleDetection[];
  processedImageUrl?: string;
}

export interface CameraAnalysisFrame {
  frameId: string;
  timestamp: number;
  ripeness: 'ripe' | 'unripe';
  defects: number;
  confidence: number;
}

export interface CameraAnalysisResult {
  totalFrames: number;
  finalRipeness: 'ripe' | 'unripe';
  totalDefects: number;
  finalGrade: 'A' | 'B' | 'C';
  frames: CameraAnalysisFrame[];
}