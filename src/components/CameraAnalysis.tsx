import { useState, useRef, useEffect } from 'react';
import { Camera, Square, Play, Pause, RotateCcw } from 'lucide-react';
import { CameraAnalysisResult, CameraAnalysisFrame } from '../types';
import { calculateGrade } from '../utils/grading';

export const CameraAnalysis = () => {
  const [isActive, setIsActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [frames, setFrames] = useState<CameraAnalysisFrame[]>([]);
  const [results, setResults] = useState<CameraAnalysisResult | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsActive(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsActive(false);
    setIsRecording(false);
  };

  const startRecording = () => {
    setIsRecording(true);
    setFrames([]);
    
    // Simulate frame analysis every 500ms
    intervalRef.current = setInterval(() => {
      const frame: CameraAnalysisFrame = {
        frameId: `frame_${Date.now()}`,
        timestamp: Date.now(),
        ripeness: Math.random() > 0.5 ? 'ripe' : 'unripe',
        defects: Math.floor(Math.random() * 6),
        confidence: 0.8 + Math.random() * 0.2
      };
      
      setFrames(prev => [...prev, frame]);
    }, 500);
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const finishScan = () => {
    if (frames.length === 0) return;

    // Analyze all frames to determine final results
    const ripeFrames = frames.filter(f => f.ripeness === 'ripe').length;
    const finalRipeness = ripeFrames > frames.length / 2 ? 'ripe' : 'unripe';
    
    // Calculate average defects
    const totalDefects = Math.round(
      frames.reduce((sum, f) => sum + f.defects, 0) / frames.length
    );
    
    const finalGrade = calculateGrade(totalDefects);

    const result: CameraAnalysisResult = {
      totalFrames: frames.length,
      finalRipeness,
      totalDefects,
      finalGrade,
      frames
    };

    setResults(result);
    stopRecording();
  };

  const reset = () => {
    setFrames([]);
    setResults(null);
    setIsRecording(false);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  if (results) {
    return (
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">Scan Results</h2>
          <button onClick={reset} className="btn-outline">
            <RotateCcw className="w-4 h-4 mr-2" />
            New Scan
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-blue-600 font-medium">Frames Analyzed</p>
            <p className="text-2xl font-bold text-blue-900">{results.totalFrames}</p>
          </div>
          
          <div className={`p-4 rounded-lg ${
            results.finalRipeness === 'ripe' ? 'bg-green-50' : 'bg-orange-50'
          }`}>
            <p className={`text-sm font-medium ${
              results.finalRipeness === 'ripe' ? 'text-green-600' : 'text-orange-600'
            }`}>
              Final Ripeness
            </p>
            <p className={`text-2xl font-bold capitalize ${
              results.finalRipeness === 'ripe' ? 'text-green-900' : 'text-orange-900'
            }`}>
              {results.finalRipeness}
            </p>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 font-medium">Total Defects</p>
            <p className="text-2xl font-bold text-gray-900">{results.totalDefects}</p>
          </div>
        </div>

        <div className="text-center">
          <div className={`inline-flex items-center px-6 py-3 rounded-full text-lg font-bold ${
            results.finalGrade === 'A' ? 'bg-green-100 text-green-800' :
            results.finalGrade === 'B' ? 'bg-yellow-100 text-yellow-800' :
            'bg-red-100 text-red-800'
          }`}>
            Final Grade: {results.finalGrade}
          </div>
        </div>

        <div className="mt-6">
          <h3 className="font-medium text-gray-900 mb-3">Frame Analysis History</h3>
          <div className="max-h-40 overflow-y-auto space-y-2">
            {results.frames.map((frame, index) => (
              <div key={frame.frameId} className="flex items-center justify-between text-sm bg-gray-50 p-2 rounded">
                <span>Frame {index + 1}</span>
                <span className={`px-2 py-1 rounded text-xs ${
                  frame.ripeness === 'ripe' ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'
                }`}>
                  {frame.ripeness}
                </span>
                <span>{frame.defects} defects</span>
                <span className="text-gray-500">{Math.round(frame.confidence * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Real-Time Camera Analysis</h2>
      
      <div className="space-y-4">
        <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '4/3' }}>
          {isActive ? (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="flex items-center justify-center h-full text-white">
              <div className="text-center">
                <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg mb-2">Camera Not Active</p>
                <p className="text-sm opacity-75">Click "Start Camera" to begin</p>
              </div>
            </div>
          )}
          
          {isRecording && (
            <div className="absolute top-4 left-4 flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-white text-sm font-medium">Recording</span>
            </div>
          )}
          
          {frames.length > 0 && (
            <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded">
              {frames.length} frames captured
            </div>
          )}
        </div>

        <div className="flex items-center justify-center space-x-4">
          {!isActive ? (
            <button onClick={startCamera} className="btn-primary">
              <Camera className="w-4 h-4 mr-2" />
              Start Camera
            </button>
          ) : (
            <>
              <button onClick={stopCamera} className="btn-secondary">
                <Square className="w-4 h-4 mr-2" />
                Stop Camera
              </button>
              
              {!isRecording ? (
                <button onClick={startRecording} className="btn-primary">
                  <Play className="w-4 h-4 mr-2" />
                  Start Scanning
                </button>
              ) : (
                <>
                  <button onClick={stopRecording} className="btn-secondary">
                    <Pause className="w-4 h-4 mr-2" />
                    Pause Scanning
                  </button>
                  
                  {frames.length > 0 && (
                    <button onClick={finishScan} className="btn-primary">
                      Finish Scan
                    </button>
                  )}
                </>
              )}
            </>
          )}
        </div>

        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-medium text-blue-900 mb-2">Instructions:</h3>
          <ol className="text-sm text-blue-800 space-y-1">
            <li>1. Start the camera and position a pineapple in view</li>
            <li>2. Click "Start Scanning" to begin analysis</li>
            <li>3. Slowly rotate the pineapple to capture all angles</li>
            <li>4. Click "Finish Scan" when done to see results</li>
          </ol>
        </div>
      </div>
    </div>
  );
};