import { Camera, Upload } from 'lucide-react';

interface ModeSelectorProps {
  selectedMode: 'upload' | 'camera';
  onModeChange: (mode: 'upload' | 'camera') => void;
}

export const ModeSelector = ({ selectedMode, onModeChange }: ModeSelectorProps) => {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Select Analysis Mode</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <button
          onClick={() => onModeChange('upload')}
          className={`p-6 rounded-lg border-2 transition-all ${
            selectedMode === 'upload'
              ? 'border-pineapple-500 bg-pineapple-50'
              : 'border-gray-200 hover:border-gray-300'
          }`}
        >
          <Upload className={`w-8 h-8 mx-auto mb-3 ${
            selectedMode === 'upload' ? 'text-pineapple-600' : 'text-gray-400'
          }`} />
          <h3 className="font-medium text-gray-900 mb-2">Image Upload</h3>
          <p className="text-sm text-gray-600">
            Upload photos with single or multiple pineapples for batch analysis
          </p>
        </button>
        
        <button
          onClick={() => onModeChange('camera')}
          className={`p-6 rounded-lg border-2 transition-all ${
            selectedMode === 'camera'
              ? 'border-pineapple-500 bg-pineapple-50'
              : 'border-gray-200 hover:border-gray-300'
          }`}
        >
          <Camera className={`w-8 h-8 mx-auto mb-3 ${
            selectedMode === 'camera' ? 'text-pineapple-600' : 'text-gray-400'
          }`} />
          <h3 className="font-medium text-gray-900 mb-2">Real-Time Camera</h3>
          <p className="text-sm text-gray-600">
            Use your webcam to analyze pineapples in real-time by rotating them
          </p>
        </button>
      </div>
    </div>
  );
};