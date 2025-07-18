import { RotateCcw, Download } from 'lucide-react';
import { AnalysisResult } from '../types';
import { getGradeColor, getRipenessColor } from '../utils/grading';

interface ResultsDisplayProps {
  results: AnalysisResult;
  onReset: () => void;
}

export const ResultsDisplay = ({ results, onReset }: ResultsDisplayProps) => {
  const downloadResults = () => {
    const data = {
      timestamp: new Date().toISOString(),
      summary: {
        totalPineapples: results.totalPineapples,
        ripeCount: results.ripeCount,
        unripeCount: results.unripeCount,
        gradeA: results.gradeACounts,
        gradeB: results.gradeBCounts,
        gradeC: results.gradeCCounts
      },
      detections: results.detections
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pineapple-analysis-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">Analysis Results</h2>
          <div className="flex space-x-2">
            <button onClick={downloadResults} className="btn-outline">
              <Download className="w-4 h-4 mr-2" />
              Export
            </button>
            <button onClick={onReset} className="btn-secondary">
              <RotateCcw className="w-4 h-4 mr-2" />
              New Analysis
            </button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-blue-900">{results.totalPineapples}</p>
            <p className="text-sm text-blue-600">Total</p>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-green-900">{results.ripeCount}</p>
            <p className="text-sm text-green-600">Ripe</p>
          </div>
          
          <div className="bg-orange-50 p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-orange-900">{results.unripeCount}</p>
            <p className="text-sm text-orange-600">Unripe</p>
          </div>
          
          <div className="bg-green-100 p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-green-800">{results.gradeACounts}</p>
            <p className="text-sm text-green-700">Grade A</p>
          </div>
          
          <div className="bg-yellow-100 p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-yellow-800">{results.gradeBCounts}</p>
            <p className="text-sm text-yellow-700">Grade B</p>
          </div>
          
          <div className="bg-red-100 p-4 rounded-lg text-center">
            <p className="text-2xl font-bold text-red-800">{results.gradeCCounts}</p>
            <p className="text-sm text-red-700">Grade C</p>
          </div>
        </div>

        {/* Processed Image */}
        {results.processedImageUrl && (
          <div className="mb-6">
            <h3 className="font-medium text-gray-900 mb-3">Annotated Image</h3>
            <div className="relative">
              <img
                src={results.processedImageUrl}
                alt="Processed pineapples"
                className="w-full max-h-96 object-contain rounded-lg border"
              />
              
              {/* Overlay bounding boxes and labels */}
              {results.detections.map((detection) => (
                <div
                  key={detection.id}
                  className="absolute border-2 border-pineapple-500"
                  style={{
                    left: `${(detection.bbox.x / 800) * 100}%`,
                    top: `${(detection.bbox.y / 600) * 100}%`,
                    width: `${(detection.bbox.width / 800) * 100}%`,
                    height: `${(detection.bbox.height / 600) * 100}%`,
                  }}
                >
                  <div className="absolute -top-8 left-0 bg-pineapple-500 text-white px-2 py-1 rounded text-xs font-medium whitespace-nowrap">
                    Pineapple {detection.id}: {detection.ripeness}, {detection.defects} defects → Grade {detection.grade}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Detection Details */}
        <div>
          <h3 className="font-medium text-gray-900 mb-3">Individual Pineapple Details</h3>
          <div className="space-y-3">
            {results.detections.map((detection) => (
              <div key={detection.id} className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <span className="font-medium text-gray-900">
                      Pineapple {detection.id}
                    </span>
                    
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRipenessColor(detection.ripeness)}`}>
                      {detection.ripeness}
                    </span>
                    
                    <span className="text-sm text-gray-600">
                      {detection.defects} defect{detection.defects !== 1 ? 's' : ''}
                    </span>
                    
                    <span className="text-sm text-gray-500">
                      {Math.round(detection.confidence * 100)}% confidence
                    </span>
                  </div>
                  
                  <span className={`px-3 py-1 rounded-full text-sm font-bold ${getGradeColor(detection.grade)}`}>
                    Grade {detection.grade}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};