import { useState, useRef } from 'react';
import { Upload, X, Loader2 } from 'lucide-react';
import { AnalysisResult } from '../types';
import { ResultsDisplay } from './ResultsDisplay';

export const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResults(null);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResults(null);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const clearSelection = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResults(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    
    // Simulate API call - replace with actual backend call
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock results for demo
      const mockResults: AnalysisResult = {
        totalPineapples: 3,
        ripeCount: 2,
        unripeCount: 1,
        gradeACounts: 1,
        gradeBCounts: 1,
        gradeCCounts: 1,
        detections: [
          {
            id: '1',
            bbox: { x: 50, y: 50, width: 150, height: 200 },
            confidence: 0.95,
            ripeness: 'ripe',
            defects: 0,
            grade: 'A'
          },
          {
            id: '2',
            bbox: { x: 250, y: 80, width: 140, height: 180 },
            confidence: 0.88,
            ripeness: 'ripe',
            defects: 2,
            grade: 'B'
          },
          {
            id: '3',
            bbox: { x: 450, y: 60, width: 160, height: 210 },
            confidence: 0.92,
            ripeness: 'unripe',
            defects: 5,
            grade: 'C'
          }
        ],
        processedImageUrl: previewUrl || undefined
      };
      
      setResults(mockResults);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (results) {
    return <ResultsDisplay results={results} onReset={clearSelection} />;
  }

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Pineapple Image</h2>
      
      {!selectedFile ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-pineapple-400 transition-colors cursor-pointer"
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-lg font-medium text-gray-900 mb-2">
            Drop your image here, or click to browse
          </p>
          <p className="text-sm text-gray-600 mb-4">
            Supports single or multiple pineapples in one image
          </p>
          <p className="text-xs text-gray-500">
            Supported formats: JPG, PNG, WebP (Max 10MB)
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative">
            <img
              src={previewUrl!}
              alt="Selected pineapple"
              className="w-full max-h-96 object-contain rounded-lg border"
            />
            <button
              onClick={clearSelection}
              className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900">{selectedFile.name}</p>
              <p className="text-sm text-gray-600">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            
            <button
              onClick={analyzeImage}
              disabled={isAnalyzing}
              className="btn-primary disabled:opacity-50"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                'Analyze Pineapples'
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};