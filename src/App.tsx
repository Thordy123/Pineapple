import { useState } from 'react';
import { Header } from './components/Header';
import { ModeSelector } from './components/ModeSelector';
import { ImageUpload } from './components/ImageUpload';
import { CameraAnalysis } from './components/CameraAnalysis';

function App() {
  const [selectedMode, setSelectedMode] = useState<'upload' | 'camera'>('upload');

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          <ModeSelector 
            selectedMode={selectedMode} 
            onModeChange={setSelectedMode} 
          />
          
          {selectedMode === 'upload' ? (
            <ImageUpload />
          ) : (
            <CameraAnalysis />
          )}
        </div>
      </main>
      
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-600">
            <p>Pineapple Grading System - Demo Version</p>
            <p className="mt-1">Built with React, TypeScript, and AI-powered computer vision</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;