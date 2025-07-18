# 🍍 Pineapple Grading Web Application

An AI-powered web application for analyzing pineapple quality through computer vision. This demo version supports both image upload and real-time camera analysis to determine ripeness, count defects, and assign quality grades.

## ✨ Features

### 🔹 Image Upload Mode
- Upload single or multiple pineapple images
- Automatic pineapple detection and segmentation
- Batch analysis with detailed results
- Annotated output images with bounding boxes and labels

### 🔹 Real-Time Camera Mode
- Live webcam analysis
- Frame-by-frame processing as you rotate the pineapple
- Accumulated analysis from multiple angles
- Final grade determination based on all captured frames

### 🔹 Grading System
- **Grade A**: 0 defects
- **Grade B**: 1-3 defects  
- **Grade C**: More than 3 defects
- Ripeness classification (ripe/unripe)
- Confidence scoring for all predictions

### 🔹 Offline Support
- Progressive Web App (PWA) capabilities
- Works without internet connection
- Local model inference

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+ and pip
- Modern web browser with camera access

### Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python app.py
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000

## 🏗️ Architecture

### Frontend (React + TypeScript)
- **React 18** with TypeScript for type safety
- **Tailwind CSS** for responsive, modern UI
- **Vite** for fast development and building
- **PWA** support for offline functionality

### Backend (Python + Flask)
- **Flask** web framework with CORS support
- **OpenCV** for image processing
- **PIL/Pillow** for image manipulation
- **NumPy** for numerical operations
- **PyTorch/Ultralytics** for ML model inference

### Key Components
- `ModeSelector`: Choose between upload and camera modes
- `ImageUpload`: Handle file uploads and batch analysis
- `CameraAnalysis`: Real-time webcam processing
- `ResultsDisplay`: Show analysis results with visualizations

## 🧠 ML Model Integration

The application is designed to work with your trained models:

1. **YOLO Detection Model**: Detects and localizes pineapples
2. **Ripeness Classifier**: Determines if pineapples are ripe/unripe
3. **Defect Counter**: Counts visible defects on the surface

### Model Placement
Place your trained models in the `backend/models/` directory:
- `pineapple_detection.pt` (YOLO model)
- `ripeness_classifier.pt` (Ripeness model)
- `defect_detector.pt` (Defect detection model)

### Integration Steps
1. Replace mock functions in `backend/app.py` with actual model loading
2. Update preprocessing/postprocessing pipelines
3. Adjust confidence thresholds and parameters

## 📱 PWA Features

The application includes Progressive Web App capabilities:
- **Offline functionality**: Works without internet
- **Installable**: Can be installed on mobile devices
- **Responsive design**: Optimized for all screen sizes
- **Service worker**: Caches resources for offline use

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
VITE_API_URL=http://localhost:5000
VITE_APP_NAME=Pineapple Grader
```

### Model Configuration
Adjust model parameters in `backend/app.py`:
- Detection confidence thresholds
- NMS (Non-Maximum Suppression) parameters
- Input image preprocessing settings

## 📊 API Endpoints

### POST `/api/analyze-image`
Analyze uploaded image with multiple pineapples
```json
{
  "image": "base64_encoded_image"
}
```

### POST `/api/analyze-frame`
Analyze single camera frame
```json
{
  "image": "base64_encoded_image"
}
```

### GET `/api/health`
Health check endpoint

## 🚀 Deployment

### Local Deployment
```bash
# Build frontend
npm run build

# Serve built files
npm run preview

# Run backend in production mode
cd backend
python app.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Cloud Deployment
The application can be deployed to:
- **Frontend**: Vercel, Netlify, or any static hosting
- **Backend**: Heroku, AWS, Google Cloud, or any Python hosting service

## 🔮 Future Enhancements

The codebase is designed to support future upgrades:
- **Factory Integration**: 3-camera conveyor system support
- **Role-based Access**: User management and permissions
- **Analytics Dashboard**: Historical data and trends
- **Batch Processing**: Large-scale image processing
- **API Integration**: External system connectivity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern web technologies for optimal performance
- Designed for scalability and future industrial integration
- Optimized for both technical and non-technical users

---

**Note**: This is a demo version. For production use, ensure proper model training, validation, and testing with your specific pineapple varieties and quality standards.