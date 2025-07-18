import os
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class PineappleAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load your custom model
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'pineapple_pen.pt')
        
        try:
            if os.path.exists(self.model_path):
                # Load your custom PyTorch model
                self.model = torch.load(self.model_path, map_location=self.device)
                self.model.eval()
                logger.info("Successfully loaded pineapple_pen.pt model")
            else:
                logger.warning(f"Model file not found at {self.model_path}, using mock analysis")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
        
        # Initialize YOLO for pineapple detection (fallback to YOLOv8n if custom model fails)
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Will download if not present
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.yolo_model = None

    def detect_pineapples(self, image):
        """Detect pineapples in the image using YOLO"""
        if self.yolo_model is None:
            # Mock detection for demo
            h, w = image.shape[:2]
            return [{
                'bbox': [w//4, h//4, w//2, h//2],
                'confidence': 0.95
            }]
        
        try:
            results = self.yolo_model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for relevant classes (you may need to adjust class IDs)
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'confidence': confidence
                            })
            
            return detections if detections else [{'bbox': [image.shape[1]//4, image.shape[0]//4, image.shape[1]//2, image.shape[0]//2], 'confidence': 0.8}]
        
        except Exception as e:
            logger.error(f"Error in pineapple detection: {e}")
            # Return mock detection
            h, w = image.shape[:2]
            return [{
                'bbox': [w//4, h//4, w//2, h//2],
                'confidence': 0.8
            }]

    def analyze_pineapple_region(self, image, bbox):
        """Analyze a specific pineapple region for ripeness and defects"""
        x, y, w, h = bbox
        pineapple_region = image[y:y+h, x:x+w]
        
        if self.model is not None:
            try:
                # Preprocess the region for your model
                pineapple_pil = Image.fromarray(cv2.cvtColor(pineapple_region, cv2.COLOR_BGR2RGB))
                
                # You'll need to adjust this preprocessing based on your model's requirements
                # This is a generic example - modify according to your model's input format
                pineapple_tensor = torch.from_numpy(np.array(pineapple_pil)).float()
                pineapple_tensor = pineapple_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW format
                pineapple_tensor = pineapple_tensor / 255.0  # Normalize to [0,1]
                pineapple_tensor = pineapple_tensor.to(self.device)
                
                with torch.no_grad():
                    # Run inference with your model
                    output = self.model(pineapple_tensor)
                    
                    # Parse model output - adjust based on your model's output format
                    if isinstance(output, torch.Tensor):
                        predictions = output.cpu().numpy()
                    elif isinstance(output, dict):
                        predictions = output
                    else:
                        predictions = output[0].cpu().numpy() if hasattr(output[0], 'cpu') else output[0]
                    
                    # Extract ripeness and defect information from predictions
                    # This is model-specific - adjust according to your model's output
                    if len(predictions.shape) > 1 and predictions.shape[-1] >= 2:
                        ripeness_score = float(predictions[0][0]) if predictions.shape[-1] > 0 else 0.7
                        defect_score = float(predictions[0][1]) if predictions.shape[-1] > 1 else 0.3
                    else:
                        ripeness_score = float(predictions[0]) if len(predictions) > 0 else 0.7
                        defect_score = 0.3
                    
                    is_ripe = ripeness_score > 0.5
                    defect_count = max(0, int(defect_score * 6))  # Scale to 0-6 defects
                    
                    return {
                        'is_ripe': is_ripe,
                        'ripeness_confidence': ripeness_score,
                        'defect_count': defect_count,
                        'defect_confidence': defect_score
                    }
                    
            except Exception as e:
                logger.error(f"Error in model inference: {e}")
                # Fall back to mock analysis
                pass
        
        # Mock analysis when model is not available or fails
        return self._mock_analysis(pineapple_region)

    def _mock_analysis(self, pineapple_region):
        """Mock analysis for demonstration purposes"""
        # Analyze color distribution for ripeness (mock)
        hsv = cv2.cvtColor(pineapple_region, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
        yellow_ratio = np.sum(yellow_mask > 0) / (pineapple_region.shape[0] * pineapple_region.shape[1])
        
        is_ripe = yellow_ratio > 0.3
        ripeness_confidence = min(0.95, yellow_ratio + 0.4)
        
        # Mock defect detection based on dark spots
        gray = cv2.cvtColor(pineapple_region, cv2.COLOR_BGR2GRAY)
        dark_spots = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to count significant defects
        significant_defects = [c for c in contours if cv2.contourArea(c) > 100]
        defect_count = min(6, len(significant_defects))
        defect_confidence = min(0.95, len(significant_defects) * 0.15 + 0.5)
        
        return {
            'is_ripe': is_ripe,
            'ripeness_confidence': ripeness_confidence,
            'defect_count': defect_count,
            'defect_confidence': defect_confidence
        }

    def grade_pineapple(self, defect_count):
        """Grade pineapple based on defect count"""
        if defect_count == 0:
            return 'A'
        elif defect_count <= 3:
            return 'B'
        else:
            return 'C'

# Initialize analyzer
analyzer = PineappleAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': analyzer.model is not None,
        'yolo_loaded': analyzer.yolo_model is not None,
        'device': str(analyzer.device)
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Detect pineapples
        detections = analyzer.detect_pineapples(image)
        
        # Analyze each detected pineapple
        results = []
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            analysis = analyzer.analyze_pineapple_region(image, bbox)
            grade = analyzer.grade_pineapple(analysis['defect_count'])
            
            results.append({
                'id': i + 1,
                'bbox': bbox,
                'detection_confidence': detection['confidence'],
                'is_ripe': analysis['is_ripe'],
                'ripeness_confidence': analysis['ripeness_confidence'],
                'defect_count': analysis['defect_count'],
                'defect_confidence': analysis['defect_confidence'],
                'grade': grade
            })
        
        # Calculate summary statistics
        total_pineapples = len(results)
        ripe_count = sum(1 for r in results if r['is_ripe'])
        unripe_count = total_pineapples - ripe_count
        
        grade_counts = {'A': 0, 'B': 0, 'C': 0}
        for result in results:
            grade_counts[result['grade']] += 1
        
        # Encode processed image with bounding boxes
        annotated_image = image.copy()
        for result in results:
            x, y, w, h = result['bbox']
            color = (0, 255, 0) if result['grade'] == 'A' else (0, 255, 255) if result['grade'] == 'B' else (0, 0, 255)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            
            label = f"P{result['id']}: {'Ripe' if result['is_ripe'] else 'Unripe'}, {result['defect_count']} defects, Grade {result['grade']}"
            cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'summary': {
                'total_pineapples': total_pineapples,
                'ripe_count': ripe_count,
                'unripe_count': unripe_count,
                'grade_counts': grade_counts
            },
            'pineapples': results,
            'annotated_image': f"data:image/jpeg;base64,{annotated_image_b64}"
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # For real-time analysis, we'll focus on the center region
        h, w = image.shape[:2]
        center_bbox = [w//4, h//4, w//2, h//2]
        
        # Analyze the center region
        analysis = analyzer.analyze_pineapple_region(image, center_bbox)
        grade = analyzer.grade_pineapple(analysis['defect_count'])
        
        return jsonify({
            'success': True,
            'is_ripe': analysis['is_ripe'],
            'ripeness_confidence': analysis['ripeness_confidence'],
            'defect_count': analysis['defect_count'],
            'defect_confidence': analysis['defect_confidence'],
            'grade': grade
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)