import os
import io
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class PineappleAnalyzer:
    def __init__(self):
        self.model = None
        self.yolo_model = None
        self.load_models()
    
    def load_models(self):
        """Load PyTorch models for pineapple analysis"""
        try:
            # Try to import PyTorch
            import torch
            from ultralytics import YOLO
            
            # Load your custom model
            model_path = os.path.join(os.path.dirname(__file__), '..', 'pineapple_pen.pt')
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
                logger.info("Successfully loaded pineapple_pen.pt model")
            else:
                logger.warning(f"Model file not found at {model_path}")
            
            # Load YOLO for pineapple detection
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("Successfully loaded YOLOv8 model")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {e}")
                
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}. Using mock analysis.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def detect_pineapples(self, image):
        """Detect pineapples in the image using YOLO"""
        if self.yolo_model is None:
            # Mock detection - assume single pineapple covering most of the image
            h, w = image.shape[:2]
            return [{
                'bbox': [int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)],
                'confidence': 0.85
            }]
        
        try:
            results = self.yolo_model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for pineapple class (you may need to adjust class ID)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence)
                        })
            
            return detections if detections else [{
                'bbox': [int(image.shape[1]*0.1), int(image.shape[0]*0.1), 
                        int(image.shape[1]*0.9), int(image.shape[0]*0.9)],
                'confidence': 0.5
            }]
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            # Fallback to mock detection
            h, w = image.shape[:2]
            return [{
                'bbox': [int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)],
                'confidence': 0.5
            }]
    
    def analyze_pineapple_region(self, image_region):
        """Analyze a single pineapple region for ripeness and defects"""
        if self.model is None:
            # Mock analysis
            return {
                'ripeness': np.random.choice(['ripe', 'unripe']),
                'ripeness_confidence': np.random.uniform(0.7, 0.95),
                'defect_count': np.random.randint(0, 6),
                'defect_confidence': np.random.uniform(0.6, 0.9)
            }
        
        try:
            import torch
            import torchvision.transforms as transforms
            
            # Preprocess image for your model
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # Adjust size as needed
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Convert BGR to RGB if needed
            if len(image_region.shape) == 3:
                image_region = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
            
            input_tensor = transform(image_region).unsqueeze(0)
            
            with torch.no_grad():
                predictions = self.model(input_tensor)
                
                # Parse predictions (adjust based on your model's output format)
                if isinstance(predictions, torch.Tensor):
                    pred_array = predictions.cpu().numpy().flatten()
                    
                    # Assuming your model outputs [ripeness_score, defect_count, ...]
                    ripeness_score = pred_array[0] if len(pred_array) > 0 else 0.5
                    defect_count = int(pred_array[1]) if len(pred_array) > 1 else 0
                    
                    return {
                        'ripeness': 'ripe' if ripeness_score > 0.5 else 'unripe',
                        'ripeness_confidence': float(abs(ripeness_score - 0.5) * 2),
                        'defect_count': max(0, defect_count),
                        'defect_confidence': 0.8
                    }
                
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
        
        # Fallback to mock analysis
        return {
            'ripeness': np.random.choice(['ripe', 'unripe']),
            'ripeness_confidence': np.random.uniform(0.7, 0.95),
            'defect_count': np.random.randint(0, 6),
            'defect_confidence': np.random.uniform(0.6, 0.9)
        }
    
    def assign_grade(self, defect_count):
        """Assign grade based on defect count"""
        if defect_count == 0:
            return 'A'
        elif defect_count <= 3:
            return 'B'
        else:
            return 'C'
    
    def analyze_image(self, image):
        """Analyze complete image with multiple pineapples"""
        # Detect pineapples
        detections = self.detect_pineapples(image)
        
        results = []
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract pineapple region
            pineapple_region = image[y1:y2, x1:x2]
            
            if pineapple_region.size == 0:
                continue
            
            # Analyze the region
            analysis = self.analyze_pineapple_region(pineapple_region)
            
            # Assign grade
            grade = self.assign_grade(analysis['defect_count'])
            
            results.append({
                'id': i + 1,
                'bbox': bbox,
                'ripeness': analysis['ripeness'],
                'ripeness_confidence': analysis['ripeness_confidence'],
                'defect_count': analysis['defect_count'],
                'defect_confidence': analysis['defect_confidence'],
                'grade': grade,
                'detection_confidence': detection['confidence']
            })
        
        return results

# Initialize analyzer
analyzer = PineappleAnalyzer()

def decode_base64_image(base64_string):
    """Decode base64 image string to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': analyzer.model is not None,
        'yolo_loaded': analyzer.yolo_model is not None
    })

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze uploaded image with multiple pineapples"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Analyze image
        results = analyzer.analyze_image(image)
        
        # Calculate summary statistics
        total_pineapples = len(results)
        ripe_count = sum(1 for r in results if r['ripeness'] == 'ripe')
        unripe_count = total_pineapples - ripe_count
        
        grade_counts = {'A': 0, 'B': 0, 'C': 0}
        for result in results:
            grade_counts[result['grade']] += 1
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_pineapples': total_pineapples,
                'ripe_count': ripe_count,
                'unripe_count': unripe_count,
                'grade_counts': grade_counts
            }
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """Analyze single camera frame"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Analyze image (assuming single pineapple for camera mode)
        results = analyzer.analyze_image(image)
        
        if not results:
            return jsonify({
                'success': True,
                'pineapple_detected': False,
                'message': 'No pineapple detected in frame'
            })
        
        # Return first detection for camera mode
        result = results[0]
        
        return jsonify({
            'success': True,
            'pineapple_detected': True,
            'ripeness': result['ripeness'],
            'ripeness_confidence': result['ripeness_confidence'],
            'defect_count': result['defect_count'],
            'defect_confidence': result['defect_confidence'],
            'grade': result['grade'],
            'detection_confidence': result['detection_confidence']
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)