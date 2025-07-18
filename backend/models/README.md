# ML Models Directory

This directory should contain your trained machine learning models:

## Required Models:

1. **YOLO Detection Model** (`pineapple_detection.pt`)
   - YOLOv5 or YOLOv8 model trained to detect pineapples
   - Should output bounding boxes and confidence scores

2. **Ripeness Classification Model** (`ripeness_classifier.pt`)
   - Model to classify pineapple ripeness (ripe/unripe)
   - Input: Cropped pineapple image
   - Output: Ripeness class and confidence

3. **Defect Detection Model** (`defect_detector.pt`)
   - Model to count and locate defects on pineapple surface
   - Input: Cropped pineapple image
   - Output: Number of defects detected

## Model Integration:

Replace the mock functions in `app.py` with actual model loading and inference:

```python
# Example model loading
import torch
from ultralytics import YOLO

class PineappleAnalyzer:
    def __init__(self):
        self.detection_model = YOLO('models/pineapple_detection.pt')
        self.ripeness_model = torch.load('models/ripeness_classifier.pt')
        self.defect_model = torch.load('models/defect_detector.pt')
```

## Performance Considerations:

- Models should be optimized for real-time inference
- Consider using TensorRT or ONNX for faster inference
- Implement model caching to avoid reloading
- Use appropriate input preprocessing and output postprocessing