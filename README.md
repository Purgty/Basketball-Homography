# Basketball Homography Mapping

## Overview
Real-time basketball player tracking and mapping onto a 2D court template using computer vision and homography transformations.  
The pipeline detects court keypoints and player positions from video footage and projects them onto a standardized court for tactical visualization.

**Inspiration:** Based on a [LinkedIn post by Piotr Skalski](https://www.linkedin.com/in/piotr-skalski/).

---

## Key Features
- **Court Keypoint Detection**: YOLOv11m-pose for 33 keypoints  
- **Player Detection**: YOLOv11m, YOLOv11-seg, RF-DETR  
- **Homography Mapping**: Court-to-template projection via OpenCV  
- **Temporal Smoothing**: One Euro filtering and homography averaging  
- **Side Detection**: Dynamic left/right court recognition  
- **Optimized Deployment**: Supports ONNX and OpenVINO for hardware-specific inference  

---

## Directory Structure
```
basketball-homography/
│
├── Notebooks/
│   ├── ModelTraining.ipynb
│   ├── ModelConversion.ipynb
│   └── HomographyPipeline.ipynb
│
├── Models/
│   ├── court_model/
│   └── player_model/
│
├── Assets/
│   ├── full-court.jpeg
│   ├── homography_config.json
│   └── clip.webm
│
├── src/
│   └── video_homography.py
│
└── requirements.txt
```

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/basketball-homography.git
cd basketball-homography
```

2. **Setup the environment**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Components

### 1. Model Training
- **Court Keypoint Detection:** YOLOv11m-pose (33 keypoints: left, center, right)  
- **Player Detection:** YOLOv11m, YOLOv11-seg, RF-DETR  
- Includes data augmentation, evaluation, and hyperparameter tuning.

### 2. Model Conversion
Converts trained PyTorch models into optimized inference formats:  
- **ONNX:** For Edge/AMD systems  
- **OpenVINO:** For Intel systems  

Enables faster, smaller, and hardware-tuned models.

### 3. Homography Pipeline
- Calibrates YOLO-detected keypoints to court template  
- Computes homography via `cv2.findHomography()`  
- Projects player feet positions onto a 2D court  

**Core Functions:**
- `detect_side()` – Determines visible court half  
- `get_player_feet()` – Extracts player foot positions  
- `process_image()` – Executes the full mapping process  

---

## Usage
### Train models in `ModelTraining.ipynb`  
### Convert models with `ModelConversion.ipynb`
- **Intel:** Use OpenVINO models  
- **AMD/NVIDIA:** Use ONNX Runtime  
- Reduce input resolution or buffer size for speed  

### Homography Setup Instructions

The notebook includes code for setting up homography on a model court map. Calibration requires mapping `src_pts` and `dst_pts` between YOLO predictions and the court template.

**Map keypoints to labels (not provided in dataset):**
```python
LEFT_INDICES   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
RIGHT_INDICES  = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
CENTER_INDICES = [15, 16, 17]
```

**Create JSON of coordinates for the model court map:**
```python
dst_pts = {
    "left": np.array([[29.0, 32.0], [30.0, 53.0], [30.0, 133.0], [29.0, 230.0], [29.0, 312.0], [28.0, 329.0],
                      [62.0, 181.0], [86.0, 52.0], [91.0, 311.0], [143.0, 134.0], [143.0, 181.0], [142.0, 230.0],
                      [190.0, 34.0], [194.0, 177.0], [193.0, 330.0]], dtype=np.float32),
    "right": np.array([[419.0, 34.0], [418.0, 183.0], [418.0, 330.0], [468.0, 135.0], [468.0, 184.0], [470.0, 228.0],
                       [521.0, 55.0], [522.0, 311.0], [550.0, 181.0], [584.0, 32.0], [585.0, 54.0], [581.0, 134.0],
                       [582.0, 229.0], [583.0, 311.0], [583.0, 330.0]], dtype=np.float32),
    "center": np.array([[305.0, 35.0], [306.0, 182.0], [305.0, 330.0]], dtype=np.float32),
}
```
### Run mapping or video inference via notebook or CLI:

```bash
python src/video_homography.py     --input path/to/video.mp4 \                      # Input video  
    --output path/to/output.mp4 \                    # Output file path  
    --court-model Models/court_model/weights/best.pt     --player-model Models/player_model/weights/best.pt     --court-map Assets/full-court.jpeg
```

---

## Video Inference Workflow
Processes videos through detection, homography, smoothing, and tracking.

**Steps:**
1. Detect court keypoints and players per frame  
2. Identify visible court side  
3. Compute and smooth homography matrices  
4. Track player positions using One Euro Filter  
5. Project player positions onto the court map  
6. Combine original video and court visualization  

**Stabilization Techniques:**
- Homography averaging (15-frame buffer)  
- Adaptive One Euro filter  
- 80% majority-vote side detection  
- Occlusion recovery up to 2 frames  

---

## Output
- **Left Panel:** Original video with detected keypoints and players  
- **Right Panel:** 2D court map with projected player positions  

---

## Future Enhancements
- Multi-camera calibration  
- Jersey and team identification  
- Ball tracking and heatmap generation  
- Tactical event analysis  

---

## Acknowledgments
- Piotr Skalski – Original concept inspiration  
- Ultralytics – YOLOv11 framework  
- OpenCV / OpenVINO – Core computer vision libraries  

---

## License
MIT License – see LICENSE for details.
