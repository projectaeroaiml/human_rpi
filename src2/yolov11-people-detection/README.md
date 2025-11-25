# YOLOv11 People Detection

A comprehensive implementation of YOLOv11 for people detection with support for training, testing, and real-time webcam inference. This project is optimized for GPU acceleration and includes modular scripts for different use cases.

## 🚀 Features

- **GPU-accelerated training** with YOLOv11n model
- **Comprehensive testing and evaluation** with detailed metrics
- **Real-time webcam detection** with performance monitoring
- **Modular architecture** with reusable components
- **Detailed logging and progress tracking**
- **Flexible configuration system**
- **Multiple detection classes**: dry-person, object, wet-swimmer

## 📁 Project Structure

```
yolov11-people-detection/
├── src/
│   ├── train.py              # GPU-accelerated training script
│   ├── test.py               # Model evaluation and testing
│   ├── predict_webcam.py     # Real-time webcam prediction
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py       # Configuration parameters
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py    # Dataset utilities
│       └── model_utils.py    # Model management utilities
├── models/                   # Trained model weights
├── results/
│   ├── training/            # Training outputs and logs
│   ├── predictions/         # Test predictions
│   └── metrics/            # Evaluation metrics
├── requirements.txt
├── setup.py
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam (for live detection)

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify dataset structure:**
   The script expects the following structure in the parent directory:
   ```
   data.yaml                    # Dataset configuration
   yolo11n.pt                  # Pretrained YOLOv11n model
   train/images/               # Training images
   train/labels/               # Training labels
   valid/images/               # Validation images  
   valid/labels/               # Validation labels
   test/images/                # Test images
   test/labels/                # Test labels
   ```

## 🎯 Usage

### Training

**Basic training:**
```bash
cd src
python train.py
```

**Advanced training options:**
```bash
python train.py --epochs 200 --batch-size 32 --imgsz 640 --lr0 0.01 --device cuda
```

**Training parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--imgsz`: Image size (default: 640)
- `--lr0`: Initial learning rate (default: 0.01)
- `--device`: Device to use (cuda/cpu, default: auto-detect)
- `--resume`: Resume from checkpoint
- `--name`: Experiment name

**Example output:**
```
=== System Information ===
PyTorch version: 2.1.0
CUDA available: True
GPU 0: NVIDIA GeForce RTX 4090

=== Training Configuration ===
Data YAML: D:\Human_Rpi\data.yaml
Epochs: 100
Batch Size: 16
Learning Rate: 0.01
Device: CUDA

=== Dataset Information ===
Total Images: 1838
Classes (3): ['dry-person', 'object', 'wet-swimmer']
Train: 1472 images, 1472 labels
Valid: 183 images, 183 labels
Test: 183 images, 183 labels
```

### Testing

**Basic testing:**
```bash
python test.py
```

**Advanced testing options:**
```bash
python test.py --model models/best_model.pt --conf 0.25 --iou 0.45 --save-images --max-images 100
```

**Testing parameters:**
- `--model`: Path to trained model (auto-detected if not specified)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold (default: 0.45)
- `--imgsz`: Image size (default: 640)
- `--device`: Device to use (cuda/cpu)
- `--save-images`: Save annotated test images
- `--max-images`: Maximum images to process (default: 50)

**Example output:**
```
=== Test Results Summary ===
Model: best_model.pt
mAP50: 0.8945
mAP50-95: 0.7123
precision: 0.8567
recall: 0.8234
fitness: 0.7891

Processed: 50 images
Total detections: 234
Average detections per image: 4.68
```

### Real-time Webcam Detection

**Basic webcam detection:**
```bash
python predict_webcam.py
```

**Advanced webcam options:**
```bash
python predict_webcam.py --model models/best_model.pt --conf 0.3 --camera 0 --save-video --output webcam_demo.mp4
```

**Webcam parameters:**
- `--model`: Path to trained model (auto-detected if not specified)
- `--conf`: Confidence threshold (default: 0.25)
- `--device`: Device to use (cuda/cpu)
- `--camera`: Camera ID (default: 0)
- `--save-video`: Save video output
- `--output`: Output video path (default: webcam_output.mp4)

**Keyboard controls during webcam detection:**
- `q`: Quit
- `s`: Save screenshot
- `r`: Reset statistics

**Real-time display includes:**
- Live FPS counter
- Detection count
- Model information
- Device status
- Bounding boxes with confidence scores

## ⚙️ Configuration

### Dataset Configuration (data.yaml)
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3
names: ['dry-person', 'object', 'wet-swimmer']
```

### Training Configuration
Edit `src/config/settings.py` to modify:
- Training hyperparameters
- Data paths
- Model parameters
- Output directories

### Key Configuration Options
```python
TRAINING_CONFIG = {
    'epochs': 100,           # Training epochs
    'batch_size': 16,        # Batch size
    'imgsz': 640,           # Image size
    'lr0': 0.01,            # Learning rate
    'device': '',           # Auto-detect GPU/CPU
    'workers': 8,           # Data loading workers
    'amp': True,            # Automatic mixed precision
    'cache': False,         # Cache images in RAM
    'plots': True           # Generate training plots
}
```

## 📊 Results and Outputs

### Training Outputs
- **Models**: `models/best_model.pt`
- **Training logs**: `results/training/`
- **Training plots**: Loss curves, metrics, validation samples
- **Tensorboard logs**: For visualization

### Testing Outputs
- **Metrics report**: `results/metrics/test_report.json`
- **Annotated images**: `results/predictions/test_images/`
- **Validation plots**: Confusion matrix, PR curves

### Webcam Outputs
- **Screenshots**: Current directory
- **Video recordings**: Specified output path
- **Real-time statistics**: FPS, detection counts

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python train.py --batch-size 8  # Reduce batch size
   ```

2. **No GPU Detected**
   - Verify CUDA installation
   - Check PyTorch CUDA compatibility
   - Use `--device cpu` for CPU training

3. **Dataset Not Found**
   - Ensure `data.yaml` exists in parent directory
   - Verify image and label paths
   - Check file permissions

4. **Webcam Not Opening**
   ```bash
   python predict_webcam.py --camera 1  # Try different camera ID
   ```

5. **Model Not Found**
   - Train model first: `python train.py`
   - Or specify model path: `--model path/to/model.pt`

### Performance Optimization

1. **Training Speed**
   - Use larger batch sizes if GPU memory allows
   - Enable AMP (Automatic Mixed Precision)
   - Use multiple workers for data loading

2. **Inference Speed**
   - Use GPU for real-time detection
   - Optimize confidence threshold
   - Reduce image size if needed

3. **Memory Usage**
   - Reduce batch size
   - Disable image caching
   - Use half precision (FP16)

## 📈 Model Performance

The model is trained to detect three classes:
- **dry-person**: People not in water
- **object**: General objects
- **wet-swimmer**: People in water/swimming

Expected performance metrics:
- **mAP@0.5**: 0.85-0.95
- **mAP@0.5:0.95**: 0.65-0.80
- **Inference speed**: 30-60 FPS (RTX 4090)
- **Training time**: 2-4 hours (100 epochs, RTX 4090)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) for dataset management
- Dataset: "Tiny people detection RPI - v1"

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description

---

**Happy Detecting! 🎯**