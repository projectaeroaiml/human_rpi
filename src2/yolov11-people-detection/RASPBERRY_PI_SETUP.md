# Raspberry Pi 5 Setup Guide

## Camera Setup for Raspberry Pi 5

### Hardware Options

1. **Raspberry Pi Camera Module (CSI)**
   - Connect to CSI camera port
   - Enable camera interface in raspi-config

2. **USB Webcam**
   - Connect to any USB port
   - Usually detected automatically

### Software Setup

#### 1. Enable Camera Interface (for Pi Camera Module)

```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
sudo reboot
```

#### 2. Install Required Packages

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Python dependencies
sudo apt install -y python3-pip python3-opencv

# Install additional video libraries
sudo apt install -y libopencv-dev v4l-utils

# For Pi Camera module support
sudo apt install -y python3-picamera2
```

#### 3. Install Python Requirements

```bash
cd ~/yolov11-people-detection
pip3 install -r requirements.txt
```

### Testing the Camera

#### Option 1: Use the Test Script

```bash
python3 test_rpi_camera.py
```

This will test all available camera backends and show which ones work.

#### Option 2: Manual Testing

**For Pi Camera Module:**
```bash
# Test with raspistill
raspistill -o test.jpg

# List camera devices
libcamera-hello --list-cameras
```

**For USB Camera:**
```bash
# List video devices
ls /dev/video*

# Get device info
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

**Test with Python:**
```python
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
ret, frame = cap.read()
print(f"Success: {ret}, Frame shape: {frame.shape if ret else 'N/A'}")
cap.release()
```

### Running YOLOv11 Detection

#### Find Your Camera Index

```bash
# List all video devices
ls -l /dev/video*

# Test different camera indices
python3 test_rpi_camera.py 0  # Try camera 0
python3 test_rpi_camera.py 1  # Try camera 1
```

#### Run Detection

```bash
# Basic usage (camera 0)
python3 src/predict_webcam.py

# Specify camera index
python3 src/predict_webcam.py --camera 0

# With custom confidence
python3 src/predict_webcam.py --camera 0 --conf 0.3

# Via main launcher
python3 main.py webcam
```

### Common Issues and Solutions

#### Issue 1: "Failed to read frame from camera"

**Causes:**
- Wrong camera index
- Camera not properly initialized
- Insufficient permissions
- Backend incompatibility

**Solutions:**

1. **Find correct camera index:**
   ```bash
   ls /dev/video*
   python3 test_rpi_camera.py
   ```

2. **Check permissions:**
   ```bash
   # Add user to video group
   sudo usermod -a -G video $USER
   
   # Check current groups
   groups
   
   # Logout and login again, or:
   newgrp video
   ```

3. **Try different camera index:**
   ```bash
   python3 src/predict_webcam.py --camera 0
   python3 src/predict_webcam.py --camera 1
   python3 src/predict_webcam.py --camera 2
   ```

4. **Test camera directly:**
   ```bash
   # For USB camera
   ffplay /dev/video0
   
   # For Pi Camera
   libcamera-hello
   ```

#### Issue 2: Camera opens but no frames

**Solutions:**

1. **Increase buffer size:**
   Edit `src/predict_webcam.py`:
   ```python
   self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
   ```

2. **Set explicit format:**
   ```python
   self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
   ```

3. **Try lower resolution:**
   ```python
   self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
   self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
   ```

#### Issue 3: Low FPS or performance issues

**Solutions:**

1. **Use hardware acceleration:**
   - Ensure TensorFlow Lite or ONNX runtime for ARM
   - Consider quantized models (INT8)

2. **Lower resolution:**
   ```bash
   python3 src/predict_webcam.py --camera 0 --conf 0.3
   ```

3. **Optimize system:**
   ```bash
   # Increase GPU memory
   sudo raspi-config
   # Advanced Options > GPU Memory > Set to 256 or 512
   
   # Overclock (if using good cooling)
   sudo raspi-config
   # Performance Options > Overclock
   ```

4. **Disable GUI if running headless:**
   The script auto-detects when GUI is unavailable.

#### Issue 4: "backend is generally available but can't be used"

**Solution:**
This is just a warning. The script will try multiple backends automatically.

If you want to force a specific backend, edit `src/predict_webcam.py`:

```python
# For V4L2 (most common on Linux)
cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

# For GStreamer
cap = cv2.VideoCapture(camera_id, cv2.CAP_GSTREAMER)
```

### Performance Optimization for Raspberry Pi 5

#### 1. Use Optimized Model

```bash
# Export to ONNX for better performance
python3 -c "from ultralytics import YOLO; model = YOLO('models/best_model.pt'); model.export(format='onnx')"

# Or use TFLite
python3 -c "from ultralytics import YOLO; model = YOLO('models/best_model.pt'); model.export(format='tflite')"
```

#### 2. System Configuration

```bash
# Enable performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### 3. Monitor Performance

```bash
# Check CPU temperature
vcgencmd measure_temp

# Monitor system resources
htop

# Watch GPU memory
vcgencmd get_mem gpu
```

### Camera Backend Priority (Raspberry Pi)

The script automatically tries backends in this order:

1. **V4L2** (Video4Linux2) - Best for USB cameras
2. **GStreamer** - Good for Pi Camera module
3. **Default** - OpenCV's automatic selection

### Headless Operation (No Display)

The script automatically detects when GUI is not available and runs in headless mode, saving frames but not displaying them.

To view results:
- Use `--save-video` flag to record output
- Access via VNC or remote desktop
- Use X11 forwarding over SSH: `ssh -X pi@raspberrypi`

### Example Commands

```bash
# Test camera
python3 test_rpi_camera.py

# Run with camera 0, confidence 0.25
python3 src/predict_webcam.py --camera 0 --conf 0.25

# Save video output
python3 src/predict_webcam.py --camera 0 --save-video --output detection_output.mp4

# Use main launcher
python3 main.py webcam
```

### Getting Help

If you still have issues:

1. Run the camera test script and share the output:
   ```bash
   python3 test_rpi_camera.py 2>&1 | tee camera_test.log
   ```

2. Check camera detection:
   ```bash
   v4l2-ctl --list-devices > devices.log
   ```

3. Check OpenCV build info:
   ```bash
   python3 -c "import cv2; print(cv2.getBuildInformation())" > opencv_info.log
   ```

### Additional Resources

- [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [OpenCV on Raspberry Pi](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
- [YOLOv11 Documentation](https://docs.ultralytics.com/)
