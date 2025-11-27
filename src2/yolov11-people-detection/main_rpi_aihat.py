#!/usr/bin/env python3
"""
YOLOv11 People Detection - Raspberry Pi 5 with AI Hat (Hailo-8L) Optimized Version

This script is optimized for running on Raspberry Pi 5 with the AI Hat accelerator.
Key optimizations:
- Hailo NPU acceleration support
- Picamera2 integration for native camera support
- Lower resolution options for better performance
- Memory-efficient processing
- Headless mode support
- Frame rate throttling

Usage:
    python main_rpi_aihat.py webcam                 # Run with default settings
    python main_rpi_aihat.py webcam --use-hailo     # Use Hailo accelerator
    python main_rpi_aihat.py webcam --picamera      # Use Pi Camera module
    python main_rpi_aihat.py webcam --resolution 320x240 --fps 15  # Lower resolution
"""

import os
import sys
import cv2
import logging
import argparse
import time
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import config and utilities
try:
    from config.settings import MODEL_DIR, CLASS_NAMES, COLORS
    from utils.model_utils import cluster_detections, calculate_group_center, calculate_pan_tilt
except ImportError as e:
    logger.warning(f"Could not import from src: {e}")
    # Fallback defaults
    MODEL_DIR = Path(__file__).parent / "models"
    CLASS_NAMES = ['person']
    COLORS = {'person': (0, 255, 0)}

# Check for Hailo availability
HAILO_AVAILABLE = False
try:
    from hailo_platform import HEF, VDevice, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams
    HAILO_AVAILABLE = True
    logger.info("Hailo SDK detected - NPU acceleration available")
except ImportError:
    logger.info("Hailo SDK not found - will use CPU/GPU inference")

# Check for Picamera2 availability
PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    logger.info("Picamera2 detected - Pi Camera support available")
except ImportError:
    logger.info("Picamera2 not found - will use OpenCV VideoCapture")


class RPiAIHatPredictor:
    """
    Optimized predictor for Raspberry Pi 5 with AI Hat
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        camera_id: int = 0,
        use_picamera: bool = False,
        use_hailo: bool = False,
        resolution: Tuple[int, int] = (640, 480),
        target_fps: int = 30,
        clustering_distance: float = 120.0,
        headless: bool = False
    ):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the YOLO model (.pt, .onnx, or .hef for Hailo)
            conf_threshold: Detection confidence threshold
            camera_id: Camera device ID (for USB cameras)
            use_picamera: Use Picamera2 for Pi Camera module
            use_hailo: Use Hailo NPU for inference
            resolution: Camera resolution (width, height)
            target_fps: Target frame rate
            clustering_distance: Distance threshold for clustering detections
            headless: Run without display (save frames to disk)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.camera_id = camera_id
        self.use_picamera = use_picamera and PICAMERA2_AVAILABLE
        self.use_hailo = use_hailo and HAILO_AVAILABLE
        self.resolution = resolution
        self.target_fps = target_fps
        self.clustering_distance = clustering_distance
        self.headless = headless
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        # Frame interval for FPS limiting
        self.frame_interval = 1.0 / target_fps
        
        # Detection results
        self.current_detections = []
        self.detection_count = 0
        
        # Class info
        self.class_names = CLASS_NAMES
        self.colors = COLORS
        
        # Initialize components
        self._load_model()
        self._initialize_camera()
        
    def _load_model(self):
        """Load the detection model"""
        model_path = Path(self.model_path)
        
        if self.use_hailo:
            self._load_hailo_model(model_path)
        else:
            self._load_ultralytics_model(model_path)
    
    def _load_hailo_model(self, model_path: Path):
        """Load model for Hailo NPU inference"""
        logger.info("Loading model for Hailo NPU...")
        
        # Check for .hef file (Hailo Executable Format)
        hef_path = model_path.with_suffix('.hef')
        
        if not hef_path.exists():
            # Try to find pre-converted HEF in models directory
            hef_candidates = list(MODEL_DIR.glob("*.hef"))
            if hef_candidates:
                hef_path = hef_candidates[0]
                logger.info(f"Using found HEF file: {hef_path}")
            else:
                logger.warning(f"No HEF file found. You need to convert your model to Hailo format.")
                logger.warning("Falling back to CPU inference with ultralytics...")
                self.use_hailo = False
                self._load_ultralytics_model(model_path)
                return
        
        try:
            # Initialize Hailo device
            self.hailo_vdevice = VDevice()
            self.hailo_hef = HEF(str(hef_path))
            
            # Configure network
            configure_params = ConfigureParams.create_from_hef(
                self.hailo_hef, 
                interface=self.hailo_vdevice.get_default_streams_interface()
            )
            
            self.hailo_network_group = self.hailo_vdevice.configure(
                self.hailo_hef, 
                configure_params
            )[0]
            
            # Get input/output info
            self.hailo_input_vstream_info = self.hailo_hef.get_input_vstream_infos()[0]
            self.hailo_output_vstream_info = self.hailo_hef.get_output_vstream_infos()
            
            logger.info(f"Hailo model loaded: {hef_path}")
            logger.info(f"Input shape: {self.hailo_input_vstream_info.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load Hailo model: {e}")
            logger.warning("Falling back to CPU inference...")
            self.use_hailo = False
            self._load_ultralytics_model(model_path)
    
    def _load_ultralytics_model(self, model_path: Path):
        """Load model using ultralytics YOLO"""
        logger.info("Loading model with ultralytics...")
        
        try:
            from ultralytics import YOLO
            
            # Prefer ONNX or TFLite for better RPi performance
            onnx_path = model_path.with_suffix('.onnx')
            tflite_path = model_path.with_suffix('.tflite')
            
            if onnx_path.exists():
                logger.info(f"Using ONNX model: {onnx_path}")
                self.model = YOLO(str(onnx_path))
            elif tflite_path.exists():
                logger.info(f"Using TFLite model: {tflite_path}")
                self.model = YOLO(str(tflite_path))
            elif model_path.exists():
                logger.info(f"Using PyTorch model: {model_path}")
                self.model = YOLO(str(model_path))
            else:
                # Try to find any model in MODEL_DIR
                model_files = list(MODEL_DIR.glob("*.pt")) + list(MODEL_DIR.glob("*.onnx"))
                if model_files:
                    logger.info(f"Using found model: {model_files[0]}")
                    self.model = YOLO(str(model_files[0]))
                else:
                    raise FileNotFoundError(f"No model found at {model_path}")
            
            # Set device to CPU for RPi (unless CUDA is somehow available)
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _initialize_camera(self):
        """Initialize the camera"""
        if self.use_picamera:
            self._initialize_picamera()
        else:
            self._initialize_opencv_camera()
    
    def _initialize_picamera(self):
        """Initialize Pi Camera using Picamera2"""
        logger.info("Initializing Pi Camera with Picamera2...")
        
        try:
            self.picam = Picamera2()
            
            # Configure for video capture
            config = self.picam.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.target_fps}
            )
            
            self.picam.configure(config)
            self.picam.start()
            
            # Warm up
            time.sleep(0.5)
            
            logger.info(f"Pi Camera initialized: {self.resolution[0]}x{self.resolution[1]} @ {self.target_fps}fps")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pi Camera: {e}")
            logger.warning("Falling back to OpenCV VideoCapture...")
            self.use_picamera = False
            self._initialize_opencv_camera()
    
    def _initialize_opencv_camera(self):
        """Initialize camera using OpenCV"""
        logger.info(f"Initializing camera {self.camera_id} with OpenCV...")
        
        # Try different backends
        backends = [
            ('V4L2', cv2.CAP_V4L2),
            ('Default', cv2.CAP_ANY),
        ]
        
        self.cap = None
        
        for name, backend in backends:
            try:
                logger.info(f"Trying {name} backend...")
                cap = cv2.VideoCapture(self.camera_id, backend)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        logger.info(f"Camera opened with {name} backend")
                        break
                    cap.release()
            except Exception as e:
                logger.warning(f"{name} backend failed: {e}")
        
        if self.cap is None:
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    def _read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the camera"""
        if self.use_picamera:
            try:
                frame = self.picam.capture_array()
                # Picamera2 returns RGB, OpenCV uses BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            except Exception as e:
                logger.error(f"Error reading Pi Camera frame: {e}")
                return None
        else:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            return None
    
    def _preprocess_for_hailo(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for Hailo inference"""
        input_shape = self.hailo_input_vstream_info.shape
        
        # Resize to model input size
        if len(input_shape) == 4:
            h, w = input_shape[1], input_shape[2]
        else:
            h, w = input_shape[0], input_shape[1]
        
        resized = cv2.resize(frame, (w, h))
        
        # Normalize if needed (depends on model training)
        # Most YOLO models expect 0-255 uint8
        return resized.astype(np.uint8)
    
    def _run_hailo_inference(self, frame: np.ndarray) -> list:
        """Run inference using Hailo NPU"""
        try:
            # Preprocess
            input_data = self._preprocess_for_hailo(frame)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            input_vstream_params = InputVStreamParams.make_from_network_group(
                self.hailo_network_group,
                quantized=False
            )
            output_vstream_params = OutputVStreamParams.make_from_network_group(
                self.hailo_network_group,
                quantized=False
            )
            
            with InferVStreams(
                self.hailo_network_group,
                input_vstream_params,
                output_vstream_params
            ) as infer_pipeline:
                input_dict = {self.hailo_input_vstream_info.name: input_data}
                output = infer_pipeline.infer(input_dict)
            
            # Post-process output
            return self._postprocess_hailo_output(output, frame.shape[:2])
            
        except Exception as e:
            logger.error(f"Hailo inference error: {e}")
            return []
    
    def _postprocess_hailo_output(self, output: dict, orig_shape: Tuple[int, int]) -> list:
        """Post-process Hailo NPU output to get detections"""
        detections = []
        
        # This is a simplified post-processing - adjust based on your model's output format
        for out_name, out_data in output.items():
            # Assuming standard YOLO output format
            out_data = np.squeeze(out_data)
            
            # Process detections
            # Format depends on the specific model - this is a general approach
            if out_data.ndim == 2 and out_data.shape[1] >= 5:
                for detection in out_data:
                    confidence = detection[4]
                    if confidence >= self.conf_threshold:
                        # Scale coordinates to original frame size
                        x_center = detection[0] * orig_shape[1]
                        y_center = detection[1] * orig_shape[0]
                        width = detection[2] * orig_shape[1]
                        height = detection[3] * orig_shape[0]
                        
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': float(confidence),
                            'class_id': 0  # person
                        })
        
        return detections
    
    def _run_ultralytics_inference(self, frame: np.ndarray) -> list:
        """Run inference using ultralytics YOLO"""
        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                imgsz=self.resolution[0]  # Use smaller inference size for speed
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        detections.append({
                            'box': tuple(map(int, box)),
                            'confidence': float(conf),
                            'class_id': 0
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Ultralytics inference error: {e}")
            return []
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process a frame and return annotated result"""
        # Run inference
        if self.use_hailo:
            detections = self._run_hailo_inference(frame)
        else:
            detections = self._run_ultralytics_inference(frame)
        
        self.current_detections = detections
        self.detection_count = len(detections)
        
        # Annotate frame
        annotated = self._annotate_frame(frame, detections)
        
        return annotated, len(detections)
    
    def _annotate_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detections on frame with clustering"""
        annotated = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        frame_center = (frame_w // 2, frame_h // 2)
        
        if not detections:
            return annotated
        
        # Collect centers for clustering
        centers = []
        boxes = []
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
            boxes.append(det['box'])
        
        # Cluster detections
        try:
            clusters = cluster_detections(centers, self.clustering_distance)
        except:
            # Fallback if clustering fails
            clusters = [[i] for i in range(len(centers))]
        
        # Define colors by group size
        colors_by_size = {
            'small': (0, 255, 0),    # Green (1-2)
            'medium': (0, 255, 255), # Yellow (3-4)
            'large': (0, 0, 255),    # Red (5+)
        }
        
        largest_group_size = 0
        largest_group_center = None
        
        for cluster_indices in clusters:
            group_size = len(cluster_indices)
            
            # Choose color
            if group_size <= 2:
                color = colors_by_size['small']
            elif group_size <= 4:
                color = colors_by_size['medium']
            else:
                color = colors_by_size['large']
            
            # Draw boxes
            for idx in cluster_indices:
                x1, y1, x2, y2 = boxes[idx]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw confidence
                conf = detections[idx]['confidence']
                cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Calculate and draw group center
            try:
                group_center = calculate_group_center(centers, cluster_indices)
            except:
                group_xs = [centers[i][0] for i in cluster_indices]
                group_ys = [centers[i][1] for i in cluster_indices]
                group_center = (sum(group_xs) // len(group_xs), sum(group_ys) // len(group_ys))
            
            cv2.circle(annotated, group_center, 6, color, -1)
            cv2.putText(annotated, f"G:{group_size}", 
                       (group_center[0] + 10, group_center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Track largest group
            if group_size > largest_group_size:
                largest_group_size = group_size
                largest_group_center = group_center
        
        # Draw frame center
        cv2.circle(annotated, frame_center, 5, (255, 0, 255), -1)
        
        # Draw line to largest group
        if largest_group_center:
            cv2.line(annotated, largest_group_center, frame_center, (255, 0, 255), 1)
            
            # Calculate pan/tilt
            try:
                pan, tilt = calculate_pan_tilt(largest_group_center, frame_center, frame_w, frame_h)
            except:
                dx = largest_group_center[0] - frame_center[0]
                dy = largest_group_center[1] - frame_center[1]
                pan = (dx / frame_w) * 60
                tilt = -(dy / frame_h) * 45
            
            cv2.putText(annotated, f"Pan:{pan:.1f} Tilt:{tilt:.1f}",
                       (10, frame_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _draw_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw info overlay"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Info text
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Detections: {self.detection_count}",
            f"Conf: {self.conf_threshold:.2f}",
            f"Device: {'Hailo' if self.use_hailo else 'CPU'}",
            f"Camera: {'PiCam' if self.use_picamera else 'USB'}"
        ]
        
        for i, text in enumerate(info_lines):
            cv2.putText(frame, text, (10, 20 + i * 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        if hasattr(self, '_last_fps_time'):
            delta = current_time - self._last_fps_time
            if delta > 0:
                self.fps_queue.append(1.0 / delta)
        self._last_fps_time = current_time
        
        return sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0
    
    def run(self, save_video: bool = False, output_path: str = None):
        """Main prediction loop"""
        logger.info("Starting prediction loop...")
        logger.info("Controls: q=quit, s=screenshot, +/-=confidence, h=help")
        
        # Setup video writer
        video_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, self.target_fps, self.resolution)
            logger.info(f"Recording to: {output_path}")
        
        # Window setup
        window_name = "YOLOv11 People Detection - RPi AI Hat"
        if not self.headless:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            except:
                logger.warning("Cannot create window - switching to headless mode")
                self.headless = True
        
        screenshot_count = 0
        failed_reads = 0
        
        try:
            while True:
                # Frame rate limiting
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                self.last_frame_time = time.time()
                
                # Read frame
                frame = self._read_frame()
                
                if frame is None:
                    failed_reads += 1
                    if failed_reads > 10:
                        logger.error("Too many failed reads - stopping")
                        break
                    time.sleep(0.1)
                    continue
                
                failed_reads = 0
                
                # Process frame
                annotated, det_count = self._process_frame(frame)
                
                # Calculate FPS
                fps = self._calculate_fps()
                self.frame_count += 1
                
                # Draw overlay
                final_frame = self._draw_overlay(annotated, fps)
                
                # Save to video
                if video_writer:
                    video_writer.write(final_frame)
                
                # Display
                if not self.headless:
                    try:
                        cv2.imshow(window_name, final_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('q') or key == 27:
                            logger.info("Quit requested")
                            break
                        elif key == ord('s'):
                            screenshot_count += 1
                            path = f"screenshot_{screenshot_count:03d}.jpg"
                            cv2.imwrite(path, final_frame)
                            logger.info(f"Screenshot saved: {path}")
                        elif key == ord('+') or key == ord('='):
                            self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                            logger.info(f"Confidence: {self.conf_threshold:.2f}")
                        elif key == ord('-'):
                            self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                            logger.info(f"Confidence: {self.conf_threshold:.2f}")
                        elif key == ord('h'):
                            logger.info("=== Controls ===")
                            logger.info("q/ESC: Quit | s: Screenshot | +/-: Confidence")
                    except cv2.error:
                        self.headless = True
                        logger.warning("Display error - switching to headless mode")
                else:
                    # Headless mode - periodic saves
                    if self.frame_count % 30 == 0:
                        path = f"frame_{self.frame_count:06d}.jpg"
                        cv2.imwrite(path, final_frame)
                    time.sleep(0.01)
                
                # Periodic stats
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed
                    logger.info(f"Frames: {self.frame_count}, Avg FPS: {avg_fps:.1f}")
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup(video_writer)
    
    def _cleanup(self, video_writer=None):
        """Clean up resources"""
        if video_writer:
            video_writer.release()
            logger.info("Video saved")
        
        if self.use_picamera:
            try:
                self.picam.stop()
            except:
                pass
        else:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        if self.use_hailo and hasattr(self, 'hailo_vdevice'):
            try:
                self.hailo_vdevice.release()
            except:
                pass
        
        logger.info("Cleanup complete")


def find_model() -> Optional[Path]:
    """Find the best available model"""
    search_paths = [
        MODEL_DIR / "best_model.pt",
        MODEL_DIR / "best_model.onnx",
        MODEL_DIR / "best_model.hef",
        MODEL_DIR / "best.pt",
    ]
    
    # Also check training results
    training_dir = MODEL_DIR.parent / "results" / "training"
    if training_dir.exists():
        for d in training_dir.glob("yolov11n_people_detection*"):
            weights = d / "weights" / "best.pt"
            if weights.exists():
                search_paths.append(weights)
    
    for path in search_paths:
        if path.exists():
            logger.info(f"Found model: {path}")
            return path
    
    return None


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """Parse resolution string like '640x480'"""
    try:
        w, h = res_str.lower().split('x')
        return (int(w), int(h))
    except:
        return (640, 480)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 People Detection - Raspberry Pi 5 with AI Hat",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('mode', choices=['webcam', 'test'], 
                       help='Operation mode')
    parser.add_argument('--model', type=str, default='',
                       help='Path to model file')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID')
    parser.add_argument('--picamera', action='store_true',
                       help='Use Pi Camera module')
    parser.add_argument('--use-hailo', action='store_true',
                       help='Use Hailo NPU for inference')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Camera resolution (WxH)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS')
    parser.add_argument('--clustering', type=float, default=120.0,
                       help='Clustering distance')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Output video path')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("🎯 YOLOv11 People Detection - Raspberry Pi 5 AI Hat Edition")
    print("=" * 60)
    
    # Print available features
    print(f"Hailo NPU: {'Available' if HAILO_AVAILABLE else 'Not Available'}")
    print(f"Picamera2: {'Available' if PICAMERA2_AVAILABLE else 'Not Available'}")
    print("=" * 60)
    
    if args.mode == 'test':
        print("Running system test...")
        print(f"Python: {sys.version}")
        try:
            import cv2
            print(f"OpenCV: {cv2.__version__}")
        except:
            print("OpenCV: Not found")
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA: {torch.cuda.is_available()}")
        except:
            print("PyTorch: Not found")
        return
    
    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_model()
    
    if not model_path or not model_path.exists():
        logger.error("No model found! Please specify with --model or place in models/")
        sys.exit(1)
    
    # Parse resolution
    resolution = parse_resolution(args.resolution)
    
    try:
        predictor = RPiAIHatPredictor(
            model_path=str(model_path),
            conf_threshold=args.conf,
            camera_id=args.camera,
            use_picamera=args.picamera,
            use_hailo=args.use_hailo,
            resolution=resolution,
            target_fps=args.fps,
            clustering_distance=args.clustering,
            headless=args.headless
        )
        
        predictor.run(
            save_video=args.save_video,
            output_path=args.output if args.save_video else None
        )
        
        print("=" * 60)
        print("✅ Detection completed!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
