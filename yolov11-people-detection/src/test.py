#!/usr/bin/env python3
"""
YOLOv11 Testing Script for People Detection
This script evaluates a trained YOLOv11 model on test data and generates detailed metrics.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import (
    DATA_YAML, MODEL_DIR, RESULTS_DIR, TESTING_CONFIG,
    CLASS_NAMES, COLORS
)
from utils.model_utils import (
    load_yolo_model, get_device, print_system_info, 
    validate_model_path, draw_predictions
)
from utils.data_loader import (
    validate_dataset_structure, get_dataset_statistics,
    load_image_safely
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_best_model() -> Path:
    """
    Find the best trained model
    
    Returns:
        Path: Path to best model
    """
    # Look for best model in multiple locations
    potential_paths = [
        MODEL_DIR / "best_model.pt",
        MODEL_DIR / "best.pt",
        RESULTS_DIR / "training" / "yolov11n_people_detection" / "weights" / "best.pt",
        RESULTS_DIR / "training" / "yolov11n_people_detection2" / "weights" / "best.pt",
        RESULTS_DIR / "training" / "yolov11n_people_detection3" / "weights" / "best.pt"
    ]
    
    for path in potential_paths:
        if path.exists():
            logger.info(f"Found trained model: {path}")
            return path
    
    # If no trained model found, look for any .pt file in models directory
    if MODEL_DIR.exists():
        pt_files = list(MODEL_DIR.glob("*.pt"))
        if pt_files:
            logger.warning(f"No best model found, using: {pt_files[0]}")
            return pt_files[0]
    
    logger.error("No trained model found!")
    return None

def setup_test_directories():
    """Create necessary directories for testing"""
    test_dirs = [
        RESULTS_DIR / "predictions",
        RESULTS_DIR / "predictions" / "test_results",
        RESULTS_DIR / "metrics"
    ]
    
    for directory in test_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Test directory ready: {directory}")

def validate_test_prerequisites(model_path: str) -> bool:
    """
    Validate prerequisites for testing
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        bool: True if all prerequisites are met
    """
    logger.info("Validating test prerequisites...")
    
    # Check if model exists
    if not validate_model_path(model_path):
        return False
    
    # Check if data.yaml exists
    if not DATA_YAML.exists():
        logger.error(f"Data YAML not found: {DATA_YAML}")
        return False
    
    # Validate dataset structure
    if not validate_dataset_structure(str(DATA_YAML)):
        logger.error("Dataset structure validation failed")
        return False
    
    logger.info("Test prerequisites validated successfully")
    return True

def run_model_validation(model: YOLO, custom_config: dict = None) -> dict:
    """
    Run model validation on test dataset
    
    Args:
        model (YOLO): Loaded YOLO model
        custom_config (dict): Custom validation configuration
        
    Returns:
        dict: Validation results
    """
    try:
        # Merge custom config with defaults
        config = TESTING_CONFIG.copy()
        if custom_config:
            config.update(custom_config)
        
        # Update config with data path
        config['data'] = str(DATA_YAML)
        
        logger.info("Running model validation...")
        logger.info(f"Validation results will be saved to: {config['project']}/{config['name']}")
        
        # Run validation
        results = model.val(**config)
        
        logger.info("Model validation completed!")
        return results
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise

def test_on_images(model: YOLO, test_images_dir: Path, output_dir: Path, 
                  conf_threshold: float = 0.25, max_images: int = 50):
    """
    Test model on individual images and save results
    
    Args:
        model (YOLO): Loaded YOLO model
        test_images_dir (Path): Directory containing test images
        output_dir (Path): Output directory for results
        conf_threshold (float): Confidence threshold
        max_images (int): Maximum number of images to process
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get test images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in test_images_dir.rglob('*') 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No images found in: {test_images_dir}")
            return
        
        # Limit number of images
        image_files = image_files[:max_images]
        logger.info(f"Processing {len(image_files)} test images...")
        
        detection_count = 0
        processed_count = 0
        
        for img_path in image_files:
            try:
                # Load image
                image = load_image_safely(str(img_path))
                if image is None:
                    continue
                
                # Run prediction
                results = model.predict(
                    source=str(img_path),
                    conf=conf_threshold,
                    save=False,
                    verbose=False
                )
                
                # Draw predictions
                annotated_image = draw_predictions(image, results, CLASS_NAMES, COLORS)
                
                # Count detections
                for result in results:
                    if result.boxes is not None:
                        detection_count += len(result.boxes)
                
                # Save annotated image
                output_path = output_dir / f"annotated_{img_path.name}"
                cv2.imwrite(str(output_path), annotated_image)
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count}/{len(image_files)} images")
                    
            except Exception as e:
                logger.warning(f"Error processing image {img_path}: {e}")
                continue
        
        logger.info(f"Image testing completed!")
        logger.info(f"Processed: {processed_count} images")
        logger.info(f"Total detections: {detection_count}")
        logger.info(f"Average detections per image: {detection_count/max(processed_count, 1):.2f}")
        
    except Exception as e:
        logger.error(f"Error in image testing: {e}")
        raise

def generate_test_report(results, model_path: str, output_dir: Path):
    """
    Generate comprehensive test report
    
    Args:
        results: Validation results from YOLO
        model_path (str): Path to tested model
        output_dir (Path): Output directory for report
    """
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'dataset': str(DATA_YAML),
            'classes': CLASS_NAMES,
            'metrics': {}
        }
        
        # Extract metrics from results
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            report['metrics'] = {
                'mAP50': float(metrics.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(metrics.get('metrics/mAP50-95(B)', 0)),
                'precision': float(metrics.get('metrics/precision(B)', 0)),
                'recall': float(metrics.get('metrics/recall(B)', 0)),
                'fitness': float(metrics.get('fitness', 0))
            }
        
        # Add dataset statistics
        dataset_stats = get_dataset_statistics(str(DATA_YAML))
        report['dataset_statistics'] = dataset_stats
        
        # Save report
        report_path = output_dir / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=== Test Results Summary ===")
        logger.info(f"Model: {Path(model_path).name}")
        for metric, value in report['metrics'].items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info(f"Detailed report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating test report: {e}")

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test YOLOv11 for People Detection")
    parser.add_argument('--model', type=str, default='', help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device (cuda/cpu)')
    parser.add_argument('--max-images', type=int, default=50, help='Max images to test')
    parser.add_argument('--save-images', action='store_true', help='Save annotated test images')
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Setup test directories
    setup_test_directories()
    
    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_best_model()
    
    if not model_path or not model_path.exists():
        logger.error("No valid model found for testing!")
        logger.info("Please train a model first or specify a valid model path with --model")
        sys.exit(1)
    
    # Validate prerequisites
    if not validate_test_prerequisites(str(model_path)):
        logger.error("Test prerequisites validation failed. Exiting.")
        sys.exit(1)
    
    try:
        # Load model
        logger.info(f"Loading model: {model_path}")
        model = load_yolo_model(str(model_path), args.device)
        
        # Custom validation config
        custom_config = {
            'conf': args.conf,
            'iou': args.iou,
            'imgsz': args.imgsz
        }
        
        if args.device:
            custom_config['device'] = args.device
        
        # Run validation
        start_time = datetime.now()
        logger.info(f"Testing started at: {start_time}")
        
        results = run_model_validation(model, custom_config)
        
        # Test on individual images if requested
        if args.save_images:
            test_images_dir = DATA_YAML.parent / "test" / "images"
            output_dir = RESULTS_DIR / "predictions" / "test_images"
            
            if test_images_dir.exists():
                test_on_images(model, test_images_dir, output_dir, 
                             args.conf, args.max_images)
            else:
                logger.warning(f"Test images directory not found: {test_images_dir}")
        
        # Generate report
        report_dir = RESULTS_DIR / "metrics"
        generate_test_report(results, str(model_path), report_dir)
        
        end_time = datetime.now()
        testing_duration = end_time - start_time
        logger.info(f"Testing completed at: {end_time}")
        logger.info(f"Total testing time: {testing_duration}")
        
        logger.info("Testing script finished successfully!")
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()