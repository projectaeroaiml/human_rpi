#!/usr/bin/env python3
"""
YOLOv11 Training Script for People Detection
This script trains a YOLOv11 model on the people detection dataset with GPU support.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from ultralytics import YOLO
from config.settings import (
    DATA_YAML, PRETRAINED_MODEL, TRAINING_CONFIG, 
    MODEL_DIR, RESULTS_DIR
)
from utils.model_utils import get_device, print_system_info, validate_model_path
from utils.data_loader import validate_dataset_structure, get_dataset_statistics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for training"""
    directories = [MODEL_DIR, RESULTS_DIR, RESULTS_DIR / "training"]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {directory}")

def validate_prerequisites():
    """Validate all prerequisites before training"""
    logger.info("Validating prerequisites...")
    
    # Check if data.yaml exists
    if not DATA_YAML.exists():
        logger.error(f"Data YAML not found: {DATA_YAML}")
        return False
    
    # Check if pretrained model exists
    if not PRETRAINED_MODEL.exists():
        logger.error(f"Pretrained model not found: {PRETRAINED_MODEL}")
        logger.info("Please download yolo11n.pt from https://github.com/ultralytics/ultralytics")
        return False
    
    # Validate dataset structure
    if not validate_dataset_structure(str(DATA_YAML)):
        logger.error("Dataset structure validation failed")
        return False
    
    logger.info("All prerequisites validated successfully")
    return True

def print_training_info():
    """Print training configuration and dataset information"""
    logger.info("=== Training Configuration ===")
    logger.info(f"Data YAML: {DATA_YAML}")
    logger.info(f"Pretrained Model: {PRETRAINED_MODEL}")
    logger.info(f"Epochs: {TRAINING_CONFIG['epochs']}")
    logger.info(f"Batch Size: {TRAINING_CONFIG['batch']}")
    logger.info(f"Image Size: {TRAINING_CONFIG['imgsz']}")
    logger.info(f"Learning Rate: {TRAINING_CONFIG['lr0']}")
    logger.info(f"Device: {TRAINING_CONFIG['device'] or 'auto-detect'}")
    
    # Print dataset statistics
    logger.info("=== Dataset Information ===")
    stats = get_dataset_statistics(str(DATA_YAML))
    logger.info(f"Total Images: {stats['total_images']}")
    logger.info(f"Total Labels: {stats['total_labels']}")
    logger.info(f"Classes ({stats['class_count']}): {stats['classes']}")
    
    for split_name, split_info in stats['splits'].items():
        logger.info(f"{split_name.capitalize()}: {split_info['images']} images, {split_info['labels']} labels")

def train_yolo_model(custom_config=None):
    """
    Train YOLOv11 model
    
    Args:
        custom_config (dict): Custom training configuration to override defaults
    """
    try:
        # Merge custom config with defaults
        config = TRAINING_CONFIG.copy()
        if custom_config:
            config.update(custom_config)
        
        # Load pretrained model
        logger.info(f"Loading pretrained model: {PRETRAINED_MODEL}")
        model = YOLO(str(PRETRAINED_MODEL))
        
        # Set device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Update config with actual paths
        config['data'] = str(DATA_YAML)
        
        # Start training
        logger.info("Starting training...")
        logger.info(f"Training will be saved to: {config['project']}/{config['name']}")
        
        results = model.train(**config)
        
        # Save final model to models directory
        final_model_path = MODEL_DIR / "best_model.pt"
        if hasattr(results, 'save_dir'):
            import shutil
            best_model = Path(results.save_dir) / "weights" / "best.pt"
            if best_model.exists():
                shutil.copy(best_model, final_model_path)
                logger.info(f"Best model saved to: {final_model_path}")
        
        logger.info("Training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train YOLOv11 for People Detection")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--device', type=str, default='', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default='', help='Resume training from checkpoint')
    parser.add_argument('--name', type=str, default='yolov11n_people_detection', help='Experiment name')
    
    args = parser.parse_args()
    
    # Print system info
    print_system_info()
    
    # Setup directories
    setup_directories()
    
    # Validate prerequisites
    if not validate_prerequisites():
        logger.error("Prerequisites validation failed. Exiting.")
        sys.exit(1)
    
    # Print training info
    print_training_info()
    
    # Create custom config from arguments
    custom_config = {
        'epochs': args.epochs,
        'batch': args.batch_size,  # Use 'batch' instead of 'batch_size'
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'name': args.name
    }
    
    if args.device:
        custom_config['device'] = args.device
    
    if args.resume:
        custom_config['resume'] = args.resume
    
    try:
        # Start training
        start_time = datetime.now()
        logger.info(f"Training started at: {start_time}")
        
        results = train_yolo_model(custom_config)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.info(f"Training completed at: {end_time}")
        logger.info(f"Total training time: {training_duration}")
        
        logger.info("Training script finished successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()