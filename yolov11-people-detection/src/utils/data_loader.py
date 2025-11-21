"""
Data loading and preprocessing utilities for YOLOv11 People Detection
"""
import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yaml_config(yaml_path: str) -> Dict:
    """
    Load YAML configuration file
    
    Args:
        yaml_path (str): Path to YAML file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"YAML config loaded from {yaml_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading YAML config: {e}")
        raise

def validate_dataset_structure(data_yaml_path: str) -> bool:
    """
    Validate dataset structure based on data.yaml
    
    Args:
        data_yaml_path (str): Path to data.yaml file
        
    Returns:
        bool: True if structure is valid
    """
    try:
        config = load_yaml_config(data_yaml_path)
        base_path = Path(data_yaml_path).parent
        
        # Check required keys
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required key in data.yaml: {key}")
                return False
        
        # Check paths exist
        for split in ['train', 'val', 'test']:
            if split in config:
                path = base_path / config[split]
                if not path.exists():
                    logger.error(f"Path does not exist: {path}")
                    return False
                else:
                    logger.info(f"Found {split} dataset at: {path}")
        
        logger.info(f"Dataset validation successful. Classes: {config['names']}")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

def count_dataset_images(data_yaml_path: str) -> Dict[str, int]:
    """
    Count images in each dataset split
    
    Args:
        data_yaml_path (str): Path to data.yaml file
        
    Returns:
        Dict[str, int]: Count of images per split
    """
    counts = {}
    try:
        config = load_yaml_config(data_yaml_path)
        base_path = Path(data_yaml_path).parent
        
        for split in ['train', 'val', 'test']:
            if split in config:
                images_path = base_path / config[split]
                if images_path.exists():
                    # Count image files
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                    count = sum(1 for f in images_path.rglob('*') 
                              if f.suffix.lower() in image_extensions)
                    counts[split] = count
                    logger.info(f"{split} split: {count} images")
                else:
                    counts[split] = 0
                    
    except Exception as e:
        logger.error(f"Error counting dataset images: {e}")
        
    return counts

def verify_labels_exist(data_yaml_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Verify corresponding label files exist for images
    
    Args:
        data_yaml_path (str): Path to data.yaml file
        
    Returns:
        Dict[str, Tuple[int, int]]: (images_count, labels_count) per split
    """
    verification = {}
    try:
        config = load_yaml_config(data_yaml_path)
        base_path = Path(data_yaml_path).parent
        
        for split in ['train', 'val', 'test']:
            if split in config:
                images_path = base_path / config[split]
                labels_path = images_path.parent / 'labels'
                
                if images_path.exists() and labels_path.exists():
                    # Count images and labels
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                    image_files = [f for f in images_path.rglob('*') 
                                 if f.suffix.lower() in image_extensions]
                    
                    label_files = list(labels_path.rglob('*.txt'))
                    
                    verification[split] = (len(image_files), len(label_files))
                    logger.info(f"{split}: {len(image_files)} images, {len(label_files)} labels")
                else:
                    verification[split] = (0, 0)
                    
    except Exception as e:
        logger.error(f"Error verifying labels: {e}")
        
    return verification

def load_image_safely(image_path: str) -> Optional[np.ndarray]:
    """
    Safely load an image file
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        Optional[np.ndarray]: Loaded image or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess image for YOLO inference
    
    Args:
        image (np.ndarray): Input image
        target_size (Tuple[int, int]): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image
    """
    try:
        # Resize image
        resized = cv2.resize(image, target_size)
        return resized
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image

def create_data_yaml(train_path: str, val_path: str, test_path: str, 
                    class_names: List[str], output_path: str) -> bool:
    """
    Create a data.yaml file for YOLO training
    
    Args:
        train_path (str): Path to training images
        val_path (str): Path to validation images  
        test_path (str): Path to test images
        class_names (List[str]): List of class names
        output_path (str): Output path for data.yaml
        
    Returns:
        bool: True if successful
    """
    try:
        data_config = {
            'train': train_path,
            'val': val_path,
            'test': test_path,
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
            
        logger.info(f"Data YAML created at: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating data YAML: {e}")
        return False

def get_dataset_statistics(data_yaml_path: str) -> Dict:
    """
    Get comprehensive dataset statistics
    
    Args:
        data_yaml_path (str): Path to data.yaml file
        
    Returns:
        Dict: Dataset statistics
    """
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'splits': {},
        'classes': [],
        'class_count': 0
    }
    
    try:
        # Load config
        config = load_yaml_config(data_yaml_path)
        stats['classes'] = config.get('names', [])
        stats['class_count'] = config.get('nc', 0)
        
        # Count images and labels
        image_counts = count_dataset_images(data_yaml_path)
        label_verification = verify_labels_exist(data_yaml_path)
        
        for split in image_counts:
            stats['splits'][split] = {
                'images': image_counts[split],
                'labels': label_verification.get(split, (0, 0))[1]
            }
            stats['total_images'] += image_counts[split]
            stats['total_labels'] += label_verification.get(split, (0, 0))[1]
            
        logger.info(f"Dataset statistics: {stats['total_images']} total images")
        
    except Exception as e:
        logger.error(f"Error getting dataset statistics: {e}")
        
    return stats