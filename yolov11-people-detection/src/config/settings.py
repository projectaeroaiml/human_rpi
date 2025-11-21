"""
Configuration settings for YOLOv11 People Detection
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATASET_DIR = BASE_DIR
MODEL_DIR = BASE_DIR / "yolov11-people-detection" / "models"
RESULTS_DIR = BASE_DIR / "yolov11-people-detection" / "results"

# Dataset configuration
DATA_YAML = DATASET_DIR / "data.yaml"
PRETRAINED_MODEL = DATASET_DIR / "yolo11n.pt"

# Training configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch': 16,  # Changed from 'batch_size' to 'batch'
    'imgsz': 640,
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 0.05,
    'cls': 0.5,
    'dfl': 1.5,
    'val': True,
    'save_period': 10,
    'cache': False,
    'device': '',  # Auto-detect GPU/CPU
    'workers': 8,
    'project': str(RESULTS_DIR / "training"),
    'name': 'yolov11n_people_detection',
    'exist_ok': False,
    'pretrained': True,
    'optimizer': 'auto',
    'verbose': True,
    'seed': 0,
    'deterministic': True,
    'single_cls': False,
    'rect': False,
    'cos_lr': False,
    'close_mosaic': 10,
    'resume': False,
    'amp': True,
    'fraction': 1.0,
    'profile': False,
    'freeze': None,
    'plots': True
}

# Testing configuration
TESTING_CONFIG = {
    'imgsz': 640,
    'conf': 0.25,
    'iou': 0.45,
    'max_det': 300,
    'half': False,
    'device': '',  # Auto-detect GPU/CPU
    'dnn': False,
    'plots': True,
    'project': str(RESULTS_DIR / "predictions"),
    'name': 'test_results',
    'exist_ok': True,
    'save_txt': True,
    'save_conf': True,
    'save_crop': False,
    'classes': None,
    'agnostic_nms': False,
    'augment': False,
    'visualize': False,
    'line_thickness': 3,
    'hide_labels': False,
    'hide_conf': False,
    'vid_stride': 1
}

# Webcam prediction configuration
WEBCAM_CONFIG = {
    'source': 0,  # Default webcam
    'imgsz': 640,
    'conf': 0.25,
    'iou': 0.45,
    'max_det': 300,
    'device': '',  # Auto-detect GPU/CPU
    'show': True,
    'save': False,
    'save_txt': False,
    'save_conf': False,
    'save_crop': False,
    'nosave': True,
    'classes': None,
    'agnostic_nms': False,
    'augment': False,
    'visualize': False,
    'update': False,
    'project': str(RESULTS_DIR / "webcam"),
    'name': 'webcam_detection',
    'exist_ok': True,
    'line_thickness': 3,
    'hide_labels': False,
    'hide_conf': False,
    'vid_stride': 1,
    'stream_buffer': False
}

# Class names from data.yaml
CLASS_NAMES = ['person']

# Colors for visualization (BGR format)
COLORS = {
    'person': (0, 255, 0)    # Green
}