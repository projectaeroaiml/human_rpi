"""
Model utilities for YOLOv11 People Detection
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from sklearn.cluster import DBSCAN
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yolo_model(model_path: str, device: str = '') -> YOLO:
    """
    Load YOLOv11 model from weights file
    
    Args:
        model_path (str): Path to model weights
        device (str): Device to use ('cuda', 'cpu', or '' for auto)
        
    Returns:
        YOLO: Loaded YOLO model
    """
    try:
        model = YOLO(model_path)
        if device:
            model.to(device)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def get_device() -> str:
    """
    Get the best available device (GPU/CPU)
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        logger.info("Using CPU for inference")
    
    return device

def validate_model_path(model_path: str) -> bool:
    """
    Validate if model path exists
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        bool: True if path exists
    """
    path = Path(model_path)
    if not path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False
    return True

def cluster_detections(centers: List[Tuple[int, int]], max_distance: float = 100.0) -> List[List[int]]:
    """
    Cluster detection centers using DBSCAN
    
    Args:
        centers: List of (x, y) center coordinates
        max_distance: Maximum distance between points in same cluster
        
    Returns:
        List of clusters, each containing indices of points in that cluster
    """
    if len(centers) == 0:
        return []
    
    # Convert to numpy array for DBSCAN
    points = np.array(centers)
    
    # Use DBSCAN clustering
    clustering = DBSCAN(eps=max_distance, min_samples=1).fit(points)
    labels = clustering.labels_
    
    # Group indices by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    return list(clusters.values())

def calculate_group_center(centers: List[Tuple[int, int]], indices: List[int]) -> Tuple[int, int]:
    """
    Calculate the center point of a group
    
    Args:
        centers: All detection centers
        indices: Indices of detections in this group
        
    Returns:
        (x, y) coordinates of group center
    """
    group_points = [centers[i] for i in indices]
    avg_x = sum(p[0] for p in group_points) // len(group_points)
    avg_y = sum(p[1] for p in group_points) // len(group_points)
    return (avg_x, avg_y)

def calculate_pan_tilt(group_center: Tuple[int, int], frame_center: Tuple[int, int], 
                      frame_width: int, frame_height: int) -> Tuple[float, float]:
    """
    Calculate pan/tilt angles needed to center a group
    
    Args:
        group_center: (x, y) of the group center
        frame_center: (x, y) of the frame center
        frame_width: Width of frame
        frame_height: Height of frame
        
    Returns:
        (pan_angle, tilt_angle) in degrees
    """
    # Calculate offset from center
    dx = group_center[0] - frame_center[0]
    dy = group_center[1] - frame_center[1]
    
    # Assume 60-degree horizontal FOV and 45-degree vertical FOV (typical webcam)
    horizontal_fov = 60.0
    vertical_fov = 45.0
    
    # Calculate pan/tilt angles
    pan_angle = (dx / frame_width) * horizontal_fov
    tilt_angle = -(dy / frame_height) * vertical_fov  # Negative because y increases downward
    
    return (pan_angle, tilt_angle)

def draw_predictions_with_clustering(image: np.ndarray, results, class_names: list, colors: dict, 
                                    clustering_distance: float = 120.0, selected_boxes: dict = None, 
                                    current_detections: list = None) -> np.ndarray:
    """
    Draw bounding boxes with clustering and group analysis
    
    Args:
        image (np.ndarray): Input image
        results: YOLO prediction results
        class_names (list): List of class names
        colors (dict): Color mapping for classes
        
    Returns:
        np.ndarray: Annotated image
    """
    # Type check to prevent the tuple error
    if not isinstance(image, np.ndarray):
        logger.error(f"Expected numpy array, got {type(image)}")
        if isinstance(image, tuple):
            logger.error(f"Tuple contents: {image}")
        raise TypeError(f"Image must be numpy array, got {type(image)}")
    
    annotated_image = image.copy()
    frame_height, frame_width = image.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)
    
    # Collect all detection centers
    detection_centers = []
    all_boxes = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                detection_centers.append((center_x, center_y))
                all_boxes.append((x1, y1, x2, y2))
    
    # Store current detections for mouse callback
    if current_detections is not None:
        current_detections.clear()
        current_detections.extend(all_boxes)
    
    if len(detection_centers) == 0:
        return annotated_image
    
    # Cluster the detections
    clusters = cluster_detections(detection_centers, max_distance=clustering_distance)
    
    # Define colors for different group sizes
    small_group_color = (0, 255, 0)    # Green for small groups (1-2 people)
    medium_group_color = (0, 255, 255) # Yellow for medium groups (3-4 people)
    large_group_color = (0, 0, 255)    # Red for large groups (5+ people)
    
    group_centers = []
    largest_group_size = 0
    largest_group_center = None
    
    # Process each cluster
    for cluster_indices in clusters:
        group_size = len(cluster_indices)
        
        # Choose color based on group size
        if group_size <= 2:
            box_color = small_group_color
        elif group_size <= 4:
            box_color = medium_group_color
        else:
            box_color = large_group_color
        
        # Draw individual bounding boxes for this group
        group_boxes = []
        for idx in cluster_indices:
            x1, y1, x2, y2 = all_boxes[idx]
            
            # Check if any selected box overlaps with this detection
            is_selected = False
            if selected_boxes:
                for sel_box in selected_boxes.values():
                    sx1, sy1, sx2, sy2 = sel_box
                    # Check for overlap
                    if not (x2 < sx1 or x1 > sx2 or y2 < sy1 or y1 > sy2):
                        is_selected = True
                        break
            
            # Draw with different opacity based on selection
            if selected_boxes and not is_selected:
                # Create semi-transparent overlay for non-selected boxes
                overlay = annotated_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
                cv2.addWeighted(overlay, 0.5, annotated_image, 0.5, 0, annotated_image)
            else:
                # Draw normally (selected or no selections)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)
            
            group_boxes.append((x1, y1, x2, y2))
        
        # Calculate and draw group bounding box
        if len(group_boxes) > 1:  # Only draw group box if more than one person
            # Find the outer bounds of all boxes in this group
            min_x = min(box[0] for box in group_boxes)
            min_y = min(box[1] for box in group_boxes)
            max_x = max(box[2] for box in group_boxes)
            max_y = max(box[3] for box in group_boxes)
            
            # Add some padding around the group
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(frame_width, max_x + padding)
            max_y = min(frame_height, max_y + padding)
            
            # Draw group bounding box with thicker line and dashed style
            thickness = 4
            dash_length = 10
            
            # Draw dashed rectangle for group boundary
            # Top line
            for x in range(min_x, max_x, dash_length * 2):
                cv2.line(annotated_image, (x, min_y), (min(x + dash_length, max_x), min_y), box_color, thickness)
            # Bottom line
            for x in range(min_x, max_x, dash_length * 2):
                cv2.line(annotated_image, (x, max_y), (min(x + dash_length, max_x), max_y), box_color, thickness)
            # Left line
            for y in range(min_y, max_y, dash_length * 2):
                cv2.line(annotated_image, (min_x, y), (min_x, min(y + dash_length, max_y)), box_color, thickness)
            # Right line
            for y in range(min_y, max_y, dash_length * 2):
                cv2.line(annotated_image, (max_x, y), (max_x, min(y + dash_length, max_y)), box_color, thickness)
        
        # Calculate group center
        group_center = calculate_group_center(detection_centers, cluster_indices)
        group_centers.append((group_center, group_size))
        
        # Track largest group
        if group_size > largest_group_size:
            largest_group_size = group_size
            largest_group_center = group_center
        
        # Draw group center point
        cv2.circle(annotated_image, group_center, 8, box_color, -1)
        cv2.circle(annotated_image, group_center, 10, (255, 255, 255), 2)
        
        # Display group center coordinates
        coord_text = f"({group_center[0]}, {group_center[1]})"
        cv2.putText(annotated_image, coord_text, 
                   (group_center[0] + 15, group_center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display group size
        size_text = f"Size: {group_size}"
        cv2.putText(annotated_image, size_text,
                   (group_center[0] + 15, group_center[1] + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw frame center
    cv2.circle(annotated_image, frame_center, 6, (255, 0, 255), -1)  # Magenta
    cv2.circle(annotated_image, frame_center, 8, (255, 255, 255), 2)
    
    # Draw lines from each group center to frame center
    for group_center, group_size in group_centers:
        line_color = large_group_color if group_size > 4 else medium_group_color if group_size > 2 else small_group_color
        cv2.line(annotated_image, group_center, frame_center, line_color, 2)
    
    # Draw selected tracker boxes with highlighting
    if selected_boxes:
        for tracker_id, (x1, y1, x2, y2) in selected_boxes.items():
            # Draw thick highlighted border for selected objects
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Thick yellow border
            
            # Draw tracker ID
            id_text = f"ID:{tracker_id}"
            cv2.putText(annotated_image, id_text, 
                       (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw corner markers for better visibility
            corner_size = 10
            # Top-left corner
            cv2.line(annotated_image, (x1, y1), (x1 + corner_size, y1), (0, 255, 255), 3)
            cv2.line(annotated_image, (x1, y1), (x1, y1 + corner_size), (0, 255, 255), 3)
            # Top-right corner
            cv2.line(annotated_image, (x2, y1), (x2 - corner_size, y1), (0, 255, 255), 3)
            cv2.line(annotated_image, (x2, y1), (x2, y1 + corner_size), (0, 255, 255), 3)
            # Bottom-left corner
            cv2.line(annotated_image, (x1, y2), (x1 + corner_size, y2), (0, 255, 255), 3)
            cv2.line(annotated_image, (x1, y2), (x1, y2 - corner_size), (0, 255, 255), 3)
            # Bottom-right corner
            cv2.line(annotated_image, (x2, y2), (x2 - corner_size, y2), (0, 255, 255), 3)
            cv2.line(annotated_image, (x2, y2), (x2, y2 - corner_size), (0, 255, 255), 3)

    # Calculate and display pan/tilt for largest group
    if largest_group_center is not None:
        pan_angle, tilt_angle = calculate_pan_tilt(largest_group_center, frame_center, frame_width, frame_height)
        
        # Display pan/tilt information
        pan_tilt_text = f"Largest Group: Pan {pan_angle:.1f}°, Tilt {tilt_angle:.1f}°"
        cv2.putText(annotated_image, pan_tilt_text,
                   (10, frame_height - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Additional info
        info_text = f"Groups: {len(clusters)}, Largest: {largest_group_size} people"
        cv2.putText(annotated_image, info_text,
                   (10, frame_height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display tracking info
    if selected_boxes:
        tracking_text = f"Tracking: {len(selected_boxes)} objects"
        cv2.putText(annotated_image, tracking_text,
                   (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return annotated_image

def draw_predictions(image: np.ndarray, results, class_names: list, colors: dict, clustering_distance: float = 120.0, 
                    selected_boxes: dict = None, current_detections: list = None) -> np.ndarray:
    """
    Wrapper function that calls the clustering version
    """
    return draw_predictions_with_clustering(image, results, class_names, colors, clustering_distance, 
                                          selected_boxes, current_detections)

def print_system_info():
    """Print system information for debugging"""
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
def calculate_model_size(model_path: str) -> float:
    """
    Calculate model file size in MB
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        float: Model size in MB
    """
    try:
        size_bytes = Path(model_path).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except Exception as e:
        logger.error(f"Error calculating model size: {e}")
        return 0.0