#!/usr/bin/env python3
"""
Model Converter for YOLOv11 People Detection
Converts PyTorch (.pt) models to ONNX format for deployment on Raspberry Pi 5 with AI Hat.

Usage:
    python model_converter.py                           # Convert default model
    python model_converter.py --model path/to/model.pt  # Convert specific model
    python model_converter.py --imgsz 320               # Use smaller image size
    python model_converter.py --all                     # Export all formats
"""

import argparse
import sys
from pathlib import Path

def find_model() -> Path:
    """Find the best available model file"""
    search_paths = [
        Path("models/best_model.pt"),
        Path("models/best.pt"),
        Path("results/training/yolov11n_people_detection/weights/best.pt"),
        Path("results/training/yolov11n_people_detection2/weights/best.pt"),
        Path("results/training/yolov11n_people_detection3/weights/best.pt"),
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    # Search recursively
    for pt_file in Path(".").rglob("*.pt"):
        if "best" in pt_file.name:
            return pt_file
    
    return None


def convert_to_onnx(
    model_path: str,
    output_dir: str = None,
    imgsz: int = 640,
    opset: int = 11,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False
) -> Path:
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to the .pt model file
        output_dir: Output directory (default: same as input)
        imgsz: Image size for export
        opset: ONNX opset version (11 recommended for Hailo)
        simplify: Simplify ONNX model
        dynamic: Use dynamic input shapes
        half: Export as FP16 (half precision)
    
    Returns:
        Path to exported ONNX file
    """
    from ultralytics import YOLO
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Exporting to ONNX (imgsz={imgsz}, opset={opset})...")
    
    # Export to ONNX
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=half
    )
    
    print(f"✅ ONNX model exported: {export_path}")
    
    # Move to output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_path = Path(export_path)
        new_path = output_dir / export_path.name
        
        if export_path != new_path:
            import shutil
            shutil.move(str(export_path), str(new_path))
            print(f"✅ Moved to: {new_path}")
            return new_path
    
    return Path(export_path)


def convert_to_tflite(model_path: str, imgsz: int = 640) -> Path:
    """Convert to TensorFlow Lite format"""
    from ultralytics import YOLO
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Exporting to TFLite (imgsz={imgsz})...")
    export_path = model.export(format="tflite", imgsz=imgsz)
    
    print(f"✅ TFLite model exported: {export_path}")
    return Path(export_path)


def convert_to_ncnn(model_path: str, imgsz: int = 640) -> Path:
    """Convert to NCNN format (good for ARM devices)"""
    from ultralytics import YOLO
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Exporting to NCNN (imgsz={imgsz})...")
    export_path = model.export(format="ncnn", imgsz=imgsz)
    
    print(f"✅ NCNN model exported: {export_path}")
    return Path(export_path)


def get_model_info(model_path: str):
    """Print model information"""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    print("\n" + "=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Path: {model_path}")
    print(f"Task: {model.task}")
    print(f"Names: {model.names}")
    
    # Get file size
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv11 model to deployment formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_converter.py                              # Auto-find and convert to ONNX
  python model_converter.py --model models/best.pt      # Convert specific model
  python model_converter.py --imgsz 320                  # Smaller image size (faster)
  python model_converter.py --output models/             # Specify output directory
  python model_converter.py --all                        # Export all formats
  python model_converter.py --info                       # Show model info only
        """
    )
    
    parser.add_argument('--model', type=str, default='',
                       help='Path to .pt model file')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640, use 320 for faster inference)')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--half', action='store_true',
                       help='Export as FP16 (half precision)')
    parser.add_argument('--dynamic', action='store_true',
                       help='Use dynamic input shapes')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Do not simplify ONNX model')
    parser.add_argument('--tflite', action='store_true',
                       help='Also export to TFLite format')
    parser.add_argument('--ncnn', action='store_true',
                       help='Also export to NCNN format')
    parser.add_argument('--all', action='store_true',
                       help='Export to all formats (ONNX, TFLite, NCNN)')
    parser.add_argument('--info', action='store_true',
                       help='Show model info only')
    
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_model()
    
    if not model_path or not model_path.exists():
        print("❌ Error: No model file found!")
        print("Please specify a model with --model or place it in models/")
        sys.exit(1)
    
    print(f"\n🎯 Found model: {model_path}\n")
    
    # Show info
    if args.info:
        get_model_info(str(model_path))
        return
    
    # Convert to ONNX (always do this)
    print("=" * 50)
    print("CONVERTING TO ONNX")
    print("=" * 50)
    
    onnx_path = convert_to_onnx(
        model_path=str(model_path),
        output_dir=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
        half=args.half
    )
    
    # Additional formats
    if args.all or args.tflite:
        print("\n" + "=" * 50)
        print("CONVERTING TO TFLITE")
        print("=" * 50)
        try:
            convert_to_tflite(str(model_path), args.imgsz)
        except Exception as e:
            print(f"⚠️ TFLite export failed: {e}")
    
    if args.all or args.ncnn:
        print("\n" + "=" * 50)
        print("CONVERTING TO NCNN")
        print("=" * 50)
        try:
            convert_to_ncnn(str(model_path), args.imgsz)
        except Exception as e:
            print(f"⚠️ NCNN export failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ CONVERSION COMPLETE")
    print("=" * 50)
    print(f"ONNX file: {onnx_path}")
    print("\nNext steps for Hailo AI Hat:")
    print("1. Copy the ONNX file to your Raspberry Pi 5")
    print("2. Use Hailo Dataflow Compiler to convert to HEF:")
    print(f"   hailo parser onnx {onnx_path.name} --hw-arch hailo8l")
    print("   hailo optimize <model>.har --hw-arch hailo8l")
    print("   hailo compiler <model>.har --hw-arch hailo8l")
    print("=" * 50)


if __name__ == "__main__":
    main()