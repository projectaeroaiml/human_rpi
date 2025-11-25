#!/usr/bin/env python3
"""
Setup and validation script for YOLOv11 People Detection
Checks prerequisites and validates the environment before running.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_gpu():
    """Check GPU availability"""
    print("🖥️  Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ {gpu_count} GPU(s) available: {gpu_name}")
            return True
        else:
            print("   ⚠️  No GPU detected - will use CPU (slower)")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'ultralytics', 'torch', 'torchvision', 'opencv-python', 
        'matplotlib', 'pandas', 'numpy', 'pyyaml', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Not installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_dataset():
    """Check dataset structure"""
    print("📊 Checking dataset structure...")
    
    base_dir = Path(__file__).parent.parent
    data_yaml = base_dir / "data.yaml"
    pretrained_model = base_dir / "yolo11n.pt"
    
    checks = [
        (data_yaml, "data.yaml configuration file"),
        (base_dir / "train" / "images", "training images directory"),
        (base_dir / "train" / "labels", "training labels directory"), 
        (base_dir / "valid" / "images", "validation images directory"),
        (base_dir / "valid" / "labels", "validation labels directory"),
        (base_dir / "test" / "images", "test images directory"),
        (base_dir / "test" / "labels", "test labels directory"),
        (pretrained_model, "pretrained YOLOv11n model")
    ]
    
    all_good = True
    for path, description in checks:
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {description} ({size_mb:.1f} MB)")
            else:
                # Count files in directory
                files = list(path.glob("*"))
                print(f"   ✅ {description} ({len(files)} files)")
        else:
            print(f"   ❌ {description} - Not found")
            all_good = False
    
    return all_good

def install_dependencies(missing_packages):
    """Install missing dependencies"""
    print(f"📥 Installing missing packages: {', '.join(missing_packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--upgrade", *missing_packages
        ])
        print("   ✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ Failed to install packages")
        return False

def download_pretrained_model():
    """Download pretrained YOLOv11n model"""
    print("⬇️  Downloading pretrained YOLOv11n model...")
    
    try:
        from ultralytics import YOLO
        # This will automatically download the model if it doesn't exist
        model = YOLO('yolo11n.pt')
        
        # Move to correct location
        base_dir = Path(__file__).parent.parent
        target_path = base_dir / "yolo11n.pt"
        
        if not target_path.exists():
            # The model is downloaded to ~/.ultralytics by default
            import shutil
            from pathlib import Path
            home_model = Path.home() / ".ultralytics" / "yolo11n.pt"
            if home_model.exists():
                shutil.copy(home_model, target_path)
                print(f"   ✅ Model downloaded to {target_path}")
            else:
                print("   ⚠️  Model downloaded but location unknown")
        else:
            print("   ✅ Pretrained model already exists")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed to download model: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 YOLOv11 People Detection - Setup & Validation")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    print()
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\n🔧 Installing missing dependencies...")
        if install_dependencies(missing):
            print("   ✅ Dependencies installed successfully")
        else:
            success = False
    
    print()
    
    # Check GPU after torch is available
    check_gpu()
    
    print()
    
    # Check dataset
    if not check_dataset():
        success = False
        print("\n📋 To fix dataset issues:")
        print("   1. Ensure data.yaml is in the parent directory")
        print("   2. Download yolo11n.pt pretrained model")
        print("   3. Verify train/valid/test directories exist")
        
        # Try to download pretrained model
        base_dir = Path(__file__).parent.parent
        if not (base_dir / "yolo11n.pt").exists():
            print("\n⬇️  Attempting to download pretrained model...")
            download_pretrained_model()
    
    print()
    print("=" * 60)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Ready to run:")
        print("   python main.py train     # Start training")
        print("   python main.py test      # Test model") 
        print("   python main.py webcam    # Live detection")
    else:
        print("❌ Setup incomplete - please fix the issues above")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())