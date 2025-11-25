#!/usr/bin/env python3
"""
YOLOv11 People Detection - Main Launcher Script
Provides a simple interface to run training, testing, or webcam prediction.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_script(script_name: str, args: list = None):
    """
    Run a script with optional arguments
    
    Args:
        script_name (str): Name of the script to run
        args (list): Additional arguments
    """
    script_path = Path(__file__).parent / "src" / script_name
    cmd = [sys.executable, str(script_path)]
    
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("Script interrupted by user")
        sys.exit(1)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="YOLOv11 People Detection - Main Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                           # Train with defaults
  python main.py train --epochs 200 --batch-size 32    # Custom training
  python main.py test                            # Test trained model  
  python main.py test --save-images              # Test and save images
  python main.py webcam                          # Real-time webcam
  python main.py webcam --save-video             # Record webcam output
        """
    )
    
    parser.add_argument(
        'mode', 
        choices=['train', 'test', 'webcam'],
        help='Operation mode: train the model, test the model, or run webcam detection'
    )
    
    # Parse known args to allow passing through to sub-scripts
    args, unknown = parser.parse_known_args()
    
    # Map modes to scripts
    script_map = {
        'train': 'train.py',
        'test': 'test.py', 
        'webcam': 'predict_webcam.py'
    }
    
    # Print header
    print("=" * 60)
    print("🎯 YOLOv11 People Detection")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    if unknown:
        print(f"Additional arguments: {' '.join(unknown)}")
    print("=" * 60)
    
    # Run the appropriate script
    script_name = script_map[args.mode]
    run_script(script_name, unknown)
    
    print("=" * 60)
    print("✅ Operation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()