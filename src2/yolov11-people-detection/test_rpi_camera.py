#!/usr/bin/env python3
"""
Raspberry Pi Camera Test Script
Tests camera functionality and helps diagnose issues on Raspberry Pi
"""

import cv2
import sys
import time
import numpy as np

def test_camera_backend(camera_id, backend_name, backend_api):
    """Test a specific camera backend"""
    print(f"\n{'='*60}")
    print(f"Testing {backend_name} backend (API: {backend_api if backend_api else 'Default'})")
    print('='*60)
    
    try:
        # Open camera
        if backend_api is None:
            cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_id, backend_api)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera with {backend_name}")
            return False
        
        print(f"✅ Camera opened successfully")
        
        # Get camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   Resolution: {int(width)}x{int(height)}")
        print(f"   FPS: {fps}")
        
        # Try to read frames
        print("\n   Testing frame capture...")
        successful_reads = 0
        failed_reads = 0
        frame_times = []
        
        for i in range(10):
            start = time.time()
            ret, frame = cap.read()
            elapsed = time.time() - start
            
            if ret and frame is not None and frame.size > 0:
                successful_reads += 1
                frame_times.append(elapsed)
                if i == 0:
                    print(f"   ✅ First frame captured: {frame.shape}")
            else:
                failed_reads += 1
                print(f"   ❌ Failed to read frame {i+1}")
        
        print(f"\n   Results: {successful_reads}/10 frames captured successfully")
        if frame_times:
            avg_time = sum(frame_times) / len(frame_times)
            print(f"   Average frame read time: {avg_time*1000:.1f}ms")
        
        cap.release()
        
        return successful_reads >= 8  # Consider success if 8/10 frames work
        
    except Exception as e:
        print(f"❌ Error testing {backend_name}: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("Raspberry Pi Camera Test Utility")
    print("="*60)
    
    # Check OpenCV version
    print(f"\nOpenCV Version: {cv2.__version__}")
    print(f"Platform: {sys.platform}")
    
    # Get camera ID from command line or use default
    camera_id = 0
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except:
            print(f"Invalid camera ID, using default: 0")
    
    print(f"\nTesting Camera ID: {camera_id}")
    
    # List of backends to test on Raspberry Pi
    backends = []
    
    if sys.platform.startswith('linux'):
        print("\n🔍 Detected Linux platform - testing Raspberry Pi backends")
        
        # V4L2 - Video4Linux2 (most common for USB cameras and some CSI cameras)
        if hasattr(cv2, 'CAP_V4L2'):
            backends.append(('V4L2 (Video4Linux2)', cv2.CAP_V4L2))
        
        # GStreamer - good for Pi Camera module
        if hasattr(cv2, 'CAP_GSTREAMER'):
            backends.append(('GStreamer', cv2.CAP_GSTREAMER))
        
        # libcamera - newer Raspberry Pi camera stack
        if hasattr(cv2, 'CAP_ANY'):
            backends.append(('libcamera/ANY', cv2.CAP_ANY))
    
    elif sys.platform.startswith('win'):
        print("\n🔍 Detected Windows platform")
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append(('DirectShow', cv2.CAP_DSHOW))
        if hasattr(cv2, 'CAP_MSMF'):
            backends.append(('Media Foundation', cv2.CAP_MSMF))
    
    # Always test default backend
    backends.append(('Default OpenCV', None))
    
    # Test each backend
    results = {}
    for backend_name, backend_api in backends:
        success = test_camera_backend(camera_id, backend_name, backend_api)
        results[backend_name] = success
        time.sleep(0.5)  # Small delay between tests
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    working_backends = [name for name, success in results.items() if success]
    
    if working_backends:
        print("\n✅ Working backends:")
        for backend in working_backends:
            print(f"   • {backend}")
        print(f"\n💡 Recommendation: Use '{working_backends[0]}' backend")
        print(f"   Add this to your script: cv2.VideoCapture(camera_id, cv2.{working_backends[0]})")
    else:
        print("\n❌ No working backends found!")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Check if camera is connected: ls /dev/video*")
        print("   2. Check camera permissions: sudo usermod -a -G video $USER")
        print("   3. For Pi Camera module, enable camera interface: sudo raspi-config")
        print("   4. For USB cameras, try different USB ports")
        print("   5. Check if camera works with: raspistill -o test.jpg (Pi Camera)")
        print("   6. Or try: v4l2-ctl --list-devices (USB cameras)")
        print("   7. Reboot after enabling camera interface")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
