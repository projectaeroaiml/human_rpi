#!/usr/bin/env python3
"""
Camera probe script to find working camera indices and backends
"""
import cv2
import sys

def try_backends(idx):
    """Try different backends for a camera index"""
    backends = []
    if sys.platform.startswith('win'):
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append(('CAP_DSHOW', cv2.CAP_DSHOW))
        if hasattr(cv2, 'CAP_MSMF'):
            backends.append(('CAP_MSMF', cv2.CAP_MSMF))
    backends.append(('DEFAULT', None))
    
    for name, backend in backends:
        try:
            cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Try to read a frame to verify it actually works
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return name, True
                else:
                    return name, False
            cap.release()
        except Exception as e:
            pass
    return None, False

def main():
    print("🔍 Probing cameras...")
    print("=" * 40)
    
    working_cameras = []
    
    for i in range(0, 6):
        backend, can_read = try_backends(i)
        if backend:
            status = "✅ WORKING" if can_read else "⚠️  OPENS (no frame)"
            print(f"Camera {i}: {status} ({backend})")
            if can_read:
                working_cameras.append(i)
        else:
            print(f"Camera {i}: ❌ Not found")
    
    print("=" * 40)
    
    if working_cameras:
        print(f"🎉 Found {len(working_cameras)} working camera(s): {working_cameras}")
        print(f"\nTo use camera {working_cameras[0]}, run:")
        print(f"python main.py webcam --camera {working_cameras[0]} --save-video --conf 0.25")
    else:
        print("❌ No working cameras found!")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check Camera app in Windows works")
        print("2. Ensure camera isn't used by another app")
        print("3. Check Windows Privacy -> Camera settings")
        print("4. Try reconnecting USB camera")
        print("5. Check Device Manager for camera devices")

if __name__ == "__main__":
    main()