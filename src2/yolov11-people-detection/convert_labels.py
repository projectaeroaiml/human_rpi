#!/usr/bin/env python3
"""
Convert multi-class labels to single 'person' class
Converts all existing classes (dry-person, object, wet-swimmer) to class 0 (person)
"""
import os
from pathlib import Path

def convert_labels_to_person_class(base_dir):
    """Convert all label files to use single person class (class 0)"""
    base_path = Path(base_dir)
    
    # Process train, valid, and test label directories
    for split in ['train', 'valid', 'test']:
        labels_dir = base_path / split / 'labels'
        
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist, skipping...")
            continue
        
        label_files = list(labels_dir.glob('*.txt'))
        print(f"Processing {len(label_files)} label files in {split}/labels/")
        
        for label_file in label_files:
            try:
                # Read the original file
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # Convert all classes to 0 (person)
                converted_lines = []
                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        if len(parts) >= 5:  # class_id x_center y_center width height
                            # Change class to 0 (person), keep bounding box coordinates
                            parts[0] = '0'
                            converted_lines.append(' '.join(parts))
                
                # Write back the converted file
                with open(label_file, 'w') as f:
                    for line in converted_lines:
                        f.write(line + '\n')
                        
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
        
        print(f"Converted {len(label_files)} files in {split}/labels/")
    
    print("Label conversion completed!")

def main():
    # Base directory containing train/valid/test folders
    base_dir = Path(__file__).parent.parent
    
    print("🔄 Converting labels to single 'person' class...")
    print(f"Base directory: {base_dir}")
    print("=" * 50)
    
    # Backup notification
    print("⚠️  This will modify your label files!")
    print("Consider backing up your labels before proceeding.")
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    convert_labels_to_person_class(base_dir)
    
    print("=" * 50)
    print("✅ Conversion complete!")
    print("All objects (dry-person, object, wet-swimmer) are now labeled as 'person' (class 0)")
    print("\nYou can now train with:")
    print("python main.py train")

if __name__ == "__main__":
    main()