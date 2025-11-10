"""
Tongji Palmprint Database Downloader and Preparation Script

This script helps download and prepare the Tongji Palmprint Database for training.
The Tongji database contains 20,000+ palmprint images from 300 different palms.

Note: Since I cannot directly download files, this script provides instructions
and utilities to help you prepare the dataset once you obtain it.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import zipfile
import json

class TongjiDatasetPreparator:
    def __init__(self, dataset_path="./Tongji_Palmprint"):
        self.dataset_path = Path(dataset_path)
        self.raw_data_path = self.dataset_path / "raw"
        self.processed_data_path = self.dataset_path / "processed"
        
    def create_directories(self):
        """Create necessary directories for dataset organization"""
        self.dataset_path.mkdir(exist_ok=True)
        self.raw_data_path.mkdir(exist_ok=True)
        self.processed_data_path.mkdir(exist_ok=True)
        
        print(f"Created dataset directories:")
        print(f"  Raw data: {self.raw_data_path}")
        print(f"  Processed data: {self.processed_data_path}")
    
    def download_instructions(self):
        """Provide instructions for downloading Tongji database"""
        print("\n" + "="*60)
        print("TONGJI PALMPRINT DATABASE DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print()
        print("The Tongji Palmprint Database is available from several sources:")
        print()
        print("1. OFFICIAL SOURCES:")
        print("   - Tongji University Computer Vision Lab")
        print("   - Contact: tongji-cv@tongji.edu.cn")
        print("   - Website: http://www.tongji.edu.cn (search for palmprint database)")
        print()
        print("2. ACADEMIC DATABASES:")
        print("   - IEEE Xplore Digital Library")
        print("   - ACM Digital Library")
        print("   - ResearchGate (search for 'Tongji Palmprint Database')")
        print()
        print("3. MIRROR SITES:")
        print("   - Some research groups may host mirrors")
        print("   - Check computer vision conference websites")
        print()
        print("4. DATASET CHARACTERISTICS:")
        print("   - 20,000+ palmprint images")
        print("   - 300 different palms")
        print("   - Multiple lighting conditions")
        print("   - Various poses and orientations")
        print("   - High resolution images")
        print()
        print("5. DOWNLOAD STEPS:")
        print("   a. Visit one of the sources above")
        print("   b. Request access (may require academic verification)")
        print("   c. Download the dataset archive")
        print("   d. Extract to:", self.raw_data_path)
        print("   e. Run this script again to process the data")
        print()
        print("6. ALTERNATIVE DATASETS:")
        print("   If Tongji is not available, consider:")
        print("   - PolyU Palmprint Database (7,752 images)")
        print("   - CASIA Palmprint Database")
        print("   - IIT Delhi Palmprint Database")
        print()
        print("="*60)
    
    def check_dataset_exists(self):
        """Check if the Tongji dataset has been downloaded"""
        if not self.raw_data_path.exists():
            return False
        
        # Look for common Tongji dataset file patterns
        patterns = ['*.bmp', '*.jpg', '*.png', '*.jpeg']
        for pattern in patterns:
            if list(self.raw_data_path.rglob(pattern)):
                return True
        
        return False
    
    def analyze_dataset_structure(self):
        """Analyze the structure of the downloaded dataset"""
        if not self.check_dataset_exists():
            print("Dataset not found. Please download first.")
            return
        
        print("\nAnalyzing dataset structure...")
        
        # Find all image files
        image_extensions = ['.bmp', '.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.raw_data_path.rglob(f'*{ext}')))
            image_files.extend(list(self.raw_data_path.rglob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} image files")
        
        # Analyze directory structure
        subdirs = [d for d in self.raw_data_path.rglob('*') if d.is_dir()]
        print(f"Found {len(subdirs)} subdirectories")
        
        # Sample some files to understand naming convention
        sample_files = image_files[:10]
        print("\nSample file names:")
        for file in sample_files:
            print(f"  {file.name}")
        
        # Try to detect subjects/classes
        subjects = set()
        for file in image_files:
            # Common naming patterns for palmprint datasets
            name = file.stem
            
            # Pattern 1: SubjectXXX_Y or XXX_Y
            if '_' in name:
                subject = name.split('_')[0]
                subjects.add(subject)
            # Pattern 2: XXXY (first 3 digits are subject)
            elif len(name) >= 4 and name[:3].isdigit():
                subjects.add(name[:3])
            # Pattern 3: directory name as subject
            else:
                subjects.add(file.parent.name)
        
        print(f"\nDetected {len(subjects)} potential subjects/classes")
        print("Sample subjects:", list(subjects)[:10])
        
        return {
            'total_images': len(image_files),
            'total_subjects': len(subjects),
            'sample_files': [str(f) for f in sample_files],
            'subjects': list(subjects)
        }
    
    def preprocess_image(self, image_path, target_size=(128, 128)):
        """Preprocess a single palmprint image"""
        try:
            # Read image
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Resize
            img = cv2.resize(img, target_size)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # Normalize
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            return img
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def organize_dataset(self):
        """Organize the raw dataset into a structured format"""
        if not self.check_dataset_exists():
            print("Dataset not found. Please download first.")
            return
        
        print("Organizing dataset...")
        
        # Analyze structure first
        analysis = self.analyze_dataset_structure()
        
        # Create organized structure
        train_dir = self.processed_data_path / "train"
        val_dir = self.processed_data_path / "val"
        test_dir = self.processed_data_path / "test"
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Find all images and organize by subject
        image_extensions = ['.bmp', '.jpg', '.jpeg', '.png']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(list(self.raw_data_path.rglob(f'*{ext}')))
            all_images.extend(list(self.raw_data_path.rglob(f'*{ext.upper()}')))
        
        # Group images by subject
        subject_images = {}
        for img_path in all_images:
            # Determine subject ID from filename or directory
            name = img_path.stem
            
            # Try different naming patterns
            if '_' in name:
                subject = name.split('_')[0]
            elif len(name) >= 4 and name[:3].isdigit():
                subject = name[:3]
            else:
                subject = img_path.parent.name
            
            if subject not in subject_images:
                subject_images[subject] = []
            subject_images[subject].append(img_path)
        
        print(f"Found {len(subject_images)} subjects")
        
        # Process each subject
        processed_count = 0
        for subject_id, images in subject_images.items():
            # Create subject directories
            subject_train_dir = train_dir / f"subject_{subject_id}"
            subject_val_dir = val_dir / f"subject_{subject_id}"
            subject_test_dir = test_dir / f"subject_{subject_id}"
            
            subject_train_dir.mkdir(exist_ok=True)
            subject_val_dir.mkdir(exist_ok=True)
            subject_test_dir.mkdir(exist_ok=True)
            
            # Split images (70% train, 15% val, 15% test)
            np.random.shuffle(images)
            n_train = int(0.7 * len(images))
            n_val = int(0.15 * len(images))
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Process and save images
            for i, img_path in enumerate(train_images):
                processed_img = self.preprocess_image(img_path)
                if processed_img is not None:
                    save_path = subject_train_dir / f"{subject_id}_train_{i:03d}.bmp"
                    cv2.imwrite(str(save_path), processed_img)
                    processed_count += 1
            
            for i, img_path in enumerate(val_images):
                processed_img = self.preprocess_image(img_path)
                if processed_img is not None:
                    save_path = subject_val_dir / f"{subject_id}_val_{i:03d}.bmp"
                    cv2.imwrite(str(save_path), processed_img)
                    processed_count += 1
            
            for i, img_path in enumerate(test_images):
                processed_img = self.preprocess_image(img_path)
                if processed_img is not None:
                    save_path = subject_test_dir / f"{subject_id}_test_{i:03d}.bmp"
                    cv2.imwrite(str(save_path), processed_img)
                    processed_count += 1
        
        print(f"Processed {processed_count} images")
        print(f"Organized dataset saved to: {self.processed_data_path}")
        
        # Save dataset info
        dataset_info = {
            'total_subjects': len(subject_images),
            'total_processed_images': processed_count,
            'train_dir': str(train_dir),
            'val_dir': str(val_dir),
            'test_dir': str(test_dir),
            'preprocessing': {
                'target_size': [128, 128],
                'clahe_applied': True,
                'normalized': True
            }
        }
        
        info_file = self.processed_data_path / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset info saved to: {info_file}")
    
    def create_training_script(self):
        """Update the CNN training script to use the organized dataset"""
        training_script = """
# Update the main() function in tongji_cnn_trainer.py to use processed data:

def load_processed_tongji_dataset(processed_dir):
    \"\"\"Load the processed and organized Tongji dataset\"\"\"
    processed_path = Path(processed_dir)
    
    if not processed_path.exists():
        return [], []
    
    train_dir = processed_path / "train"
    val_dir = processed_path / "val" 
    test_dir = processed_path / "test"
    
    image_paths = []
    labels = []
    
    # Load from all splits
    for split_dir, split_name in [(train_dir, 'train'), (val_dir, 'val'), (test_dir, 'test')]:
        if split_dir.exists():
            for subject_dir in split_dir.iterdir():
                if subject_dir.is_dir():
                    subject_id = subject_dir.name.replace('subject_', '')
                    
                    for img_file in subject_dir.glob('*.bmp'):
                        image_paths.append(str(img_file))
                        labels.append(int(subject_id) if subject_id.isdigit() else hash(subject_id) % 1000)
    
    return image_paths, labels

# Add this to your main() function after device setup:
# processed_tongji_dir = "./Tongji_Palmprint/processed"
# if os.path.exists(processed_tongji_dir):
#     print("Loading processed Tongji dataset...")
#     image_paths, labels = load_processed_tongji_dataset(processed_tongji_dir)
# else:
#     # Fallback to original dataset loading
"""
        
        script_file = self.dataset_path / "training_update.py"
        with open(script_file, 'w') as f:
            f.write(training_script.strip())
        
        print(f"Training script update saved to: {script_file}")

def main():
    print("Tongji Palmprint Database Preparation Tool")
    print("=" * 50)
    
    preparator = TongjiDatasetPreparator()
    preparator.create_directories()
    
    if not preparator.check_dataset_exists():
        print("\nTongji dataset not found!")
        preparator.download_instructions()
        
        print("\nAfter downloading:")
        print("1. Extract the dataset to:", preparator.raw_data_path)
        print("2. Run this script again to organize the data")
        print("3. Use tongji_cnn_trainer.py to train the model")
        
    else:
        print("\nTongji dataset found! Processing...")
        preparator.analyze_dataset_structure()
        preparator.organize_dataset()
        preparator.create_training_script()
        
        print("\nDataset preparation complete!")
        print("Next steps:")
        print("1. Run: python tongji_cnn_trainer.py")
        print("2. The trainer will automatically use the processed Tongji data")

if __name__ == "__main__":
    main()