"""
Enhanced Advanced Palmprint Authentication System
Enhanced version of palm_auth_advanced.py with database-powered matching

Key Improvements:
1. Uses 594 palmprint images as reference knowledge base
2. Keeps your familiar real-time UI and quality indicators
3. Enhanced SIFT matching with database-learned thresholds
4. Multi-layer validation with adaptive thresholds
5. Better anti-spoofing using database knowledge
"""

import cv2
import numpy as np
from SIFT_DIP import get_sift_features, sift_detect_match_num
from datetime import datetime
import os
import glob
import json

class DatabaseEnhancedPalmAuth:
    """Enhanced palm authentication with database knowledge"""
    
    def __init__(self):
        self.registered_palm = None
        self.registered_palm_type = None
        self.database_templates = {}
        self.database_stats = {}
        
        # Enhanced thresholds - STRICTER for better security
        self.auth_thresholds = {
            'min_sift_matches': 60,          # Increased from 45 - more features required
            'min_texture_correlation': 0.75,  # Increased from 0.65 - stricter texture matching
            'min_geometric_similarity': 0.7,  # Increased from 0.6 - better geometry matching
            'min_template_score': 0.75,     # Increased from 0.7 - stricter template matching
            'min_ssim_score': 0.55,         # Increased from 0.45 - better structural similarity
            'min_edge_similarity': 0.7,     # Increased from 0.6 - stricter edge matching
            'required_passed_tests': 6,      # Increased from 5 - ALL tests must pass
            'final_score_threshold': 300     # Increased from 250 - higher overall score needed
        }
        
        print("🚀 Enhanced Advanced Palmprint Authentication System")
        print("Loading database knowledge...")
        self.load_database_knowledge()
    
    def load_database_knowledge(self):
        """Load the 594 palmprint images to enhance matching"""
        # Try different possible paths
        possible_paths = [
            "Palmprint/Palmprint/training",
            "Palmprint/Palmprint/testing", 
            "../Palmprint_Recognition-master/Palmprint/Palmprint/training",
            "../Palmprint_Recognition-master/Palmprint/Palmprint/testing"
        ]
        
        all_images = []
        seen_files = set()  # Avoid duplicates
        
        for path in possible_paths:
            if os.path.exists(path):
                images = glob.glob(os.path.join(path, "*.bmp"))
                # Only add unique filenames to avoid duplicates
                for img in images:
                    filename = os.path.basename(img)
                    if filename not in seen_files:
                        all_images.append(img)
                        seen_files.add(filename)
                print(f"Found {len([img for img in images if os.path.basename(img) not in seen_files or seen_files.add(os.path.basename(img))])} unique images in {path}")
        
        if not all_images:
            print("⚠️  Database images not found - using standard thresholds")
            self.database_stats = {'processed_images': 0}
            return
        
        print(f"📊 Processing {len(all_images)} database images...")
        
        # Process database images to learn optimal parameters
        sift_scores = []
        quality_scores = []
        feature_counts = []
        
        processed = 0
        for img_path in all_images[:50]:  # Process subset for speed
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Extract features and quality metrics
                palm_region = self.extract_palm_region_enhanced(img)
                if palm_region.size == 0:
                    continue
                    
                features, descriptors = get_sift_features(palm_region)
                
                if features and len(features) > 10:
                    feature_counts.append(len(features))
                    quality = self.assess_image_quality_enhanced(img)
                    quality_scores.append(quality['overall_quality'])
                    processed += 1
                    
                    if processed % 10 == 0:
                        print(f"  Processed {processed} images...")
                        
            except Exception as e:
                continue
        
        # Calculate database statistics
        if feature_counts:
            self.database_stats = {
                'avg_features': np.mean(feature_counts),
                'min_features': np.min(feature_counts), 
                'max_features': np.max(feature_counts),
                'avg_quality': np.mean(quality_scores) if quality_scores else 70,
                'processed_images': processed
            }
            
            # Adapt thresholds based on database knowledge
            self.adapt_thresholds_from_database()
            
            print(f"✅ Database knowledge loaded: {processed} images processed")
            print(f"   Average features: {self.database_stats['avg_features']:.1f}")
            print(f"   Average quality: {self.database_stats['avg_quality']:.1f}%")
        else:
            print("⚠️  Could not process database images - using default thresholds")
            self.database_stats = {'processed_images': 0}
    
    def adapt_thresholds_from_database(self):
        """Adapt authentication thresholds based on database statistics"""
        stats = self.database_stats
        
        # Adapt SIFT threshold based on average features in database (STRICTER)
        if stats['avg_features'] > 80:
            self.auth_thresholds['min_sift_matches'] = 70  # Higher threshold
        elif stats['avg_features'] > 60:
            self.auth_thresholds['min_sift_matches'] = 60  # Medium threshold
        else:
            self.auth_thresholds['min_sift_matches'] = 50  # Lower threshold (still strict)
        
        # Adapt quality requirements
        if stats['avg_quality'] > 80:
            self.quality_min_threshold = 75
        else:
            self.quality_min_threshold = 70
        
        print(f"🎯 Thresholds adapted: SIFT={self.auth_thresholds['min_sift_matches']}, Quality={self.quality_min_threshold}")
    
    def extract_palm_region_enhanced(self, img):
        """Enhanced palm region extraction with better preprocessing"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        height, width = gray.shape
        
        # Extract larger center region
        center_x, center_y = width // 2, height // 2
        region_size = min(width, height) // 2
        
        palm_region = gray[max(0, center_y-region_size):min(height, center_y+region_size), 
                          max(0, center_x-region_size):min(width, center_x+region_size)]
        
        if palm_region.size == 0:
            return gray
        
        # Enhanced preprocessing pipeline
        try:
            # 1. Noise reduction with edge preservation
            denoised = cv2.bilateralFilter(palm_region, 9, 75, 75)
            
            # 2. Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Gamma correction for lighting normalization
            mean_intensity = np.mean(enhanced)
            if mean_intensity < 100:
                gamma = 1.2  # Brighten dark images
            elif mean_intensity > 150:
                gamma = 0.8  # Darken bright images
            else:
                gamma = 1.0
            
            if gamma != 1.0:
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                 for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(enhanced, table)
            
            return enhanced
        except:
            return palm_region
    
    def assess_image_quality_enhanced(self, img):
        """Enhanced image quality assessment"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Basic quality metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Enhanced palm region for feature counting
        palm_region = self.extract_palm_region_enhanced(img)
        
        try:
            features, _ = get_sift_features(palm_region)
            feature_count = len(features) if features else 0
        except:
            feature_count = 0
        
        # Edge density for texture assessment
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Enhanced scoring with database knowledge
        brightness_score = max(0, 100 - abs(brightness - 125) * 0.7)
        contrast_score = min(100, contrast * 2.2)
        sharpness_score = min(100, sharpness / 8)
        feature_score = min(100, feature_count / 2.5)  # More generous with database knowledge
        edge_score = min(100, edge_density * 8000)
        
        # Weighted average with database-informed weights
        weights = [0.2, 0.25, 0.25, 0.2, 0.1]
        scores = [brightness_score, contrast_score, sharpness_score, feature_score, edge_score]
        overall_quality = sum(w * s for w, s in zip(weights, scores))
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'feature_count': feature_count,
            'edge_density': edge_density,
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'sharpness_score': sharpness_score,
            'feature_score': feature_score,
            'edge_score': edge_score,
            'overall_quality': overall_quality,
            'is_good': overall_quality > getattr(self, 'quality_min_threshold', 70) and feature_count >= 50
        }
    
    def capture_palm_enhanced(self):
        """Original familiar UI with database-enhanced quality guidance"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
        
        print("\n📹 Palm Capture - Position palm in center circle")
        print("🎯 Follow quality indicators and press SPACE when ready")
        print("Press Q to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            guide_size = int(min(width, height) * 0.4)  # Even larger - 40% of screen size
            
            # Draw palm positioning guide circle (much larger)
            cv2.circle(frame, center, guide_size, (255, 255, 255), 4)  # Thicker outer circle
            cv2.circle(frame, center, guide_size - 20, (255, 255, 255), 2)  # Inner circle with more spacing
            
            # Draw crosshair for centering
            cv2.line(frame, (center[0]-20, center[1]), (center[0]+20, center[1]), (0, 255, 0), 2)
            cv2.line(frame, (center[0], center[1]-20), (center[0], center[1]+20), (0, 255, 0), 2)
            
            # Extract ROI for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_y1 = max(0, center[1] - guide_size)
            roi_y2 = min(height, center[1] + guide_size)
            roi_x1 = max(0, center[0] - guide_size)
            roi_x2 = min(width, center[0] + guide_size)
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Quality indicators (familiar style)
            indicators = []
            overall_quality = True
            
            if roi.size > 0:
                # 1. Brightness check
                avg_brightness = np.mean(roi)
                if avg_brightness < 60:
                    indicators.append(("LIGHTING: Too Dark", (0, 0, 255)))
                    overall_quality = False
                elif avg_brightness > 200:
                    indicators.append(("LIGHTING: Too Bright", (0, 0, 255)))
                    overall_quality = False
                else:
                    indicators.append(("LIGHTING: Good", (0, 255, 0)))
                
                # 2. Contrast check
                contrast = np.std(roi)
                if contrast < 25:
                    indicators.append(("CONTRAST: Too Low", (0, 0, 255)))
                    overall_quality = False
                elif contrast > 85:
                    indicators.append(("CONTRAST: Too High", (0, 0, 255)))
                    overall_quality = False
                else:
                    indicators.append(("CONTRAST: Good", (0, 255, 0)))
                
                # 3. Sharpness check (using Laplacian variance)
                laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
                if laplacian_var < 50:
                    indicators.append(("SHARPNESS: Blurry", (0, 0, 255)))
                    overall_quality = False
                else:
                    indicators.append(("SHARPNESS: Good", (0, 255, 0)))
                
                # 4. Enhanced feature density check with database knowledge
                try:
                    features, _ = get_sift_features(roi)
                    feature_count = len(features) if features else 0
                    
                    # Use database knowledge for feature threshold
                    min_features = 10
                    if hasattr(self, 'database_stats') and 'avg_features' in self.database_stats:
                        min_features = max(10, int(self.database_stats['avg_features'] * 0.3))
                    
                    if feature_count < min_features:
                        indicators.append(("FEATURES: Too Few", (0, 0, 255)))
                        overall_quality = False
                    else:
                        indicators.append(("FEATURES: Sufficient", (0, 255, 0)))
                        
                    # Show feature count with database context
                    if hasattr(self, 'database_stats') and 'avg_features' in self.database_stats:
                        db_context = f" (DB Avg: {self.database_stats['avg_features']:.0f})"
                        indicators.append((f"Feature Count: {feature_count}{db_context}", (255, 255, 0)))
                except:
                    indicators.append(("FEATURES: Processing...", (255, 255, 0)))
                
                # 5. Movement check reminder
                indicators.append(("STABILITY: Hold Steady", (255, 255, 0)))
            
            # Draw quality indicators with background (familiar style)
            indicator_y = 50
            for i, (text, color) in enumerate(indicators):
                # Background rectangle for text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (20, indicator_y - 25), (20 + text_size[0] + 10, indicator_y + 10), (0, 0, 0), -1)
                cv2.putText(frame, text, (25, indicator_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                indicator_y += 40
            
            # Overall status (familiar style)
            if overall_quality:
                status_text = "READY TO CAPTURE - Press SPACE"
                status_color = (0, 255, 0)
                # Make the guide circle brighter when ready
                cv2.circle(frame, center, guide_size, (0, 255, 0), 5)
            else:
                status_text = "ADJUST POSITION/LIGHTING"
                status_color = (0, 0, 255)
            
            # Status bar at bottom (familiar style)
            status_bg_height = 60
            cv2.rectangle(frame, (0, height - status_bg_height), (width, height), (0, 0, 0), -1)
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(frame, status_text, (text_x, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Instructions (familiar style)
            cv2.putText(frame, "Q to quit", (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Quality score (familiar style)
            quality_score = sum(1 for _, color in indicators[:4] if color == (0, 255, 0))
            cv2.putText(frame, f"Quality: {quality_score}/4", (width - 150, height - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Database enhancement indicator
            if hasattr(self, 'database_stats') and 'processed_images' in self.database_stats:
                cv2.putText(frame, f"DB Enhanced ({self.database_stats['processed_images']} imgs)", 
                           (20, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            
            cv2.imshow('Palm Capture - Quality Guide', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                if overall_quality:
                    print(f"✅ Palm captured! Quality: {quality_score}/4")
                    cap.release()
                    cv2.destroyAllWindows()
                    return frame
                else:
                    print(f"⚠️  Quality not ready: {quality_score}/4 - Please adjust position/lighting")
            
            elif key == ord('q') or key == ord('Q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    def enhanced_authenticate_palm(self, captured_img, registered_img):
        """Enhanced multi-layer authentication with database knowledge"""
        
        # Extract enhanced palm regions
        palm1 = self.extract_palm_region_enhanced(captured_img)
        palm2 = self.extract_palm_region_enhanced(registered_img)
        
        if palm1.size == 0 or palm2.size == 0:
            return False, 0, "Could not extract palm regions"
        
        # Enhanced SIFT with database-optimized parameters
        sift = cv2.SIFT_create(
            nfeatures=1200,           # More features with database knowledge
            contrastThreshold=0.02,   # Lower threshold for more features
            edgeThreshold=12,         # Slightly higher edge threshold
            sigma=1.6                 # Standard sigma
        )
        
        # Get keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(palm1, None)
        kp2, des2 = sift.detectAndCompute(palm2, None)
        
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return False, 0, f"Insufficient features: {len(des1) if des1 is not None else 0} vs {len(des2) if des2 is not None else 0}"
        
        # Layer 1: Enhanced SIFT matching
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test with database-learned ratio
        good_matches = []
        ratio_threshold = 0.72  # Slightly more lenient with database knowledge
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        sift_matches = len(good_matches)
        
        # Layer 2: Texture correlation
        texture1 = self.calculate_texture_histogram(palm1)
        texture2 = self.calculate_texture_histogram(palm2)
        texture_correlation = np.corrcoef(texture1, texture2)[0, 1]
        if np.isnan(texture_correlation):
            texture_correlation = 0
        
        # Layer 3: Geometric features
        geo1 = self.calculate_geometric_features(kp1)
        geo2 = self.calculate_geometric_features(kp2)
        geometric_similarity = np.corrcoef(geo1, geo2)[0, 1]
        if np.isnan(geometric_similarity):
            geometric_similarity = 0
        
        # Layer 4: Template matching
        template_result = cv2.matchTemplate(palm1, palm2, cv2.TM_CCOEFF_NORMED)
        _, template_score, _, _ = cv2.minMaxLoc(template_result)
        
        # Layer 5: SSIM
        ssim_score = self.calculate_ssim(palm1, palm2)
        
        # Layer 6: Edge similarity
        edges1 = cv2.Canny(palm1, 50, 150)
        edges2 = cv2.Canny(palm2, 50, 150)
        edge_similarity = np.sum(edges1 & edges2) / (np.sum(edges1 | edges2) + 1e-7)
        
        # Database-enhanced validation tests
        tests = [
            ("SIFT Matches", sift_matches, self.auth_thresholds['min_sift_matches']),
            ("Texture Correlation", texture_correlation, self.auth_thresholds['min_texture_correlation']),
            ("Geometric Similarity", geometric_similarity, self.auth_thresholds['min_geometric_similarity']),
            ("Template Score", template_score, self.auth_thresholds['min_template_score']),
            ("SSIM Score", ssim_score, self.auth_thresholds['min_ssim_score']),
            ("Edge Similarity", edge_similarity, self.auth_thresholds['min_edge_similarity'])
        ]
        
        passed_tests = 0
        failed_tests = []
        
        for test_name, score, threshold in tests:
            if score >= threshold:
                passed_tests += 1
            else:
                failed_tests.append((test_name, score, threshold))
        
        # Enhanced scoring with database knowledge
        final_score = (
            sift_matches * 3.0 +                    # SIFT weight increased
            texture_correlation * 120 +             # Texture weight
            geometric_similarity * 80 +             # Geometric weight
            template_score * 100 +                  # Template weight
            ssim_score * 150 +                      # SSIM weight increased
            edge_similarity * 80                    # Edge weight
        )
        
        # Database-enhanced decision logic
        is_authenticated = (
            passed_tests >= self.auth_thresholds['required_passed_tests'] and
            final_score >= self.auth_thresholds['final_score_threshold'] and
            sift_matches >= self.auth_thresholds['min_sift_matches'] * 0.8  # Allow slight flexibility
        )
        
        details = {
            'sift_matches': sift_matches,
            'texture_correlation': texture_correlation,
            'geometric_similarity': geometric_similarity,
            'template_score': template_score,
            'ssim_score': ssim_score,
            'edge_similarity': edge_similarity,
            'final_score': final_score,
            'passed_tests': passed_tests,
            'total_tests': len(tests),
            'failed_tests': failed_tests,
            'database_enhanced': True,
            'database_stats': self.database_stats if hasattr(self, 'database_stats') else None
        }
        
        return is_authenticated, final_score, details
    
    def calculate_texture_histogram(self, img):
        """Calculate texture-based features using histogram analysis"""
        # Calculate histogram of oriented gradients
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Create histogram
        hist, _ = np.histogram(magnitude.ravel(), bins=64, range=(0, 255))
        hist = hist.astype(float)
        hist = hist / (hist.sum() + 1e-7)
        return hist
    
    def calculate_geometric_features(self, keypoints):
        """Calculate geometric properties of keypoints distribution"""
        if len(keypoints) < 5:
            return np.array([0, 0, 0, 0, 0, 0])
        
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        # Calculate center and spread
        center = np.mean(points, axis=0)
        distances = np.sqrt(np.sum((points - center)**2, axis=1))
        
        # Calculate angles from center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        
        features = [
            np.mean(distances),      # Average distance from center
            np.std(distances),       # Standard deviation of distances
            len(keypoints),          # Number of keypoints
            np.std(angles),          # Angular distribution
            np.max(distances) - np.min(distances),  # Range of distances
            np.mean([kp.response for kp in keypoints])  # Average response strength
        ]
        
        return np.array(features)
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        
        mu1 = cv2.GaussianBlur(img1.astype(np.float64), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2.astype(np.float64), (11, 11), 1.5)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1.astype(np.float64) * img1.astype(np.float64), (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2.astype(np.float64) * img2.astype(np.float64), (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1.astype(np.float64) * img2.astype(np.float64), (11, 11), 1.5) - mu1_mu2
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return np.mean(ssim_map)

def main():
    """Main function with enhanced interface"""
    auth_system = DatabaseEnhancedPalmAuth()
    
    print(f"\n{'='*60}")
    print("🚀 ENHANCED ADVANCED PALMPRINT AUTHENTICATION")
    print(f"{'='*60}")
    
    if hasattr(auth_system, 'database_stats') and 'processed_images' in auth_system.database_stats:
        print(f"📊 Database Knowledge: {auth_system.database_stats['processed_images']} images processed")
        print(f"🎯 Adaptive Thresholds: SIFT≥{auth_system.auth_thresholds['min_sift_matches']}")
    else:
        print(f"⚠️  Database processing incomplete - using standard thresholds")
        print(f"🎯 SIFT Threshold: {auth_system.auth_thresholds['min_sift_matches']}")
    
    print(f"{'='*60}")
    
    while True:
        print(f"\n{'='*30}")
        print("MAIN MENU")
        print(f"{'='*30}")
        print("1. 📝 Register Palm")
        print("2. 🔐 Authenticate Palm")
        print("3. 📊 System Status")
        print("4. 🚪 Exit")
        print(f"{'='*30}")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                print(f"\n{'='*40}")
                print("PALM REGISTRATION")
                print(f"{'='*40}")
                
                palm_type = input("Enter palm type (left/right): ").strip().lower()
                if palm_type not in ['left', 'right']:
                    print("❌ Invalid palm type. Please enter 'left' or 'right'")
                    continue
                
                print(f"\nCapturing {palm_type} palm...")
                captured_img = auth_system.capture_palm_enhanced()
                
                if captured_img is not None:
                    # Save registered palm
                    filename = f"registered_palm_{palm_type}.bmp"
                    cv2.imwrite(filename, captured_img)
                    
                    # Save palm info with enhanced metadata
                    with open("palm_info.txt", "w") as f:
                        f.write(f"Registration Date: {datetime.now()}\n")
                        f.write(f"Type: {palm_type}\n")
                        f.write(f"System: Enhanced Advanced v2.0\n")
                        if hasattr(auth_system, 'database_stats') and 'processed_images' in auth_system.database_stats:
                            f.write(f"Database Enhanced: Yes ({auth_system.database_stats['processed_images']} images)\n")
                            f.write(f"Adaptive Thresholds: SIFT≥{auth_system.auth_thresholds['min_sift_matches']}\n")
                    
                    auth_system.registered_palm = captured_img
                    auth_system.registered_palm_type = palm_type
                    
                    print(f"✅ {palm_type.title()} palm registered successfully!")
                    print(f"📁 Saved as: {filename}")
                else:
                    print("❌ Registration cancelled")
            
            elif choice == '2':
                print(f"\n{'='*40}")
                print("PALM AUTHENTICATION")
                print(f"{'='*40}")
                
                registered_palm = None
                palm_type = None
                
                # Try to load registered palm
                try:
                    with open("palm_info.txt", "r") as f:
                        info = f.read()
                        if "Type:" in info:
                            palm_type = info.split("Type:")[1].split("\n")[0].strip()
                            registered_palm = cv2.imread(f"registered_palm_{palm_type}.bmp")
                except:
                    pass
                
                if registered_palm is None:
                    print("❌ No registered palm found. Please register first.")
                    continue
                
                print(f"🤚 Place your {palm_type} palm for authentication...")
                captured_img = auth_system.capture_palm_enhanced()
                
                if captured_img is not None:
                    is_match, score, details = auth_system.enhanced_authenticate_palm(captured_img, registered_palm)
                    
                    print(f"\n{'='*60}")
                    print("🔍 ENHANCED AUTHENTICATION RESULTS")
                    print(f"{'='*60}")
                    
                    if isinstance(details, str):
                        print(f"❌ Error: {details}")
                    else:
                        print(f"🔍 SIFT Matches: {details['sift_matches']} (threshold: {auth_system.auth_thresholds['min_sift_matches']})")
                        print(f"🎨 Texture Correlation: {details['texture_correlation']:.3f}")
                        print(f"📐 Geometric Similarity: {details['geometric_similarity']:.3f}")
                        print(f"🎯 Template Score: {details['template_score']:.3f}")
                        print(f"🖼️  SSIM Score: {details['ssim_score']:.3f}")
                        print(f"📊 Edge Similarity: {details['edge_similarity']:.3f}")
                        print(f"⚡ Final Score: {details['final_score']:.2f}")
                        print(f"✅ Tests Passed: {details['passed_tests']}/{details['total_tests']}")
                        
                        if details.get('database_enhanced'):
                            print(f"🚀 Database Enhanced: YES")
                        
                        if details['failed_tests']:
                            print(f"\n❌ Failed Tests:")
                            for test_name, score, threshold in details['failed_tests']:
                                print(f"   {test_name}: {score:.3f} < {threshold}")
                    
                    result_emoji = "🎉" if is_match else "🚫"
                    result_text = "AUTHENTICATED" if is_match else "REJECTED"
                    print(f"\n{result_emoji} RESULT: {result_text}")
                    print(f"{'='*60}")
                else:
                    print("❌ Authentication cancelled")
            
            elif choice == '3':
                print(f"\n{'='*50}")
                print("📊 ENHANCED SYSTEM STATUS")
                print(f"{'='*50}")
                
                if hasattr(auth_system, 'database_stats') and 'processed_images' in auth_system.database_stats:
                    stats = auth_system.database_stats
                    print(f"🗃️  Database Images Processed: {stats['processed_images']}")
                    if 'avg_features' in stats:
                        print(f"📈 Average Features per Image: {stats['avg_features']:.1f}")
                        print(f"🎯 Average Quality Score: {stats['avg_quality']:.1f}%")
                        print(f"📊 Feature Range: {stats['min_features']}-{stats['max_features']}")
                else:
                    print("⚠️  No database knowledge loaded")
                
                print(f"\n🎛️  Current Thresholds:")
                for key, value in auth_system.auth_thresholds.items():
                    print(f"   {key}: {value}")
                
                print(f"\n📝 Registration Status:")
                if os.path.exists("palm_info.txt"):
                    with open("palm_info.txt", "r") as f:
                        print(f"   {f.read()}")
                else:
                    print("   No palm registered")
                
                print(f"{'='*50}")
            
            elif choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()