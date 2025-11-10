"""
Tongji-Style SIFT Palmprint Authentication System

This system uses SIFT (Scale-Invariant Feature Transform) for palmprint recognition
with enhanced preprocessing and matching techniques inspired by Tongji database approaches.

Features:
- Advanced SIFT feature extraction
- Multi-template enrollment 
- Robust matching with geometric verification
- Real-time quality assessment
- Enhanced preprocessing for various lighting conditions
"""

import cv2
import numpy as np
from datetime import datetime
import os
import json
import pickle
from SIFT_DIP import get_sift_features, sift_detect_match_num
import warnings
warnings.filterwarnings('ignore')

class TongjiSIFTAuthenticator:
    """Advanced SIFT-based palmprint authentication system"""
    
    def __init__(self):
        self.enrolled_users = {}  # Dictionary to store multiple users
        self.current_user_id = None
        
        # SIFT parameters
        self.sift_detector = cv2.SIFT_create(
            nfeatures=0,          # Keep all features
            nOctaveLayers=3,      # More octave layers for better detection
            contrastThreshold=0.04,  # Lower threshold for more features
            edgeThreshold=10,     # Edge threshold
            sigma=1.6            # Gaussian blur sigma
        )
        
        # Quality thresholds
        self.min_brightness = 40
        self.max_brightness = 220
        self.min_contrast = 25
        self.min_sharpness = 80
        self.min_sift_features = 150
        
        # Authentication thresholds
        self.min_matches = 60
        self.ratio_threshold = 0.75
        self.ransac_threshold = 5.0
        self.min_inliers = 30
        
        print("Tongji SIFT Palmprint Authentication System Initialized")
        print(f"Minimum SIFT features required: {self.min_sift_features}")
        print(f"Minimum matches for authentication: {self.min_matches}")
    
    def enhance_image(self, image):
        """Enhanced preprocessing for palmprint images"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Gamma correction for lighting normalization
        gamma = self.estimate_gamma(enhanced)
        gamma_corrected = self.adjust_gamma(enhanced, gamma)
        
        # 4. Histogram equalization
        equalized = cv2.equalizeHist(gamma_corrected)
        
        # 5. Final normalization
        normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def estimate_gamma(self, image):
        """Estimate optimal gamma value for the image"""
        # Calculate mean intensity
        mean_intensity = np.mean(image)
        
        # Optimal gamma based on mean intensity
        if mean_intensity < 100:
            gamma = 1.2  # Brighten dark images
        elif mean_intensity > 150:
            gamma = 0.8  # Darken bright images
        else:
            gamma = 1.0  # No correction needed
        
        return gamma
    
    def adjust_gamma(self, image, gamma=1.0):
        """Apply gamma correction to the image"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def assess_image_quality(self, image):
        """Comprehensive image quality assessment"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Basic quality metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # SIFT features count
        enhanced = self.enhance_image(image)
        keypoints = self.sift_detector.detect(enhanced, None)
        feature_count = len(keypoints)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Quality scores (0-100)
        brightness_score = max(0, 100 - abs(brightness - 125) * 0.6)
        contrast_score = min(100, contrast * 2.5)
        sharpness_score = min(100, sharpness / 8)
        feature_score = min(100, feature_count / 3)
        edge_score = min(100, edge_density * 10000)
        
        # Overall quality (weighted average)
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
            'is_good': overall_quality > 75 and feature_count >= self.min_sift_features
        }
    
    def extract_sift_features_advanced(self, image):
        """Advanced SIFT feature extraction with multiple scales"""
        enhanced = self.enhance_image(image)
        
        # Extract features at original scale
        keypoints1, descriptors1 = self.sift_detector.detectAndCompute(enhanced, None)
        
        # Extract features at different scales for robustness
        scale_factors = [0.8, 1.2]
        all_keypoints = list(keypoints1) if keypoints1 else []
        all_descriptors = [descriptors1] if descriptors1 is not None else []
        
        for scale in scale_factors:
            h, w = enhanced.shape
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(enhanced, (new_w, new_h))
            
            kp, desc = self.sift_detector.detectAndCompute(scaled, None)
            
            if kp and desc is not None:
                # Scale keypoints back to original size
                for keypoint in kp:
                    keypoint.pt = (keypoint.pt[0] / scale, keypoint.pt[1] / scale)
                
                all_keypoints.extend(kp)
                all_descriptors.append(desc)
        
        # Combine all descriptors
        if all_descriptors:
            combined_descriptors = np.vstack(all_descriptors)
            return all_keypoints, combined_descriptors
        else:
            return [], None
    
    def match_features_robust(self, desc1, desc2):
        """Robust feature matching with geometric verification"""
        if desc1 is None or desc2 is None:
            return [], 0
        
        # Use FLANN matcher for better performance
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(desc1, desc2, k=2)
        except:
            # Fallback to brute force matcher
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches, len(good_matches)
    
    def geometric_verification(self, kp1, kp2, matches):
        """Perform geometric verification using RANSAC"""
        if len(matches) < 4:
            return 0, None
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        try:
            H, mask = cv2.findHomography(
                src_pts, dst_pts, 
                cv2.RANSAC, 
                self.ransac_threshold,
                confidence=0.99,
                maxIters=2000
            )
            
            if H is not None:
                inliers = np.sum(mask)
                return inliers, H
            else:
                return 0, None
                
        except:
            return 0, None
    
    def enroll_user(self, user_id, palm_images):
        """Enroll a user with multiple palm images for robustness"""
        print(f"\nEnrolling user {user_id}...")
        
        if not isinstance(palm_images, list):
            palm_images = [palm_images]
        
        user_templates = []
        total_features = 0
        
        for i, image in enumerate(palm_images):
            print(f"Processing template {i+1}/{len(palm_images)}...")
            
            # Quality check
            quality = self.assess_image_quality(image)
            print(f"  Image quality: {quality['overall_quality']:.1f}%")
            
            if not quality['is_good']:
                print(f"  WARNING: Template {i+1} has poor quality, skipping...")
                continue
            
            # Extract SIFT features
            keypoints, descriptors = self.extract_sift_features_advanced(image)
            
            if descriptors is None or len(keypoints) < self.min_sift_features:
                print(f"  WARNING: Template {i+1} has insufficient features ({len(keypoints)}), skipping...")
                continue
            
            # Store template
            template = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'quality': quality,
                'processed_image': self.enhance_image(image),
                'enrollment_time': datetime.now().isoformat()
            }
            
            user_templates.append(template)
            total_features += len(keypoints)
            print(f"  Template {i+1}: {len(keypoints)} features extracted")
        
        if not user_templates:
            print("ERROR: No valid templates could be created!")
            return False
        
        # Store user data
        self.enrolled_users[user_id] = {
            'templates': user_templates,
            'enrollment_date': datetime.now().isoformat(),
            'total_templates': len(user_templates),
            'total_features': total_features
        }
        
        print(f"User {user_id} enrolled successfully!")
        print(f"Templates: {len(user_templates)}, Total features: {total_features}")
        
        # Save enrollment data
        self.save_user_data(user_id)
        return True
    
    def authenticate_user(self, user_id, test_image):
        """Authenticate a user against their enrolled templates"""
        if user_id not in self.enrolled_users:
            return False, {"error": f"User {user_id} not enrolled"}
        
        print(f"\nAuthenticating user {user_id}...")
        
        # Quality check
        quality = self.assess_image_quality(test_image)
        print(f"Test image quality: {quality['overall_quality']:.1f}%")
        
        if not quality['is_good']:
            return False, {
                "result": "REJECTED",
                "reason": "Poor image quality",
                "quality": quality['overall_quality']
            }
        
        # Extract test features
        test_kp, test_desc = self.extract_sift_features_advanced(test_image)
        
        if test_desc is None or len(test_kp) < self.min_sift_features:
            return False, {
                "result": "REJECTED", 
                "reason": "Insufficient features in test image",
                "features": len(test_kp) if test_kp else 0
            }
        
        # Match against all templates
        user_data = self.enrolled_users[user_id]
        best_score = 0
        best_matches = 0
        best_inliers = 0
        template_scores = []
        
        for i, template in enumerate(user_data['templates']):
            template_kp = template['keypoints']
            template_desc = template['descriptors']
            
            # Feature matching
            matches, match_count = self.match_features_robust(test_desc, template_desc)
            
            if match_count < self.min_matches:
                template_scores.append(0)
                continue
            
            # Geometric verification
            inliers, homography = self.geometric_verification(test_kp, template_kp, matches)
            
            # Calculate score based on matches and geometric consistency
            geometric_ratio = inliers / match_count if match_count > 0 else 0
            template_score = match_count * geometric_ratio
            
            template_scores.append(template_score)
            
            print(f"Template {i+1}: {match_count} matches, {inliers} inliers, score: {template_score:.2f}")
            
            if template_score > best_score:
                best_score = template_score
                best_matches = match_count
                best_inliers = inliers
        
        # Decision based on best template match
        is_authenticated = (
            best_score > 0 and
            best_matches >= self.min_matches and
            best_inliers >= self.min_inliers
        )
        
        result = {
            "authenticated": is_authenticated,
            "user_id": user_id,
            "best_score": best_score,
            "best_matches": best_matches,
            "best_inliers": best_inliers,
            "template_scores": template_scores,
            "test_features": len(test_kp),
            "quality": quality,
            "thresholds": {
                "min_matches": self.min_matches,
                "min_inliers": self.min_inliers
            },
            "timestamp": datetime.now().isoformat()
        }
        
        status = "AUTHENTICATED" if is_authenticated else "REJECTED"
        print(f"\nResult: {status}")
        print(f"Best score: {best_score:.2f}")
        print(f"Best matches: {best_matches}")
        print(f"Best inliers: {best_inliers}")
        
        return is_authenticated, result
    
    def save_user_data(self, user_id):
        """Save user enrollment data to file"""
        try:
            # Prepare data for saving (remove OpenCV objects)
            save_data = {
                'user_id': user_id,
                'enrollment_date': self.enrolled_users[user_id]['enrollment_date'],
                'total_templates': self.enrolled_users[user_id]['total_templates'],
                'total_features': self.enrolled_users[user_id]['total_features'],
                'templates': []
            }
            
            for template in self.enrolled_users[user_id]['templates']:
                template_data = {
                    'descriptors': template['descriptors'].tolist(),
                    'keypoints_data': [(kp.pt, kp.angle, kp.size, kp.response) for kp in template['keypoints']],
                    'quality': template['quality'],
                    'enrollment_time': template['enrollment_time']
                }
                save_data['templates'].append(template_data)
            
            # Save to file
            filename = f'user_{user_id}_enrollment.json'
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"User data saved to {filename}")
            
        except Exception as e:
            print(f"Error saving user data: {e}")
    
    def load_user_data(self, user_id):
        """Load user enrollment data from file"""
        try:
            filename = f'user_{user_id}_enrollment.json'
            
            if not os.path.exists(filename):
                return False
            
            with open(filename, 'r') as f:
                save_data = json.load(f)
            
            # Reconstruct user data
            templates = []
            for template_data in save_data['templates']:
                # Reconstruct keypoints
                keypoints = []
                for pt, angle, size, response in template_data['keypoints_data']:
                    kp = cv2.KeyPoint(pt[0], pt[1], size, angle, response)
                    keypoints.append(kp)
                
                template = {
                    'keypoints': keypoints,
                    'descriptors': np.array(template_data['descriptors'], dtype=np.float32),
                    'quality': template_data['quality'],
                    'enrollment_time': template_data['enrollment_time']
                }
                templates.append(template)
            
            self.enrolled_users[user_id] = {
                'templates': templates,
                'enrollment_date': save_data['enrollment_date'],
                'total_templates': save_data['total_templates'],
                'total_features': save_data['total_features']
            }
            
            print(f"User {user_id} data loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading user data: {e}")
            return False
    
    def real_time_authentication(self):
        """Real-time palmprint authentication using camera"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("\nTongji SIFT Palmprint Authentication")
        print("Commands:")
        print("  '1-9' - Enroll user with ID 1-9")
        print("  'a' - Authenticate (will ask for user ID)")
        print("  's' - Show enrolled users")
        print("  'q' - Quit")
        
        enrollment_frames = []
        enrolling = False
        target_user_id = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get quality assessment
            quality = self.assess_image_quality(frame)
            
            # Draw quality indicators
            display_frame = frame.copy()
            
            # Quality bar
            bar_width = int(300 * quality['overall_quality'] / 100)
            color = (0, 255, 0) if quality['is_good'] else (0, 165, 255)
            cv2.rectangle(display_frame, (20, 20), (320, 50), (0, 0, 0), 2)
            cv2.rectangle(display_frame, (20, 20), (20 + bar_width, 50), color, -1)
            
            # Quality text
            cv2.putText(display_frame, f"Quality: {quality['overall_quality']:.1f}%", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Feature count
            cv2.putText(display_frame, f"Features: {quality['feature_count']}", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Enrollment status
            if enrolling:
                cv2.putText(display_frame, f"Enrolling user {target_user_id}: {len(enrollment_frames)}/3 frames", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, "Press SPACE to capture frame", 
                           (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Instructions
            instructions = "1-9: Enroll user | A: Authenticate | S: Show users | Q: Quit"
            cv2.putText(display_frame, instructions, 
                       (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow('Tongji SIFT Palm Authentication', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle enrollment
            if key >= ord('1') and key <= ord('9') and not enrolling:
                target_user_id = key - ord('0')
                enrolling = True
                enrollment_frames = []
                print(f"Starting enrollment for user {target_user_id}")
                print("Press SPACE to capture frames (need 3 frames)")
            
            elif key == ord(' ') and enrolling:
                if quality['is_good']:
                    enrollment_frames.append(frame.copy())
                    print(f"Frame {len(enrollment_frames)}/3 captured")
                    
                    if len(enrollment_frames) >= 3:
                        success = self.enroll_user(target_user_id, enrollment_frames)
                        if success:
                            print(f"User {target_user_id} enrolled successfully!")
                        else:
                            print(f"Failed to enroll user {target_user_id}")
                        enrolling = False
                        enrollment_frames = []
                        target_user_id = None
                else:
                    print("Image quality too low, please try again")
            
            elif key == ord('a') and not enrolling:
                if quality['is_good']:
                    print("Enter user ID to authenticate (1-9): ", end='')
                    try:
                        auth_user_id = int(input())
                        if 1 <= auth_user_id <= 9:
                            authenticated, result = self.authenticate_user(auth_user_id, frame)
                            if authenticated:
                                print("✓ AUTHENTICATION SUCCESSFUL")
                            else:
                                print("✗ AUTHENTICATION FAILED")
                                print(f"Reason: {result.get('reason', 'Unknown')}")
                        else:
                            print("Invalid user ID")
                    except:
                        print("Invalid input")
                else:
                    print("Image quality too low for authentication")
            
            elif key == ord('s'):
                print(f"\nEnrolled users: {list(self.enrolled_users.keys())}")
                for uid, data in self.enrolled_users.items():
                    print(f"User {uid}: {data['total_templates']} templates, {data['total_features']} features")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("Tongji SIFT Palmprint Authentication System")
    print("=" * 50)
    
    # Initialize authenticator
    auth = TongjiSIFTAuthenticator()
    
    # Try to load existing user data
    for user_id in range(1, 10):
        auth.load_user_data(user_id)
    
    # Start real-time authentication
    auth.real_time_authentication()

if __name__ == "__main__":
    main()