import face_recognition
import os
import cv2
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)


class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.registered_encoding = None
        self.registered_name = None
        
    def capture_face_encoding(self, frame):
        """Capture and return face encoding from frame"""
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            return None, None
        
        if len(face_locations) > 1:
            return None, "multiple"
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_encodings[0], face_locations[0]
    
    def register_face(self):
        """Register a new face"""
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print('Camera not found!')
            return False
        
        print("\n=== FACE REGISTRATION ===")
        name = input("Enter your name: ").strip()
        
        if not name:
            print("Name cannot be empty!")
            video_capture.release()
            return False
        
        print(f"\nRegistering face for: {name}")
        print("Position your face in the center of the frame")
        print("Press SPACE to capture, 'q' to quit\n")
        
        face_captured = False
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Mirror the frame for natural viewing
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Try to detect face
            encoding, location = self.capture_face_encoding(frame)
            
            if location:
                top, right, bottom, left = location
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face Detected - Press SPACE", 
                           (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Face Registration', display_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord(' ') and encoding is not None:
                self.registered_encoding = encoding
                self.registered_name = name
                print(f"✓ Face registered successfully for {name}!")
                face_captured = True
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        return face_captured
    
    def verify_face(self):
        """Verify face recognition"""
        if self.registered_encoding is None:
            print("No face registered! Please register first.")
            return False
        
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print('Camera not found!')
            return False
        
        print("\n=== FACE VERIFICATION ===")
        print("Position your face in front of the camera")
        print("Press 'q' to quit\n")
        
        verified = False
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Mirror the frame for natural viewing
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # Capture face encoding and location
            encoding, location = self.capture_face_encoding(frame)
            
            # Draw face rectangle and verification status
            if location:
                top, right, bottom, left = location
                
                if encoding is not None:
                    face_distance = face_recognition.face_distance([self.registered_encoding], encoding)[0]
                    confidence = face_confidence(face_distance)
                    
                    if face_distance < 0.6:  # Match found
                        color = (0, 255, 0)
                        status = f"Verified: {self.registered_name}"
                        confidence_text = f"Confidence: {confidence}%"
                        verified = True
                        
                        # Show success message
                        cv2.putText(display_frame, "ACCESS GRANTED!", 
                                   (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    else:
                        color = (0, 0, 255)
                        status = "Not Recognized"
                        confidence_text = f"Confidence: {confidence}%"
                else:
                    color = (0, 0, 255)
                    status = "Processing..."
                    confidence_text = ""
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, status, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if confidence_text:
                    cv2.putText(display_frame, confidence_text, (left, bottom + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(display_frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Face Verification', display_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        # Appears in the terminal
        if verified:
            print(f"\n✓ Welcome, {self.registered_name}!")
            return True
        else:
            print("\n✗ Verification failed")
            return False
    
    def run(self):
        """Main program flow"""
        while True:
            print("\n" + "="*50)
            print("FACE RECOGNITION SYSTEM")
            print("="*50)
            print("1. Register Face")
            print("2. Verify Face")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                self.register_face()
            elif choice == '2':
                self.verify_face()
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()
    