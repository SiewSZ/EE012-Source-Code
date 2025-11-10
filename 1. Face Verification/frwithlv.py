import cv2
import dlib
import os
import math
import time
import numpy as np
import face_recognition
from imutils import face_utils
from scipy.spatial import distance as dist
import warnings
import json

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------- Helper Function: Face Confidence ---------------------- #
def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)
    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)


# ---------------------- Eye Aspect Ratio (for Blink Detection) ---------------------- #
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 2

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# ---------------------- Load Pretrained Models ---------------------- #
frontal_face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('dataset/shape_predictor_68_face_landmarks.dat')


# ---------------------- Main Class ---------------------- #
class FaceRecognitionSystem:
    def __init__(self):
        self.data_file = "registered_faces.json"
        self.registered_encoding = None
        self.registered_name = None
        self.load_registered_faces()

# ---------------------- Persistent Storage ---------------------- #
    def save_registered_faces(self):
        """Save registered face encoding and name."""
        if self.registered_encoding is not None:
            data = {
                "name": self.registered_name,
                "encoding": self.registered_encoding.tolist()
            }
            with open(self.data_file, "w") as f:
                json.dump(data, f)
            print(f"Saved registered face for {self.registered_name}")

    def load_registered_faces(self):
        """Load registered face from file if exists."""
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                data = json.load(f)
                self.registered_name = data["name"]
                self.registered_encoding = np.array(data["encoding"])
            print(f"Loaded registered face for {self.registered_name}")
        else:
            print("No registered face found — please register first.")

    # ---------------------- Capture + Register ---------------------- #
    def capture_face_encoding(self, frame):
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb)
        if len(face_locations) != 1:
            return None, None
        encoding = face_recognition.face_encodings(rgb, face_locations)[0]
        return encoding, face_locations[0]

    def register_face(self):
        """Register a new face only if not already registered."""
        # Prevent re-registration if a face is already stored
        if os.path.exists(self.data_file) and self.registered_encoding is not None:
            print(f"Already registered as {self.registered_name}.")
            return True

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not found!")
            return False

        name = input("Enter your name: ").strip()
        print("Align your face in the center. Press SPACE to capture.")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            encoding, loc = self.capture_face_encoding(frame)

            if loc:
                top, right, bottom, left = loc
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Register Face", frame)

            key = cv2.waitKey(1)
            if key == ord(' '):
                if encoding is not None:
                    self.registered_encoding = encoding
                    self.registered_name = name
                    self.save_registered_faces()
                    print(f"✓ Face registered successfully for {name}!")
                    break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return True


    # ---------------------- Enhanced Liveness Detection ---------------------- #
    def liveness_check(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not found!")
            return False

        COUNTER = 0
        blink_detected = False
        head_turns = {"LEFT": False, "CENTER": False, "RIGHT": False}
        start_time = time.time()
        TIME_LIMIT = 30
        direction = "CENTER"
        current_instruction = "Face forward (CENTER) to begin"

        print("\n=== LIVENESS DETECTION STARTED ===")
        print("Instructions: Look CENTER, then turn LEFT, then RIGHT, and blink once.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = frontal_face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = predictor(gray, rect)
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                # ---------- Blink Detection ----------
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        blink_detected = True
                    COUNTER = 0

                # ---------- Head Movement ----------
                nose_x = landmarks[33][0]
                left_cheek = landmarks[0][0]
                right_cheek = landmarks[16][0]
                face_center = (left_cheek + right_cheek) / 2
                offset = nose_x - face_center

                if offset < -28:
                    head_turns["LEFT"] = True
                    direction = "LEFT"
                elif offset > 28:
                    head_turns["RIGHT"] = True
                    direction = "RIGHT"
                else:
                    head_turns["CENTER"] = True
                    direction = "CENTER"

                # ---------- Instructions ----------
                if not head_turns["CENTER"]:
                    current_instruction = "Face forward (CENTER)"
                elif not head_turns["LEFT"]:
                    current_instruction = "Turn your head LEFT"
                elif not head_turns["RIGHT"]:
                    current_instruction = "Turn your head RIGHT"
                elif not blink_detected:
                    current_instruction = "Blink your eyes"
                else:
                    current_instruction = "All actions completed !"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Dir: {direction}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ---------- Timer ----------
            elapsed = time.time() - start_time
            remaining = TIME_LIMIT - int(elapsed)
            cv2.putText(frame, f"Time Left: {remaining}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255) if remaining > 10 else (0, 0, 255), 2)

            progress = int(blink_detected) + sum(head_turns.values())
            cv2.putText(frame, f"Progress: {progress}/4 actions done", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if progress == 4 else (255, 255, 0), 2)

            cv2.putText(frame, current_instruction, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if "✅" in current_instruction else (0, 255, 255), 2)

            cv2.putText(frame, f"Blink: {'Yes' if blink_detected else 'No'}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if blink_detected else (0, 0, 255), 2)
            cv2.putText(frame, f"L:{head_turns['LEFT']} C:{head_turns['CENTER']} R:{head_turns['RIGHT']}",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Liveness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False

            if blink_detected and all(head_turns.values()):
                print("\n✅ Liveness confirmed (Blink + Head Movements Detected)!")
                self.show_final_overlay("Access Granted !", (0, 255, 0))
                return True

            if elapsed > TIME_LIMIT:
                print("\n❌ Liveness failed: 30s time limit exceeded — access denied.")
                self.show_final_overlay("Access Denied !", (0, 0, 255))
                return False


    # ---------------------- Result Overlay ---------------------- #
    def show_final_overlay(self, message, color):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        cv2.putText(frame, message, (80, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        cv2.imshow("Liveness Detection", frame)
        cv2.waitKey(3000)
        cap.release()
        cv2.destroyAllWindows()


    # ---------------------- Face Verification ---------------------- #
    def verify_face(self):
        """Verify identity, show result longer, and only proceed to liveness if match is positive."""
        if self.registered_encoding is None:
            print("No face registered! Please register first.")
            return False

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not found!")
            return False

        verified = False
        print("Align your face with the camera for verification...")

        start_time = time.time()
        HOLD_DURATION = 8  # seconds to display result before next step

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            encoding, loc = self.capture_face_encoding(frame)

            if encoding is not None:
                face_distance = face_recognition.face_distance(
                    [self.registered_encoding], encoding
                )[0]
                confidence = face_confidence(face_distance)

                top, right, bottom, left = loc
                if face_distance < 0.6:
                    verified = True
                    color = (0, 255, 0)
                    message = f"Verified: {self.registered_name}"
                    status_text = f"Confidence: {confidence}%"
                else:
                    verified = False
                    color = (0, 0, 255)
                    message = "Access Denied"
                    status_text = f"Confidence: {confidence}%"

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, message, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, status_text, (left, bottom + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Verification", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Verification cancelled by user.")
                return False

            if encoding is not None:
                time.sleep(HOLD_DURATION)
                break

        cap.release()
        cv2.destroyAllWindows()

        if verified:
            print(f"\nFace recognized as {self.registered_name}. Proceeding to liveness test...")
            if self.liveness_check():
                print(f"\n✅ Access Granted: {self.registered_name}")
                return True
            else:
                print("\n❌ Liveness Failed: Access Denied.")
                return False
        else:
            print("\n❌ Face not recognized — Access Denied.")
            self.show_final_overlay("Access Denied ✖", (0, 0, 255))
            return False


    # ---------------------- Main Program ---------------------- #
    def run(self):
        while True:
            print("\n==============================")
            print(" FACE RECOGNITION SYSTEM ")
            print("==============================")
            print("1. Register Face")
            print("2. Verify Face")
            print("3. Exit")
            choice = input("\nEnter choice: ").strip()
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
    FaceRecognitionSystem().run()
