import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import mediapipe as mp  # --- Import MediaPipe ---

# --- Configuration Constants ---

# --- New Dynamic Calibration Constants ---
# We no longer use a fixed EAR threshold.
# Instead, we set a personal threshold as a percentage of the user's open-eye value.
# e.g., if open EAR is 0.30, threshold will be 0.30 * 0.75 = 0.225
THRESHOLD_CALIBRATION_FACTOR = 0.75

# Sanity check: if the user's "open" EAR is below this,
# they probably tried to register with their eyes closed.
MIN_OPEN_EYE_EAR = 0.20

# How many consecutive frames of "closed" eyes trigger an alarm?
# User requested 10 seconds. Assuming ~20fps, 10 * 20 = 200 frames.
EYE_AR_CONSEC_FRAMES = 200

# Tracking constant: max distance a face can "jump" between frames.
MAX_TRACKING_JUMP_PX = 150

class SleepingAlertApp:
    def __init__(self, window):
        """
        Initialize the application.
        """
        self.window = window
        self.window.title("Sleeping Alert System (Calibrated)")
        self.window.geometry("800x700")

        # --- State Variables ---
        self.cap = None
        self.video_thread = None
        self.is_running = False

        # --- MediaPipe Face Mesh Initialization ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,  # Detect up to 5 faces
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Logic & Registration State ---
        self.eye_closed_counter = 0
        self.is_monitoring = False
        self.registered_face_center_guess = None
        self.registered_person_name = None
        self.current_frame_for_registration = None
        self.registration_lock = threading.Lock()

        # --- New Calibration State Variables ---
        # These are set during registration and are unique to the user.
        self.calibrated_open_ear = 0.0
        self.calibrated_threshold = 0.0

        # --- GUI Elements ---
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- GUI Layout Fix: Anchor controls to BOTTOM ---
        
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.BOTTOM, pady=10)

        self.status_text = tk.StringVar()
        self.status_text.set("Ready. Press 'Start Camera' to begin.")
        self.status_label = tk.Label(self.main_frame, textvariable=self.status_text, font=("Arial", 14))
        self.status_label.pack(side=tk.BOTTOM, pady=5)

        # Video label expands to fill the remaining space
        self.video_label = tk.Label(self.main_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # --- Buttons ---
        self.start_button = tk.Button(self.button_frame, text="Start Camera", command=self.start_video_stream, font=("Arial", 12), width=15)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(self.button_frame, text="Stop Camera", command=self.stop_video_stream, font=("Arial", 12), width=15)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.register_button = tk.Button(self.button_frame, text="Register & Monitor", command=self.register_person, font=("Arial", 12), width=18)
        self.register_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear Registration", command=self.clear_registration, font=("Arial", 12), width=18)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_video_stream(self):
        """
        Starts the video capture in a new thread.
        """
        if self.is_running:
            return

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("Cannot open webcam.")
            
            self.is_running = True
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            self.status_text.set("Camera running. Click 'Register & Monitor'.")
        
        except IOError as e:
            messagebox.showerror("Webcam Error", str(e))
            if self.cap:
                self.cap.release()

    def stop_video_stream(self):
        """
        Signals the video loop to stop and resets all states.
        """
        if not self.is_running:
            return
            
        self.is_running = False
        if self.video_thread:
            self.video_thread.join(timeout=0.5) 
            
        if self.cap:
            self.cap.release()

        self.video_label.config(image=None)
        self.status_text.set("Camera stopped.")
        
        # Reset all logic states
        self.is_monitoring = False
        self.registered_face_center_guess = None
        self.registered_person_name = None
        self.eye_closed_counter = 0
        self.calibrated_open_ear = 0.0
        self.calibrated_threshold = 0.0

    def video_loop(self):
        """
        The main loop for video processing. Runs in a separate thread.
        """
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_text.set("Error: Can't read from camera.")
                    time.sleep(0.5)
                    continue

                frame = cv2.flip(frame, 1)
                
                with self.registration_lock:
                    self.current_frame_for_registration = frame.copy()
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame and get status
                processed_frame, status = self.process_frame_logic(frame, rgb_frame)

                self.status_text.set(status)

                # Convert for Tkinter display
                cv_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_img)
                
                w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
                if w > 1 and h > 1:
                     pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)

                imgtk = ImageTk.PhotoImage(image=pil_img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            except Exception as e:
                print(f"Error in video loop: {e}")
                self.is_running = False
            
            time.sleep(0.01)

        print("Video loop stopped.")

    def get_ear(self, eye_points, w, h):
        """
        Calculates the Eye Aspect Ratio (EAR) given 6 landmark points.
        """
        try:
            # Convert landmarks to (x, y) tuples
            p1 = (int(eye_points[0].x * w), int(eye_points[0].y * h))
            p4 = (int(eye_points[1].x * w), int(eye_points[1].y * h))
            p2 = (int(eye_points[2].x * w), int(eye_points[2].y * h))
            p6 = (int(eye_points[3].x * w), int(eye_points[3].y * h))
            p3 = (int(eye_points[4].x * w), int(eye_points[4].y * h))
            p5 = (int(eye_points[5].x * w), int(eye_points[5].y * h))

            def get_dist(p_a, p_b):
                return np.linalg.norm(np.array(p_a) - np.array(p_b))

            v_dist_1 = get_dist(p2, p6)
            v_dist_2 = get_dist(p3, p5)
            h_dist = get_dist(p1, p4)

            if h_dist == 0:
                return 0.3 # Default "open"

            ear = (v_dist_1 + v_dist_2) / (2.0 * h_dist)
            return ear
        except Exception:
            return 0.3 # Default "open"

    def get_faces_from_results(self, frame, results):
        """
        Extracts face data (box, center, EAR) from MediaPipe results.
        """
        faces_data = []
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Get Bounding Box
                x_min, y_min, x_max, y_max = w, h, 0, 0
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    if x < x_min: x_min = x
                    if y < y_min: y_min = y
                    if x > x_max: x_max = x
                    if y > y_max: y_max = y
                
                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                box = (x_min, y_min, x_max - x_min, y_max - y_min)

                # 2. Get Nose Tip (Landmark 1) for tracking
                nose_tip = face_landmarks.landmark[1]
                center = (int(nose_tip.x * w), int(nose_tip.y * h))
                
                # 3. Calculate Eye Aspect Ratio (EAR)
                RIGHT_EYE_EAR_POINTS_INDICES = [33, 133, 159, 145, 158, 153]
                LEFT_EYE_EAR_POINTS_INDICES = [362, 263, 386, 374, 385, 380]

                landmarks = face_landmarks.landmark
                right_eye_points = [landmarks[i] for i in RIGHT_EYE_EAR_POINTS_INDICES]
                left_eye_points = [landmarks[i] for i in LEFT_EYE_EAR_POINTS_INDICES]

                right_ear = self.get_ear(right_eye_points, w, h)
                left_ear = self.get_ear(left_eye_points, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                
                faces_data.append({'box': box, 'center': center, 'avg_ear': avg_ear})
        
        return faces_data

    def register_person(self):
        """
        Registers a person and dynamically calibrates the EAR threshold.
        """
        with self.registration_lock:
            if self.current_frame_for_registration is None:
                messagebox.showwarning("Registration Error", "Camera not ready. Please try again.")
                return
            
            rgb_frame_reg = cv2.cvtColor(self.current_frame_for_registration, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame_reg)
            faces = self.get_faces_from_results(self.current_frame_for_registration, results)

        if len(faces) == 0:
            messagebox.showwarning("Registration Error", "No person detected. Please face the camera and try again.")
            return

        faces_by_area = sorted(faces, key=lambda f: f['box'][2] * f['box'][3], reverse=True)
        largest_face = faces_by_area[0]

        # --- DYNAMIC CALIBRATION ---
        current_ear = largest_face['avg_ear']
        
        if current_ear < MIN_OPEN_EYE_EAR:
            messagebox.showwarning("Registration Error", f"Eyes seem to be closed (EAR: {current_ear:.2f}).\nPlease open your eyes and try again.")
            return

        self.calibrated_open_ear = current_ear
        self.calibrated_threshold = current_ear * THRESHOLD_CALIBRATION_FACTOR
        
        name = simpledialog.askstring("Register Person", "Enter the person's name:", parent=self.window)
        if not name:
            messagebox.showwarning("Registration Cancelled", "Registration was cancelled.")
            return
        
        # --- Lock in the registration ---
        (x, y, w, h) = largest_face['box']
        self.is_monitoring = True
        self.registered_person_name = name
        self.registered_face_center_guess = largest_face['center']
        self.eye_closed_counter = 0

        status = f"Calibrated for {name}. Open EAR: {self.calibrated_open_ear:.2f}, Threshold: {self.calibrated_threshold:.2f}"
        self.status_text.set(status)
        messagebox.showinfo("Registration Complete", status)

    def clear_registration(self):
        """
        Clears the current registration and stops monitoring.
        """
        if not self.is_monitoring:
            messagebox.showwarning("Info", "No person is currently registered.")
            return
        
        # Reset all logic states
        self.is_monitoring = False
        self.registered_face_center_guess = None
        self.registered_person_name = None
        self.eye_closed_counter = 0
        self.calibrated_open_ear = 0.0
        self.calibrated_threshold = 0.0

        self.status_text.set("Monitoring stopped. Ready to register a new person.")


    def process_frame_logic(self, frame, rgb_frame):
        """
        Main logic: finds faces, tracks registered person, checks EAR.
        """
        results = self.face_mesh.process(rgb_frame)
        faces = self.get_faces_from_results(frame, results)

        # If not monitoring, just show all faces
        if not self.is_monitoring:
            current_status = "Click 'Register & Monitor' to begin."
            for face_data in faces:
                (x, y, w, h) = face_data['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 192, 0), 2)
            return frame, current_status

        # --- If we ARE monitoring ---
        
        if len(faces) == 0:
            current_status = f"MONITORING: {self.registered_person_name} lost!"
            self.eye_closed_counter = 0
            return frame, current_status

        # Find the face closest to our last known center
        min_dist = float('inf')
        tracked_face_data = None
        current_center = None

        for face_data in faces:
            center = face_data['center']
            dist = np.linalg.norm(np.array(center) - np.array(self.registered_face_center_guess))
            if dist < min_dist:
                min_dist = dist
                tracked_face_data = face_data
                current_center = center
        
        if min_dist > MAX_TRACKING_JUMP_PX:
             current_status = f"MONITORING: {self.registered_person_name} lost! (New face detected)"
             self.eye_closed_counter = 0
             (x, y, w, h) = tracked_face_data['box']
             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
             return frame, current_status

        # --- We have re-acquired the tracked face ---
        (x, y, w, h) = tracked_face_data['box']
        self.registered_face_center_guess = current_center
        avg_ear = tracked_face_data['avg_ear']
        alert_triggered = False
        
        # --- CALIBRATED Eye Aspect Ratio (EAR) Logic ---
        if avg_ear < self.calibrated_threshold:
            self.eye_closed_counter += 1
        else:
            self.eye_closed_counter = 0
        
        if self.eye_closed_counter > EYE_AR_CONSEC_FRAMES:
            current_status = f"!!! ALERT: {self.registered_person_name} EYES CLOSED !!!"
            alert_triggered = True
        else:
            current_status = f"Monitoring: {self.registered_person_name} (Eyes Open)"
        
        # --- Draw on the frame ---
        if alert_triggered:
            color = (0, 0, 255) # Red for Alert
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"ALERT: {self.registered_person_name} (EYES CLOSED)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            color = (255, 0, 0) # Blue for Tracking
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, self.registered_person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show the live data for tuning
            ear_text = f"EAR: {avg_ear:.2f} (Thresh: {self.calibrated_threshold:.2f})"
            count_text = f"Eye Closed Count: {self.eye_closed_counter}/{EYE_AR_CONSEC_FRAMES}"
            cv2.putText(frame, ear_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, count_text, (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, current_status


    def on_closing(self):
        """
        Handles the window close event.
        """
        if messagebox.askokcancel("Quit", "Do you want to exit the application?"):
            self.stop_video_stream()
            self.window.destroy()

# --- Main execution ---
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SleepingAlertApp(root)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")