import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import mediapipe as mp  # --- Import MediaPipe ---
import winsound        # --- Using built-in winsound for Windows ---

# --- Configuration Constants ---

# --- Path to the alert sound ---
# Make sure to use double backslashes (\\) on Windows!
ALERT_SOUND_FILE_PATH = "D:\\GIT_HUB\\12_Final_Projects_of_all\\03_deep_learning\\beep-01a.wav"

# --- NEW: Sound loop configuration ---
# Time in seconds between playing the alert sound
ALERT_SOUND_INTERVAL = 2.0 # Play sound every 2 seconds

# --- Dynamic Calibration Constants ---
THRESHOLD_CALIBRATION_FACTOR = 0.75
MIN_OPEN_EYE_EAR = 0.20
EYE_AR_CONSEC_FRAMES = 200 # ~10 seconds at 20fps

# --- Tracking Constants ---
LOST_GRACE_FRAMES = 50 # ~2.5 seconds
MAX_TRACKING_JUMP_PX = 150 # Max distance a face can move between frames

class SleepingAlertApp:
    def __init__(self, window):
        """
        Initialize the application.
        """
        self.window = window
        self.window.title("Multi-Person Sleeping Alert System (v2.0)")
        self.window.geometry("1000x700") # Made window wider for sidebar
        self.window.minsize(800, 600) # Set a minimum size

        # --- State Variables ---
        self.cap = None
        self.video_thread = None
        self.is_running = False

        # --- MediaPipe Face Mesh Initialization ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,  # Detect more faces
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Multi-Person Logic & Registration State ---
        self.registered_people = {} # Main dictionary to hold all registered people
        self.registration_lock = threading.Lock()
        self.current_frame_for_registration = None

        # --- GUI REBUILD: Using .grid() for stable layout ---
        
        # Configure the main window's grid
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.rowconfigure(0, weight=1) # Content (video/sidebar) expands
        self.main_frame.rowconfigure(1, weight=0) # Bottom bar does not expand
        self.main_frame.columnconfigure(0, weight=1) # Main column expands
        
        # --- 1. Content Frame (Video & Sidebar) ---
        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.content_frame.rowconfigure(0, weight=1)
        self.content_frame.columnconfigure(0, weight=1) # Video expands
        self.content_frame.columnconfigure(1, weight=0) # Sidebar does not expand
        
        # Video Label
        self.video_label = tk.Label(self.content_frame, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Sidebar
        self.sidebar_frame = tk.Frame(self.content_frame, width=250, bg="#f0f0f0", relief=tk.SUNKEN, borderwidth=2)
        self.sidebar_frame.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.sidebar_frame.pack_propagate(False)

        self.sidebar_title = tk.Label(self.sidebar_frame, text="Registered People", font=("Arial", 16, "bold"), bg="#f0f0f0")
        self.sidebar_title.pack(pady=10, padx=10, anchor="w")

        self.registered_list_var = tk.StringVar()
        self.registered_list_display = tk.Label(self.sidebar_frame, textvariable=self.registered_list_var, font=("Arial", 12), bg="#f0f0f0", justify=tk.LEFT, anchor="nw")
        self.registered_list_display.pack(pady=5, padx=10, fill=tk.X, anchor="nw")
        self.registered_list_var.set("None")

        # --- 2. Bottom Frame (Status & Buttons) ---
        self.bottom_frame = tk.Frame(self.main_frame, bg="#e0e0e0")
        self.bottom_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.bottom_frame.columnconfigure(0, weight=1) # Status label expands
        self.bottom_frame.columnconfigure(1, weight=0) # Buttons do not expand

        self.status_text = tk.StringVar()
        self.status_text.set("Ready. Press 'Start Camera' to begin.")
        self.status_label = tk.Label(self.bottom_frame, textvariable=self.status_text, font=("Arial", 14), bg="#e0e0e0", anchor="w")
        self.status_label.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.button_frame = tk.Frame(self.bottom_frame, bg="#e0e0e0")
        self.button_frame.grid(row=0, column=1, sticky="e", padx=5, pady=5)

        self.start_button = tk.Button(self.button_frame, text="Start Camera", command=self.start_video_stream, font=("Arial", 12), width=15)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(self.button_frame, text="Stop Camera", command=self.stop_video_stream, font=("Arial", 12), width=15)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.register_button = tk.Button(self.button_frame, text="Register & Monitor", command=self.register_person, font=("Arial", 12), width=18)
        self.register_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear All", command=self.clear_all_registrations, font=("Arial", 12), width=18)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_video_stream(self):
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
            if self.cap: self.cap.release()

    def stop_video_stream(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.video_thread:
            self.video_thread.join(timeout=0.5) 
        if self.cap:
            self.cap.release()
        self.video_label.config(image=None)
        self.status_text.set("Camera stopped.")
        self.clear_all_registrations() # Clear people when camera stops

    def video_loop(self):
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
                processed_frame, status, list_text = self.process_frame_logic(frame, rgb_frame)
                
                # Update GUI from main thread
                self.status_text.set(status)
                self.registered_list_var.set(list_text if list_text else "None")

                # Resize image
                cv_img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_img)
                w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
                if w > 1 and h > 1:
                     pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=pil_img)
                
                # Update image
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            except Exception as e:
                print(f"Error in video loop: {e}")
                self.is_running = False
            
            time.sleep(0.01) # ~100fps theoretical max, but processing slows it
        print("Video loop stopped.")

    def get_ear(self, eye_points, w, h):
        """Calculates Eye Aspect Ratio (EAR)."""
        try:
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
            if h_dist == 0: return 0.3
            ear = (v_dist_1 + v_dist_2) / (2.0 * h_dist)
            return ear
        except Exception:
            return 0.3

    def get_faces_from_results(self, frame, results):
        """Extracts all face data (box, center, EAR) from MediaPipe results."""
        faces_data = []
        h, w, _ = frame.shape
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
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
                nose_tip = face_landmarks.landmark[1]
                center = (int(nose_tip.x * w), int(nose_tip.y * h))
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
        """Registers a new person based on the largest face in the frame."""
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
        current_ear = largest_face['avg_ear']
        
        if current_ear < MIN_OPEN_EYE_EAR:
            messagebox.showwarning("Registration Error", f"Eyes seem to be closed (EAR: {current_ear:.2f}).\nPlease open your eyes and try again.")
            return

        calibrated_threshold = current_ear * THRESHOLD_CALIBRATION_FACTOR
        name = simpledialog.askstring("Register Person", "Enter a unique name:", parent=self.window)
        
        if not name:
            messagebox.showwarning("Registration Cancelled", "Registration was cancelled.")
            return
            
        if name in self.registered_people:
            messagebox.showwarning("Registration Error", f"The name '{name}' is already registered. Please use a unique name.")
            return
        
        # --- Add new person to the dictionary ---
        self.registered_people[name] = {
            'center': largest_face['center'],
            'open_ear': current_ear,
            'threshold': calibrated_threshold,
            'eye_counter': 0,
            'lost_counter': 0,
            'status': 'Calibrated', # For the sidebar list
            'last_alert_time': 0.0 # NEW: For sound loop
        }

        status = f"Calibrated for {name}. Open EAR: {current_ear:.2f}, Threshold: {calibrated_threshold:.2f}"
        self.status_text.set(f"Registered {name}. Monitoring {len(self.registered_people)} person(s).")
        messagebox.showinfo("Registration Complete", status)

    def clear_all_registrations(self):
        """Clears all registered people from monitoring."""
        if not self.registered_people:
            return
        self.registered_people = {}
        self.status_text.set("All registrations cleared. Ready to register.")
        self.registered_list_var.set("None")

    def play_alert_sound(self):
        """Plays the alert sound asynchronously using winsound."""
        if not os.path.exists(ALERT_SOUND_FILE_PATH):
            print(f"Alert Sound Error: File not found at {ALERT_SOUND_FILE_PATH}")
            print("!!! ALERT (SOUND FILE MISSING) !!!")
            return
        try:
            winsound.PlaySound(ALERT_SOUND_FILE_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Error playing sound file {ALERT_SOUND_FILE_PATH}: {e}")
            print("!!! ALERT BEEP !!!")

    def process_person(self, frame, person_name, person_data, face_data):
        """
        Processes a single person (found or lost), updates state, and draws on frame.
        """
        alert_triggered = False
        
        if face_data is None:
            # --- Person is LOST ---
            person_data['lost_counter'] += 1
            if person_data['lost_counter'] > LOST_GRACE_FRAMES:
                person_data['status'] = 'Lost'
                person_data['eye_counter'] = 0
            else:
                person_data['status'] = 'Searching...'
            return # Stop processing
        
        # --- Person is FOUND ---
        person_data['lost_counter'] = 0
        person_data['center'] = face_data['center']
        avg_ear = face_data['avg_ear']
        
        # --- Check Eye Status ---
        if avg_ear < person_data['threshold']:
            person_data['eye_counter'] += 1
        else:
            person_data['eye_counter'] = 0
            # Continuous Re-calibration
            alpha = 0.01
            person_data['open_ear'] = (person_data['open_ear'] * (1 - alpha)) + (avg_ear * alpha)
            person_data['threshold'] = person_data['open_ear'] * THRESHOLD_CALIBRATION_FACTOR
        
        # --- Check Alert Status ---
        if person_data['eye_counter'] > EYE_AR_CONSEC_FRAMES:
            person_data['status'] = '!!! ALERT !!!'
            alert_triggered = True
        else:
            person_data['status'] = 'Tracking'
        
        # --- Draw on Frame ---
        (x, y, w, h) = face_data['box']
        
        if alert_triggered:
            color = (0, 0, 255) # Red
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"ALERT: {person_name} (EYES CLOSED)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # --- NEW: LOOPING SOUND LOGIC ---
            current_time = time.time()
            if current_time - person_data['last_alert_time'] > ALERT_SOUND_INTERVAL:
                threading.Thread(target=self.play_alert_sound, daemon=True).start()
                person_data['last_alert_time'] = current_time
        else:
            color = (255, 0, 0) # Blue
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            ear_text = f"EAR: {avg_ear:.2f} (Th: {person_data['threshold']:.2f})"
            count_text = f"Count: {person_data['eye_counter']}/{EYE_AR_CONSEC_FRAMES}"
            cv2.putText(frame, ear_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, count_text, (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    def process_frame_logic(self, frame, rgb_frame):
        """Main logic loop for processing a frame."""
        results = self.face_mesh.process(rgb_frame)
        faces = self.get_faces_from_results(frame, results)

        if not self.registered_people:
            # Not monitoring anyone, just draw all faces
            for face_data in faces:
                (x, y, w, h) = face_data['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 192, 0), 2)
            return frame, "Click 'Register & Monitor' to begin.", ""

        # --- Multi-Person Tracking Logic ---
        unmatched_face_indices = set(range(len(faces)))
        matches = {} # Stores {person_name: face_index}
        
        # Greedily assign best match for each person
        for name, person_data in self.registered_people.items():
            best_dist = float('inf')
            best_face_idx = -1
            for i in unmatched_face_indices: # Only check against unused faces
                dist = np.linalg.norm(np.array(faces[i]['center']) - np.array(person_data['center']))
                if dist < best_dist and dist < MAX_TRACKING_JUMP_PX:
                    best_dist = dist
                    best_face_idx = i
            
            if best_face_idx != -1:
                matches[name] = best_face_idx
                unmatched_face_indices.remove(best_face_idx) # This face is now taken

        list_display_text = []
        # Process all people (matched or lost)
        for name, person_data in self.registered_people.items():
            face_data = None
            if name in matches:
                face_data = faces[matches[name]] # Get the matched face data
            
            # This function updates the person's state and draws on the frame
            self.process_person(frame, name, person_data, face_data)
            
            list_display_text.append(f"{name}: {person_data['status']}")

        # Draw remaining unmatched (unregistered) faces
        for i in unmatched_face_indices:
            (x, y, w, h) = faces[i]['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 192, 0), 2)

        status = f"Monitoring {len(self.registered_people)} person(s)."
        return frame, status, "\n".join(list_display_text)

    def on_closing(self):
        """Handles the window close event."""
        if messagebox.askokcancel("Quit", "Do you want to exit the application?"):
            self.stop_video_stream()
            self.window.destroy()

# --- Main execution ---
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SleepingAlertApp(root)
        root.mainloop()
    except ImportError:
        print("\n--- ERROR ---")
        print("Could not import one or more required libraries.")
        print("Please ensure you have the following installed:")
        print("pip install opencv-python pillow numpy mediapipe")
        print("---------------")
    except Exception as e:
        print(f"An error occurred: {e}")