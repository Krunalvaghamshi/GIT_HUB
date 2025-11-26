import cv2
import mediapipe as mp
import csv
import time
import tkinter as tk
from tkinter import font, simpledialog, messagebox, filedialog
from PIL import Image, ImageTk
import os
import glob
import face_recognition
import numpy as np
import threading
import platform
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from collections import deque, Counter
import json
import gc
import psutil

# --- Sound Imports ---
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("WARNING: Pygame not installed. Sound may rely on system beeps.")

try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# --- GLOBAL SOUND PATHS ---
# Verified raw strings for Windows paths
ALERT_SOUND_PATH = r"D:\GIT_HUB\12_Final_Projects_of_all\03_deep_learning\experimental alert system\emergency-siren-351963.mp3"
FUGITIVE_SOUND_PATH = r"D:\GIT_HUB\12_Final_Projects_of_all\03_deep_learning\experimental alert system\Fugitive.mp3"

# --- 1. Configuration Loading ---
def load_config():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Config load error: {e}. Using defaults.")
        return {
            "detection": {"min_detection_confidence": 0.5, "min_tracking_confidence": 0.5, 
                         "face_recognition_tolerance": 0.5, "re_detect_interval": 60},
            "alert": {"default_interval_seconds": 10, "alert_cooldown_seconds": 2.5},
            "performance": {"gui_refresh_ms": 30, "pose_buffer_size": 12, "frame_skip_interval": 2, 
                           "min_buffer_for_classification": 5, "enable_frame_skipping": False},
            "logging": {"log_directory": "logs", "max_log_size_mb": 10, "auto_flush_interval": 50},
            "storage": {"alert_snapshots_dir": "alert_snapshots", "snapshot_retention_days": 30,
                       "guard_profiles_dir": "guard_profiles", "capture_snapshots_dir": "capture_snapshots"},
            "monitoring": {"mode": "pose", "session_restart_prompt_hours": 8},
            "sleeping": {
                "min_ear_threshold_floor": 0.18,
                "calibration_alpha": 0.05
            }
        }

CONFIG = load_config()

# --- 2. Logging Setup ---
if not os.path.exists(CONFIG["logging"]["log_directory"]):
    os.makedirs(CONFIG["logging"]["log_directory"])

logger = logging.getLogger("PoseGuard")
logger.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

file_handler = RotatingFileHandler(
    os.path.join(CONFIG["logging"]["log_directory"], "session.log"),
    maxBytes=CONFIG["logging"]["max_log_size_mb"] * 1024 * 1024,
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- 3. File Storage Utilities ---
def get_storage_paths():
    paths = {
        "guard_profiles": CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles"),
        "pose_references": CONFIG.get("storage", {}).get("pose_references_dir", "pose_references"),
        "capture_snapshots": CONFIG.get("storage", {}).get("capture_snapshots_dir", "capture_snapshots"),
        "logs": CONFIG["logging"]["log_directory"]
    }
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
    return paths

def save_guard_face(face_image, guard_name):
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    profile_path = os.path.join(paths["guard_profiles"], f"target_{safe_name}_face.jpg")
    cv2.imwrite(profile_path, face_image)
    return profile_path

def save_capture_snapshot(face_image, guard_name):
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(paths["capture_snapshots"], f"{safe_name}_capture_{timestamp}.jpg")
    cv2.imwrite(snapshot_path, face_image)
    return snapshot_path

def save_pose_landmarks_json(guard_name, poses_dict):
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    pose_path = os.path.join(paths["pose_references"], f"{safe_name}_poses.json")
    with open(pose_path, 'w') as f:
        json.dump(poses_dict, f, indent=2)
    return pose_path

def load_pose_landmarks_json(guard_name):
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    pose_path = os.path.join(paths["pose_references"], f"{safe_name}_poses.json")
    if os.path.exists(pose_path):
        with open(pose_path, 'r') as f:
            return json.load(f)
    return {}

# Directory Initialization
if not os.path.exists(CONFIG["storage"]["alert_snapshots_dir"]):
    os.makedirs(CONFIG["storage"]["alert_snapshots_dir"])
get_storage_paths()

csv_file = os.path.join(CONFIG["logging"]["log_directory"], "events.csv")
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name", "Action", "Status", "Image_Path", "Confidence"])

# --- 4. Cleanup Old Snapshots ---
def cleanup_old_snapshots():
    try:
        retention_days = CONFIG["storage"]["snapshot_retention_days"]
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        snapshot_dir = CONFIG["storage"]["alert_snapshots_dir"]
        for filename in os.listdir(snapshot_dir):
            filepath = os.path.join(snapshot_dir, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_time < cutoff_time:
                    os.remove(filepath)
    except Exception as e:
        logger.error(f"Snapshot cleanup error: {e}")

threading.Thread(target=cleanup_old_snapshots, daemon=True).start()

# --- MediaPipe Solutions Setup ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Sound Logic (Rewritten for Reliability) ---
def play_siren_sound(stop_event=None, duration_seconds=30, sound_file_path=None):
    """Play alert sound with verbose debugging and reliable fallbacks"""
    def _sound_worker():
        # 1. Validation
        if not sound_file_path:
            print("SOUND ERROR: No file path provided.")
        elif not os.path.exists(sound_file_path):
            print(f"SOUND ERROR: File not found at {sound_file_path}")
        else:
            # 2. Try Pygame (Best for MP3 on Windows)
            if PYGAME_AVAILABLE:
                try:
                    print(f"SOUND: Attempting to play {os.path.basename(sound_file_path)} via Pygame...")
                    if pygame.mixer.get_init():
                        pygame.mixer.quit() # Reset mixer to clear errors
                    
                    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                    pygame.mixer.music.load(sound_file_path)
                    pygame.mixer.music.set_volume(1.0)
                    pygame.mixer.music.play(-1) # Loop
                    
                    start_t = time.time()
                    while True:
                        if stop_event and stop_event.is_set(): 
                            print("SOUND: Stop event received.")
                            break
                        if time.time() - start_t > duration_seconds: 
                            print("SOUND: Duration expired.")
                            break
                        if not pygame.mixer.music.get_busy():
                            # If music stopped unexpectedly
                            print("SOUND: Playback stopped unexpectedly.")
                            break
                        time.sleep(0.1)
                    
                    pygame.mixer.music.stop()
                    return # Success
                except Exception as e: 
                    print(f"SOUND ERROR (Pygame): {e}")
        
        # 3. Fallback: Windows System Beep (Reliable Panic Mode)
        print("SOUND: Falling back to System Beep.")
        if platform.system() == "Windows":
            import winsound
            start = time.time()
            while True:
                if stop_event and stop_event.is_set(): break
                if time.time() - start > duration_seconds: break
                try:
                    # Siren pattern
                    winsound.Beep(2000, 300)
                    winsound.Beep(1500, 300)
                except: break
        else:
            # Linux/Mac Beep
            start = time.time()
            while True:
                if stop_event and stop_event.is_set(): break
                if time.time() - start > duration_seconds: break
                print('\a')
                time.sleep(0.5)

    t = threading.Thread(target=_sound_worker, daemon=True)
    t.start()
    return t

# --- EAR Calculation Helper ---
def calculate_ear(landmarks, width, height):
    """Calculates Eye Aspect Ratio (EAR)."""
    RIGHT_EYE = [33, 133, 159, 145, 158, 153]
    LEFT_EYE = [362, 263, 386, 374, 385, 380]

    def get_eye_ear(indices):
        p1 = np.array([landmarks[indices[0]].x * width, landmarks[indices[0]].y * height])
        p2 = np.array([landmarks[indices[1]].x * width, landmarks[indices[1]].y * height])
        p3 = np.array([landmarks[indices[2]].x * width, landmarks[indices[2]].y * height])
        p4 = np.array([landmarks[indices[3]].x * width, landmarks[indices[3]].y * height])
        p5 = np.array([landmarks[indices[4]].x * width, landmarks[indices[4]].y * height])
        p6 = np.array([landmarks[indices[5]].x * width, landmarks[indices[5]].y * height])

        v1 = np.linalg.norm(p3 - p4)
        v2 = np.linalg.norm(p5 - p6)
        h1 = np.linalg.norm(p1 - p2)

        if h1 == 0: return 0.0
        return (v1 + v2) / (2.0 * h1)

    ear_right = get_eye_ear(RIGHT_EYE)
    ear_left = get_eye_ear(LEFT_EYE)
    return (ear_right + ear_left) / 2.0

# --- Styled Drawing Helper ---
def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                 mp_drawing.DrawingSpec(color=(80,255,121), thickness=1, circle_radius=1)) 
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

# --- Action Classification ---
def classify_action(landmarks, h, w):
    try:
        NOSE = mp_holistic.PoseLandmark.NOSE.value
        L_WRIST = mp_holistic.PoseLandmark.LEFT_WRIST.value
        R_WRIST = mp_holistic.PoseLandmark.RIGHT_WRIST.value
        L_ELBOW = mp_holistic.PoseLandmark.LEFT_ELBOW.value
        R_ELBOW = mp_holistic.PoseLandmark.RIGHT_ELBOW.value
        L_SHOULDER = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        R_SHOULDER = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        L_HIP = mp_holistic.PoseLandmark.LEFT_HIP.value
        R_HIP = mp_holistic.PoseLandmark.RIGHT_HIP.value
        L_KNEE = mp_holistic.PoseLandmark.LEFT_KNEE.value
        R_KNEE = mp_holistic.PoseLandmark.RIGHT_KNEE.value

        nose = landmarks[NOSE]; l_wrist = landmarks[L_WRIST]; r_wrist = landmarks[R_WRIST]
        l_elbow = landmarks[L_ELBOW]; r_elbow = landmarks[R_ELBOW]
        l_shoulder = landmarks[L_SHOULDER]; r_shoulder = landmarks[R_SHOULDER]
        l_hip = landmarks[L_HIP]; r_hip = landmarks[R_HIP]
        l_knee = landmarks[L_KNEE]; r_knee = landmarks[R_KNEE]

        nose_y = nose.y * h
        lw_y = l_wrist.y * h; rw_y = r_wrist.y * h
        lw_x = l_wrist.x * w; rw_x = r_wrist.x * w
        ls_y = l_shoulder.y * h; rs_y = r_shoulder.y * h
        ls_x = l_shoulder.x * w; rs_x = r_shoulder.x * w
        
        l_wrist_visible = l_wrist.visibility > 0.6
        r_wrist_visible = r_wrist.visibility > 0.6
        l_knee_visible = l_knee.visibility > 0.6
        r_knee_visible = r_knee.visibility > 0.6

        # 1. Hands Up
        if (l_wrist_visible and r_wrist_visible and lw_y < (nose_y - 0.1 * h) and rw_y < (nose_y - 0.1 * h)):
            return "Hands Up"
        # 2. Hands Crossed
        if (l_wrist_visible and r_wrist_visible):
            chest_y = (ls_y + rs_y) / 2
            if (abs(lw_y - chest_y) < 0.2 * h and abs(rw_y - chest_y) < 0.2 * h):
                if ((lw_x > (ls_x + rs_x)/2 and rw_x < (ls_x + rs_x)/2) or (lw_x < (ls_x + rs_x)/2 and rw_x > (ls_x + rs_x)/2)):
                    return "Hands Crossed"
        # 3. T-Pose
        if (l_wrist_visible and r_wrist_visible and l_elbow.visibility > 0.6 and r_elbow.visibility > 0.6):
            if (abs(lw_y - ls_y) < 0.15 * h and abs(rw_y - rs_y) < 0.15 * h):
                if (lw_x < (ls_x - 0.2 * w) and rw_x > (rs_x + 0.2 * w)):
                    return "T-Pose"
        # 4. One Hand Raised
        if l_wrist_visible and lw_y < (nose_y - 0.1 * h) and not r_wrist_visible:
            return "One Hand Raised (Left)"
        if r_wrist_visible and rw_y < (nose_y - 0.1 * h) and not l_wrist_visible:
            return "One Hand Raised (Right)"
        if l_wrist_visible and r_wrist_visible:
            chest_y = (ls_y + rs_y) / 2
            if lw_y < (nose_y - 0.1 * h) and rw_y > (chest_y + 0.15 * h): return "One Hand Raised (Left)"
            if rw_y < (nose_y - 0.1 * h) and lw_y > (chest_y + 0.15 * h): return "One Hand Raised (Right)"
        # 5. Sit/Stand
        if l_knee_visible and r_knee_visible:
            avg_thigh_angle = (abs(l_knee.y - l_hip.y) + abs(r_knee.y - r_hip.y)) / 2
            if avg_thigh_angle < 0.15: return "Sit"
            else: return "Standing"

        return "Standing" 
    except Exception:
        return "Unknown"

# --- Helper: Calculate Dynamic Body Box ---
def calculate_body_box(face_box, frame_h, frame_w, expansion_factor=3.0):
    x1, y1, x2, y2 = face_box
    face_w = x2 - x1
    face_h = y2 - y1
    face_cx = x1 + (face_w // 2)
    bx1 = max(0, int(face_cx - (face_w * expansion_factor)))
    bx2 = min(frame_w, int(face_cx + (face_w * expansion_factor)))
    by1 = max(0, int(y1 - (face_h * 0.5)))
    by2 = frame_h
    return (bx1, by1, bx2, by2)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def detect_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: available_cameras.append(i)
            cap.release()
    return available_cameras

# --- Tkinter Application Class ---
class PoseApp:
    def __init__(self, window_title="Pose Guard Pro"):
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1800x1000")
        self.root.configure(bg="black")
        try: self.root.state('zoomed')
        except: pass
        
        self.cap = None
        self.unprocessed_frame = None 
        self.is_running = False
        self.is_logging = False
        self.camera_index = 0
        
        self.is_alert_mode = False
        self.alert_interval = 10
        # New Sleep Alert Settings
        self.sleep_alert_delay_seconds = 1.5 # Default 1.5 seconds
        
        self.is_in_capture_mode = False
        self.frame_w = 640 
        self.frame_h = 480 

        self.target_map = {}
        self.targets_status = {} 
        self.re_detect_counter = 0    
        self.RE_DETECT_INTERVAL = CONFIG["detection"]["re_detect_interval"]
        self.temp_log = []
        self.temp_log_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 30 # Default guess
        self.last_process_frame = None
        self.last_action_cache = {}
        self.session_start_time = time.time()
        self.onboarding_mode = False
        self.onboarding_step = 0
        self.onboarding_name = None
        self.onboarding_poses = {}
        self.onboarding_detection_results = None
        self.onboarding_face_box = None
        
        self.fugitive_mode = False
        self.fugitive_image = None
        self.fugitive_face_encoding = None
        self.fugitive_name = "Unknown Fugitive"
        self.fugitive_detected_log_done = False
        self.last_fugitive_snapshot_time = 0
        self.fugitive_alert_sound_thread = None
        self.fugitive_alert_stop_event = None
        
        try:
            self.holistic = mp_holistic.Holistic(
                min_detection_confidence=CONFIG["detection"]["min_detection_confidence"],
                min_tracking_confidence=CONFIG["detection"]["min_tracking_confidence"],
                static_image_mode=False
            )
            logger.warning("System initialized")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Holistic Model: {e}")
            self.root.destroy()
            return

        self.frame_timestamp_ms = 0 

        # --- GUI Layout ---
        self.root.grid_rowconfigure(0, weight=10)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.red_zone = tk.Frame(self.root, bg="red", bd=2)
        self.red_zone.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)
        self.video_container = tk.Frame(self.red_zone, bg="black")
        self.video_container.pack(fill="both", expand=True, padx=0, pady=0)
        self.video_label = tk.Label(self.video_container, bg="black", text="Camera Feed Off", fg="white")
        self.video_label.pack(fill="both", expand=True)
        
        self.guard_preview_frame = tk.Frame(self.video_container, bg="darkgreen", bd=2, relief="raised")
        self.guard_preview_frame.place(in_=self.video_container, relx=0.02, rely=0.02, anchor="nw")
        tk.Label(self.guard_preview_frame, text="GUARD", bg="darkgreen", fg="white", font=("Arial", 8, "bold")).pack(fill="x")
        self.guard_preview_label = tk.Label(self.guard_preview_frame, bg="black", fg="white", text="No Guard Selected", font=("Arial", 8), width=20, height=10)
        self.guard_preview_label.pack(fill="both", expand=True, padx=2, pady=2)
        
        self.fugitive_preview_frame = tk.Frame(self.video_container, bg="darkred", bd=2, relief="raised")
        self.fugitive_preview_frame.place(in_=self.video_container, relx=0.98, rely=0.02, anchor="ne")
        tk.Label(self.fugitive_preview_frame, text="FUGITIVE", bg="darkred", fg="white", font=("Arial", 8, "bold")).pack(fill="x")
        self.fugitive_preview_label = tk.Label(self.fugitive_preview_frame, bg="black", fg="white", text="No Fugitive Selected", font=("Arial", 8), width=20, height=10)
        self.fugitive_preview_label.pack(fill="both", expand=True, padx=2, pady=2)

        self.bottom_container = tk.Frame(self.root, bg="black")
        self.bottom_container.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        self.bottom_container.grid_columnconfigure(0, weight=6) 
        self.bottom_container.grid_columnconfigure(1, weight=4) 
        self.bottom_container.grid_rowconfigure(0, weight=1)

        self.yellow_zone = tk.Frame(self.bottom_container, bg="gold", bd=4)
        self.yellow_zone.grid(row=0, column=0, sticky="nsew", padx=2)
        self.yellow_zone.grid_rowconfigure(1, weight=1)
        
        self.controls_frame = tk.Frame(self.yellow_zone, bg="gold")
        self.controls_frame.pack(side="top", fill="x", padx=5, pady=5)
        
        btn_font = font.Font(family='Helvetica', size=10, weight='bold')
        btn_font_small = font.Font(family='Helvetica', size=9, weight='bold')

        # Row 0
        self.btn_start = tk.Button(self.controls_frame, text="Start Camera", command=self.start_camera, font=btn_font, bg="#27ae60", fg="white", width=12)
        self.btn_start.grid(row=0, column=0, padx=3, pady=3)
        self.btn_stop = tk.Button(self.controls_frame, text="Stop Camera", command=self.stop_camera, font=btn_font, bg="#c0392b", fg="white", width=12, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=3, pady=3)
        self.btn_add_guard = tk.Button(self.controls_frame, text="Add Guard", command=self.add_guard_dialog, font=btn_font, bg="#8e44ad", fg="white", width=12, state="disabled")
        self.btn_add_guard.grid(row=0, column=2, padx=3, pady=3)
        self.btn_remove_guard = tk.Button(self.controls_frame, text="Remove Guard", command=self.remove_guard_dialog, font=btn_font, bg="#e74c3c", fg="white", width=12)
        self.btn_remove_guard.grid(row=0, column=3, padx=3, pady=3)
        self.btn_exit = tk.Button(self.controls_frame, text="Exit", command=self.graceful_exit, font=btn_font, bg="#34495e", fg="white", width=12)
        self.btn_exit.grid(row=0, column=4, padx=3, pady=3)

        # Row 1
        tk.Label(self.controls_frame, text="Action:", bg="gold", font=btn_font_small).grid(row=1, column=0, sticky="e", padx=3)
        self.required_action_var = tk.StringVar(self.root)
        self.required_action_var.set("Hands Up")
        self.action_dropdown = tk.OptionMenu(self.controls_frame, self.required_action_var, 
                                            "Hands Up", "Hands Crossed", "One Hand Raised (Left)", 
                                            "One Hand Raised (Right)", "T-Pose", "Sit", "Standing", command=self.on_action_change)
        self.action_dropdown.grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        
        self.btn_set_interval = tk.Button(self.controls_frame, text=f"Action Timer ({self.alert_interval}s)", command=self.set_alert_interval, font=btn_font_small, bg="#7f8c8d", fg="white", width=15)
        self.btn_set_interval.grid(row=1, column=2, padx=3, pady=3)

        # NEW: Sleep Timer Button
        self.btn_set_sleep = tk.Button(self.controls_frame, text=f"Sleep Timer ({self.sleep_alert_delay_seconds}s)", command=self.set_sleep_interval, font=btn_font_small, bg="#546e7a", fg="white", width=15)
        self.btn_set_sleep.grid(row=1, column=3, padx=3, pady=3)
        
        self.btn_toggle_alert = tk.Button(self.controls_frame, text="Start Alert Mode", command=self.toggle_alert_mode, font=btn_font, bg="#e67e22", fg="white", width=18, state="disabled")
        self.btn_toggle_alert.grid(row=1, column=4, columnspan=1, padx=3, pady=3)
        
        # Row 2: Monitoring Mode & Fugitive
        tk.Label(self.controls_frame, text="Monitoring Mode:", bg="gold", font=btn_font_small).grid(row=2, column=0, sticky="e", padx=3)
        self.monitor_mode_var = tk.StringVar(self.root)
        self.monitor_mode_var.set("All Alerts (Action + Sleep)")
        self.monitor_mode_dropdown = tk.OptionMenu(self.controls_frame, self.monitor_mode_var, 
                                                  "All Alerts (Action + Sleep)", 
                                                  "Action Alerts Only", 
                                                  "Sleeping Alerts Only")
        self.monitor_mode_dropdown.grid(row=2, column=1, sticky="ew", padx=3, pady=3)

        # Fugitive Button
        self.btn_fugitive = tk.Button(self.controls_frame, text="Enable Fugitive Mode", command=self.toggle_fugitive_mode, font=btn_font, bg="#8b0000", fg="white", width=20, state="disabled")
        self.btn_fugitive.grid(row=2, column=2, columnspan=3, padx=3, pady=3, sticky="ew")
        
        self.listbox_frame = tk.Frame(self.yellow_zone, bg="gold")
        self.listbox_frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        
        tk.Label(self.listbox_frame, text="Select Targets to Track (Multi-Select):", bg="gold", font=btn_font_small).pack(anchor="w")
        self.target_listbox = tk.Listbox(self.listbox_frame, selectmode=tk.MULTIPLE, height=8, font=('Helvetica', 10))
        self.target_listbox.pack(side="left", fill="both", expand=True)
        self.target_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        scrollbar = tk.Scrollbar(self.listbox_frame)
        scrollbar.pack(side="right", fill="y")
        self.target_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.target_listbox.yview)
        self.btn_apply_targets = tk.Button(self.listbox_frame, text="TRACK SELECTED", command=self.apply_target_selection, font=btn_font_small, bg="black", fg="gold")
        self.btn_apply_targets.pack(side="bottom", fill="x", pady=2)
        self.btn_refresh = tk.Button(self.listbox_frame, text="Refresh List", command=self.load_targets, font=btn_font_small, bg="#e67e22", fg="white")
        self.btn_refresh.pack(side="bottom", fill="x", pady=2)

        self.green_zone = tk.Frame(self.bottom_container, bg="#00FF00", bd=4)
        self.green_zone.grid(row=0, column=1, sticky="nsew", padx=2)
        self.green_zone.grid_rowconfigure(1, weight=1)
        self.green_zone.grid_columnconfigure(0, weight=1)
        
        preview_header = tk.Label(self.green_zone, text="Guard & Fugitive Previews", bg="#00AA00", fg="white", font=btn_font_small)
        preview_header.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        self.preview_container = tk.Frame(self.green_zone, bg="black")
        self.preview_container.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.green_zone.grid_rowconfigure(1, weight=1)
        self.preview_display = tk.Frame(self.preview_container, bg="black")
        self.preview_display.pack(fill="both", expand=True)

        self.btn_snap = tk.Button(self.controls_frame, text="Snap Photo", command=self.snap_photo, font=btn_font, bg="#d35400", fg="white")
        self.btn_cancel_capture = tk.Button(self.controls_frame, text="Cancel", command=self.exit_onboarding_mode, font=btn_font, bg="#7f8c8d", fg="white")
        
        self.status_label = tk.Label(self.controls_frame, text="FPS: 0 | MEM: 0 MB", bg="gold", font=('Helvetica', 9))
        self.status_label.grid(row=3, column=0, columnspan=5, sticky="w", padx=3)

        self.load_targets()
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        self.root.mainloop()
    
    def graceful_exit(self):
        try:
            if self.is_running or self.is_alert_mode:
                if not messagebox.askyesno("Confirm Exit", "Camera is running. Exit?"): return
            logger.warning("Initiating graceful shutdown...")
            if self.is_running:
                self.is_running = False
                if self.cap: self.cap.release()
            if self.is_logging: self.save_log_to_file()
            if hasattr(self, 'holistic'): self.holistic.close()
            gc.collect()
            self.root.quit()
            self.root.destroy()
        except:
            try: self.root.destroy()
            except: pass

    def add_guard_dialog(self):
        if not self.is_running:
            messagebox.showwarning("Camera Required", "Please start the camera first.")
            return
        choice = messagebox.askquestion("Add Guard", "Yes = Take Photo with Camera\nNo = Upload Existing Image", icon='question')
        if choice == 'yes': self.enter_onboarding_mode()
        else: self.upload_guard_image()
    
    def remove_guard_dialog(self):
        if not self.target_map:
            messagebox.showwarning("No Guards", "No guards available to remove.")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Guard")
        dialog.geometry("400x300")
        dialog.grab_set()
        tk.Label(dialog, text="Select guard to remove:", font=('Helvetica', 11, 'bold')).pack(pady=10)
        listbox_frame = tk.Frame(dialog)
        listbox_frame.pack(fill="both", expand=True, padx=20, pady=10)
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side="right", fill="y")
        guard_listbox = tk.Listbox(listbox_frame, font=('Helvetica', 10), yscrollcommand=scrollbar.set)
        guard_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=guard_listbox.yview)
        for name in sorted(self.target_map.keys()): guard_listbox.insert(tk.END, name)
        
        def on_remove():
            selection = guard_listbox.curselection()
            if not selection: return
            guard_name = guard_listbox.get(selection[0])
            if messagebox.askyesno("Confirm", f"Remove '{guard_name}'?"):
                self.remove_guard(guard_name)
                dialog.destroy()
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Remove", command=on_remove, bg="#e74c3c", fg="white").pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#7f8c8d", fg="white").pack(side="left", padx=5)
    
    def remove_guard(self, guard_name):
        try:
            safe_name = guard_name.replace(" ", "_")
            files_to_remove = [
                f"target_{safe_name}_face.jpg",
                os.path.join(CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles"), f"target_{safe_name}_face.jpg"),
                os.path.join(CONFIG.get("storage", {}).get("pose_references_dir", "pose_references"), f"{safe_name}_poses.json")
            ]
            for f in files_to_remove:
                if os.path.exists(f): os.remove(f)
            if guard_name in self.targets_status: del self.targets_status[guard_name]
            self.load_targets()
            messagebox.showinfo("Success", f"Removed {guard_name}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def upload_guard_image(self):
        if not self.is_running: return
        filepath = filedialog.askopenfilename(title="Select Guard Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not filepath: return
        try:
            name = simpledialog.askstring("Guard Name", "Enter guard name:")
            if not name: return
            img = face_recognition.load_image_file(filepath)
            if len(face_recognition.face_locations(img)) != 1:
                messagebox.showerror("Error", "Image must contain exactly one face.")
                return
            save_guard_face(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), name)
            # Legacy copy for backward compat
            import shutil
            shutil.copy(filepath, f"target_{name.strip().replace(' ', '_')}_face.jpg")
            self.load_targets()
            messagebox.showinfo("Success", f"Guard '{name}' added!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def load_pose_references(self, guard_name):
        return load_pose_landmarks_json(guard_name)
    
    def save_pose_references(self, guard_name, poses_data):
        save_pose_landmarks_json(guard_name, poses_data)

    def load_targets(self):
        self.target_map = {}
        target_files = glob.glob("target_*.jpg")
        gp_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles")
        if os.path.exists(gp_dir): target_files.extend(glob.glob(os.path.join(gp_dir, "target_*.jpg")))
        
        display_names = []
        for f in target_files:
            try:
                base = os.path.basename(f).replace(".jpg", "")
                parts = base.split('_')
                if len(parts) >= 3 and parts[-1] == "face":
                    name = " ".join(parts[1:-1])
                    self.target_map[name] = f
                    display_names.append(name)
            except: pass

        self.target_listbox.delete(0, tk.END)
        if not display_names:
             self.target_listbox.insert(tk.END, "No targets found")
             self.target_listbox.config(state=tk.DISABLED)
        else:
             self.target_listbox.config(state=tk.NORMAL)
             for name in sorted(list(set(display_names))):
                 self.target_listbox.insert(tk.END, name)

    def on_listbox_select(self, event):
        for widget in self.preview_display.winfo_children(): widget.destroy()
        selections = self.target_listbox.curselection()
        if not selections:
            self.guard_preview_label.config(image='', text="No Guard Selected")
            return
        
        first_name = self.target_listbox.get(selections[0])
        if first_name in self.target_map:
            try:
                img = cv2.imread(self.target_map[first_name])
                h, w = img.shape[:2]
                scale = min(150/w, 150/h)
                img = cv2.resize(img, (int(w*scale), int(h*scale)))
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                self.guard_preview_label.config(image=imgtk, text=first_name)
                self.guard_preview_label.image = imgtk
            except: pass
        
        MAX_PREVIEW = 4
        display_idx = selections[:MAX_PREVIEW]
        cols = 1 if len(display_idx) == 1 else 2
        for i, idx in enumerate(display_idx):
            name = self.target_listbox.get(idx)
            filename = self.target_map.get(name)
            if filename:
                try:
                    img = cv2.imread(filename)
                    img = cv2.resize(img, (100, 100))
                    imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    lbl = tk.Label(self.preview_display, image=imgtk, bg="black", text=name, compound="bottom", fg="white")
                    lbl.image = imgtk 
                    lbl.grid(row=i//cols, column=i%cols, padx=5, pady=5)
                except: pass

    def apply_target_selection(self):
        self.targets_status = {} 
        selections = self.target_listbox.curselection()
        if not selections:
            messagebox.showwarning("Selection", "No targets selected.")
            return
        count = 0
        for idx in selections:
            name = self.target_listbox.get(idx)
            filename = self.target_map.get(name)
            if filename:
                try:
                    img = face_recognition.load_image_file(filename)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        self.targets_status[name] = {
                            "encoding": encs[0],
                            "tracker": None,
                            "face_box": None, 
                            "visible": False,
                            "last_action_time": time.time(),
                            "alert_cooldown": 0,
                            "alert_triggered_state": False,
                            "last_logged_action": None,
                            "pose_buffer": deque(maxlen=CONFIG["performance"]["pose_buffer_size"]),
                            "missing_pose_counter": 0,
                            "face_confidence": 0.0,
                            "pose_references": self.load_pose_references(name),
                            "last_snapshot_time": 0,
                            "last_log_time": 0,
                            "alert_sound_thread": None,
                            "alert_stop_event": None,
                            "alert_logged_timeout": False,
                            # --- SLEEPING ALERT STATE ---
                            "eye_counter_closed": 0,
                            "ear_threshold": 0.22, # Start safe
                            "open_ear_baseline": 0.30,
                            "is_sleeping": False
                        }
                        count += 1
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
        if count > 0:
            logger.warning(f"Tracking initialized for {count} targets.")
            messagebox.showinfo("Tracking Updated", f"Now scanning for {count} selected targets.")

    def toggle_alert_mode(self):
        self.is_alert_mode = not self.is_alert_mode
        if self.is_alert_mode:
            self.btn_toggle_alert.config(text="Stop Alert Mode", bg="#c0392b")
            if not self.is_logging:
                self.is_logging = True
                self.temp_log.clear()
                self.temp_log_counter = 0
            for name in self.targets_status:
                self.targets_status[name]["last_action_time"] = time.time()
                self.targets_status[name]["alert_triggered_state"] = False
        else:
            self.btn_toggle_alert.config(text="Start Alert Mode", bg="#e67e22")
            if self.is_logging:
                self.save_log_to_file()
                self.is_logging = False

    def set_alert_interval(self):
        val = simpledialog.askinteger("Set Interval", "Enter seconds for Action Alert:", minvalue=1, maxvalue=3600, initialvalue=self.alert_interval)
        if val:
            self.alert_interval = val
            self.btn_set_interval.config(text=f"Action Timer ({self.alert_interval}s)")
    
    def set_sleep_interval(self):
        # User enters SECONDS, we convert to frames roughly in the loop
        val = simpledialog.askfloat("Set Sleep Timer", "Enter seconds eyes must be closed (e.g. 1.5):", minvalue=0.1, maxvalue=10.0, initialvalue=self.sleep_alert_delay_seconds)
        if val:
            self.sleep_alert_delay_seconds = val
            self.btn_set_sleep.config(text=f"Sleep Timer ({self.sleep_alert_delay_seconds}s)")

    def on_action_change(self, value):
        if self.is_alert_mode:
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_action_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False

    def toggle_fugitive_mode(self):
        if not self.fugitive_mode:
            file_path = filedialog.askopenfilename(title="Select Fugitive Image")
            if not file_path: return
            try:
                self.fugitive_image = cv2.imread(file_path)
                rgb = cv2.cvtColor(self.fugitive_image, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb)
                if not locs: raise Exception("No face found")
                self.fugitive_face_encoding = face_recognition.face_encodings(rgb, locs)[0]
                self.fugitive_name = simpledialog.askstring("Fugitive Name", "Enter fugitive name:") or "Unknown"
                self.fugitive_mode = True
                self.fugitive_detected_log_done = False
                self.btn_fugitive.config(text="Disable Fugitive Mode", bg="#ff6b6b")
                self._update_fugitive_preview()
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            self.fugitive_mode = False
            self.fugitive_image = None
            self.btn_fugitive.config(text="Enable Fugitive Mode", bg="#8b0000")
            self.fugitive_preview_label.config(image='', text="No Fugitive Selected")

    def _update_fugitive_preview(self):
        if self.fugitive_image is None: return
        try:
            rgb = cv2.cvtColor(self.fugitive_image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(150/w, 150/h)
            rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.fugitive_preview_label.config(image=imgtk, text='')
            self.fugitive_preview_label.image = imgtk
        except: pass

    def start_camera(self):
        if not self.is_running:
            cams = detect_available_cameras()
            if not cams:
                messagebox.showerror("Error", "No cameras detected!")
                return
            self.camera_index = cams[0] 
            # If multiples, could ask user here (omitted for brevity)
            
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened(): return
            
            self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.is_running = True
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.btn_add_guard.config(state="normal")
            self.btn_toggle_alert.config(state="normal")
            self.btn_fugitive.config(state="normal")
            self.update_video_feed()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap: self.cap.release(); self.cap = None
            if self.is_logging: self.save_log_to_file()
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.btn_add_guard.config(state="disabled")
            self.btn_fugitive.config(state="disabled")
            self.video_label.config(image='')

    def auto_flush_logs(self):
        if self.is_logging and len(self.temp_log) >= CONFIG["logging"]["auto_flush_interval"]:
            self.save_log_to_file()

    def save_log_to_file(self):
        if self.temp_log:
            try:
                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(self.temp_log)
                self.temp_log.clear()
                self.temp_log_counter = 0
            except: pass
            
    def capture_alert_snapshot(self, frame, target_name, check_rate_limit=False):
        current_time = time.time()
        if check_rate_limit and target_name in self.targets_status:
            if (current_time - self.targets_status[target_name].get("last_snapshot_time", 0)) < 60:
                return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = target_name.replace(" ", "_")
        filename = os.path.join(CONFIG["storage"]["alert_snapshots_dir"], f"alert_{safe_name}_{timestamp}.jpg")
        try:
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if target_name in self.targets_status:
                self.targets_status[target_name]["last_snapshot_time"] = current_time
            return filename
        except: return "Error"

    def enter_onboarding_mode(self):
        if not self.is_running: return
        self.onboarding_mode = True
        self.onboarding_step = 0
        self.onboarding_poses = {}
        self.is_in_capture_mode = True
        name = simpledialog.askstring("New Guard", "Enter guard name:")
        if not name:
            self.onboarding_mode = False; self.is_in_capture_mode = False
            return
        self.onboarding_name = name.strip()
        self.btn_start.grid_remove(); self.btn_stop.grid_remove(); self.btn_add_guard.grid_remove()
        self.btn_snap.grid(row=0, column=0); self.btn_cancel_capture.grid(row=0, column=1)
        messagebox.showinfo("Step 1", "Stand in front of camera. Click 'Snap Photo' when ready.")

    def exit_onboarding_mode(self):
        self.is_in_capture_mode = False
        self.onboarding_mode = False
        self.onboarding_step = 0
        self.btn_snap.grid_remove(); self.btn_cancel_capture.grid_remove()
        self.btn_start.grid(); self.btn_stop.grid(); self.btn_add_guard.grid()

    def snap_photo(self):
        if self.unprocessed_frame is None: return
        if self.onboarding_step == 0:
            if self.onboarding_face_box is None:
                messagebox.showwarning("Error", "No face detected.")
                return
            top, right, bottom, left = self.onboarding_face_box
            h, w = self.unprocessed_frame.shape[:2]
            # Crop logic
            crop_top = max(0, top - int((bottom-top)*0.3))
            crop_bottom = min(h, bottom + int((bottom-top)*0.5))
            crop_left = max(0, left - int((right-left)*0.3))
            crop_right = min(w, right + int((right-left)*0.3))
            cropped = self.unprocessed_frame[crop_top:crop_bottom, crop_left:crop_right]
            
            save_guard_face(cropped, self.onboarding_name)
            save_capture_snapshot(cropped, self.onboarding_name)
            
            # Legacy
            cv2.imwrite(f"target_{self.onboarding_name.replace(' ', '_')}_face.jpg", cropped)
            
            self.onboarding_step = 1
            messagebox.showinfo("Step 2", "Perform: ONE HAND RAISED LEFT and click Snap")
        else:
            pose_actions = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing"]
            action = pose_actions[self.onboarding_step - 1]
            if not self.onboarding_detection_results or not self.onboarding_detection_results.pose_landmarks:
                messagebox.showwarning("Error", "No pose detected.")
                return
            
            current = classify_action(self.onboarding_detection_results.pose_landmarks.landmark, self.frame_h, self.frame_w)
            if current != action:
                messagebox.showwarning("Mismatch", f"Perform {action}. Detected: {current}")
                return
                
            landmarks = [{"x": l.x, "y": l.y, "z": l.z, "visibility": l.visibility} for l in self.onboarding_detection_results.pose_landmarks.landmark]
            self.onboarding_poses[action] = landmarks
            self.onboarding_step += 1
            if self.onboarding_step <= 4:
                messagebox.showinfo("Next", f"Perform: {pose_actions[self.onboarding_step - 1]} and snap.")
            else:
                save_pose_landmarks_json(self.onboarding_name, self.onboarding_poses)
                self.load_targets()
                self.exit_onboarding_mode()
                messagebox.showinfo("Done", "Onboarding complete!")

    def update_video_feed(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if not ret: 
            self.stop_camera()
            return
        self.unprocessed_frame = frame.copy()
        
        # Frame skipping logic
        self.frame_counter += 1
        if self.is_in_capture_mode:
            self.process_capture_frame(frame)
        else:
            if CONFIG["performance"]["enable_frame_skipping"] and self.frame_counter % CONFIG["performance"]["frame_skip_interval"] != 0:
                if self.last_process_frame is not None: frame = self.last_process_frame.copy()
            else:
                self.process_tracking_frame_optimized(frame)
                self.last_process_frame = frame.copy()

        if self.frame_counter % 30 == 0:
            fps = 30 / (time.time() - self.last_fps_time)
            self.current_fps = max(1, fps) # Prevent div by zero
            self.last_fps_time = time.time()
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            self.status_label.config(text=f"FPS: {fps:.1f} | MEM: {mem:.0f} MB")

        self.auto_flush_logs()
        
        if self.video_label.winfo_exists():
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Resize logic
            w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if w > 10:
                img_w, img_h = img.size
                scale = min(w/img_w, h/img_h)
                img = img.resize((int(img_w*scale), int(img_h*scale)))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        
        self.root.after(CONFIG["performance"]["gui_refresh_ms"], self.update_video_feed)

    def process_capture_frame(self, frame):
        # Onboarding logic (simplified for brevity, same as before)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        self.onboarding_detection_results = results
        
        if self.onboarding_step == 0:
            locs = face_recognition.face_locations(rgb)
            if len(locs) == 1:
                top, right, bottom, left = locs[0]
                self.onboarding_face_box = locs[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        
        cv2.putText(frame, f"Step {self.onboarding_step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # --- MAIN TRACKING LOGIC WITH PROFESSIONAL SLEEP DETECTION ---
    def process_tracking_frame_optimized(self, frame):
        if not self.targets_status:
            cv2.putText(frame, "SELECT TARGETS TO START", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        self.re_detect_counter += 1
        if self.re_detect_counter > self.RE_DETECT_INTERVAL:
            self.re_detect_counter = 0
        
        rgb_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # --- FUGITIVE LOGIC (Unchanged) ---
        if self.fugitive_mode and self.fugitive_face_encoding is not None:
            face_locations = face_recognition.face_locations(rgb_full_frame)
            if face_locations:
                encs = face_recognition.face_encodings(rgb_full_frame, face_locations)
                for enc, loc in zip(encs, face_locations):
                    match = face_recognition.compare_faces([self.fugitive_face_encoding], enc, tolerance=0.5)
                    if match[0]:
                        top, right, bottom, left = loc
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                        cv2.putText(frame, f"FUGITIVE: {self.fugitive_name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        if not self.fugitive_detected_log_done:
                            if self.fugitive_alert_stop_event is None: self.fugitive_alert_stop_event = threading.Event()
                            self.fugitive_alert_stop_event.clear()
                            self.fugitive_alert_sound_thread = play_siren_sound(self.fugitive_alert_stop_event, sound_file_path=FUGITIVE_SOUND_PATH)
                            self.capture_alert_snapshot(frame, f"FUGITIVE_{self.fugitive_name}")
                            self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), f"FUGITIVE_{self.fugitive_name}", "DETECTED", "ALERT", "N/A", "1.00"))
                            self.fugitive_detected_log_done = True
                    else:
                        self.fugitive_detected_log_done = False

        # 1. Update Trackers
        for name, status in self.targets_status.items():
            if status["tracker"]:
                success, box = status["tracker"].update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    status["face_box"] = (x, y, x + w, y + h)
                    status["visible"] = True
                else:
                    status["visible"] = False; status["tracker"] = None

        # 2. Detection
        untracked = [n for n, s in self.targets_status.items() if not s["visible"]]
        if untracked and self.re_detect_counter == 0:
            locs = face_recognition.face_locations(rgb_full_frame)
            if locs:
                encs = face_recognition.face_encodings(rgb_full_frame, locs)
                for i, enc in enumerate(encs):
                    for name in untracked:
                        dist = face_recognition.face_distance([self.targets_status[name]["encoding"]], enc)[0]
                        if dist < CONFIG["detection"]["face_recognition_tolerance"]:
                            top, right, bottom, left = locs[i]
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(frame, (left, top, right-left, bottom-top))
                            self.targets_status[name]["tracker"] = tracker
                            self.targets_status[name]["face_box"] = (left, top, right, bottom)
                            self.targets_status[name]["visible"] = True
                            self.targets_status[name]["face_confidence"] = 1.0 - dist

        # 3. Process Found Targets (Pose + Professional Sleeping Alert)
        required_act = self.required_action_var.get()
        monitor_mode = self.monitor_mode_var.get()
        current_time = time.time()

        # DYNAMIC SLEEP FRAMES CALCULATION
        # frames = seconds * fps
        required_sleep_frames = int(self.sleep_alert_delay_seconds * self.current_fps)

        for name, status in self.targets_status.items():
            if status["visible"]:
                fx1, fy1, fx2, fy2 = status["face_box"]
                bx1, by1, bx2, by2 = calculate_body_box((fx1, fy1, fx2, fy2), frame_h, frame_w)

                pose_found_in_box = False
                if bx1 < bx2 and by1 < by2:
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size != 0:
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        rgb_crop.flags.writeable = False
                        results_crop = self.holistic.process(rgb_crop)
                        rgb_crop.flags.writeable = True
                        
                        # --- SLEEPING ALERT LOGIC (IMPROVED) ---
                        is_sleeping_detected = False
                        
                        # Only process sleep if mode allows
                        if monitor_mode in ["All Alerts (Action + Sleep)", "Sleeping Alerts Only"]:
                            if results_crop.face_landmarks:
                                crop_h, crop_w = crop.shape[:2]
                                ear = calculate_ear(results_crop.face_landmarks.landmark, crop_w, crop_h)
                                
                                # LOGIC IMPROVEMENT: Hard Floor & Only adapt up
                                # 1. If EAR is very small (closed), count it
                                if ear < status["ear_threshold"]:
                                    status["eye_counter_closed"] += 1
                                else:
                                    status["eye_counter_closed"] = 0
                                    # 2. Only recalibrate baseline if eyes are WIDER open than current average (e.g. > 0.35)
                                    # This prevents the system from learning "closed eyes" as normal
                                    if ear > 0.35:
                                        status["open_ear_baseline"] = (status["open_ear_baseline"] * 0.95) + (ear * 0.05)
                                        
                                        # 3. Update threshold but NEVER drop below the hard floor (0.20 default)
                                        new_thresh = status["open_ear_baseline"] * 0.70 
                                        status["ear_threshold"] = max(0.20, new_thresh)
                                
                                # Check Alert Trigger
                                if status["eye_counter_closed"] > required_sleep_frames:
                                    is_sleeping_detected = True
                                    status["is_sleeping"] = True
                                    
                                    # VISUAL 1: Thick Red Border around the Person
                                    cv2.rectangle(frame, (fx1-15, fy1-15), (fx2+15, fy2+15), (0, 0, 255), 6)
                                    
                                    # VISUAL 2: Center Screen FLASHING Warning
                                    if int(time.time() * 4) % 2 == 0: # Flash effect
                                        cv2.putText(frame, "WAKE UP!", (frame_w//2 - 200, frame_h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

                                    # AUDIO
                                    if status["alert_stop_event"] is None: status["alert_stop_event"] = threading.Event()
                                    status["alert_stop_event"].clear()
                                    if not status.get("alert_sound_thread") or not status["alert_sound_thread"].is_alive():
                                        status["alert_sound_thread"] = play_siren_sound(status["alert_stop_event"], duration_seconds=10, sound_file_path=ALERT_SOUND_PATH)
                                    
                                    # LOGGING
                                    if self.is_logging and (current_time - status["last_log_time"] > 5):
                                        self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, "SLEEPING", "ALERT", "N/A", f"{ear:.2f}"))
                                        status["last_log_time"] = current_time
                                else:
                                    status["is_sleeping"] = False
                                    # Debug Display for EAR
                                    # cv2.putText(frame, f"EAR: {ear:.2f}", (fx1, fy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # --------------------------------------

                        current_action = "Unknown"
                        if results_crop.pose_landmarks:
                            pose_found_in_box = True
                            status["missing_pose_counter"] = 0
                            draw_styled_landmarks(crop, results_crop)
                            raw_action = classify_action(results_crop.pose_landmarks.landmark, (by2-by1), (bx2-bx1))
                            status["pose_buffer"].append(raw_action)
                            if len(status["pose_buffer"]) >= CONFIG["performance"]["min_buffer_for_classification"]:
                                current_action = Counter(status["pose_buffer"]).most_common(1)[0][0]
                            else: current_action = raw_action
                            self.last_action_cache[name] = current_action

                            # ACTION ALERT LOGIC
                            if monitor_mode in ["All Alerts (Action + Sleep)", "Action Alerts Only"]:
                                if current_action == required_act and not status["is_sleeping"]:
                                    if self.is_alert_mode:
                                        status["last_action_time"] = current_time
                                        status["alert_triggered_state"] = False
                                        if status["alert_stop_event"]: status["alert_stop_event"].set()
                                    if self.is_logging and status["last_logged_action"] != required_act:
                                        if (current_time - status["last_log_time"] > 60):
                                            self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, current_action, "Action Performed", "N/A", "1.0"))
                                            status["last_log_time"] = current_time
                                        status["last_logged_action"] = required_act
                            
                            # Draw Tracking Box
                            if not status["is_sleeping"]:
                                color = (0, 255, 0) 
                                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                                cv2.putText(frame, f"{name}: {current_action}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if not pose_found_in_box:
                    status["missing_pose_counter"] += 1
                if status["missing_pose_counter"] > 5:
                    status["tracker"] = None; status["visible"] = False

            # Alert Logic (Time-based Pose Timeout) - Only if mode enables it
            if self.is_alert_mode and monitor_mode in ["All Alerts (Action + Sleep)", "Action Alerts Only"]:
                time_diff = current_time - status["last_action_time"]
                time_left = max(0, self.alert_interval - time_diff)
                y_offset = 50 + (list(self.targets_status.keys()).index(name) * 30)
                color = (0, 255, 0) if time_left > 3 else (0, 0, 255)
                status_txt = "OK" if status["visible"] else "MISSING"
                cv2.putText(frame, f"{name} ({status_txt}): {time_left:.1f}s", (frame_w - 300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if time_diff > self.alert_interval:
                    if (current_time - status["alert_cooldown"]) > 2.5:
                        if status["alert_stop_event"] is None: status["alert_stop_event"] = threading.Event()
                        status["alert_stop_event"].clear()
                        if not status.get("alert_sound_thread") or not status["alert_sound_thread"].is_alive():
                            status["alert_sound_thread"] = play_siren_sound(status["alert_stop_event"], sound_file_path=ALERT_SOUND_PATH)
                        status["alert_cooldown"] = current_time
                        if self.is_logging:
                            log_s = "ALERT TRIGGERED" if status["visible"] else "MISSING"
                            self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, "TIMEOUT", log_s, "N/A", "0.0"))
                            self.temp_log_counter += 1

        return frame 

if __name__ == "__main__":
    app = PoseApp()