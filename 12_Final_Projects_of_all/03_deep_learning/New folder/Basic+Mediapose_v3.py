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
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

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
            "performance": {"gui_refresh_ms": 30, "pose_buffer_size": 12, "frame_skip_interval": 2,"enable_frame_skipping": True,"min_buffer_for_classification": 5},
            "logging": {"log_directory": "logs", "max_log_size_mb": 10, "auto_flush_interval": 50},
            "storage": {"alert_snapshots_dir": "alert_snapshots", "snapshot_retention_days": 30,
                       "guard_profiles_dir": "guard_profiles", "capture_snapshots_dir": "capture_snapshots"},
            "monitoring": {"mode": "pose", "session_restart_prompt_hours": 8}
        }

CONFIG = load_config()

# --- 2. Logging Setup with Rotation ---

if not os.path.exists(CONFIG["logging"]["log_directory"]):
    os.makedirs(CONFIG["logging"]["log_directory"])

logger = logging.getLogger("PoseGuard")
logger.setLevel(logging.WARNING)  # Only log warnings and errors by default

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Rotating file handler
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

# --- 3. File Storage Utilities (Systematic Organization) ---
def get_storage_paths():
    """
    Get all organized storage directory paths.
    Structure:
    - guard_profiles/: Face images for recognition
    - pose_references/: Pose landmark JSON files
    - capture_snapshots/: Timestamped captures
    - logs/: CSV events and session logs
    """
    paths = {
        "guard_profiles": CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles"),
        "pose_references": CONFIG.get("storage", {}).get("pose_references_dir", "pose_references"),
        "capture_snapshots": CONFIG.get("storage", {}).get("capture_snapshots_dir", "capture_snapshots"),
        "logs": CONFIG["logging"]["log_directory"]
    }
    
    # Create all directories
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
    
    return paths

def save_guard_face(face_image, guard_name):
    """Save guard face image to guard_profiles directory."""
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    profile_path = os.path.join(paths["guard_profiles"], f"target_{safe_name}_face.jpg")
    cv2.imwrite(profile_path, face_image)
    return profile_path

def save_capture_snapshot(face_image, guard_name):
    """Save timestamped capture snapshot to capture_snapshots directory."""
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(paths["capture_snapshots"], f"{safe_name}_capture_{timestamp}.jpg")
    cv2.imwrite(snapshot_path, face_image)
    return snapshot_path

def save_pose_landmarks_json(guard_name, poses_dict):
    """Save pose landmarks to pose_references directory as JSON."""
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    pose_path = os.path.join(paths["pose_references"], f"{safe_name}_poses.json")
    with open(pose_path, 'w') as f:
        json.dump(poses_dict, f, indent=2)
    return pose_path

def load_pose_landmarks_json(guard_name):
    """Load pose landmarks from pose_references directory."""
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    pose_path = os.path.join(paths["pose_references"], f"{safe_name}_poses.json")
    if os.path.exists(pose_path):
        with open(pose_path, 'r') as f:
            return json.load(f)
    return {}

# --- Directory Setup (using systematic functions) ---
if not os.path.exists(CONFIG["storage"]["alert_snapshots_dir"]):
    os.makedirs(CONFIG["storage"]["alert_snapshots_dir"])

if not os.path.exists(CONFIG.get("storage", {}).get("pose_references_dir", "pose_references")):
    os.makedirs(CONFIG.get("storage", {}).get("pose_references_dir", "pose_references"))

if not os.path.exists(CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles")):
    os.makedirs(CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles"))

if not os.path.exists(CONFIG.get("storage", {}).get("capture_snapshots_dir", "capture_snapshots")):
    os.makedirs(CONFIG.get("storage", {}).get("capture_snapshots_dir", "capture_snapshots"))

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

# --- Sound Logic ---
def play_siren_sound(stop_event=None, duration_seconds=30, sound_file="emergency-siren-351963.mp3"):
    """Play alert sound looping for up to duration_seconds or until stop_event is set
    
    Args:
        stop_event: threading.Event to signal stop playback
        duration_seconds: Maximum duration to play (default 30 seconds)
        sound_file: Name of audio file (default 'emergency-siren-351963.mp3' for action, 'Fugitive.mp3' for fugitive)
    """
    def _sound_worker():
        mp3_path = rf"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Web_Cam\{sound_file}"
        start_time = time.time()
        
        # Option 1: Try pygame (PRIMARY - most reliable for MP3 on Windows)
        if PYGAME_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                pygame.mixer.music.load(mp3_path)
                pygame.mixer.music.set_volume(1.0)
                
                # Play in loop until stop_event or duration_seconds
                pygame.mixer.music.play(-1)  # -1 means infinite loop
                logger.info(f"Alert sound started via pygame (max {duration_seconds}s)")
                
                # Wait until stop_event or duration expired
                while True:
                    elapsed = time.time() - start_time
                    
                    # Check if stop_event is set (action performed)
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Check if duration expired
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    time.sleep(0.1)
                
                pygame.mixer.music.stop()
                return
            except Exception as e:
                logger.warning(f"Pygame playback failed: {e}")
        
        # Option 2: Try pydub (requires ffmpeg/avconv)
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_mp3(mp3_path)
                logger.info(f"Alert sound started via pydub (max {duration_seconds}s)")
                
                while True:
                    elapsed = time.time() - start_time
                    
                    # Check if stop_event is set
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Check if duration expired
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Play audio clip
                    play(audio)
                
                logger.info("Alert sound via pydub completed")
                return
            except Exception as e:
                logger.warning(f"Pydub playback failed: {e}")
        
        # Fallback: Use system beeps (Windows winsound - always available)
        try:
            if platform.system() == "Windows":
                import winsound
                logger.info(f"Alert sound started via winsound (max {duration_seconds}s)")
                
                # Simulate emergency siren with pulsing high-low tones
                while True:
                    elapsed = time.time() - start_time
                    
                    # Check if stop_event is set
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Check if duration expired
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Play siren pattern
                    winsound.Beep(2500, 150)  # High beep
                    time.sleep(0.05)
                    winsound.Beep(1800, 150)  # Lower beep
                    time.sleep(0.05)
            else:
                # Unix/Linux fallback
                logger.info(f"Alert sound started via beep (max {duration_seconds}s)")
                while True:
                    elapsed = time.time() - start_time
                    
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    print('\a')
                    time.sleep(0.3)
        except Exception as e:
            logger.error(f"Sound Error: {e}")

    t = threading.Thread(target=_sound_worker, daemon=True)
    t.start()
    return t

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

# --- classify_action with improved detection ---
def classify_action(landmarks, h, w):
    """
    Classify pose action with robust detection and confidence scoring.
    Supports: Hands Up, Hands Crossed, One Hand Raised (Left/Right), T-Pose, Sit, Standing
    """
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
        L_ANKLE = mp_holistic.PoseLandmark.LEFT_ANKLE.value
        R_ANKLE = mp_holistic.PoseLandmark.RIGHT_ANKLE.value

        nose = landmarks[NOSE]
        l_wrist = landmarks[L_WRIST]
        r_wrist = landmarks[R_WRIST]
        l_elbow = landmarks[L_ELBOW]
        r_elbow = landmarks[R_ELBOW]
        l_shoulder = landmarks[L_SHOULDER]
        r_shoulder = landmarks[R_SHOULDER]
        l_hip = landmarks[L_HIP]
        r_hip = landmarks[R_HIP]
        l_knee = landmarks[L_KNEE]
        r_knee = landmarks[R_KNEE]
        l_ankle = landmarks[L_ANKLE]
        r_ankle = landmarks[R_ANKLE]

        nose_y = nose.y * h
        nose_x = nose.x * w
        lw_y = l_wrist.y * h
        rw_y = r_wrist.y * h
        lw_x = l_wrist.x * w
        rw_x = r_wrist.x * w
        ls_y = l_shoulder.y * h
        rs_y = r_shoulder.y * h
        ls_x = l_shoulder.x * w
        rs_x = r_shoulder.x * w
        lh_y = l_hip.y * h
        rh_y = r_hip.y * h
        
        # Calculate visibility thresholds
        l_wrist_visible = l_wrist.visibility > 0.6
        r_wrist_visible = r_wrist.visibility > 0.6
        l_elbow_visible = l_elbow.visibility > 0.6
        r_elbow_visible = r_elbow.visibility > 0.6
        nose_visible = nose.visibility > 0.5
        l_knee_visible = l_knee.visibility > 0.6
        r_knee_visible = r_knee.visibility > 0.6
        
        # 1. Hands Up Detection (both hands clearly above head)
        if (l_wrist_visible and r_wrist_visible and 
            lw_y < (nose_y - 0.1 * h) and rw_y < (nose_y - 0.1 * h)):
            return "Hands Up"
        
        # 2. Hands Crossed Detection (wrists cross at chest level)
        if (l_wrist_visible and r_wrist_visible):
            chest_y = (ls_y + rs_y) / 2
            body_center_x = (ls_x + rs_x) / 2
            # Check if both wrists are at chest level
            if (abs(lw_y - chest_y) < 0.2 * h and abs(rw_y - chest_y) < 0.2 * h):
                # Check if wrists are crossed (left hand on right side, vice versa)
                if ((lw_x > body_center_x and rw_x < body_center_x) or 
                    (lw_x < body_center_x and rw_x > body_center_x)):
                    return "Hands Crossed"
        
        # 3. T-Pose Detection (arms extended sideways at shoulder height)
        if (l_wrist_visible and r_wrist_visible and l_elbow_visible and r_elbow_visible):
            # Check if both elbows and wrists are at shoulder level
            if (abs(lw_y - ls_y) < 0.15 * h and abs(rw_y - rs_y) < 0.15 * h and
                abs(l_elbow.y * h - ls_y) < 0.15 * h and abs(r_elbow.y * h - rs_y) < 0.15 * h):
                # Check if arms are extended outward
                if (lw_x < (ls_x - 0.2 * w) and rw_x > (rs_x + 0.2 * w)):
                    return "T-Pose"
        
        # 4. One Hand Raised Detection (only one hand above head, clearly)
        if l_wrist_visible and lw_y < (nose_y - 0.1 * h) and not r_wrist_visible:
            return "One Hand Raised (Left)"
        if r_wrist_visible and rw_y < (nose_y - 0.1 * h) and not l_wrist_visible:
            return "One Hand Raised (Right)"
        
        # Alternative: one hand raised while other is down
        if l_wrist_visible and r_wrist_visible:
            chest_y = (ls_y + rs_y) / 2
            if lw_y < (nose_y - 0.1 * h) and rw_y > (chest_y + 0.15 * h):
                return "One Hand Raised (Left)"
            if rw_y < (nose_y - 0.1 * h) and lw_y > (chest_y + 0.15 * h):
                return "One Hand Raised (Right)"
        
        # 5. Sit/Stand Detection
        if l_knee_visible and r_knee_visible:
            # Calculate angle of thigh (knee to hip)
            thigh_angle_l = abs(l_knee.y - l_hip.y)
            thigh_angle_r = abs(r_knee.y - r_hip.y)
            avg_thigh_angle = (thigh_angle_l + thigh_angle_r) / 2
            
            # If thigh is nearly horizontal, person is sitting
            if avg_thigh_angle < 0.15:
                return "Sit"
            else:
                return "Standing"
        else:
            # Default to standing if knee not visible
            return "Standing"

        return "Standing" 

    except Exception as e:
        return "Unknown"

# --- Helper: Calculate Dynamic Body Box from Face ---
def calculate_body_box(face_box, frame_h, frame_w, expansion_factor=3.0):
    """
    Calculate dynamic body bounding box from detected face box.
    
    Args:
        face_box: tuple (x1, y1, x2, y2) - face coordinates
        frame_h, frame_w: frame dimensions
        expansion_factor: how many face widths to expand (default 3x)
    
    Returns:
        tuple (bx1, by1, bx2, by2) - body box coordinates
    """
    x1, y1, x2, y2 = face_box
    face_w = x2 - x1
    face_h = y2 - y1
    face_cx = x1 + (face_w // 2)
    
    # Expand horizontally based on face width
    bx1 = max(0, int(face_cx - (face_w * expansion_factor)))
    bx2 = min(frame_w, int(face_cx + (face_w * expansion_factor)))
    
    # Expand vertically: slightly above face, down to feet
    by1 = max(0, int(y1 - (face_h * 0.5)))
    by2 = frame_h
    
    return (bx1, by1, bx2, by2)

# --- Helper: IoU for Overlap Check ---
def calculate_iou(boxA, boxB):
    # box = (x, y, w, h) -> convert to (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

# --- Helper: Detect Available Cameras ---
def detect_available_cameras(max_cameras=10):
    """Detect all available camera indices"""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

# --- Tkinter Application Class ---
class PoseApp:
    def __init__(self, window_title="Pose Guard (Multi-Target)"):
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1800x1000")  # Larger default size
        self.root.configure(bg="black")
        try:
            self.root.state('zoomed')  # Start maximized (Windows only)
        except Exception:
            pass
        
        self.cap = None
        self.unprocessed_frame = None 
        self.is_running = False
        self.is_logging = False
        self.camera_index = 0  # Default camera
        
        self.is_alert_mode = False
        self.alert_interval = 10  
        self.is_in_capture_mode = False
        self.frame_w = 640 
        self.frame_h = 480 

        self.target_map = {}
        self.targets_status = {} 
        self.re_detect_counter = 0    
        self.RE_DETECT_INTERVAL = CONFIG["detection"]["re_detect_interval"]
        self.RESIZE_SCALE = 1.0 
        self.temp_log = []
        self.temp_log_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.last_process_frame = None
        self.last_action_cache = {}
        self.session_start_time = time.time()
        self.onboarding_mode = False
        self.onboarding_step = 0
        self.onboarding_name = None
        self.onboarding_poses = {}
        self.onboarding_detection_results = None  # Store detection results for capture
        self.onboarding_face_box = None  # Store face box for capture
        
        # Fugitive Mode Fields
        self.fugitive_mode = False
        self.fugitive_image = None
        self.fugitive_face_encoding = None
        self.fugitive_name = "Unknown Fugitive"
        self.fugitive_detected_log_done = False  # Prevent duplicate logs
        self.last_fugitive_snapshot_time = 0  # Rate limiting for snapshots
        self.fugitive_alert_sound_thread = None
        self.fugitive_alert_stop_event = None
        
        try:
            # Single Holistic instance for efficiency
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

        # --- Layout ---
        self.root.grid_rowconfigure(0, weight=10)  # Maximize top row (camera)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # 1. Red Zone
        self.red_zone = tk.Frame(self.root, bg="red", bd=2)
        self.red_zone.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)
        self.video_container = tk.Frame(self.red_zone, bg="black")
        self.video_container.pack(fill="both", expand=True, padx=0, pady=0)
        self.video_label = tk.Label(self.video_container, bg="black", text="Camera Feed Off", fg="white")
        self.video_label.pack(fill="both", expand=True)
        
        # Guard Preview (overlaid in top-left corner)
        self.guard_preview_frame = tk.Frame(self.video_container, bg="darkgreen", bd=2, relief="raised")
        self.guard_preview_frame.place(in_=self.video_container, relx=0.02, rely=0.02, anchor="nw")
        tk.Label(self.guard_preview_frame, text="GUARD", bg="darkgreen", fg="white", 
                font=("Arial", 8, "bold")).pack(fill="x", padx=2, pady=1)
        self.guard_preview_label = tk.Label(self.guard_preview_frame, bg="black", fg="white", 
                                            text="No Guard Selected", font=("Arial", 8), width=20, height=10)
        self.guard_preview_label.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Fugitive Preview (overlaid in top-right corner)
        self.fugitive_preview_frame = tk.Frame(self.video_container, bg="darkred", bd=2, relief="raised")
        self.fugitive_preview_frame.place(in_=self.video_container, relx=0.98, rely=0.02, anchor="ne")
        tk.Label(self.fugitive_preview_frame, text="FUGITIVE", bg="darkred", fg="white", 
                font=("Arial", 8, "bold")).pack(fill="x", padx=2, pady=1)
        self.fugitive_preview_label = tk.Label(self.fugitive_preview_frame, bg="black", fg="white", 
                                               text="No Fugitive Selected", font=("Arial", 8), width=20, height=10)
        self.fugitive_preview_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Bottom Container
        self.bottom_container = tk.Frame(self.root, bg="black")
        self.bottom_container.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        self.bottom_container.grid_columnconfigure(0, weight=6) 
        self.bottom_container.grid_columnconfigure(1, weight=4) 
        self.bottom_container.grid_rowconfigure(0, weight=1)

        # 2. Yellow Zone - Control Panel
        self.yellow_zone = tk.Frame(self.bottom_container, bg="gold", bd=4)
        self.yellow_zone.grid(row=0, column=0, sticky="nsew", padx=2)
        self.yellow_zone.grid_rowconfigure(1, weight=1)
        
        # Top control buttons
        self.controls_frame = tk.Frame(self.yellow_zone, bg="gold")
        self.controls_frame.pack(side="top", fill="x", padx=5, pady=5)
        
        # Widgets
        btn_font = font.Font(family='Helvetica', size=10, weight='bold')
        btn_font_small = font.Font(family='Helvetica', size=9, weight='bold')

        # Row 0: Start/Stop and Add/Remove buttons
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

        # Row 1: Action selection and interval
        tk.Label(self.controls_frame, text="Action:", bg="gold", font=btn_font_small).grid(row=1, column=0, sticky="e", padx=3)
        self.required_action_var = tk.StringVar(self.root)
        self.required_action_var.set("Hands Up")
        self.action_dropdown = tk.OptionMenu(self.controls_frame, self.required_action_var, 
                                            "Hands Up", "Hands Crossed", 
                                            "One Hand Raised (Left)", "One Hand Raised (Right)", 
                                            "T-Pose", "Sit", "Standing", command=self.on_action_change)
        self.action_dropdown.grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        self.btn_set_interval = tk.Button(self.controls_frame, text=f"Set Interval ({self.alert_interval}s)", command=self.set_alert_interval, font=btn_font_small, bg="#7f8c8d", fg="white", width=15)
        self.btn_set_interval.grid(row=1, column=2, padx=3, pady=3)
        self.btn_toggle_alert = tk.Button(self.controls_frame, text="Start Alert Mode", command=self.toggle_alert_mode, font=btn_font, bg="#e67e22", fg="white", width=18, state="disabled")
        self.btn_toggle_alert.grid(row=1, column=3, columnspan=2, padx=3, pady=3)
        
        # Row 2: Fugitive Mode Button (highlighted section)
        fugitive_frame = tk.Frame(self.controls_frame, bg="gold")
        fugitive_frame.grid(row=2, column=0, columnspan=5, sticky="ew", padx=3, pady=5)
        tk.Label(fugitive_frame, text="Fugitive Mode:", bg="gold", font=btn_font_small, fg="#8b0000").pack(side="left", padx=5)
        self.btn_fugitive = tk.Button(fugitive_frame, text="Enable Fugitive Mode", command=self.toggle_fugitive_mode, font=btn_font, bg="#8b0000", fg="white", width=25, state="disabled")
        self.btn_fugitive.pack(side="left", padx=5, fill="x", expand=True)
        
        # Target selection frame
        self.listbox_frame = tk.Frame(self.yellow_zone, bg="gold")
        self.listbox_frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        
        # Listbox widgets
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

        # 3. Green Zone - Preview Panel
        self.green_zone = tk.Frame(self.bottom_container, bg="#00FF00", bd=4)
        self.green_zone.grid(row=0, column=1, sticky="nsew", padx=2)
        self.green_zone.grid_rowconfigure(1, weight=1)
        self.green_zone.grid_columnconfigure(0, weight=1)
        
        # Preview header
        preview_header = tk.Label(self.green_zone, text="Guard & Fugitive Previews", bg="#00AA00", fg="white", font=btn_font_small)
        preview_header.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        self.preview_container = tk.Frame(self.green_zone, bg="black")
        self.preview_container.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.green_zone.grid_rowconfigure(1, weight=1)
        self.preview_display = tk.Frame(self.preview_container, bg="black")
        self.preview_display.pack(fill="both", expand=True)

        self.btn_snap = tk.Button(self.controls_frame, text="Snap Photo", command=self.snap_photo, font=btn_font, bg="#d35400", fg="white")
        self.btn_cancel_capture = tk.Button(self.controls_frame, text="Cancel", command=self.exit_onboarding_mode, font=btn_font, bg="#7f8c8d", fg="white")
        
        # FPS and Memory Display
        self.status_label = tk.Label(self.controls_frame, text="FPS: 0 | MEM: 0 MB", bg="gold", font=('Helvetica', 9))
        self.status_label.grid(row=2, column=0, columnspan=5, sticky="w", padx=3)

        self.load_targets()
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        
        self.root.mainloop()
    
    def graceful_exit(self):
        """Gracefully exit the application with proper cleanup"""
        try:
            # Confirm exit if camera is running or alert mode is active
            if self.is_running or self.is_alert_mode:
                response = messagebox.askyesno(
                    "Confirm Exit",
                    "Camera is running. Are you sure you want to exit?"
                )
                if not response:
                    return
            
            logger.warning("Initiating graceful shutdown...")
            
            # Stop camera if running
            if self.is_running:
                self.is_running = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
            
            # Save logs if logging
            if self.is_logging:
                self.save_log_to_file()
            
            # Cleanup trackers
            for status in self.targets_status.values():
                if status.get("tracker"):
                    status["tracker"] = None
            
            # Release holistic model
            if hasattr(self, 'holistic'):
                self.holistic.close()
            
            # Force garbage collection
            gc.collect()
            
            logger.warning("Shutdown complete")
            
            # Destroy window
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error during exit: {e}")
            # Force exit even if there's an error
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

    def add_guard_dialog(self):
        """Show dialog to choose between capturing or uploading guard image"""
        if not self.is_running:
            messagebox.showwarning("Camera Required", "Please start the camera first.")
            return
        
        # Create custom dialog
        choice = messagebox.askquestion(
            "Add Guard",
            "How would you like to add the guard?\n\nYes = Take Photo with Camera\nNo = Upload Existing Image",
            icon='question'
        )
        
        if choice == 'yes':
            self.enter_onboarding_mode()
        else:
            self.upload_guard_image()
    
    def remove_guard_dialog(self):
        """Show dialog to select and remove a guard"""
        if not self.target_map:
            messagebox.showwarning("No Guards", "No guards available to remove.")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Guard")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Select guard to remove:", 
                font=('Helvetica', 11, 'bold')).pack(pady=10)
        
        # Listbox for guard selection
        listbox_frame = tk.Frame(dialog)
        listbox_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side="right", fill="y")
        
        guard_listbox = tk.Listbox(listbox_frame, font=('Helvetica', 10), 
                                   yscrollcommand=scrollbar.set)
        guard_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=guard_listbox.yview)
        
        # Populate listbox
        guard_names = sorted(self.target_map.keys())
        for name in guard_names:
            guard_listbox.insert(tk.END, name)
        
        def on_remove():
            selection = guard_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a guard to remove.")
                return
            
            guard_name = guard_listbox.get(selection[0])
            
            # Confirm deletion
            response = messagebox.askyesno(
                "Confirm Removal",
                f"Are you sure you want to remove '{guard_name}'?\n\nThis will delete:\n" +
                "- Face image\n- Pose references\n- All associated data\n\nThis action cannot be undone!"
            )
            
            if response:
                self.remove_guard(guard_name)
                dialog.destroy()
        
        # Buttons
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Remove", command=on_remove, bg="#e74c3c", 
                 fg="white", font=('Helvetica', 10, 'bold'), width=12).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#7f8c8d", 
                 fg="white", font=('Helvetica', 10, 'bold'), width=12).pack(side="left", padx=5)
    
    def remove_guard(self, guard_name):
        """Remove guard profile and all associated data"""
        try:
            safe_name = guard_name.replace(" ", "_")
            guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles")
            pose_references_dir = CONFIG.get("storage", {}).get("pose_references_dir", "pose_references")
            
            deleted_items = []
            
            # Remove face image from root
            face_image = f"target_{safe_name}_face.jpg"
            if os.path.exists(face_image):
                os.remove(face_image)
                deleted_items.append("Face image (root)")
            
            # Remove face image from guard_profiles directory
            profile_image = os.path.join(guard_profiles_dir, f"target_{safe_name}_face.jpg")
            if os.path.exists(profile_image):
                os.remove(profile_image)
                deleted_items.append("Face image (profiles)")
            
            # Remove pose references
            pose_file = os.path.join(pose_references_dir, f"{safe_name}_poses.json")
            if os.path.exists(pose_file):
                os.remove(pose_file)
                deleted_items.append("Pose references")
            
            # Remove from tracking if currently tracked
            if guard_name in self.targets_status:
                if self.targets_status[guard_name].get("tracker"):
                    self.targets_status[guard_name]["tracker"] = None
                del self.targets_status[guard_name]
                deleted_items.append("Active tracking")
            
            # Reload targets list
            self.load_targets()
            
            logger.warning(f"Guard removed: {guard_name} ({', '.join(deleted_items)})")
            messagebox.showinfo(
                "Guard Removed",
                f"'{guard_name}' has been successfully removed.\n\nDeleted: {', '.join(deleted_items)}"
            )
            
        except Exception as e:
            logger.error(f"Error removing guard {guard_name}: {e}")
            messagebox.showerror("Error", f"Failed to remove guard: {e}")
    
    def upload_guard_image(self):
        """Upload an existing image for guard onboarding"""
        if not self.is_running: return
        
        filepath = filedialog.askopenfilename(
            title="Select Guard Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if not filepath: return
        
        try:
            name = simpledialog.askstring("Guard Name", "Enter guard name:")
            if not name: return
            
            safe_name = name.strip().replace(" ", "_")
            guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles")
            target_path = os.path.join(guard_profiles_dir, f"target_{safe_name}_face.jpg")
            
            # Load and verify face
            img = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(img)
            
            if len(face_locations) != 1:
                messagebox.showerror("Error", "Image must contain exactly one face.")
                return
            
            # Copy image
            import shutil
            shutil.copy(filepath, target_path)
            
            # Also copy to root for backward compatibility
            shutil.copy(filepath, f"target_{safe_name}_face.jpg")
            
            self.load_targets()
            messagebox.showinfo("Success", f"Guard '{name}' added successfully!\n(Pose capture skipped for uploaded images)")
            logger.warning(f"Guard added via upload: {name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload image: {e}")
            logger.error(f"Upload error: {e}")
    
    def load_pose_references(self, guard_name):
        """Load saved pose references for a guard"""
        try:
            pose_dir = CONFIG.get("storage", {}).get("pose_references_dir", "pose_references")
            safe_name = guard_name.replace(" ", "_")
            pose_file = os.path.join(pose_dir, f"{safe_name}_poses.json")
            
            if os.path.exists(pose_file):
                with open(pose_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load pose references for {guard_name}: {e}")
        return {}
    
    def save_pose_references(self, guard_name, poses_data):
        """Save pose references for a guard using systematic storage"""
        try:
            pose_file = save_pose_landmarks_json(guard_name, poses_data)
            logger.warning(f"Pose references saved for {guard_name} at {pose_file}")
        except Exception as e:
            logger.error(f"Failed to save pose references: {e}")

    def load_targets(self):
        self.target_map = {}
        # Search in both root and guard_profiles directory
        target_files = glob.glob("target_*.jpg")
        guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles")
        if os.path.exists(guard_profiles_dir):
            target_files.extend(glob.glob(os.path.join(guard_profiles_dir, "target_*.jpg")))
        display_names = []
        for f in target_files:
            try:
                # Parse filename: target_[Name]_face.jpg or target_[First_Last]_face.jpg
                base_name = os.path.basename(f).replace(".jpg", "")
                parts = base_name.split('_')
                
                # Remove 'target' prefix and 'face' suffix
                if len(parts) >= 3 and parts[-1] == "face":
                    # Join all parts between 'target' and 'face' as the name
                    display_name = " ".join(parts[1:-1])
                    self.target_map[display_name] = f
                    display_names.append(display_name)
            except Exception as e:
                logger.error(f"Error parsing {f}: {e}")

        self.target_listbox.delete(0, tk.END)
        if not display_names:
             self.target_listbox.insert(tk.END, "No targets found")
             self.target_listbox.config(state=tk.DISABLED)
             logger.warning("No target files found")
        else:
             self.target_listbox.config(state=tk.NORMAL)
             for name in sorted(list(set(display_names))):
                 self.target_listbox.insert(tk.END, name)
             logger.warning(f"Loaded {len(set(display_names))} guards")

    def on_listbox_select(self, event):
        for widget in self.preview_display.winfo_children():
            widget.destroy()
        selections = self.target_listbox.curselection()
        if not selections:
            tk.Label(self.preview_display, text="No Selection", bg="black", fg="white").pack(expand=True)
            # Clear guard preview
            self.guard_preview_label.config(image='', text="No Guard Selected")
            return
        
        # Show first selected guard in guard preview (top-left corner)
        first_name = self.target_listbox.get(selections[0])
        first_filename = self.target_map.get(first_name)
        if first_filename:
            try:
                img = cv2.imread(first_filename)
                h, w = img.shape[:2]
                scale = min(150 / w, 150 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=pil_img)
                self.guard_preview_label.config(image=imgtk, text=first_name)
                self.guard_preview_label.image = imgtk
            except Exception:
                self.guard_preview_label.config(text=f"Error: {first_name}")
        
        # Show all selected guards in preview area (bottom-right)
        MAX_PREVIEW = 4
        display_idx = selections[:MAX_PREVIEW]
        cols = 1 if len(display_idx) == 1 else 2
        for i, idx in enumerate(display_idx):
            name = self.target_listbox.get(idx)
            filename = self.target_map.get(name)
            if filename:
                try:
                    img = cv2.imread(filename)
                    target_h = 130 if len(display_idx) > 1 else 260
                    target_w = 180 if len(display_idx) > 1 else 360
                    h, w = img.shape[:2]
                    scale = min(target_w/w, target_h/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    img = cv2.resize(img, (new_w, new_h))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=pil_img)
                    lbl = tk.Label(self.preview_display, image=imgtk, bg="black", text=name, compound="bottom", fg="white", font=("Arial", 9, "bold"))
                    lbl.image = imgtk 
                    lbl.grid(row=i//cols, column=i%cols, padx=5, pady=5)
                except Exception: pass

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
                    target_image_file = face_recognition.load_image_file(filename)
                    encodings = face_recognition.face_encodings(target_image_file)
                    if encodings:
                        self.targets_status[name] = {
                            "encoding": encodings[0],
                            "tracker": None,
                            "face_box": None, 
                            "visible": False,
                            "last_action_time": time.time(),  # Renamed from last_wave_time
                            "alert_cooldown": 0,
                            "alert_triggered_state": False,
                            "last_logged_action": None,
                            "pose_buffer": deque(maxlen=CONFIG["performance"]["pose_buffer_size"]),
                            "missing_pose_counter": 0,
                            "face_confidence": 0.0,
                            "pose_references": self.load_pose_references(name),
                            "last_snapshot_time": 0,  # Rate limiting: one snapshot per minute
                            "last_log_time": 0,  # Rate limiting: one log entry per minute
                            "alert_sound_thread": None,  # Track current alert sound thread
                            "alert_stop_event": None,  # Event to signal sound to stop when action performed
                            "alert_logged_timeout": False  # Track if timeout alert was logged
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
            # Auto-start logging
            if not self.is_logging:
                self.is_logging = True
                self.temp_log.clear()
                self.temp_log_counter = 0
                logger.warning("Alert mode started - logging enabled")
            
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_action_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False
        else:
            self.btn_toggle_alert.config(text="Start Alert Mode", bg="#e67e22")
            # Auto-stop logging and save
            if self.is_logging:
                self.save_log_to_file()
                self.is_logging = False
                logger.warning("Alert mode stopped - logging saved")

    def set_alert_interval(self):
        val = simpledialog.askinteger("Set Interval", "Enter seconds:", minvalue=1, maxvalue=3600, initialvalue=self.alert_interval)
        if val:
            self.alert_interval = val
            self.btn_set_interval.config(text=f"Set Interval ({self.alert_interval}s)")
            
    def on_action_change(self, value):
        if self.is_alert_mode:
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_action_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False

    def toggle_fugitive_mode(self):
        """Toggle Fugitive Mode - Search for a specific person in live feed"""
        if not self.fugitive_mode:
            # Start Fugitive Mode
            file_path = filedialog.askopenfilename(
                title="Select Fugitive Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            try:
                # Load and display fugitive image
                self.fugitive_image = cv2.imread(file_path)
                if self.fugitive_image is None:
                    messagebox.showerror("Error", "Failed to load image")
                    return
                
                # Extract face encoding from fugitive image
                rgb_image = cv2.cvtColor(self.fugitive_image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                
                if not face_locations:
                    messagebox.showerror("Error", "No face detected in selected image")
                    return
                
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if not face_encodings:
                    messagebox.showerror("Error", "Failed to extract face encoding")
                    return
                
                self.fugitive_face_encoding = face_encodings[0]
                self.fugitive_name = simpledialog.askstring("Fugitive Name", "Enter fugitive name:") or "Unknown Fugitive"
                
                # Start Fugitive Mode
                self.fugitive_mode = True
                self.fugitive_detected_log_done = False
                self.btn_fugitive.config(text="Disable Fugitive Mode", bg="#ff6b6b")
                
                # Display fugitive image in preview
                self._update_fugitive_preview()
                
                logger.warning(f"Fugitive Mode Started - Searching for: {self.fugitive_name}")
                messagebox.showinfo("Fugitive Mode", f"Searching for: {self.fugitive_name}")
                
            except Exception as e:
                logger.error(f"Fugitive Mode Error: {e}")
                messagebox.showerror("Error", f"Failed to process image: {e}")
        else:
            # Stop Fugitive Mode
            self.fugitive_mode = False
            self.fugitive_image = None
            self.fugitive_face_encoding = None
            self.fugitive_detected_log_done = False
            self.btn_fugitive.config(text="Enable Fugitive Mode", bg="#8b0000")
            logger.warning("Fugitive Mode Stopped")
            messagebox.showinfo("Fugitive Mode", "Fugitive Mode Stopped")
            # Clear preview
            self.fugitive_preview_label.config(image='', text="No Fugitive Selected")

    def _update_fugitive_preview(self):
        """Update fugitive preview image display"""
        if self.fugitive_image is None:
            self.fugitive_preview_label.config(image='', text="No Fugitive")
            return
        
        try:
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(self.fugitive_image, cv2.COLOR_BGR2RGB)
            
            # Resize for preview (150x150)
            preview_size = 150
            h, w = rgb_image.shape[:2]
            aspect = w / h
            if aspect > 1:
                new_w = preview_size
                new_h = int(preview_size / aspect)
            else:
                new_h = preview_size
                new_w = int(preview_size * aspect)
            
            rgb_resized = cv2.resize(rgb_image, (new_w, new_h))
            
            # Convert to PIL
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(rgb_resized)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.fugitive_preview_label.config(image=photo, text='')
            self.fugitive_preview_label.image = photo  # Keep reference
            
        except Exception as e:
            logger.error(f"Failed to update fugitive preview: {e}")
            self.fugitive_preview_label.config(text="Preview Error")

    def start_camera(self):
        if not self.is_running:
            try:
                # Detect available cameras
                available_cameras = detect_available_cameras()
                
                if not available_cameras:
                    messagebox.showerror("Camera Error", "No cameras detected!")
                    return
                
                # If multiple cameras, let user choose
                if len(available_cameras) > 1:
                    camera_options = [f"Camera {i}" for i in available_cameras]
                    dialog = tk.Toplevel(self.root)
                    dialog.title("Select Camera")
                    dialog.geometry("300x200")
                    dialog.transient(self.root)
                    dialog.grab_set()
                    
                    tk.Label(dialog, text="Multiple cameras detected.\nSelect which camera to use:", 
                            font=('Helvetica', 10)).pack(pady=10)
                    
                    selected_camera = tk.IntVar(value=available_cameras[0])
                    
                    for idx in available_cameras:
                        tk.Radiobutton(dialog, text=f"Camera {idx}", variable=selected_camera, 
                                      value=idx, font=('Helvetica', 10)).pack(anchor="w", padx=20)
                    
                    def on_select():
                        self.camera_index = selected_camera.get()
                        dialog.destroy()
                    
                    tk.Button(dialog, text="Select", command=on_select, bg="#27ae60", 
                             fg="white", font=('Helvetica', 10, 'bold')).pack(pady=10)
                    
                    dialog.wait_window()
                else:
                    self.camera_index = available_cameras[0]
                
                # Open selected camera
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    messagebox.showerror("Camera Error", f"Failed to open camera {self.camera_index}")
                    return
                    
                self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.is_running = True
                self.btn_start.config(state="disabled")
                self.btn_stop.config(state="normal")
                self.btn_add_guard.config(state="normal")
                self.btn_toggle_alert.config(state="normal")
                self.btn_fugitive.config(state="normal")
                logger.warning(f"Camera {self.camera_index} started successfully")
                self.update_video_feed()
            except Exception as e:
                logger.error(f"Camera start error: {e}")
                messagebox.showerror("Error", f"Failed to start camera: {e}")

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.is_logging:
                self.save_log_to_file()
            
            # Stop Fugitive Mode if running
            if self.fugitive_mode:
                self.fugitive_mode = False
                self.fugitive_image = None
                self.fugitive_face_encoding = None
                self.fugitive_detected_log_done = False
                self.btn_fugitive.config(text="Enable Fugitive Mode", bg="#8b0000")
                self.fugitive_preview_label.config(image='', text="No Fugitive Selected")
            
            # Clear guard preview
            self.guard_preview_label.config(image='', text="No Guard Selected")
            
            # Cleanup
            for status in self.targets_status.values():
                if status["tracker"]:
                    status["tracker"] = None
            
            gc.collect()
            
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.btn_add_guard.config(state="disabled")
            self.btn_fugitive.config(state="disabled")
            self.video_label.config(image='')

    def auto_flush_logs(self):
        """Automatically flush logs when threshold reached"""
        if self.is_logging and len(self.temp_log) >= CONFIG["logging"]["auto_flush_interval"]:
            self.save_log_to_file()

    def save_log_to_file(self):
        if self.temp_log:
            try:
                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(self.temp_log)
                logger.warning(f"Saved {len(self.temp_log)} log entries")
                self.temp_log.clear()
                self.temp_log_counter = 0
            except Exception as e:
                logger.error(f"Log save error: {e}")
            
    def capture_alert_snapshot(self, frame, target_name, check_rate_limit=False):
        """
        Capture alert snapshot with optional rate limiting (1 per minute).
        
        Args:
            frame: Image frame to save
            target_name: Name of the target
            check_rate_limit: If True, only capture if 60+ seconds since last snapshot
        
        Returns:
            filename if saved, None if rate limited, "Error" if failed
        """
        current_time = time.time()
        
        # Rate limiting check: only one snapshot per minute per target
        if check_rate_limit and target_name in self.targets_status:
            last_snap_time = self.targets_status[target_name].get("last_snapshot_time", 0)
            if (current_time - last_snap_time) < 60:  # Less than 60 seconds
                return None  # Skip snapshot due to rate limit
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = target_name.replace(" ", "_")
        snapshot_dir = CONFIG["storage"]["alert_snapshots_dir"]
        filename = os.path.join(snapshot_dir, f"alert_{safe_name}_{timestamp}.jpg")
        try:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr_frame)
            
            # Update last snapshot time for this target
            if target_name in self.targets_status:
                self.targets_status[target_name]["last_snapshot_time"] = current_time
            
            return filename
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
            return "Error"

    def enter_onboarding_mode(self):
        if not self.is_running: return
        self.onboarding_mode = True
        self.onboarding_step = 0  # 0=face, 1-4=poses
        self.onboarding_poses = {}
        self.is_in_capture_mode = True
        
        name = simpledialog.askstring("New Guard", "Enter guard name:")
        if not name:
            self.onboarding_mode = False
            self.is_in_capture_mode = False
            return
        self.onboarding_name = name.strip()
        
        self.btn_start.grid_remove()
        self.btn_stop.grid_remove()
        self.btn_add_guard.grid_remove()
        self.btn_snap.grid(row=0, column=0)
        self.btn_cancel_capture.grid(row=0, column=1)
        messagebox.showinfo("Step 1", "Stand in front of camera (green box will appear when detected). Click 'Snap Photo' when ready.")

    def exit_onboarding_mode(self):
        self.is_in_capture_mode = False
        self.onboarding_mode = False
        self.onboarding_step = 0
        self.onboarding_poses = {}
        self.onboarding_detection_results = None
        self.onboarding_face_box = None
        self.btn_snap.grid_remove()
        self.btn_cancel_capture.grid_remove()
        self.btn_start.grid()
        self.btn_stop.grid()
        self.btn_add_guard.grid()

    def snap_photo(self):
        if self.unprocessed_frame is None: return
        
        if not self.onboarding_mode:
            # Legacy simple capture - now with dynamic detection
            rgb_frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 1:
                name = simpledialog.askstring("Name", "Enter Name:")
                if name:
                    # Get face box for better cropping
                    top, right, bottom, left = face_locations[0]
                    face_h = bottom - top
                    face_w = right - left
                    
                    # Expand to include shoulders/upper body
                    crop_top = max(0, top - int(face_h * 0.3))
                    crop_bottom = min(self.unprocessed_frame.shape[0], bottom + int(face_h * 0.5))
                    crop_left = max(0, left - int(face_w * 0.3))
                    crop_right = min(self.unprocessed_frame.shape[1], right + int(face_w * 0.3))
                    
                    cropped_face = self.unprocessed_frame[crop_top:crop_bottom, crop_left:crop_right]
                    
                    # Save using systematic helpers
                    save_guard_face(cropped_face, name)
                    save_capture_snapshot(cropped_face, name)
                    
                    # Backward compatibility
                    safe_name = name.strip().replace(" ", "_")
                    cv2.imwrite(f"target_{safe_name}_face.jpg", cropped_face)
                    
                    self.load_targets()
                    self.exit_onboarding_mode()
            else:
                messagebox.showwarning("Error", "Ensure exactly one face is visible. Move closer to camera.")
            return
        
        # Onboarding mode with pose capture
        if self.onboarding_step == 0:
            # Capture face - use cached detection results
            if self.onboarding_face_box is None:
                messagebox.showwarning("Error", "No face detected. Please stand in front of camera and wait for green box.")
                return
            
            rgb_frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) != 1:
                messagebox.showwarning("Error", "Ensure exactly one face is visible. Move closer to camera.")
                return
            
            # Use detected face box to crop intelligently
            top, right, bottom, left = face_locations[0]
            face_h = bottom - top
            face_w = right - left
            
            # Check if face is large enough (person is close)
            frame_h, frame_w = self.unprocessed_frame.shape[:2]
            face_area_ratio = (face_h * face_w) / (frame_h * frame_w)
            
            if face_area_ratio < 0.02:  # Face is too small
                messagebox.showwarning("Error", "Please move closer to the camera. Face is too small.")
                return
            
            # Expand to include shoulders/upper body for better recognition
            crop_top = max(0, top - int(face_h * 0.3))
            crop_bottom = min(frame_h, bottom + int(face_h * 0.5))
            crop_left = max(0, left - int(face_w * 0.3))
            crop_right = min(frame_w, right + int(face_w * 0.3))
            
            cropped_face = self.unprocessed_frame[crop_top:crop_bottom, crop_left:crop_right]
            
            # Save using systematic helpers
            save_guard_face(cropped_face, self.onboarding_name)
            save_capture_snapshot(cropped_face, self.onboarding_name)
            
            # Backward compatibility - save to root
            safe_name = self.onboarding_name.replace(" ", "_")
            cv2.imwrite(f"target_{safe_name}_face.jpg", cropped_face)
            
            self.onboarding_step = 1
            messagebox.showinfo("Step 2", "Good! Now perform: ONE HAND RAISED LEFT (raise your left hand) and click Snap")
        else:
            # Capture pose - use cached detection results
            pose_actions = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing"]
            action = pose_actions[self.onboarding_step - 1]
            
            if self.onboarding_detection_results is None or not self.onboarding_detection_results.pose_landmarks:
                messagebox.showwarning("Error", f"No pose detected. Step back so full body is visible and perform {action}")
                return
            
            # Verify pose quality
            results = self.onboarding_detection_results
            visible_landmarks = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
            
            if visible_landmarks < 20:  # Need at least 20 visible landmarks for good pose
                messagebox.showwarning("Error", f"Pose not clear enough. Ensure full body is visible and well-lit. ({visible_landmarks}/33 landmarks visible)")
                return
            
            # Verify the action matches what we're capturing
            rgb_frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2RGB)
            current_action = classify_action(results.pose_landmarks.landmark, self.frame_h, self.frame_w)
            
            if current_action != action:
                messagebox.showwarning("Pose Mismatch", f"Please perform {action.upper()}. Currently detecting: {current_action}")
                return
            
            # Save pose landmarks
            landmarks_data = []
            for lm in results.pose_landmarks.landmark:
                landmarks_data.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
            self.onboarding_poses[action] = landmarks_data
            
            self.onboarding_step += 1
            if self.onboarding_step <= 4:
                pose_actions_local = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing"]
                next_action = pose_actions_local[self.onboarding_step - 1]
                messagebox.showinfo(f"Step {self.onboarding_step + 1}", f"Perfect! Now perform: {next_action.upper()} and click Snap when ready")
            else:
                # Save all pose references
                self.save_pose_references(self.onboarding_name, self.onboarding_poses)
                self.load_targets()
                self.exit_onboarding_mode()
                messagebox.showinfo("Complete", f"{self.onboarding_name} onboarding complete with {len(self.onboarding_poses)} poses!")
                messagebox.showinfo("Complete", f"{self.onboarding_name} onboarding complete with {len(self.onboarding_poses)} poses!")

    def update_video_feed(self):
        if not self.is_running: return
        
        try:
            if not self.cap or not self.cap.isOpened():
                logger.error("Camera not available")
                self.stop_camera()
                return
            
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame, attempting reconnect...")
                # Try to reconnect camera
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.stop_camera()
                    messagebox.showerror("Camera Error", "Camera disconnected")
                return
        except Exception as e:
            logger.error(f"Camera read error: {e}")
            self.stop_camera()
            return
        
        self.unprocessed_frame = frame.copy()
        
        # Frame skipping for performance
        self.frame_counter += 1
        skip_interval = CONFIG["performance"]["frame_skip_interval"]
        
        if self.is_in_capture_mode:
            self.process_capture_frame(frame)
        else:
            # Skip processing every N frames when enabled
            # if CONFIG["performance"]["enable_frame_skipping"] and self.frame_counter % skip_interval != 0:
            # Skip processing every N frames when enabled
            if CONFIG["performance"].get("enable_frame_skipping", True) and self.frame_counter % skip_interval != 0:
                # Use cached frame
                if self.last_process_frame is not None:
                    frame = self.last_process_frame.copy()
            else:
                self.process_tracking_frame_optimized(frame)
                self.last_process_frame = frame.copy()
        
        # FPS calculation
        if self.frame_counter % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
            self.last_fps_time = current_time
            
            # Memory monitoring
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            self.status_label.config(text=f"FPS: {self.current_fps:.1f} | MEM: {mem_mb:.0f} MB")
            
            # Session time check
            session_hours = (current_time - self.session_start_time) / 3600
            if session_hours >= CONFIG["monitoring"]["session_restart_prompt_hours"]:
                response = messagebox.askyesno(
                    "Long Session",
                    f"Session running for {session_hours:.1f} hours. Restart recommended. Continue?"
                )
                if not response:
                    self.stop_camera()
                    return
                else:
                    self.session_start_time = current_time
        
        # Auto flush logs
        self.auto_flush_logs()
        
        if self.video_label.winfo_exists():
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- Full Fill Resize Logic ---
                lbl_w = self.video_label.winfo_width()
                lbl_h = self.video_label.winfo_height()
                if lbl_w > 10 and lbl_h > 10:
                    h, w = frame.shape[:2]
                    # Maintain aspect ratio
                    scale = min(lbl_w/w, lbl_h/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
                
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            except Exception as e:
                logger.error(f"Frame display error: {e}")
        
        refresh_ms = CONFIG["performance"]["gui_refresh_ms"]
        self.root.after(refresh_ms, self.update_video_feed)

    def process_capture_frame(self, frame):
        """Process frame during onboarding capture mode with dynamic detection"""
        h, w = frame.shape[:2]
        
        # Detect face and pose from entire frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Use holistic model to detect both face and pose
        results = self.holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Store detection results for snap_photo to use
        self.onboarding_detection_results = results
        self.onboarding_face_box = None
        
        detection_status = ""
        box_color = (0, 0, 255)  # Red by default
        
        if self.onboarding_step == 0:
            # Step 0: Face capture
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) == 1:
                top, right, bottom, left = face_locations[0]
                self.onboarding_face_box = (top, right, bottom, left)
                
                # Check if face is large enough (person is close)
                face_area_ratio = ((bottom - top) * (right - left)) / (h * w)
                
                if face_area_ratio >= 0.02:  # Good size
                    box_color = (0, 255, 0)  # Green
                    detection_status = "READY - Click Snap Photo"
                    # Draw face box
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
                else:
                    box_color = (0, 165, 255)  # Orange
                    detection_status = "Move Closer to Camera"
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            elif len(face_locations) == 0:
                detection_status = "No Face Detected - Stand in front of camera"
            else:
                detection_status = "Multiple Faces - Only one person should be visible"
                
        else:
            # Steps 1-4: Pose capture
            pose_actions = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing"]
            target_action = pose_actions[self.onboarding_step - 1]
            
            if results.pose_landmarks:
                # Draw pose landmarks
                draw_styled_landmarks(frame, results)
                
                # Count visible landmarks
                visible_landmarks = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
                
                # Classify current action
                current_action = classify_action(results.pose_landmarks.landmark, h, w)
                
                # Draw bounding box around detected pose
                x_coords = [lm.x * w for lm in results.pose_landmarks.landmark if lm.visibility > 0.5]
                y_coords = [lm.y * h for lm in results.pose_landmarks.landmark if lm.visibility > 0.5]
                
                if x_coords and y_coords:
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    # Check quality
                    if visible_landmarks >= 20:
                        if current_action == target_action or target_action in ["Sit", "Standing"]:
                            box_color = (0, 255, 0)  # Green - ready
                            detection_status = f"READY - {current_action} detected - Click Snap Photo"
                        else:
                            box_color = (0, 165, 255)  # Orange - wrong pose
                            detection_status = f"Perform {target_action} (currently: {current_action})"
                    else:
                        box_color = (0, 165, 255)  # Orange - poor quality
                        detection_status = f"Pose unclear ({visible_landmarks}/33 landmarks) - Step back to show full body"
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)
            else:
                detection_status = f"No Pose Detected - Step back and perform {target_action}"
        
        # Display instructions and status
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)  # Black background for text
        
        if self.onboarding_step == 0:
            instruction = f"STEP 1/5: FACE CAPTURE"
        else:
            pose_actions = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing"]
            instruction = f"STEP {self.onboarding_step + 1}/5: {pose_actions[self.onboarding_step - 1].upper()}"
        
        cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, detection_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        return frame

    # --- TRACKING LOGIC ---
    def process_tracking_frame_optimized(self, frame):
        if not self.targets_status:
            cv2.putText(frame, "SELECT TARGETS TO START", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        self.re_detect_counter += 1
        if self.re_detect_counter > self.RE_DETECT_INTERVAL:
            self.re_detect_counter = 0
        
        rgb_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # ==================== FUGITIVE MODE ====================
        # Search for Fugitive in frame (always active when enabled, regardless of alert mode)
        if self.fugitive_mode and self.fugitive_face_encoding is not None:
            face_locations = face_recognition.face_locations(rgb_full_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_full_frame, face_locations)
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Compare with fugitive face
                    match = face_recognition.compare_faces([self.fugitive_face_encoding], face_encoding, tolerance=0.5)
                    face_distance = face_recognition.face_distance([self.fugitive_face_encoding], face_encoding)
                    
                    if match[0]:  # If face matches
                        # Draw bounding box
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                        cv2.putText(frame, f"FUGITIVE: {self.fugitive_name}", (left, top - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        # Execute all three operations simultaneously (once per detection, regardless of logging status)
                        if not self.fugitive_detected_log_done:
                            # 1. Play Fugitive Alert Sound (always)
                            if self.fugitive_alert_stop_event is None:
                                self.fugitive_alert_stop_event = threading.Event()
                            self.fugitive_alert_stop_event.clear()
                            self.fugitive_alert_sound_thread = play_siren_sound(
                                stop_event=self.fugitive_alert_stop_event,
                                sound_file="Fugitive.mp3", 
                                duration_seconds=30
                            )
                            
                            # 2. Capture snapshot (always)
                            snapshot_path = self.capture_alert_snapshot(frame, f"FUGITIVE_{self.fugitive_name}", check_rate_limit=False)
                            img_path = snapshot_path if snapshot_path else "N/A"
                            
                            # 3. Create CSV log entry (always, regardless of logging enabled/disabled)
                            confidence = 1.0 - face_distance[0]
                            self.temp_log.append((
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                f"FUGITIVE_{self.fugitive_name}",
                                "FUGITIVE_DETECTED",
                                "FUGITIVE ALERT",
                                img_path,
                                f"{confidence:.2f}"
                            ))
                            self.temp_log_counter += 1
                            
                            # Log all three actions
                            logger.warning(f"🚨 FUGITIVE DETECTED - All operations executed:")
                            logger.warning(f"   ├─ 🔊 Alert Sound: Fugitive.mp3 (30s)")
                            logger.warning(f"   ├─ 📸 Snapshot: {img_path}")
                            logger.warning(f"   └─ 📋 CSV Logged: {self.fugitive_name} (Confidence: {confidence:.2f})")
                            
                            self.fugitive_detected_log_done = True
                            self.last_fugitive_snapshot_time = time.time()
                    else:
                        # Reset flag when fugitive not in frame
                        self.fugitive_detected_log_done = False
        # ===================================================

        # 1. Update Trackers
        for name, status in self.targets_status.items():
            if status["tracker"]:
                success, box = status["tracker"].update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    status["face_box"] = (x, y, x + w, y + h)
                    status["visible"] = True
                else:
                    status["visible"] = False
                    status["tracker"] = None

        # 2. Detection (GREEDY BEST MATCH) - Fixes Target Switching
        untracked_targets = [name for name, s in self.targets_status.items() if not s["visible"]]
        
        if untracked_targets and self.re_detect_counter == 0:
            face_locations = face_recognition.face_locations(rgb_full_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_full_frame, face_locations)
                possible_matches = []
                
                for i, unknown_encoding in enumerate(face_encodings):
                    for name in untracked_targets:
                        target_encoding = self.targets_status[name]["encoding"]
                        dist = face_recognition.face_distance([target_encoding], unknown_encoding)[0]
                        confidence = 1.0 - dist
                        if dist < CONFIG["detection"]["face_recognition_tolerance"]:
                            possible_matches.append((dist, i, name))
                
                possible_matches.sort(key=lambda x: x[0])
                assigned_faces = set()
                assigned_targets = set()
                
                for dist, face_idx, name in possible_matches:
                    if face_idx in assigned_faces or name in assigned_targets: continue
                    
                    assigned_faces.add(face_idx)
                    assigned_targets.add(name)
                    (top, right, bottom, left) = face_locations[face_idx]
                    
                    # Confidence is 1 - distance
                    match_confidence = 1.0 - dist
                    
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, (left, top, right-left, bottom-top))
                    self.targets_status[name]["tracker"] = tracker
                    self.targets_status[name]["face_box"] = (left, top, right, bottom)
                    self.targets_status[name]["visible"] = True
                    self.targets_status[name]["missing_pose_counter"] = 0
                    self.targets_status[name]["face_confidence"] = match_confidence

        # 3. Overlap Check (Fixes Merging Targets)
        active_names = [n for n, s in self.targets_status.items() if s["visible"]]
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                nameA = active_names[i]
                nameB = active_names[j]
                
                # Check Face Box IoU
                boxA = self.targets_status[nameA]["face_box"]
                boxB = self.targets_status[nameB]["face_box"]
                # Convert to x,y,w,h format for IoU check
                rectA = (boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1])
                rectB = (boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1])
                
                iou = calculate_iou(rectA, rectB)
                if iou > 0.5: # Significant overlap
                    # Force re-detection for both
                    self.targets_status[nameA]["tracker"] = None
                    self.targets_status[nameA]["visible"] = False
                    self.targets_status[nameB]["tracker"] = None
                    self.targets_status[nameB]["visible"] = False

        # 4. Processing & Drawing
        required_act = self.required_action_var.get()
        current_time = time.time()

        for name, status in self.targets_status.items():
            if status["visible"]:
                fx1, fy1, fx2, fy2 = status["face_box"]
                
                # --- USE DYNAMIC BODY BOX HELPER (consistent across all modes) ---
                bx1, by1, bx2, by2 = calculate_body_box((fx1, fy1, fx2, fy2), frame_h, frame_w, expansion_factor=3.0)

                # Ghost Box Check: Only draw if tracker is confident AND pose is found
                pose_found_in_box = False
                
                if bx1 < bx2 and by1 < by2:
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size != 0:
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        rgb_crop.flags.writeable = False
                        results_crop = self.holistic.process(rgb_crop)
                        rgb_crop.flags.writeable = True
                        
                        current_action = "Unknown"
                        if results_crop.pose_landmarks:
                            pose_found_in_box = True
                            status["missing_pose_counter"] = 0 # Reset
                            
                            draw_styled_landmarks(crop, results_crop)
                            raw_action = classify_action(results_crop.pose_landmarks.landmark, (by2-by1), (bx2-bx1))
                            
                            status["pose_buffer"].append(raw_action)
                            # min_buffer = CONFIG["performance"]["min_buffer_for_classification"]
                            # Use .get() with a default value of 3 to prevent crashing
                            min_buffer = CONFIG["performance"].get("min_buffer_for_classification", 3)
                            
                            if len(status["pose_buffer"]) >= min_buffer:
                                most_common = Counter(status["pose_buffer"]).most_common(1)[0][0]
                                current_action = most_common
                            else:
                                current_action = raw_action
                            
                            # Cache action for logging
                            self.last_action_cache[name] = current_action

                            if current_action == required_act:
                                if self.is_alert_mode:
                                    status["last_action_time"] = current_time
                                    status["alert_triggered_state"] = False
                                    # STOP ALERT SOUND when action is performed
                                    if status["alert_stop_event"] is not None:
                                        status["alert_stop_event"].set()  # Signal sound to stop
                                        logger.info(f"Alert sound stopped for {name} - action performed: {required_act}")
                                if self.is_logging and status["last_logged_action"] != required_act:
                                    # Rate limiting: only log once per minute per target
                                    time_since_last_log = current_time - status["last_log_time"]
                                    if time_since_last_log > 60:
                                        self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, current_action, "Action Performed", "N/A", f"{status['face_confidence']:.2f}"))
                                        status["last_log_time"] = current_time
                                        self.temp_log_counter += 1
                                    status["last_logged_action"] = required_act
                            elif status["last_logged_action"] == required_act:
                                status["last_logged_action"] = None
                            
                            # Draw Box ONLY if pose found
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{name}: {current_action}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Ghost Box Removal Logic
                if not pose_found_in_box:
                    status["missing_pose_counter"] += 1
                else:
                    # Cache for later use
                    if name not in self.last_action_cache:
                        self.last_action_cache[name] = "Unknown"
                    # If tracker says visible, but no pose for 5 frames -> Kill Tracker
                    if status["missing_pose_counter"] > 5:
                        status["tracker"] = None
                        status["visible"] = False

            # Alert Logic
            if self.is_alert_mode:
                time_diff = current_time - status["last_action_time"]
                time_left = max(0, self.alert_interval - time_diff)
                y_offset = 50 + (list(self.targets_status.keys()).index(name) * 30)
                color = (0, 255, 0) if time_left > 3 else (0, 0, 255)
                
                # Only show status on screen if target is genuinely lost or safe
                status_txt = "OK" if status["visible"] else "MISSING"
                cv2.putText(frame, f"{name} ({status_txt}): {time_left:.1f}s", (frame_w - 300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if time_diff > self.alert_interval:
                    if (current_time - status["alert_cooldown"]) > 2.5:
                        # Start alert sound (30 seconds or until action performed)
                        if status["alert_stop_event"] is None:
                            status["alert_stop_event"] = threading.Event()
                        status["alert_stop_event"].clear()  # Reset stop flag
                        status["alert_sound_thread"] = play_siren_sound(
                            stop_event=status["alert_stop_event"], 
                            duration_seconds=30
                        )
                        status["alert_cooldown"] = current_time
                        
                        img_path = "N/A"
                        if status["visible"]:
                            # Snapshot logic with rate limiting (use calculate_body_box helper)
                            fx1, fy1, fx2, fy2 = status["face_box"]
                            bx1, by1, bx2, by2 = calculate_body_box((fx1, fy1, fx2, fy2), frame_h, frame_w, expansion_factor=3.0)
                            if bx1 < bx2:
                                snapshot_result = self.capture_alert_snapshot(frame[by1:by2, bx1:bx2], name, check_rate_limit=True)
                                img_path = snapshot_result if snapshot_result else "N/A"
                        else:
                            snapshot_result = self.capture_alert_snapshot(frame, name, check_rate_limit=True)
                            img_path = snapshot_result if snapshot_result else "N/A"

                        if self.is_logging:
                            # Determine log status based on visibility and action
                            if not status["visible"]:
                                log_s = "ALERT TRIGGERED - TARGET MISSING"
                                log_a = "MISSING"
                            else:
                                log_s = "ALERT CONTINUED" if status["alert_triggered_state"] else "ALERT TRIGGERED"
                                log_a = self.last_action_cache.get(name, "Unknown")
                            
                            confidence = status.get("face_confidence", 0.0)
                            self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, img_path, f"{confidence:.2f}"))
                            status["alert_triggered_state"] = True
                            self.temp_log_counter += 1
                
                # LOG: Action NOT performed within alert interval
                elif time_diff > (self.alert_interval - 1) and time_diff <= self.alert_interval:
                    # Log when approaching or at the end of alert interval without action
                    if self.is_logging and not status.get("alert_logged_timeout", False):
                        if status["visible"]:
                            log_s = "ACTION NOT PERFORMED (TIMEOUT)"
                            log_a = self.last_action_cache.get(name, "Unknown")
                            confidence = status.get("face_confidence", 0.0)
                        else:
                            log_s = "MISSING - NO ACTION"
                            log_a = "MISSING"
                            confidence = 0.0
                        
                        self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, "N/A", f"{confidence:.2f}"))
                        status["alert_logged_timeout"] = True
                        self.temp_log_counter += 1
                
                # RESET: When action is performed or target reset
                if time_diff <= 0:
                    status["alert_logged_timeout"] = False

        return frame 

if __name__ == "__main__":
    app = PoseApp()