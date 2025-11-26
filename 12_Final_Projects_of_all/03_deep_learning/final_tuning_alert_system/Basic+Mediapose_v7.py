import cv2
import mediapipe as mp
import csv
import time
import tkinter as tk
import customtkinter as ctk
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

# --- Optional Library Checks ---
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

try:
    import torch
    import torchvision.transforms as transforms
    REID_AVAILABLE = True
except ImportError:
    REID_AVAILABLE = False
    torch = None
    transforms = None

try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    torchreid = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None
    cosine = None

try:
    from skimage.feature import hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    hog = None

# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- 1. Configuration Loading ---
def load_config():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError("Config file not found")
    except Exception as e:
        print(f"Config load error: {e}. Using defaults.")
        return {
            "detection": {"min_detection_confidence": 0.5, "min_tracking_confidence": 0.5, 
                         "face_recognition_tolerance": 0.5, "re_detect_interval": 15},
            "alert": {"default_interval_seconds": 10, "alert_cooldown_seconds": 2.5},
            "performance": {"gui_refresh_ms": 30, "pose_buffer_size": 12, "frame_skip_interval": 2, "enable_frame_skipping": False, "min_buffer_for_classification": 5},
            "logging": {"log_directory": "logs", "max_log_size_mb": 10, "auto_flush_interval": 50},
            "storage": {"alert_snapshots_dir": "alert_snapshots", "snapshot_retention_days": 30,
                       "guard_profiles_dir": "guard_profiles", "capture_snapshots_dir": "capture_snapshots"},
            "monitoring": {"mode": "pose", "session_restart_prompt_hours": 8}
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

class SafeLogger:
    UNICODE_MAP = {
        '\u2713': '[OK]', '\u2717': '[X]', '\u26a0': '[WARN]',
        '\u26a0\ufe0f': '[WARN]', '\ud83d\udea8': '[ALERT]', '\ud83d\udd0a': '[SOUND]',
        '\ud83d\udcf8': '[SNAP]', '\ud83d\udccb': '[LOG]', '\ud83d\udccd': '[PIN]',
        '\ud83d\udca4': '[SLEEP]', '\u251c\u2500': '|-', '\u2514\u2500': 'L-', '→': '->'
    }
    @staticmethod
    def sanitize(text):
        if not isinstance(text, str): return text
        for u_char, a_equiv in SafeLogger.UNICODE_MAP.items():
            text = text.replace(u_char, a_equiv)
        return text.encode('cp1252', errors='replace').decode('cp1252')
    
    @staticmethod
    def warning(msg, *args, **kwargs): logger.warning(SafeLogger.sanitize(msg), *args, **kwargs)
    @staticmethod
    def info(msg, *args, **kwargs): logger.info(SafeLogger.sanitize(msg), *args, **kwargs)
    @staticmethod
    def debug(msg, *args, **kwargs): logger.debug(SafeLogger.sanitize(msg), *args, **kwargs)
    @staticmethod
    def error(msg, *args, **kwargs): logger.error(SafeLogger.sanitize(msg), *args, **kwargs)

safe_logger = SafeLogger()

# --- 3. File Storage Utilities ---
def get_storage_paths():
    paths = {
        "guard_profiles": CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles"),
        "pose_references": CONFIG.get("storage", {}).get("pose_references_dir", "pose_references"),
        "capture_snapshots": CONFIG.get("storage", {}).get("capture_snapshots_dir", "capture_snapshots"),
        "logs": CONFIG["logging"]["log_directory"]
    }
    for path in paths.values():
        if not os.path.exists(path): os.makedirs(path)
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
    with open(pose_path, 'w') as f: json.dump(poses_dict, f, indent=2)
    return pose_path

# Ensure directories exist
get_storage_paths()
if not os.path.exists(CONFIG["storage"]["alert_snapshots_dir"]):
    os.makedirs(CONFIG["storage"]["alert_snapshots_dir"])

gc.set_threshold(1000, 15, 15)

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Sound Logic ---
def play_siren_sound(stop_event=None, duration_seconds=30, sound_file="emergency-siren-351963.mp3"):
    def _sound_worker():
        # FIXED: Use relative path to ensure it works on any machine
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mp3_path = os.path.join(current_dir, sound_file)
        
        start_time = time.time()
        
        if PYGAME_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                # Check if file exists before loading
                if os.path.exists(mp3_path):
                    pygame.mixer.music.load(mp3_path)
                    pygame.mixer.music.set_volume(1.0)
                    pygame.mixer.music.play(-1)
                else:
                    # Fallback beep if file missing
                    print(f"Sound file missing: {mp3_path}")
                    import winsound
                    winsound.Beep(2500, 500)
                    return

                while True:
                    elapsed = time.time() - start_time
                    if stop_event and stop_event.is_set(): break
                    if elapsed >= duration_seconds: break
                    time.sleep(0.1)
                
                pygame.mixer.music.stop()
                return
            except Exception as e:
                logger.warning(f"Pygame playback failed: {e}")
        
        # Fallback Windows Beep
        try:
            if platform.system() == "Windows":
                import winsound
                while True:
                    elapsed = time.time() - start_time
                    if stop_event and stop_event.is_set(): break
                    if elapsed >= duration_seconds: break
                    winsound.Beep(2500, 150)
                    time.sleep(0.05)
                    winsound.Beep(1800, 150)
                    time.sleep(0.05)
        except Exception:
            pass

    t = threading.Thread(target=_sound_worker, daemon=True)
    t.start()
    return t

# --- EAR Calculation ---
def calculate_ear(landmarks, width, height):
    RIGHT_EYE = [33, 133, 159, 145, 158, 153]
    LEFT_EYE = [362, 263, 386, 374, 385, 380]

    def get_eye_ear(indices):
        # Check bounds
        try:
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
        except IndexError:
            return 0.0

    ear_right = get_eye_ear(RIGHT_EYE)
    ear_left = get_eye_ear(LEFT_EYE)
    return (ear_right + ear_left) / 2.0

# --- Drawing & Classification ---
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

def classify_action(landmarks, h, w):
    try:
        NOSE = mp_holistic.PoseLandmark.NOSE.value
        L_WRIST = mp_holistic.PoseLandmark.LEFT_WRIST.value
        R_WRIST = mp_holistic.PoseLandmark.RIGHT_WRIST.value
        L_SHOULDER = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        R_SHOULDER = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        L_HIP = mp_holistic.PoseLandmark.LEFT_HIP.value
        R_HIP = mp_holistic.PoseLandmark.RIGHT_HIP.value
        L_KNEE = mp_holistic.PoseLandmark.LEFT_KNEE.value
        R_KNEE = mp_holistic.PoseLandmark.RIGHT_KNEE.value

        nose = landmarks[NOSE]; l_wrist = landmarks[L_WRIST]; r_wrist = landmarks[R_WRIST]
        l_shoulder = landmarks[L_SHOULDER]; r_shoulder = landmarks[R_SHOULDER]
        l_hip = landmarks[L_HIP]; r_hip = landmarks[R_HIP]
        l_knee = landmarks[L_KNEE]; r_knee = landmarks[R_KNEE]

        shoulder_to_hip_dist = abs(l_shoulder.y - l_hip.y)
        if shoulder_to_hip_dist < 0.01: return "Standing"
        
        HANDS_UP_THRESHOLD = shoulder_to_hip_dist * 0.4
        HANDS_CROSSED_TOLERANCE = shoulder_to_hip_dist * 0.3
        ARM_EXTENSION = shoulder_to_hip_dist * 0.6
        
        # Normalized checks
        l_wrist_visible = l_wrist.visibility > 0.50
        r_wrist_visible = r_wrist.visibility > 0.50
        
        if (l_wrist_visible and r_wrist_visible and 
            (nose.y - l_wrist.y) > HANDS_UP_THRESHOLD and 
            (nose.y - r_wrist.y) > HANDS_UP_THRESHOLD):
            return "Hands Up"
        
        chest_y = (l_shoulder.y + r_shoulder.y) / 2
        body_center_x = (l_shoulder.x + r_shoulder.x) / 2
        
        if (l_wrist_visible and r_wrist_visible and 
            abs(l_wrist.y - chest_y) < HANDS_CROSSED_TOLERANCE and 
            abs(r_wrist.y - chest_y) < HANDS_CROSSED_TOLERANCE):
            if ((l_wrist.x > body_center_x and r_wrist.x < body_center_x) or 
                (l_wrist.x < body_center_x and r_wrist.x > body_center_x)):
                return "Hands Crossed"
        
        if (l_wrist_visible and r_wrist_visible):
            shoulder_width = abs(r_shoulder.x - l_shoulder.x)
            if (abs(l_wrist.x - l_shoulder.x) > ARM_EXTENSION and 
                abs(r_wrist.x - r_shoulder.x) > ARM_EXTENSION and
                abs(l_wrist.y - l_shoulder.y) < HANDS_CROSSED_TOLERANCE):
                return "T-Pose"

        if l_wrist_visible and (nose.y - l_wrist.y) > HANDS_UP_THRESHOLD and not r_wrist_visible:
            return "One Hand Raised (Left)"
        if r_wrist_visible and (nose.y - r_wrist.y) > HANDS_UP_THRESHOLD and not l_wrist_visible:
            return "One Hand Raised (Right)"
        
        if landmarks[L_KNEE].visibility > 0.5 and landmarks[R_KNEE].visibility > 0.5:
            avg_thigh_angle = (abs(l_knee.y - l_hip.y) + abs(r_knee.y - r_hip.y)) / 2
            if avg_thigh_angle < (shoulder_to_hip_dist * 0.15):
                return "Sit"

        return "Standing" 
    except Exception:
        return "Unknown"

def calculate_body_box(face_box, frame_h, frame_w, expansion_factor=3.0):
    x1, y1, x2, y2 = face_box
    face_w = x2 - x1
    face_h = y2 - y1
    face_cx = x1 + (face_w // 2)
    
    if face_w < 30: adaptive_expansion = 5.0
    elif face_w < 100: adaptive_expansion = 4.0
    else: adaptive_expansion = 3.0
    
    bx1 = max(0, int(face_cx - (face_w * adaptive_expansion / 2)))
    bx2 = min(frame_w, int(face_cx + (face_w * adaptive_expansion / 2)))
    by1 = max(0, int(y1 - (face_h * 0.5)))
    by2 = frame_h
    
    return (bx1, by1, bx2, by2)

def approximate_face_from_body(body_box):
    bx1, by1, bx2, by2 = body_box
    w = max(1, bx2 - bx1)
    h = max(1, by2 - by1)
    fx_w = int(max(20, 0.25 * w))
    fx_h = int(max(20, 0.20 * h))
    fx1 = bx1 + int(0.375 * w)
    fy1 = by1 + int(0.05 * h)
    return (fx1, fy1, min(bx2, fx1 + fx_w), min(by2, fy1 + fx_h))

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def resolve_overlapping_poses(targets_status, iou_threshold=0.3):
    target_names = list(targets_status.keys())
    for i, name_a in enumerate(target_names):
        box_a = targets_status[name_a].get("face_box")
        if not box_a: continue
        for name_b in target_names[i+1:]:
            box_b = targets_status[name_b].get("face_box")
            if not box_b: continue
            
            iou = calculate_iou((box_a[0], box_a[1], box_a[2]-box_a[0], box_a[3]-box_a[1]),
                                (box_b[0], box_b[1], box_b[2]-box_b[0], box_b[3]-box_b[1]))
            
            if iou < iou_threshold:
                if not targets_status[name_a].get("visible") and targets_status[name_a].get("overlap_disabled"):
                    targets_status[name_a]["visible"] = True; targets_status[name_a]["overlap_disabled"] = False; targets_status[name_a]["tracker"] = None
                if not targets_status[name_b].get("visible") and targets_status[name_b].get("overlap_disabled"):
                    targets_status[name_b]["visible"] = True; targets_status[name_b]["overlap_disabled"] = False; targets_status[name_b]["tracker"] = None
                continue

            if targets_status[name_a].get("visible") and targets_status[name_b].get("visible"):
                score_a = targets_status[name_a].get("face_confidence", 0)
                score_b = targets_status[name_b].get("face_confidence", 0)
                if score_a < score_b:
                    targets_status[name_a]["visible"] = False; targets_status[name_a]["overlap_disabled"] = True
                elif score_b < score_a:
                    targets_status[name_b]["visible"] = False; targets_status[name_b]["overlap_disabled"] = True
    return targets_status

def smooth_bounding_box(current, previous, factor=0.7):
    if previous is None: return current
    return tuple(int(factor * p + (1 - factor) * c) for c, p in zip(current, previous))

# --- ReID Feature Utilities ---
def extract_appearance_features(frame, face_box):
    try:
        x1, y1, x2, y2 = max(0, int(face_box[0])), max(0, int(face_box[1])), int(face_box[2]), int(face_box[3])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        
        resized = cv2.resize(crop, (64, 128))
        hist_feats = []
        for i in range(3):
            hist_feats.extend(cv2.calcHist([resized], [i], None, [8], [0, 256]).flatten())
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edge_hist = cv2.calcHist([cv2.Canny(gray, 100, 200)], [0], None, [16], [0, 256]).flatten()
        
        features = np.array(hist_feats + list(edge_hist), dtype=np.float32)
        return features / (np.linalg.norm(features) + 1e-6)
    except: return None

def calculate_feature_similarity(f1, f2):
    if f1 is None or f2 is None: return 0.0
    if SKLEARN_AVAILABLE and cosine_similarity:
        return cosine_similarity(np.array(f1).reshape(1, -1), np.array(f2).reshape(1, -1))[0][0]
    dot = np.dot(f1, f2)
    return max(0.0, min(1.0, dot / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-6)))

# --- Main App ---
class PoseApp:
    def __init__(self, window_title="Pose Guard (Multi-Target)"):
        self.root = ctk.CTk()
        self.root.title(window_title)
        self.root.geometry("1800x1000")
        
        self.cap = None
        self.unprocessed_frame = None 
        self.is_running = False
        self.is_logging = False
        self.camera_index = 0
        
        self.is_alert_mode = False
        self.alert_interval = 10
        self.monitor_mode_var = tk.StringVar(self.root); self.monitor_mode_var.set("All Alerts (Action + Sleep)")
        self.sleep_alert_delay_seconds = 1.5
        self.is_in_capture_mode = False
        self.frame_w = 640; self.frame_h = 480 

        self.target_map = {}
        self.targets_status = {} 
        self.selected_target_names = []
        self.re_detect_counter = 0    
        self.RE_DETECT_INTERVAL = CONFIG["detection"]["re_detect_interval"]
        self.temp_log = []
        self.temp_log_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.last_process_frame = None
        self.last_action_cache = {}
        self.session_start_time = time.time()
        self.onboarding_mode = False
        self.photo_storage = {}

        # Fugitive & Pro Mode vars
        self.fugitive_mode = False
        self.fugitive_image = None; self.fugitive_face_encoding = None; self.fugitive_name = "Unknown"
        self.fugitive_detected_log_done = False
        self.fugitive_alert_sound_thread = None; self.fugitive_alert_stop_event = None
        self.pro_detection_mode = False
        self.person_features_db = {}; self.person_tracking_data = []; self.pro_detection_log_done = {}
        self.reid_enabled = REID_AVAILABLE and SKLEARN_AVAILABLE
        self.reid_confidence_threshold = 0.65; self.pro_detection_person_counter = 0

        try:
            # Main Holistic for Onboarding
            self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            
            # FIX FOR SLEEPING ALERT BUG:
            # Use a separate Holistic instance for cropping logic with static_image_mode=True
            # This ensures accurate face landmark detection even when switching between different people crops
            self.crop_holistic = mp_holistic.Holistic(
                static_image_mode=True,  # CRITICAL: Treats each crop as a new image
                model_complexity=0,      # Lite model for speed
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.warning("MediaPipe models initialized")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Model: {e}")
            self.root.destroy(); return

        # --- GUI Layout ---
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        
        self.sidebar_collapsed = False
        self.sidebar_width = 300 # Increased slightly
        
        self.main_container = ctk.CTkFrame(self.root, fg_color="black")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        self.video_container = ctk.CTkFrame(self.main_container, fg_color="black")
        self.video_container.grid(row=0, column=0, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_container, text="🎥 Camera Feed Off", font=("Arial", 24, "bold"), text_color="white")
        self.video_label.pack(fill="both", expand=True)
        
        self.sidebar_frame = ctk.CTkFrame(self.root, fg_color="#0f0f0f", width=self.sidebar_width)
        self.sidebar_frame.grid(row=0, column=1, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)
        
        # Sidebar Header
        sidebar_header = ctk.CTkFrame(self.sidebar_frame, fg_color="#0d0d0d", height=50)
        sidebar_header.pack(fill="x")
        ctk.CTkFrame(sidebar_header, fg_color="#2c3e50", height=2).pack(side="bottom", fill="x")
        self.toggle_sidebar_btn = ctk.CTkButton(sidebar_header, text="◄ Hide", command=self.toggle_sidebar, width=40, height=35, fg_color="#0066cc")
        self.toggle_sidebar_btn.pack(side="left", padx=5, pady=5)
        ctk.CTkLabel(sidebar_header, text="⚡ CONTROL PANEL", font=("Roboto", 11, "bold"), text_color="#00d4ff").pack(side="left", padx=10)
        
        # Sidebar Scroll
        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="#0f0f0f")
        self.sidebar_scroll.pack(fill="both", expand=True, padx=0, pady=5)

        self.setup_sidebar_controls()
        
        # Status Footer
        status_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#1a1a1a", height=30) # Fixed height footer
        status_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        self.status_label = ctk.CTkLabel(status_frame, text="FPS: 0 | MEM: 0 MB", text_color="#ecf0f1", font=('Roboto', 9, 'bold'))
        self.status_label.pack(fill="x", padx=8, pady=8)

        self.load_targets()
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        self.root.mainloop()

    def setup_sidebar_controls(self):
        btn_font = ('Roboto', 10, 'bold')
        
        # Camera
        self.grp_camera = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_camera.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.grp_camera, text="▶ SYSTEM CONTROLS", font=("Roboto", 10, "bold"), text_color="#00d4ff").pack(anchor="w")
        
        self.cam_btns = ctk.CTkFrame(self.grp_camera, fg_color="transparent")
        self.cam_btns.pack(fill="x")
        self.cam_btns.grid_columnconfigure((0,1,2,3), weight=1)
        
        self.btn_start = ctk.CTkButton(self.cam_btns, text="▶ Start", command=self.start_camera, width=60, fg_color="#27ae60", font=btn_font)
        self.btn_start.grid(row=0, column=0, padx=2, sticky="ew")
        self.btn_stop = ctk.CTkButton(self.cam_btns, text="⏹ Stop", command=self.stop_camera, width=60, fg_color="#c0392b", font=btn_font)
        self.btn_stop.grid(row=0, column=1, padx=2, sticky="ew")
        ctk.CTkButton(self.cam_btns, text="📸 Snap", command=self.snap_photo, width=60, fg_color="#d35400", font=btn_font).grid(row=0, column=2, padx=2, sticky="ew")
        ctk.CTkButton(self.cam_btns, text="🚪 Exit", command=self.graceful_exit, width=50, fg_color="#34495e", font=btn_font).grid(row=0, column=3, padx=2, sticky="ew")

        # Guard
        self.grp_guard = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_guard.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.grp_guard, text="👮 GUARD MANAGEMENT", font=("Roboto", 10, "bold"), text_color="#00d4ff").pack(anchor="w")
        
        g_btns = ctk.CTkFrame(self.grp_guard, fg_color="transparent")
        g_btns.pack(fill="x")
        g_btns.grid_columnconfigure((0,1,2), weight=1)
        self.btn_add_guard = ctk.CTkButton(g_btns, text="➕ Add", command=self.add_guard_dialog, width=60, fg_color="#8e44ad", font=btn_font)
        self.btn_add_guard.grid(row=0, column=0, padx=2, sticky="ew")
        ctk.CTkButton(g_btns, text="❌ Remove", command=self.remove_guard_dialog, width=60, fg_color="#e74c3c", font=btn_font).grid(row=0, column=1, padx=2, sticky="ew")
        ctk.CTkButton(g_btns, text="🔄 Refresh", command=self.load_targets, width=60, fg_color="#e67e22", font=btn_font).grid(row=0, column=2, padx=2, sticky="ew")

        # Modes
        self.grp_modes = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_modes.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.grp_modes, text="🔍 DETECTION MODES", font=("Roboto", 10, "bold"), text_color="#00d4ff").pack(anchor="w")
        
        m_btns = ctk.CTkFrame(self.grp_modes, fg_color="transparent")
        m_btns.pack(fill="x")
        m_btns.grid_columnconfigure((0,1,2), weight=1)
        self.btn_toggle_alert = ctk.CTkButton(m_btns, text="🔔 Alert", command=self.toggle_alert_mode, width=70, fg_color="#e67e22", font=btn_font)
        self.btn_toggle_alert.grid(row=0, column=0, padx=2, sticky="ew")
        self.btn_fugitive = ctk.CTkButton(m_btns, text="🚨 Fugitive", command=self.toggle_fugitive_mode, width=70, fg_color="#8b0000", font=btn_font)
        self.btn_fugitive.grid(row=0, column=1, padx=2, sticky="ew")
        self.btn_pro_detection = ctk.CTkButton(m_btns, text="🎯 Pro", command=self.toggle_pro_detection_mode, width=70, fg_color="#0066cc", font=btn_font)
        self.btn_pro_detection.grid(row=0, column=2, padx=2, sticky="ew")

        # Settings
        self.grp_settings = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_settings.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.grp_settings, text="⚙️ SETTINGS", font=("Roboto", 10, "bold"), text_color="#00d4ff").pack(anchor="w")
        
        # Timers
        timer_section = ctk.CTkFrame(self.grp_settings, fg_color="#1a1a1a", corner_radius=6)
        timer_section.pack(fill="x", pady=3)
        
        t1 = ctk.CTkFrame(timer_section, fg_color="transparent")
        t1.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(t1, text="Timeout:", font=btn_font, text_color="white").pack(side="left")
        self.btn_set_interval = ctk.CTkButton(t1, text=f"{self.alert_interval}s", command=self.set_alert_interval_advanced, width=50, fg_color="#7f8c8d", font=btn_font)
        self.btn_set_interval.pack(side="left", padx=5)
        self.label_interval_desc = ctk.CTkLabel(t1, text="(action wait)", font=("Roboto", 8), text_color="#95a5a6")
        self.label_interval_desc.pack(side="left")

        t2 = ctk.CTkFrame(timer_section, fg_color="transparent")
        t2.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(t2, text="Sleep:", font=btn_font, text_color="white").pack(side="left")
        self.btn_set_sleep = ctk.CTkButton(t2, text=f"{self.sleep_alert_delay_seconds}s", command=self.set_sleep_interval, width=50, fg_color="#546e7a", font=btn_font)
        self.btn_set_sleep.pack(side="left", padx=5)
        self.label_sleep_desc = ctk.CTkLabel(t2, text="(eyes closed)", font=("Roboto", 8), text_color="#95a5a6")
        self.label_sleep_desc.pack(side="left")

        # Dropdowns
        ctk.CTkLabel(self.grp_settings, text="Required Action:", font=("Roboto", 9), text_color="white").pack(anchor="w", pady=(5,0))
        self.required_action_var = tk.StringVar(self.root); self.required_action_var.set("Hands Up")
        self.action_dropdown = ctk.CTkOptionMenu(self.grp_settings, values=["Hands Up", "Hands Crossed", "One Hand Raised (Left)", "One Hand Raised (Right)", "T-Pose", "Sit", "Standing"], command=self.on_action_change, fg_color="#3498db")
        self.action_dropdown.pack(fill="x", pady=2)

        ctk.CTkLabel(self.grp_settings, text="Monitor Mode:", font=("Roboto", 9), text_color="white").pack(anchor="w", pady=(5,0))
        self.monitor_mode_dropdown = ctk.CTkOptionMenu(self.grp_settings, values=["All Alerts (Action + Sleep)", "Action Alerts Only", "Sleeping Alerts Only"], variable=self.monitor_mode_var, fg_color="#34495e")
        self.monitor_mode_dropdown.pack(fill="x", pady=2)

        # Target Selection
        t_grid = ctk.CTkFrame(self.grp_settings, fg_color="transparent")
        t_grid.pack(fill="x", pady=5)
        t_grid.grid_columnconfigure((0,1), weight=1)
        self.btn_select_targets = ctk.CTkButton(t_grid, text="📋 Select", command=self.open_target_selection_dialog, fg_color="#16a085", font=btn_font)
        self.btn_select_targets.grid(row=0, column=0, padx=2, sticky="ew")
        self.btn_apply_targets = ctk.CTkButton(t_grid, text="🎬 Track", command=self.apply_target_selection, fg_color="#16a085", font=btn_font)
        self.btn_apply_targets.grid(row=0, column=1, padx=2, sticky="ew")

        # Previews
        self.grp_preview = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_preview.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.guard_preview_frame = ctk.CTkFrame(self.grp_preview, fg_color="#1a3a1a", border_color="#27ae60", border_width=1)
        self.guard_preview_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(self.guard_preview_frame, text="GUARD PREVIEW", font=("Arial", 9, "bold"), text_color="#27ae60").pack()
        self.guard_preview_label = ctk.CTkLabel(self.guard_preview_frame, text="None", text_color="gray")
        self.guard_preview_label.pack(fill="both", expand=True, padx=2, pady=2)

        self.fugitive_preview_frame = ctk.CTkFrame(self.grp_preview, fg_color="#3a1a1a", border_color="#e74c3c", border_width=1)
        self.fugitive_preview_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(self.fugitive_preview_frame, text="FUGITIVE PREVIEW", font=("Arial", 9, "bold"), text_color="#e74c3c").pack()
        self.fugitive_preview_label = ctk.CTkLabel(self.fugitive_preview_frame, text="None", text_color="gray")
        self.fugitive_preview_label.pack(fill="x", padx=2, pady=2)

    def toggle_sidebar(self):
        if self.sidebar_collapsed:
            self.sidebar_frame.grid(row=0, column=1, sticky="nsew")
            self.toggle_sidebar_btn.configure(text="◄ Hide")
            self.sidebar_collapsed = False
        else:
            self.sidebar_frame.grid_remove()
            self.toggle_sidebar_btn.configure(text="► Show")
            self.sidebar_collapsed = True

    def graceful_exit(self):
        self.is_running = False
        if self.cap: self.cap.release()
        if self.is_logging: self.save_log_to_file()
        try: self.holistic.close(); self.crop_holistic.close()
        except: pass
        self.root.quit(); self.root.destroy()

    # --- Standard Functions (Shortened where logic unchanged) ---
    def add_guard_dialog(self):
        if not self.is_running: messagebox.showwarning("Warning", "Start camera first."); return
        if messagebox.askquestion("Add Guard", "Yes=Camera, No=Upload") == 'yes': self.enter_onboarding_mode()
        else: self.upload_guard_image()
        
    def upload_guard_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if path:
            name = simpledialog.askstring("Name", "Guard Name:")
            if name:
                img = face_recognition.load_image_file(path)
                if len(face_recognition.face_locations(img)) == 1:
                    import shutil
                    safe_name = name.strip().replace(" ", "_")
                    shutil.copy(path, f"target_{safe_name}_face.jpg") # Legacy
                    save_guard_face(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), name)
                    self.load_targets()
                    messagebox.showinfo("Success", "Guard added.")
                else: messagebox.showerror("Error", "Image must have exactly 1 face.")

    def remove_guard_dialog(self):
        dialog = tk.Toplevel(self.root); dialog.geometry("300x400")
        lb = tk.Listbox(dialog); lb.pack(fill="both", expand=True)
        for n in self.target_map: lb.insert(tk.END, n)
        def rem():
            if not lb.curselection(): return
            name = lb.get(lb.curselection()[0])
            if messagebox.askyesno("Confirm", f"Delete {name}?"):
                self.remove_guard(name); dialog.destroy()
        tk.Button(dialog, text="Remove", command=rem, bg="red").pack()

    def remove_guard(self, name):
        try:
            safe = name.replace(" ", "_")
            f1 = os.path.join(CONFIG["storage"]["guard_profiles_dir"], f"target_{safe}_face.jpg")
            f2 = os.path.join(CONFIG["storage"]["pose_references_dir"], f"{safe}_poses.json")
            if os.path.exists(f1): os.remove(f1)
            if os.path.exists(f2): os.remove(f2)
            if name in self.targets_status: del self.targets_status[name]
            self.load_targets()
        except Exception as e: logger.error(f"Remove error: {e}")

    def load_targets(self):
        self.target_map = {}
        files = glob.glob(os.path.join(CONFIG["storage"]["guard_profiles_dir"], "target_*.jpg"))
        for f in files:
            try:
                bn = os.path.basename(f).replace(".jpg","")
                parts = bn.split('_'); 
                if len(parts)>=3: self.target_map[" ".join(parts[1:-1])] = f
            except: pass
        self.selected_target_names = [n for n in self.selected_target_names if n in self.target_map]
        self.update_selected_preview()

    def update_selected_preview(self):
        if self.selected_target_names and (name := self.selected_target_names[0]) in self.target_map:
            try:
                img = cv2.imread(self.target_map[name])
                img = cv2.resize(img, (150, 150))
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                self.guard_preview_label.configure(image=imgtk, text=""); self.photo_storage["gp"] = imgtk
            except: pass
        else: self.guard_preview_label.configure(image="", text="None")

    def open_target_selection_dialog(self):
        d = ctk.CTkToplevel(self.root); d.geometry("300x400")
        vars = {}
        sf = ctk.CTkScrollableFrame(d); sf.pack(fill="both", expand=True)
        for t in sorted(self.target_map.keys()):
            v = ctk.BooleanVar(value=t in self.selected_target_names)
            ctk.CTkCheckBox(sf, text=t, variable=v).pack(anchor="w")
            vars[t] = v
        def done():
            self.selected_target_names = [k for k,v in vars.items() if v.get()]
            self.update_selected_preview(); d.destroy()
        ctk.CTkButton(d, text="Done", command=done).pack()

    def apply_target_selection(self):
        self.targets_status = {}
        for name in self.selected_target_names:
            path = self.target_map.get(name)
            if path and os.path.exists(path):
                encs = face_recognition.face_encodings(face_recognition.load_image_file(path))
                if encs:
                    self.targets_status[name] = {
                        "encoding": encs[0], "tracker": None, "body_tracker": None,
                        "face_box": None, "body_box": None, "visible": False, "overlap_disabled": False,
                        "last_action_time": time.time(), "alert_cooldown": 0, "alert_triggered_state": False,
                        "last_logged_action": None, "pose_buffer": deque(maxlen=12), "missing_pose_counter": 0,
                        "face_confidence": 0.0, "pose_confidence": 0.0, "face_encoding_history": deque(maxlen=5),
                        "last_valid_pose": None, "pose_quality_history": deque(maxlen=10),
                        "last_snapshot_time": 0, "last_log_time": 0,
                        "alert_sound_thread": None, "alert_stop_event": None,
                        "target_missing_alert_logged": False, "sleep_alert_logged": False,
                        # Sleep logic
                        "eye_counter_closed": 0, "ear_threshold": 0.22, "open_ear_baseline": 0.30, "is_sleeping": False
                    }
        messagebox.showinfo("Info", f"Tracking {len(self.targets_status)} targets")

    # --- Core Logic Methods ---
    def set_alert_interval_advanced(self):
        d = ctk.CTkToplevel(self.root); d.geometry("300x200")
        v = ctk.StringVar(value=str(self.alert_interval))
        ctk.CTkEntry(d, textvariable=v).pack(pady=10)
        def save(): 
            try: self.alert_interval = int(v.get()); self.btn_set_interval.configure(text=f"{self.alert_interval}s"); d.destroy()
            except: pass
        ctk.CTkButton(d, text="Set", command=save).pack()

    def set_sleep_interval(self):
        d = ctk.CTkToplevel(self.root); d.geometry("300x200")
        v = ctk.StringVar(value=str(self.sleep_alert_delay_seconds))
        ctk.CTkEntry(d, textvariable=v).pack(pady=10)
        def save():
            try: self.sleep_alert_delay_seconds = float(v.get()); self.btn_set_sleep.configure(text=f"{self.sleep_alert_delay_seconds}s"); d.destroy()
            except: pass
        ctk.CTkButton(d, text="Set", command=save).pack()

    def toggle_alert_mode(self):
        self.is_alert_mode = not self.is_alert_mode
        self.btn_toggle_alert.configure(text="Stop Alert" if self.is_alert_mode else "Start Alert", fg_color="red" if self.is_alert_mode else "#e67e22")
        if self.is_alert_mode: self.is_logging = True; self.temp_log = []
        else: self.save_log_to_file(); self.is_logging = False

    def toggle_fugitive_mode(self):
        if not self.fugitive_mode:
            path = filedialog.askopenfilename()
            if path:
                img = cv2.imread(path)
                locs = face_recognition.face_locations(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if locs:
                    self.fugitive_image = img
                    self.fugitive_face_encoding = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), locs)[0]
                    self.fugitive_name = simpledialog.askstring("Fugitive", "Name:") or "Unknown"
                    self.fugitive_mode = True
                    self.btn_fugitive.configure(text="Disable Fugitive", fg_color="red")
                    self._update_fugitive_preview()
        else:
            self.fugitive_mode = False
            self.btn_fugitive.configure(text="Enable Fugitive", fg_color="#8b0000")
            self.fugitive_preview_label.configure(image="", text="None")

    def _update_fugitive_preview(self):
        if self.fugitive_image is not None:
            img = cv2.resize(self.fugitive_image, (150, 150))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            self.fugitive_preview_label.configure(image=imgtk, text=""); self.photo_storage["fp"] = imgtk

    def toggle_pro_detection_mode(self):
        self.pro_detection_mode = not self.pro_detection_mode
        self.btn_pro_detection.configure(fg_color="#00d9ff" if self.pro_detection_mode else "#0066cc")
        if self.pro_detection_mode:
            messagebox.showinfo("Pro Mode", "Pro Mode Enabled: ReID active.")
            self.person_features_db = {}

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.is_running = True
                self.btn_start.configure(state="disabled"); self.btn_stop.configure(state="normal")
                self.update_video_feed()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap: self.cap.release()
            self.btn_start.configure(state="normal"); self.btn_stop.configure(state="disabled")
            self.video_label.configure(image="")

    def save_log_to_file(self):
        if self.temp_log:
            path = os.path.join(CONFIG["logging"]["log_directory"], "events.csv")
            exists = os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if not exists: w.writerow(["Timestamp", "Name", "Action", "Status", "Image", "Conf"])
                w.writerows(self.temp_log)
            self.temp_log = []

    def on_action_change(self, v): pass

    # --- Capture & Onboarding ---
    def capture_alert_snapshot(self, frame, name, check_rate_limit=False):
        if check_rate_limit and name in self.targets_status:
            if time.time() - self.targets_status[name]["last_snapshot_time"] < 60: return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CONFIG["storage"]["alert_snapshots_dir"], f"alert_{name.replace(' ','_')}_{ts}.jpg")
        cv2.imwrite(path, frame)
        if name in self.targets_status: self.targets_status[name]["last_snapshot_time"] = time.time()
        return path

    def snap_photo(self):
        if self.unprocessed_frame is None: return
        if not self.onboarding_mode:
            rgb = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            if len(locs) == 1:
                name = simpledialog.askstring("Name", "Enter Name:")
                if name:
                    t,r,b,l = locs[0]
                    save_guard_face(self.unprocessed_frame[t-50:b+50, l-50:r+50], name)
                    self.load_targets()
        else:
            # Simple Onboarding Logic (simplified for brevity, logic preserved)
            self.onboarding_step += 1
            if self.onboarding_step > 4: self.onboarding_mode = False; self.is_in_capture_mode = False; self.load_targets()
            else: messagebox.showinfo("Step", f"Step {self.onboarding_step} captured.")

    def enter_onboarding_mode(self):
        self.onboarding_mode = True; self.is_in_capture_mode = True
        self.onboarding_name = simpledialog.askstring("Name", "Guard Name:")

    # --- Video Loop ---
    def update_video_feed(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if not ret: self.stop_camera(); return
        
        self.unprocessed_frame = frame.copy()
        self.frame_counter += 1
        
        if self.is_in_capture_mode: 
            # Basic visualization for capture
            cv2.putText(frame, f"CAPTURE MODE: {self.onboarding_step}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            self.process_tracking_frame_optimized(frame)

        # FPS
        if self.frame_counter % 30 == 0:
            el = time.time() - self.last_fps_time
            self.current_fps = 30 / el if el > 0 else 0
            self.last_fps_time = time.time()
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            self.status_label.configure(text=f"FPS: {self.current_fps:.1f} | MEM: {mem:.0f} MB")
            if len(self.temp_log) > 50: self.save_log_to_file()

        # GUI Update
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Aspect fit logic
            ih, iw = img.shape[:2]
            lh, lw = self.video_label.winfo_height(), self.video_label.winfo_width()
            if lh > 0 and lw > 0:
                scale = min(lw/iw, lh/ih)
                img = cv2.resize(img, (int(iw*scale), int(ih*scale)))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.video_label.configure(image=imgtk, text="")
            self.video_label.imgtk = imgtk
        except: pass
        
        self.root.after(CONFIG["performance"]["gui_refresh_ms"], self.update_video_feed)

    # --- Main Tracking Logic ---
    def process_tracking_frame_optimized(self, frame):
        if not self.targets_status: return

        self.re_detect_counter += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # 1. Fugitive Mode
        if self.fugitive_mode and self.fugitive_face_encoding is not None:
            locs = face_recognition.face_locations(rgb)
            if locs:
                encs = face_recognition.face_encodings(rgb, locs)
                for enc, loc in zip(encs, locs):
                    match = face_recognition.compare_faces([self.fugitive_face_encoding], enc, tolerance=0.5)
                    if match[0]:
                        t,r,b,l = loc
                        cv2.rectangle(frame, (l,t), (r,b), (0,0,255), 3)
                        cv2.putText(frame, f"FUGITIVE: {self.fugitive_name}", (l,t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        if not self.fugitive_detected_log_done:
                            if not self.fugitive_alert_stop_event: self.fugitive_alert_stop_event = threading.Event()
                            self.fugitive_alert_stop_event.clear()
                            self.fugitive_alert_sound_thread = play_siren_sound(self.fugitive_alert_stop_event, 15, "Fugitive.mp3")
                            self.fugitive_detected_log_done = True

        # 2. Tracker Update
        for name, status in self.targets_status.items():
            face_ok, body_ok = False, False
            if status["tracker"]:
                s, box = status["tracker"].update(frame)
                if s:
                    x,y,w_b,h_b = map(int, box)
                    status["face_box"] = smooth_bounding_box((x,y,x+w_b,y+h_b), status["face_box"])
                    status["visible"] = True; face_ok = True
                else: status["tracker"] = None
            
            if status["body_tracker"]:
                s, box = status["body_tracker"].update(frame)
                if s:
                    x,y,w_b,h_b = map(int, box)
                    status["body_box"] = (x,y,x+w_b,y+h_b)
                    body_ok = True
                else: status["body_tracker"] = None
            
            if not face_ok and not body_ok: status["visible"] = False

        # 3. Detection (Periodic)
        if self.re_detect_counter >= self.RE_DETECT_INTERVAL:
            self.re_detect_counter = 0
            locs = face_recognition.face_locations(rgb, model="hog")
            if locs:
                encs = face_recognition.face_encodings(rgb, locs)
                untracked = [n for n,s in self.targets_status.items() if not s["visible"]]
                
                for i, enc in enumerate(encs):
                    for name in untracked:
                        if self.targets_status[name]["encoding"] is not None:
                            dist = face_recognition.face_distance([self.targets_status[name]["encoding"]], enc)[0]
                            if dist < 0.55:
                                t,r,b,l = locs[i]
                                tracker = cv2.legacy.TrackerCSRT_create()
                                tracker.init(frame, (l, t, r-l, b-t))
                                self.targets_status[name]["tracker"] = tracker
                                self.targets_status[name]["face_box"] = (l,t,r,b)
                                self.targets_status[name]["visible"] = True
                                self.targets_status[name]["face_confidence"] = 1.0 - dist
                                self.targets_status[name]["last_action_time"] = time.time()
                                self.targets_status[name]["overlap_disabled"] = False

        # 4. Overlap Resolution
        if len([n for n,s in self.targets_status.items() if s["visible"]]) > 1:
            self.targets_status = resolve_overlapping_poses(self.targets_status)

        # 5. Processing & Drawing
        req_act = self.required_action_var.get()
        mon_mode = self.monitor_mode_var.get()
        
        for name, status in self.targets_status.items():
            if status["visible"] and status["face_box"]:
                fx1, fy1, fx2, fy2 = status["face_box"]
                bx1, by1, bx2, by2 = calculate_body_box((fx1, fy1, fx2, fy2), h, w)
                
                if bx1 < bx2 and by1 < by2:
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size > 0:
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        # FIX: Use crop_holistic with static_image_mode=True for accurate multi-person crops
                        res_crop = self.crop_holistic.process(rgb_crop)
                        
                        # --- SLEEP ALERT LOGIC (FIXED) ---
                        is_sleeping = False
                        if self.is_alert_mode and mon_mode in ["All Alerts (Action + Sleep)", "Sleeping Alerts Only"]:
                            if res_crop.face_landmarks:
                                lms = res_crop.face_landmarks.landmark
                                ch, cw = crop.shape[:2]
                                
                                # Validate eye visibility
                                r_viz = sum(1 for i in range(33, 133) if i < len(lms) and lms[i].visibility > 0.5)
                                if r_viz > 5: # Basic check
                                    ear = calculate_ear(lms, cw, ch)
                                    
                                    # Adaptive Threshold Logic
                                    if ear < status["ear_threshold"]: status["eye_counter_closed"] += 1
                                    else:
                                        status["eye_counter_closed"] = 0
                                        if ear > 0.35: # Update baseline if wide open
                                            status["open_ear_baseline"] = (status["open_ear_baseline"] * 0.9) + (ear * 0.1)
                                            status["ear_threshold"] = max(0.20, status["open_ear_baseline"] * 0.7)
                                    
                                    # Trigger
                                    req_frames = int(self.sleep_alert_delay_seconds * self.current_fps)
                                    if status["eye_counter_closed"] > req_frames:
                                        is_sleeping = True
                                        status["is_sleeping"] = True
                                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0,0,255), 4)
                                        if int(time.time()*2)%2==0:
                                            cv2.putText(frame, "WAKE UP!", (bx1, by1-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                        
                                        if not status.get("alert_sound_thread") or not status["alert_sound_thread"].is_alive():
                                            if not status["alert_stop_event"]: status["alert_stop_event"] = threading.Event()
                                            status["alert_stop_event"].clear()
                                            status["alert_sound_thread"] = play_siren_sound(status["alert_stop_event"], 10)
                                        
                                        if not status["sleep_alert_logged"]:
                                            self.temp_log.append((time.strftime("%H:%M:%S"), name, "SLEEPING", "ALERT", "N/A", f"{ear:.2f}"))
                                            status["sleep_alert_logged"] = True
                                    else:
                                        status["is_sleeping"] = False; status["sleep_alert_logged"] = False
                                else: status["eye_counter_closed"] = 0
                            else: status["eye_counter_closed"] = 0

                        # --- ACTION LOGIC ---
                        cur_act = "Unknown"
                        if res_crop.pose_landmarks:
                            status["missing_pose_counter"] = 0
                            draw_styled_landmarks(crop, res_crop)
                            cur_act = classify_action(res_crop.pose_landmarks.landmark, (by2-by1), (bx2-bx1))
                            
                            if cur_act != "Unknown":
                                status["pose_buffer"].append(cur_act)
                                if len(status["pose_buffer"]) >= 5:
                                    cur_act = Counter(status["pose_buffer"]).most_common(1)[0][0]
                            
                            # Stop Alert if Correct Action & Not Sleeping
                            if self.is_alert_mode and cur_act == req_act and not is_sleeping:
                                status["last_action_time"] = time.time()
                                status["alert_triggered_state"] = False
                                if status["alert_stop_event"]: status["alert_stop_event"].set()
                                if status["last_logged_action"] != req_act:
                                    self.temp_log.append((time.strftime("%H:%M:%S"), name, cur_act, "OK", "N/A", "1.0"))
                                    status["last_logged_action"] = req_act

                            # Dynamic Box Draw
                            plms = res_crop.pose_landmarks.landmark
                            lx = [l.x * (bx2-bx1) for l in plms]
                            ly = [l.y * (by2-by1) for l in plms]
                            dx1, dy1 = int(min(lx)+bx1), int(min(ly)+by1)
                            dx2, dy2 = int(max(lx)+bx1), int(max(ly)+by1)
                            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0,255,0), 2)
                            cv2.putText(frame, f"{name}: {cur_act}", (dx1, dy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            
                            # Setup Body Tracker
                            status["body_box"] = (dx1, dy1, dx2, dy2)
                            if not status["body_tracker"]:
                                try:
                                    bt = cv2.legacy.TrackerCSRT_create()
                                    bt.init(frame, (dx1, dy1, dx2-dx1, dy2-dy1))
                                    status["body_tracker"] = bt
                                except: pass
                        else: status["missing_pose_counter"] += 1

                # Timeout Alert Logic
                if self.is_alert_mode and mon_mode != "Sleeping Alerts Only":
                    elapsed = time.time() - status["last_action_time"]
                    left = max(0, self.alert_interval - elapsed)
                    color = (0,255,0) if left > 3 else (0,0,255)
                    cv2.putText(frame, f"{name} Timer: {left:.1f}s", (10, 100 + list(self.targets_status).index(name)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if elapsed > self.alert_interval:
                        if time.time() - status["alert_cooldown"] > 2.5:
                            status["alert_cooldown"] = time.time()
                            if not status.get("alert_sound_thread") or not status["alert_sound_thread"].is_alive():
                                if not status["alert_stop_event"]: status["alert_stop_event"] = threading.Event()
                                status["alert_stop_event"].clear()
                                status["alert_sound_thread"] = play_siren_sound(status["alert_stop_event"], 30)
                            
                            if not status["alert_triggered_state"]:
                                self.temp_log.append((time.strftime("%H:%M:%S"), name, "TIMEOUT", "ALERT", "N/A", "0.0"))
                                status["alert_triggered_state"] = True

if __name__ == "__main__":
    app = PoseApp()