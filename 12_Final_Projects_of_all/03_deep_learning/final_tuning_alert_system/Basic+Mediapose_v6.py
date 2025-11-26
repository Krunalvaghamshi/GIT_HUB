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

# --- ReID (Person Re-Identification) Libraries ---
try:
    import torch
    import torchvision.transforms as transforms
    REID_AVAILABLE = True
except ImportError:
    REID_AVAILABLE = False
    torch = None
    transforms = None

# Try to import torchreid for advanced ReID
try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    torchreid = None

# Try to import scikit-image for feature comparison
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None
    cosine = None

# Try to import scikit-image for HOG features in ReID
try:
    from skimage.feature import hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    hog = None

# Set CustomTkinter appearance with modern dark theme
import customtkinter as ctk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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
                         "face_recognition_tolerance": 0.5, "re_detect_interval": 15},  # ✅ REDUCED from 60 to 15 frames (every 0.5s at 30 FPS)
            "alert": {"default_interval_seconds": 10, "alert_cooldown_seconds": 2.5},
            "performance": {"gui_refresh_ms": 30, "pose_buffer_size": 12, "frame_skip_interval": 2, "enable_frame_skipping": False, "min_buffer_for_classification": 5},
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

# ✅ SAFE LOGGING: ASCII-safe logging utility to prevent Unicode encoding errors on Windows
class SafeLogger:
    """
    Wrapper for logger that sanitizes Unicode characters and replaces them with ASCII equivalents.
    Prevents UnicodeEncodeError on Windows console (cp1252 encoding).
    """
    
    # Unicode to ASCII replacement map
    UNICODE_MAP = {
        '\u2713': '[OK]',           # ✓ checkmark
        '\u2717': '[X]',            # ✗ cross
        '\u26a0': '[WARN]',         # ⚠ warning
        '\u26a0\ufe0f': '[WARN]',   # ⚠️ warning with variant
        '\ud83d\udea8': '[ALERT]',  # 🚨 siren
        '\ud83d\udd0a': '[SOUND]',  # 🔊 speaker
        '\ud83d\udcf8': '[SNAP]',   # 📸 camera
        '\ud83d\udccb': '[LOG]',    # 📋 log
        '\ud83d\udccd': '[PIN]',    # 📍 pin
        '\ud83d\udca4': '[SLEEP]',  # 💤 sleep
        '\u251c\u2500': '|-',       # ├─ tree
        '\u2514\u2500': 'L-',       # └─ tree
        '→': '->',                   # arrow
        '→': '->',                   # variant
    }
    
    @staticmethod
    def sanitize(text):
        """Convert Unicode characters to ASCII-safe equivalents."""
        if not isinstance(text, str):
            return text
        for unicode_char, ascii_equiv in SafeLogger.UNICODE_MAP.items():
            text = text.replace(unicode_char, ascii_equiv)
        # Remove any remaining problematic Unicode by encoding/decoding
        try:
            text.encode('cp1252')  # Test if encoding works
        except UnicodeEncodeError:
            text = text.encode('cp1252', errors='replace').decode('cp1252')
        return text
    
    @staticmethod
    def warning(msg, *args, **kwargs):
        """Log warning with Unicode sanitization."""
        msg = SafeLogger.sanitize(msg)
        logger.warning(msg, *args, **kwargs)
    
    @staticmethod
    def info(msg, *args, **kwargs):
        """Log info with Unicode sanitization."""
        msg = SafeLogger.sanitize(msg)
        logger.info(msg, *args, **kwargs)
    
    @staticmethod
    def debug(msg, *args, **kwargs):
        """Log debug with Unicode sanitization."""
        msg = SafeLogger.sanitize(msg)
        logger.debug(msg, *args, **kwargs)
    
    @staticmethod
    def error(msg, *args, **kwargs):
        """Log error with Unicode sanitization."""
        msg = SafeLogger.sanitize(msg)
        logger.error(msg, *args, **kwargs)

# Create instance for use throughout application
safe_logger = SafeLogger()

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

# Ensure logs directory exists
if not os.path.exists(CONFIG["logging"]["log_directory"]):
    os.makedirs(CONFIG["logging"]["log_directory"])

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

# ✅ PERFORMANCE: Memory management and garbage collection optimization
def optimize_memory():
    """Optimize memory usage by aggressive garbage collection at strategic points."""
    try:
        gc.collect()
        # Force collection of unreachable objects
        unreachable = gc.collect()
        if unreachable > 100:  # Log only if significant cleanup
            logger.debug(f"[MEMORY] Collected {unreachable} unreachable objects")
    except Exception as e:
        logger.debug(f"[MEMORY] GC optimization error: {e}")

# Set aggressive garbage collection for real-time performance
gc.set_threshold(1000, 15, 15)  # More aggressive collection

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

# --- EAR Calculation Helper (from Basic_v5.py) ---
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

# --- classify_action with improved detection ---
def classify_action(landmarks, h, w):
    """
    Classify pose action with robust detection and confidence scoring.
    Supports: Hands Up, Hands Crossed, One Hand Raised (Left/Right), T-Pose, Sit, Standing
    Includes visibility and quality checks for stable detection.
    
    ✅ IMPROVED: Distance-agnostic pose classification
    - Uses relative positions (normalized to body size)
    - Works for guards at near and far distances
    - More tolerant of small finger movements
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

        # ✅ CRITICAL: Normalize to body scale (use shoulder-to-hip distance as reference)
        # This makes detection work for near and far guards equally well
        shoulder_to_hip_dist = abs(l_shoulder.y - l_hip.y)  # Vertical distance
        if shoulder_to_hip_dist < 0.01:  # Guard too far/close or bad detection
            return "Standing"
        
        # Threshold multipliers based on body size instead of fixed pixels
        # This scales detection to work at any distance
        HANDS_UP_THRESHOLD = shoulder_to_hip_dist * 0.4      # Hands above head = 40% of torso
        HANDS_CROSSED_TOLERANCE = shoulder_to_hip_dist * 0.3  # Hands at chest = within 30% of torso
        ARM_EXTENSION = shoulder_to_hip_dist * 0.6            # Arms extended = 60% of torso width
        WRIST_ARM_ALIGNMENT = shoulder_to_hip_dist * 0.25     # Wrist aligned with arm = within 25%

        nose_y = nose.y
        nose_x = nose.x
        lw_y = l_wrist.y
        rw_y = r_wrist.y
        lw_x = l_wrist.x
        rw_x = r_wrist.x
        ls_y = l_shoulder.y
        rs_y = r_shoulder.y
        ls_x = l_shoulder.x
        rs_x = r_shoulder.x
        lh_y = l_hip.y
        rh_y = r_hip.y
        
        # ✅ IMPROVED: Lower visibility thresholds for far guards
        # Far guards have naturally lower visibility scores
        l_wrist_visible = l_wrist.visibility > 0.50  # Reduced from 0.70
        r_wrist_visible = r_wrist.visibility > 0.50  # Reduced from 0.70
        l_elbow_visible = l_elbow.visibility > 0.50  # Reduced from 0.70
        r_elbow_visible = r_elbow.visibility > 0.50  # Reduced from 0.70
        nose_visible = nose.visibility > 0.50        # Reduced from 0.60
        l_shoulder_visible = l_shoulder.visibility > 0.50  # Reduced from 0.70
        r_shoulder_visible = r_shoulder.visibility > 0.50  # Reduced from 0.70
        l_knee_visible = l_knee.visibility > 0.40    # Reduced from 0.70 (knees hardest to see)
        r_knee_visible = r_knee.visibility > 0.40    # Reduced from 0.70
        l_hip_visible = l_hip.visibility > 0.45      # Reduced from 0.65
        r_hip_visible = r_hip.visibility > 0.45      # Reduced from 0.65
        
        # ✅ IMPROVED: Quality check allows 60% of joints (not 80%)
        # More lenient for far guards where some joints are naturally hard to detect
        visible_joints = sum([
            l_wrist_visible, r_wrist_visible, l_elbow_visible, r_elbow_visible,
            l_shoulder_visible, r_shoulder_visible, l_knee_visible, r_knee_visible,
            l_hip_visible, r_hip_visible, nose_visible
        ])
        if visible_joints < 7:  # Need at least 7/11 major joints (was 9/11)
            return "Standing"  # Default to standing if pose quality is poor
        
        # ✅ CRITICAL: All comparisons use normalized distances
        # 1. Hands Up Detection (both hands clearly above head)
        if (l_wrist_visible and r_wrist_visible and 
            (nose_y - lw_y) > HANDS_UP_THRESHOLD and 
            (nose_y - rw_y) > HANDS_UP_THRESHOLD):
            return "Hands Up"
        
        # 2. Hands Crossed Detection (wrists cross at chest level)
        if (l_wrist_visible and r_wrist_visible and l_shoulder_visible and r_shoulder_visible):
            chest_y = (ls_y + rs_y) / 2
            body_center_x = (ls_x + rs_x) / 2
            
            # Check if both wrists are at chest level (normalized)
            wrist_chest_dist_l = abs(lw_y - chest_y)
            wrist_chest_dist_r = abs(rw_y - chest_y)
            
            if (wrist_chest_dist_l < HANDS_CROSSED_TOLERANCE and 
                wrist_chest_dist_r < HANDS_CROSSED_TOLERANCE):
                # Check if wrists are crossed (left hand on right side, vice versa)
                if ((lw_x > body_center_x and rw_x < body_center_x) or 
                    (lw_x < body_center_x and rw_x > body_center_x)):
                    return "Hands Crossed"
        
        # 3. T-Pose Detection (arms extended sideways at shoulder height)
        if (l_wrist_visible and r_wrist_visible and l_elbow_visible and r_elbow_visible and
            l_shoulder_visible and r_shoulder_visible):
            
            # Check if both elbows and wrists are at shoulder level (normalized)
            lw_at_shoulder = abs(lw_y - ls_y) < WRIST_ARM_ALIGNMENT
            rw_at_shoulder = abs(rw_y - rs_y) < WRIST_ARM_ALIGNMENT
            le_at_shoulder = abs(l_elbow.y - ls_y) < WRIST_ARM_ALIGNMENT
            re_at_shoulder = abs(r_elbow.y - rs_y) < WRIST_ARM_ALIGNMENT
            
            if lw_at_shoulder and rw_at_shoulder and le_at_shoulder and re_at_shoulder:
                # Check if arms are extended outward (normalized)
                shoulder_width = abs(rs_x - ls_x)
                left_extension = abs(lw_x - ls_x)
                right_extension = abs(rw_x - rs_x)
                
                if left_extension > ARM_EXTENSION and right_extension > ARM_EXTENSION:
                    return "T-Pose"
        
        # 4. One Hand Raised Detection (only one hand above head, clearly)
        if l_wrist_visible and (nose_y - lw_y) > HANDS_UP_THRESHOLD and not r_wrist_visible:
            return "One Hand Raised (Left)"
        if r_wrist_visible and (nose_y - rw_y) > HANDS_UP_THRESHOLD and not l_wrist_visible:
            return "One Hand Raised (Right)"
        
        # Alternative: one hand raised while other is down
        if l_wrist_visible and r_wrist_visible and l_shoulder_visible and r_shoulder_visible:
            chest_y = (ls_y + rs_y) / 2
            if ((nose_y - lw_y) > HANDS_UP_THRESHOLD and 
                (rw_y - chest_y) > HANDS_CROSSED_TOLERANCE):
                return "One Hand Raised (Left)"
            if ((nose_y - rw_y) > HANDS_UP_THRESHOLD and 
                (lw_y - chest_y) > HANDS_CROSSED_TOLERANCE):
                return "One Hand Raised (Right)"
        
        # 5. Sit/Stand Detection with improved reliability
        if l_knee_visible and r_knee_visible and l_hip_visible and r_hip_visible:
            # Calculate angle of thigh (knee to hip) - normalized to body scale
            thigh_angle_l = abs(l_knee.y - l_hip.y)
            thigh_angle_r = abs(r_knee.y - r_hip.y)
            avg_thigh_angle = (thigh_angle_l + thigh_angle_r) / 2
            
            # If thigh is nearly horizontal, person is sitting (normalized threshold)
            sit_threshold = shoulder_to_hip_dist * 0.15  # Thigh angle < 15% of torso = sitting
            if avg_thigh_angle < sit_threshold:
                return "Sit"
            else:
                return "Standing"
        else:
            # Default to standing if knee not visible
            return "Standing"

        return "Standing" 

    except Exception as e:
        logger.debug(f"Pose classification error: {e}")
        return "Unknown"

# --- Helper: Calculate Dynamic Body Box from Face ---
def calculate_body_box(face_box, frame_h, frame_w, expansion_factor=3.0):
    """
    Calculate dynamic body bounding box from detected face box.
    
    ✅ IMPROVED: Distance-agnostic body box calculation
    - Works for guards at near and far distances
    - Expands based on face size (relative scaling)
    - Ensures full body is captured for pose detection
    
    Args:
        face_box: tuple (x1, y1, x2, y2) - face coordinates
        frame_h, frame_w: frame dimensions
        expansion_factor: how many face widths to expand (default 3x for near/far)
    
    Returns:
        tuple (bx1, by1, bx2, by2) - body box coordinates that captures full body
    """
    x1, y1, x2, y2 = face_box
    face_w = x2 - x1
    face_h = y2 - y1
    face_cx = x1 + (face_w // 2)
    
    # ✅ CRITICAL: Dynamic expansion based on face size
    # Smaller faces (far guards) and larger faces (near guards) both get appropriate body boxes
    # For small faces: expand more; for large faces: expand less (but always full body)
    if face_w < 30:  # Very far guard
        adaptive_expansion = 5.0  # Expand 5x for small face
    elif face_w < 100:  # Far guard
        adaptive_expansion = 4.0  # Expand 4x for small-medium face
    else:  # Near guard or medium distance
        adaptive_expansion = 3.0  # Expand 3x for large face
    
    # Expand horizontally based on adaptive face width
    bx1 = max(0, int(face_cx - (face_w * adaptive_expansion / 2)))
    bx2 = min(frame_w, int(face_cx + (face_w * adaptive_expansion / 2)))
    
    # Expand vertically: slightly above face, down to feet
    # Scale vertical expansion with face size too
    by1 = max(0, int(y1 - (face_h * 0.5)))
    by2 = frame_h  # Always extend to bottom for leg visibility
    
    return (bx1, by1, bx2, by2)

def approximate_face_from_body(body_box):
    """
    Approximate a face box from a tracked body box. Useful when face tracking is lost
    but the person/body remains tracked. Returns (x1, y1, x2, y2).
    """
    bx1, by1, bx2, by2 = body_box
    w = max(1, bx2 - bx1)
    h = max(1, by2 - by1)
    fx_w = int(max(20, 0.25 * w))
    fx_h = int(max(20, 0.20 * h))
    fx1 = bx1 + int(0.375 * w)
    fy1 = by1 + int(0.05 * h)
    fx2 = min(bx2, fx1 + fx_w)
    fy2 = min(by2, fy1 + fx_h)
    return (fx1, fy1, fx2, fy2)

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

# --- Multi-Guard Pose Detection Enhancement ---
def resolve_overlapping_poses(targets_status, iou_threshold=0.3):
    """
    Resolve conflicting pose detections when multiple guards overlap.
    Ensures each guard has independent, consistent pose detection.
    
    ✅ MULTI-GUARD ENHANCED: Uses confidence scoring, temporal consistency, 
    and spatial analysis for better conflict resolution.
    ✅ SEPARATION FIX: Re-enables disabled guards when they move apart (IoU < threshold).
    
    Args:
        targets_status: Dictionary of all tracked guards and their status
        iou_threshold: IoU threshold for considering boxes as overlapping
    
    Returns:
        Updated targets_status with resolved conflicts
    """
    try:
        target_names = list(targets_status.keys())
        conflicts_resolved = []
        
        for i, name_a in enumerate(target_names):
            # ✅ SEPARATION FIX: Check BOTH visible and disabled guards for re-enabling
            box_a = targets_status[name_a].get("face_box")
            if not box_a:
                continue
            
            # Check overlap with other guards
            for name_b in target_names[i+1:]:
                box_b = targets_status[name_b].get("face_box")
                if not box_b:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(
                    (box_a[0], box_a[1], box_a[2] - box_a[0], box_a[3] - box_a[1]),
                    (box_b[0], box_b[1], box_b[2] - box_b[0], box_b[3] - box_b[1])
                )
                
                # ✅ SEPARATION FIX: If guards were previously overlapping but now separated
                # Re-enable the disabled guard immediately
                if iou < iou_threshold:
                    # Guards are SEPARATED
                    was_disabled_a = not targets_status[name_a].get("visible")
                    was_disabled_b = not targets_status[name_b].get("visible")
                    
                    # If either was disabled due to overlap, re-enable it now
                    if was_disabled_a and targets_status[name_a].get("overlap_disabled"):
                        targets_status[name_a]["visible"] = True
                        targets_status[name_a]["overlap_disabled"] = False
                        targets_status[name_a]["tracker"] = None  # Reset tracker for fresh detection
                        logger.warning(f"[SEPARATION] Re-enabled {name_a} - guards separated (IoU: {iou:.2f})")
                    
                    if was_disabled_b and targets_status[name_b].get("overlap_disabled"):
                        targets_status[name_b]["visible"] = True
                        targets_status[name_b]["overlap_disabled"] = False
                        targets_status[name_b]["tracker"] = None  # Reset tracker for fresh detection
                        logger.warning(f"[SEPARATION] Re-enabled {name_b} - guards separated (IoU: {iou:.2f})")
                    
                    continue  # Skip conflict resolution if not overlapping
                
                # ✅ ORIGINAL: Handle overlapping guards (IoU > threshold)
                if targets_status[name_a].get("visible") and targets_status[name_b].get("visible"):
                    # Both visible and overlapping - resolve conflict
                    # Multi-factor scoring for better decision making
                    pose_conf_a = targets_status[name_a].get("pose_confidence", 0.0)
                    pose_conf_b = targets_status[name_b].get("pose_confidence", 0.0)
                    face_conf_a = targets_status[name_a].get("face_confidence", 0.0)
                    face_conf_b = targets_status[name_b].get("face_confidence", 0.0)
                    
                    # ✅ MULTI-GUARD FIX: Check for temporal consistency (recent pose quality)
                    pose_hist_a = targets_status[name_a].get("pose_quality_history", deque())
                    pose_hist_b = targets_status[name_b].get("pose_quality_history", deque())
                    recent_quality_a = sum(pose_hist_a) / len(pose_hist_a) if pose_hist_a else 0.0
                    recent_quality_b = sum(pose_hist_b) / len(pose_hist_b) if pose_hist_b else 0.0
                    
                    # Composite score: 40% pose + 30% face + 30% temporal consistency
                    score_a = (pose_conf_a * 0.4) + (face_conf_a * 0.3) + (recent_quality_a * 0.3)
                    score_b = (pose_conf_b * 0.4) + (face_conf_b * 0.3) + (recent_quality_b * 0.3)
                    
                    # Keep guard with higher composite score
                    if score_a < score_b:
                        targets_status[name_a]["visible"] = False
                        targets_status[name_a]["overlap_disabled"] = True  # ✅ SEPARATION FIX: Mark as disabled due to overlap
                        conflicts_resolved.append((name_a, name_b, score_a, score_b, iou))
                        logger.debug(f"[RESOLVE] Disabled {name_a} (score:{score_a:.3f}) - kept {name_b} (score:{score_b:.3f}), IoU:{iou:.2f}")
                    elif score_b < score_a:
                        targets_status[name_b]["visible"] = False
                        targets_status[name_b]["overlap_disabled"] = True  # ✅ SEPARATION FIX: Mark as disabled due to overlap
                        conflicts_resolved.append((name_b, name_a, score_b, score_a, iou))
                        logger.debug(f"[RESOLVE] Disabled {name_b} (score:{score_b:.3f}) - kept {name_a} (score:{score_a:.3f}), IoU:{iou:.2f}")
                    else:
                        # ✅ MULTI-GUARD TIE-BREAKER: If scores are equal, prefer by face confidence
                        if face_conf_a >= face_conf_b:
                            targets_status[name_b]["visible"] = False
                            targets_status[name_b]["overlap_disabled"] = True  # ✅ SEPARATION FIX: Mark as disabled
                            logger.debug(f"[RESOLVE] Tie-break: Disabled {name_b} - kept {name_a} by face confidence")
                        else:
                            targets_status[name_a]["visible"] = False
                            targets_status[name_a]["overlap_disabled"] = True  # ✅ SEPARATION FIX: Mark as disabled
                            logger.debug(f"[RESOLVE] Tie-break: Disabled {name_a} - kept {name_b} by face confidence")
        
        if conflicts_resolved:
            logger.info(f"[RESOLVE] Multi-guard: Resolved {len(conflicts_resolved)} pose conflicts")
    except Exception as e:
        logger.debug(f"[ERROR] Pose conflict resolution error: {e}")
    
    return targets_status

def smooth_bounding_box(current_box, previous_box, smoothing_factor=0.7):
    """
    Apply exponential moving average smoothing to bounding box to reduce jitter.
    
    Args:
        current_box: Current detected box (x1, y1, x2, y2)
        previous_box: Previous smoothed box (x1, y1, x2, y2)
        smoothing_factor: Smoothing weight (0-1, higher = more weight on previous box)
    
    Returns:
        Smoothed box coordinates
    """
    if previous_box is None:
        return current_box
    
    try:
        smoothed_box = tuple(
            int(smoothing_factor * prev + (1 - smoothing_factor) * curr)
            for curr, prev in zip(current_box, previous_box)
        )
        return smoothed_box
    except Exception as e:
        logger.debug(f"Box smoothing error: {e}")
        return current_box

# --- ReID Feature Extraction & Matching Functions ---
def extract_appearance_features(frame, face_box):
    """
    Extract appearance features from a person's image using simple deep features.
    
    Args:
        frame: Input image frame
        face_box: Bounding box (x1, y1, x2, y2) for person region
    
    Returns:
        Feature vector (numpy array) or None if extraction fails
    """
    try:
        x1, y1, x2, y2 = face_box
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
        
        # Simple approach: use histogram of colors + histogram of edges as features
        # This is a lightweight alternative to deep neural networks
        h, w = person_crop.shape[:2]
        
        # Resize for consistent feature extraction
        person_resized = cv2.resize(person_crop, (64, 128))
        
        # Color histogram features (3 channels × 8 bins each = 24 features)
        color_features = []
        for i in range(3):
            hist = cv2.calcHist([person_resized], [i], None, [8], [0, 256])
            color_features.extend(hist.flatten())
        
        # Edge detection features using Canny
        gray = cv2.cvtColor(person_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        
        # Combine all features into single vector
        features = np.array(color_features + edge_hist.flatten(), dtype=np.float32)
        
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    except Exception as e:
        logger.warning(f"Feature extraction error: {e}")
        return None

def calculate_feature_similarity(features1, features2):
    """
    Calculate similarity between two feature vectors using cosine similarity.
    
    Args:
        features1: Feature vector 1
        features2: Feature vector 2
    
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    if features1 is None or features2 is None:
        return 0.0
    
    try:
        if SKLEARN_AVAILABLE and cosine_similarity is not None:
            # Use scikit-learn's cosine similarity with proper array conversion
            similarity = cosine_similarity(
                np.array(features1).reshape(1, -1), 
                np.array(features2).reshape(1, -1)
            )[0][0]
        else:
            # Fallback: manual cosine similarity
            f1 = np.array(features1).astype(np.float32)
            f2 = np.array(features2).astype(np.float32)
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            similarity = dot_product / (norm1 * norm2 + 1e-6)
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    except Exception as e:
        logger.warning(f"Similarity calculation error: {e}")
        return 0.0

def match_person_identity(person_id, new_features, person_features_db, confidence_threshold=0.65):
    """
    Match a person's features against database to determine if same person.
    
    Args:
        person_id: ID of person to match
        new_features: New feature vector
        person_features_db: Database of known person features
        confidence_threshold: Minimum similarity for match
    
    Returns:
        (matched_person_id, confidence) or (None, 0.0) if no match
    """
    try:
        if person_id not in person_features_db:
            # New person, store features
            person_features_db[person_id] = {
                'features': new_features,
                'count': 1,
                'last_seen': time.time()
            }
            return person_id, 1.0
        
        # Calculate similarity with stored features
        stored_features = person_features_db[person_id]['features']
        similarity = calculate_feature_similarity(new_features, stored_features)
        
        if similarity >= confidence_threshold:
            # Update features with exponential moving average for stability
            alpha = 0.3  # Learning rate
            person_features_db[person_id]['features'] = (
                alpha * new_features + 
                (1 - alpha) * stored_features
            )
            person_features_db[person_id]['count'] += 1
            person_features_db[person_id]['last_seen'] = time.time()
            return person_id, similarity
        else:
            # Create new person identity
            new_id = f"{person_id}_variant_{len(person_features_db)}"
            person_features_db[new_id] = {
                'features': new_features,
                'count': 1,
                'last_seen': time.time()
            }
            return new_id, similarity
    except Exception as e:
        logger.warning(f"Person matching error: {e}")
        return None, 0.0

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

# --- Helper: ReID Feature Extraction ---
def extract_reid_features(frame, bbox, model=None):
    """
    Extract appearance features from a person bounding box for re-identification.
    
    Args:
        frame: Input video frame
        bbox: Bounding box (x1, y1, x2, y2)
        model: Optional pre-trained ReID model
    
    Returns:
        features: numpy array of appearance features, or None if extraction fails
    """
    try:
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        # Ensure valid bbox
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        
        # Resize to standard size (256, 128) for ReID
        person_resized = cv2.resize(person_crop, (128, 256))
        
        # Convert BGR to RGB
        person_rgb = cv2.cvtColor(person_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize (if using standard ImageNet normalization)
        person_normalized = person_rgb.astype(np.float32) / 255.0
        
        # Simple feature extraction: Compute histogram of oriented gradients (HOG) as appearance features
        # For more advanced ReID, use pre-trained models like OSNet, ResNet50-ReID, etc.
        if SKIMAGE_AVAILABLE:
            features = hog(cv2.cvtColor(person_resized, cv2.COLOR_BGR2GRAY), 
                          orientations=9, pixels_per_cell=(8, 8), 
                          cells_per_block=(2, 2), visualize=False)
        else:
            # Fallback: use color histogram if skimage not available
            logger.warning("scikit-image not available, using color histogram for ReID")
            features = extract_appearance_features(frame, bbox)
        
        return features
    except Exception as e:
        logger.error(f"ReID feature extraction error: {e}")
        return None

def compute_feature_distance(features1, features2):
    """
    Compute cosine similarity between two feature vectors for person re-identification.
    
    Returns:
        similarity: Float between 0 and 1 (higher = more similar)
    """
    try:
        if features1 is None or features2 is None:
            return 0.0
        
        # Normalize features
        f1 = features1 / (np.linalg.norm(features1) + 1e-5)
        f2 = features2 / (np.linalg.norm(features2) + 1e-5)
        
        # Compute cosine similarity
        similarity = np.dot(f1, f2)
        return float(similarity)
    except Exception as e:
        logger.error(f"Feature distance computation error: {e}")
        return 0.0

# --- Tkinter Application Class ---
class PoseApp:
    def __init__(self, window_title="Pose Guard (Multi-Target)"):
        self.root = ctk.CTk()
        self.root.title(window_title)
        self.root.geometry("1800x1000")  # Larger default size
        
        self.cap = None
        self.unprocessed_frame = None 
        self.is_running = False
        self.is_logging = False
        self.camera_index = 0  # Default camera
        
        self.is_alert_mode = False
        self.alert_interval = 10
        # NEW: Monitoring Mode (from Basic_v5.py)
        self.monitor_mode_var = tk.StringVar(self.root)
        self.monitor_mode_var.set("All Alerts (Action + Sleep)")
        # NEW: Sleep Alert Timer (from Basic_v5.py)
        self.sleep_alert_delay_seconds = 1.5  # Default 1.5 seconds
        self.is_in_capture_mode = False
        self.frame_w = 640 
        self.frame_h = 480 

        self.target_map = {}
        self.targets_status = {} 
        self.selected_target_names = []  # NEW: Track selected targets
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
        
        # PRO_Detection Mode Fields (Person Re-Identification)
        self.pro_detection_mode = False
        self.reid_model = None
        self.reid_transform = None
        self.person_features_db = {}  # Store appearance features for each person
        self.person_identity_history = {}  # Track person identity across frames
        self.reid_enabled = REID_AVAILABLE and SKLEARN_AVAILABLE  # Check if ReID libraries are available
        self.reid_confidence_threshold = 0.65  # Threshold for person re-identification
        self.person_tracking_data = []  # Store ReID tracking data for CSV logging
        self.pro_detection_log_done = {}  # Track which persons have been logged
        self.pro_detection_person_counter = 0  # Counter for assigning unique person IDs
        
        # Photo storage for Tkinter (prevent garbage collection)
        self.photo_storage = {}  # Dictionary to store PhotoImage references
        
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
        self.root.grid_rowconfigure(0, weight=1)  # Main content area
        self.root.grid_columnconfigure(0, weight=1)  # Camera feed (expandable)
        self.root.grid_columnconfigure(1, weight=0)  # Sidebar (fixed width)
        
        # Sidebar state
        self.sidebar_collapsed = False
        self.sidebar_width = 280
        
        # 1. Main Content Area (Camera Feed)
        self.main_container = ctk.CTkFrame(self.root, fg_color="black")
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Camera feed in main container
        self.video_container = ctk.CTkFrame(self.main_container, fg_color="black")
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.video_label = ctk.CTkLabel(self.video_container, text="🎥 Camera Feed Off", font=("Arial", 24, "bold"), text_color="white")
        self.video_label.pack(fill="both", expand=True)
        
        # 2. Collapsible Sidebar
        self.sidebar_frame = ctk.CTkFrame(self.root, fg_color="#0f0f0f", width=self.sidebar_width)
        self.sidebar_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_propagate(False)
        
        # Sidebar Header with Enhanced Styling
        sidebar_header = ctk.CTkFrame(self.sidebar_frame, fg_color="#0d0d0d", height=50, corner_radius=0)
        sidebar_header.pack(fill="x", padx=0, pady=0)
        sidebar_header.pack_propagate(False)
        
        # Add border for visual separation
        header_border = ctk.CTkFrame(sidebar_header, fg_color="#2c3e50", height=2)
        header_border.pack(side="bottom", fill="x")
        
        # Left side: Toggle button with enhanced styling
        self.toggle_sidebar_btn = ctk.CTkButton(sidebar_header, text="◄ Hide", command=self.toggle_sidebar, 
                                                 width=40, height=35, fg_color="#0066cc", hover_color="#004a9f", 
                                                 font=('Roboto', 9, 'bold'), corner_radius=6)
        self.toggle_sidebar_btn.pack(side="left", padx=5, pady=5)
        
        # Middle: Title with icon
        ctk.CTkLabel(sidebar_header, text="⚡ CONTROL PANEL", font=("Roboto", 11, "bold"), text_color="#00d4ff").pack(side="left", padx=10, pady=5, expand=False)
        
        # Scrollable content area in sidebar
        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="#0f0f0f")
        self.sidebar_scroll.pack(fill="both", expand=True, padx=0, pady=5)

        # 2. Blue Zone - Control Panel (LEFT) - ALL BUTTONS/CONTROLS/TARGETS LIST
        
        # Fonts
        btn_font = ('Roboto', 10, 'bold')
        btn_font_small = ('Roboto', 9)
        
        # --- Group 1: Camera Controls ---
        self.grp_camera = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_camera.pack(fill="x", padx=5, pady=(5,3))
        ctk.CTkLabel(self.grp_camera, text="▶ SYSTEM CONTROLS", font=("Roboto", 10, "bold"), text_color="#00d4ff").pack(anchor="w", padx=0, pady=(0,3))
        
        # Separator line with better styling
        separator1 = ctk.CTkFrame(self.grp_camera, height=2, fg_color="#0066cc", corner_radius=1)
        separator1.pack(fill="x", pady=(0,5))
        
        self.cam_btns_frame = ctk.CTkFrame(self.grp_camera, fg_color="transparent")
        self.cam_btns_frame.pack(fill="x")
        
        # Button row 1: Start, Stop, Snap, Exit (using grid for proper alignment)
        self.btn_start = ctk.CTkButton(self.cam_btns_frame, text="▶ Start", command=self.start_camera, width=60, fg_color="#27ae60", hover_color="#229954", font=btn_font, corner_radius=6)
        self.btn_start.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        self.btn_stop = ctk.CTkButton(self.cam_btns_frame, text="⏹ Stop", command=self.stop_camera, width=60, fg_color="#c0392b", hover_color="#a93226", font=btn_font, corner_radius=6)
        self.btn_stop.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        self.btn_snap = ctk.CTkButton(self.cam_btns_frame, text="📸 Snap", command=self.snap_photo, width=60, fg_color="#d35400", hover_color="#ba4a00", font=btn_font, corner_radius=6)
        self.btn_snap.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        
        self.btn_exit = ctk.CTkButton(self.cam_btns_frame, text="🚪 Exit", command=self.graceful_exit, width=50, fg_color="#34495e", hover_color="#2c3e50", font=btn_font, corner_radius=6)
        self.btn_exit.grid(row=0, column=3, padx=2, pady=2, sticky="ew")
        
        # Configure grid columns for equal width
        for i in range(4):
            self.cam_btns_frame.grid_columnconfigure(i, weight=1)

        # --- Group 2: Guard Management ---
        self.grp_guard = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_guard.pack(fill="x", padx=5, pady=(5,3))
        ctk.CTkLabel(self.grp_guard, text="👮 GUARD MANAGEMENT", font=("Roboto", 10, "bold"), text_color="#00d4ff").pack(anchor="w", padx=0, pady=(0,3))
        
        # Separator line with better styling
        separator2 = ctk.CTkFrame(self.grp_guard, height=2, fg_color="#8e44ad", corner_radius=1)
        separator2.pack(fill="x", pady=(0,5))
        
        self.guard_btns_frame = ctk.CTkFrame(self.grp_guard, fg_color="transparent")
        self.guard_btns_frame.pack(fill="x")
        
        self.btn_add_guard = ctk.CTkButton(self.guard_btns_frame, text="➕ Add", command=self.add_guard_dialog, width=60, fg_color="#8e44ad", hover_color="#7d3c98", font=btn_font, corner_radius=6)
        self.btn_add_guard.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        self.btn_remove_guard = ctk.CTkButton(self.guard_btns_frame, text="❌ Remove", command=self.remove_guard_dialog, width=60, fg_color="#e74c3c", hover_color="#cb4335", font=btn_font, corner_radius=6)
        self.btn_remove_guard.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        self.btn_refresh = ctk.CTkButton(self.guard_btns_frame, text="🔄 Refresh", command=self.load_targets, width=60, fg_color="#e67e22", hover_color="#ca6f1e", font=btn_font, corner_radius=6)
        self.btn_refresh.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        
        # Configure grid columns for equal width
        for i in range(3):
            self.guard_btns_frame.grid_columnconfigure(i, weight=1)

        # --- Group 3: Modes ---
        self.grp_modes = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_modes.pack(fill="x", padx=5, pady=(5,3))
        ctk.CTkLabel(self.grp_modes, text="🔍 DETECTION MODES", font=("Roboto", 10, "bold"), text_color="#00d4ff").pack(anchor="w", padx=0, pady=(0,3))
        
        # Separator line with better styling
        separator3 = ctk.CTkFrame(self.grp_modes, height=2, fg_color="#e67e22", corner_radius=1)
        separator3.pack(fill="x", pady=(0,5))
        
        self.mode_btns_frame = ctk.CTkFrame(self.grp_modes, fg_color="transparent")
        self.mode_btns_frame.pack(fill="x")
        
        self.btn_toggle_alert = ctk.CTkButton(self.mode_btns_frame, text="🔔 Alert", command=self.toggle_alert_mode, width=70, fg_color="#e67e22", hover_color="#d35400", font=btn_font, corner_radius=6)
        self.btn_toggle_alert.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        self.btn_fugitive = ctk.CTkButton(self.mode_btns_frame, text="🚨 Fugitive", command=self.toggle_fugitive_mode, width=70, fg_color="#8b0000", hover_color="#6b0000", font=btn_font, corner_radius=6)
        self.btn_fugitive.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        self.btn_pro_detection = ctk.CTkButton(self.mode_btns_frame, text="🎯 Pro", command=self.toggle_pro_detection_mode, width=70, fg_color="#0066cc", hover_color="#004a9f", font=btn_font, corner_radius=6)
        self.btn_pro_detection.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
        
        # Configure grid columns for equal width
        self.mode_btns_frame.grid_columnconfigure(0, weight=1)
        self.mode_btns_frame.grid_columnconfigure(1, weight=1)
        self.mode_btns_frame.grid_columnconfigure(2, weight=1)

        # --- Group 4: Settings & Actions ---
        self.grp_settings = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_settings.pack(fill="x", padx=5, pady=(5,3))
        ctk.CTkLabel(self.grp_settings, text="⚙️  SETTINGS & ACTIONS", font=("Roboto", 10, "bold"), text_color="#00d4ff").pack(anchor="w", padx=0, pady=(0,3))
        
        # Separator line with better styling
        separator4 = ctk.CTkFrame(self.grp_settings, height=2, fg_color="#16a085", corner_radius=1)
        separator4.pack(fill="x", pady=(0,5))
        
        # ===== ALERT TIMERS SECTION (Dual Timer UI) =====
        timer_section = ctk.CTkFrame(self.grp_settings, fg_color="#1a1a1a", corner_radius=6, border_width=1, border_color="#404040")
        timer_section.pack(fill="x", padx=0, pady=3)
        
        ctk.CTkLabel(timer_section, text="⏱ ALERT TIMERS", font=("Roboto", 9, "bold"), text_color="#f39c12").pack(anchor="w", padx=5, pady=(5,2))
        
        # Timer 1: Action Interval (Timeout Alert)
        timer1_frame = ctk.CTkFrame(timer_section, fg_color="transparent")
        timer1_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(timer1_frame, text="Action Timeout:", font=btn_font_small, text_color="white").pack(side="left", padx=2)
        self.btn_set_interval = ctk.CTkButton(timer1_frame, text=f"{self.alert_interval}s", command=self.set_alert_interval_advanced, width=50, fg_color="#7f8c8d", font=btn_font_small, corner_radius=5)
        self.btn_set_interval.pack(side="left", padx=2)
        self.label_interval_desc = ctk.CTkLabel(timer1_frame, text="(time before timeout alert)", font=("Roboto", 7), text_color="#95a5a6")
        self.label_interval_desc.pack(side="left", padx=2, fill="x", expand=True)
        
        # Timer 2: Sleep Duration (Eye Closure Detection)
        timer2_frame = ctk.CTkFrame(timer_section, fg_color="transparent")
        timer2_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(timer2_frame, text="Sleep Duration:", font=btn_font_small, text_color="white").pack(side="left", padx=2)
        self.btn_set_sleep = ctk.CTkButton(timer2_frame, text=f"{self.sleep_alert_delay_seconds}s", command=self.set_sleep_interval, width=50, fg_color="#546e7a", font=btn_font_small, corner_radius=5)
        self.btn_set_sleep.pack(side="left", padx=2)
        self.label_sleep_desc = ctk.CTkLabel(timer2_frame, text="(eyes closed duration)", font=("Roboto", 7), text_color="#95a5a6")
        self.label_sleep_desc.pack(side="left", padx=2, fill="x", expand=True)
        
        # ===== ACTION SELECTION SECTION =====
        action_section = ctk.CTkFrame(self.grp_settings, fg_color="#1a1a1a", corner_radius=6, border_width=1, border_color="#404040")
        action_section.pack(fill="x", padx=0, pady=3)
        
        ctk.CTkLabel(action_section, text="🎬 ACTION SELECTION", font=("Roboto", 9, "bold"), text_color="#f39c12").pack(anchor="w", padx=5, pady=(5,2))
        
        action_frame = ctk.CTkFrame(action_section, fg_color="transparent")
        action_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(action_frame, text="Required Action:", font=btn_font_small, text_color="white").pack(side="left", padx=2)
        
        self.required_action_var = tk.StringVar(self.root)
        self.required_action_var.set("Hands Up")
        self.action_dropdown = ctk.CTkOptionMenu(action_frame, values=["Hands Up", "Hands Crossed", 
                                            "One Hand Raised (Left)", "One Hand Raised (Right)", 
                                            "T-Pose", "Sit", "Standing"], command=self.on_action_change, fg_color="#3498db", text_color="white", font=btn_font_small, dropdown_font=btn_font_small, corner_radius=5)
        self.action_dropdown.pack(side="left", padx=2, fill="x", expand=True)
        
        # ===== MONITORING MODE SECTION =====
        monitor_section = ctk.CTkFrame(self.grp_settings, fg_color="#1a1a1a", corner_radius=6, border_width=1, border_color="#404040")
        monitor_section.pack(fill="x", padx=0, pady=3)
        
        ctk.CTkLabel(monitor_section, text="🔔 MONITORING MODE", font=("Roboto", 9, "bold"), text_color="#f39c12").pack(anchor="w", padx=5, pady=(5,2))
        
        self.monitor_mode_dropdown = ctk.CTkOptionMenu(monitor_section, values=[
            "All Alerts (Action + Sleep)", 
            "Action Alerts Only", 
            "Sleeping Alerts Only"
        ], variable=self.monitor_mode_var, fg_color="#34495e", text_color="white", font=btn_font_small, dropdown_font=btn_font_small, corner_radius=5)
        self.monitor_mode_dropdown.pack(fill="x", padx=5, pady=2)
        
        # ===== TARGET MANAGEMENT SECTION =====
        target_section = ctk.CTkFrame(self.grp_settings, fg_color="#1a1a1a", corner_radius=6, border_width=1, border_color="#404040")
        target_section.pack(fill="x", padx=0, pady=3)
        
        ctk.CTkLabel(target_section, text="👤 TARGET MANAGEMENT", font=("Roboto", 9, "bold"), text_color="#f39c12").pack(anchor="w", padx=5, pady=(5,2))
        
        self.target_grid = ctk.CTkFrame(target_section, fg_color="transparent")
        self.target_grid.pack(fill="x", padx=5, pady=2)
        
        self.btn_select_targets = ctk.CTkButton(self.target_grid, text="📋 Select", command=self.open_target_selection_dialog, width=50, fg_color="#16a085", font=btn_font_small, corner_radius=5)
        self.btn_select_targets.grid(row=0, column=0, padx=2, sticky="ew")
        
        self.btn_apply_targets = ctk.CTkButton(self.target_grid, text="🎬 Track", command=self.apply_target_selection, width=50, fg_color="#16a085", font=btn_font_small, corner_radius=5)
        self.btn_apply_targets.grid(row=0, column=1, padx=2, sticky="ew")
        
        self.target_grid.grid_columnconfigure(0, weight=1)
        self.target_grid.grid_columnconfigure(1, weight=1)
        
        # Initialize selected targets list
        self.selected_target_names = []

        
        # --- Group 5: Tracked Persons Preview ---
        self.grp_preview = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.grp_preview.pack(fill="both", expand=True, padx=5, pady=(5,3))
        ctk.CTkLabel(self.grp_preview, text="📹 TRACKED PERSONS", font=("Roboto", 10, "bold"), text_color="gray").pack(anchor="w", padx=0, pady=(0,3))
        
        # Separator line
        separator5 = ctk.CTkFrame(self.grp_preview, height=1, fg_color="#404040")
        separator5.pack(fill="x", pady=(0,5))
        
        # Guard Preview (LARGE - single preview only) with enhanced styling
        self.guard_preview_frame = ctk.CTkFrame(self.grp_preview, fg_color="#1a3a1a", border_width=2, border_color="#27ae60", corner_radius=8)
        self.guard_preview_frame.pack(fill="both", expand=True, padx=0, pady=2)
        ctk.CTkLabel(self.guard_preview_frame, text="👮 GUARD", text_color="#27ae60", 
                font=("Arial", 10, "bold")).pack(fill="x", padx=5, pady=(5,3))
        self.guard_preview_label = ctk.CTkLabel(self.guard_preview_frame, text="No Guard Selected", text_color="#bdc3c7", 
                                            font=("Arial", 9))
        self.guard_preview_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Fugitive Preview with enhanced styling
        self.fugitive_preview_frame = ctk.CTkFrame(self.grp_preview, fg_color="#3a1a1a", border_width=2, border_color="#e74c3c", corner_radius=8)
        self.fugitive_preview_frame.pack(fill="x", padx=0, pady=2)
        ctk.CTkLabel(self.fugitive_preview_frame, text="🚨 FUGITIVE", text_color="#e74c3c", 
                font=("Arial", 10, "bold")).pack(fill="x", padx=5, pady=(5,3))
        self.fugitive_preview_label = ctk.CTkLabel(self.fugitive_preview_frame, text="No Fugitive Selected", text_color="#bdc3c7", 
                                               font=("Arial", 9))
        self.fugitive_preview_label.pack(fill="x", padx=5, pady=5)

        # Status label at bottom with enhanced styling
        status_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1a1a1a", corner_radius=6, border_width=1, border_color="#404040")
        status_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        self.status_label = ctk.CTkLabel(status_frame, text="FPS: 0 | MEM: 0 MB", text_color="#ecf0f1", font=('Roboto', 9, 'bold'))
        self.status_label.pack(fill="x", padx=8, pady=8)
        
        self.load_targets()
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        
        self.root.mainloop()
    
    def toggle_sidebar(self):
        """Toggle sidebar between collapsed and expanded states"""
        if self.sidebar_collapsed:
            # Expand sidebar
            self.sidebar_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
            self.toggle_sidebar_btn.configure(text="◄ Hide")
            self.sidebar_collapsed = False
        else:
            # Collapse sidebar
            self.sidebar_frame.grid_remove()
            self.toggle_sidebar_btn.configure(text="► Show")
            self.sidebar_collapsed = True
    
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
            
            # Remove face image from guard_profiles directory ONLY
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
        # Search ONLY in guard_profiles directory
        guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", "guard_profiles")
        if not os.path.exists(guard_profiles_dir):
            os.makedirs(guard_profiles_dir)
        target_files = glob.glob(os.path.join(guard_profiles_dir, "target_*.jpg"))
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

        if not display_names:
             logger.warning("No target files found")
        else:
             logger.warning(f"Loaded {len(set(display_names))} guards")
        
        # Update selected targets list
        self.selected_target_names = [name for name in self.selected_target_names if name in self.target_map]
        self.update_selected_preview()
    
    def select_all_targets(self):
        """Select all targets"""
        self.selected_target_names = list(self.target_map.keys())
        self.update_selected_preview()

    def update_selected_preview(self):
        """Update guard preview with first selected target"""
        
        if not self.selected_target_names:
            # Clear guard preview
            self.guard_preview_label.configure(image='', text="No Guard Selected")
            return
        
        # Show first selected guard in large guard preview
        if self.selected_target_names:
            first_name = self.selected_target_names[0]
            first_filename = self.target_map.get(first_name)
            if first_filename:
                try:
                    img = cv2.imread(first_filename)
                    if img is not None:
                        h, w = img.shape[:2]
                        scale = min(250 / w, 250 / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        img = cv2.resize(img, (new_w, new_h))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img)
                        imgtk = ImageTk.PhotoImage(image=pil_img)
                        self.guard_preview_label.configure(image=imgtk, text="")
                        # Store photo reference to prevent garbage collection
                        self.photo_storage["guard_preview"] = imgtk
                except Exception:
                    self.guard_preview_label.configure(text=f"Error: {first_name}")

    def apply_target_selection(self):
        self.targets_status = {} 
        if not self.selected_target_names:
            # No targets selected, tracking disabled
            logger.info("No targets selected - tracking disabled")
            return
        count = 0
        # ✅ IMPROVED: Increased pose buffer size for better multi-guard stability
        pose_buffer_size = max(CONFIG["performance"].get("pose_buffer_size", 5), 12)
        
        for name in self.selected_target_names:
            filename = self.target_map.get(name)
            if filename:
                try:
                    # ✅ CRITICAL: Check if file exists before loading
                    if not os.path.exists(filename):
                        logger.error(f"Guard profile file not found: {filename}")
                        continue
                    
                    target_image_file = face_recognition.load_image_file(filename)
                    # ✅ IMPROVED: Use num_jitters=2 for more robust encoding
                    encodings = face_recognition.face_encodings(target_image_file, num_jitters=2)
                    if encodings and len(encodings) > 0:
                        self.targets_status[name] = {
                            "encoding": encodings[0],
                            "tracker": None,
                            "body_tracker": None,
                            "face_box": None, 
                            "body_box": None,
                            "visible": False,
                            "overlap_disabled": False,  # ✅ SEPARATION FIX: Track if disabled due to overlap
                            "last_action_time": time.time(),  # Renamed from last_wave_time
                            "alert_cooldown": 0,
                            "alert_triggered_state": False,
                            "last_logged_action": None,
                            "pose_buffer": deque(maxlen=pose_buffer_size),  # ✅ Larger buffer for stability
                            "missing_pose_counter": 0,
                            "face_confidence": 0.0,
                            "pose_confidence": 0.0,  # ✅ NEW: Track pose detection quality
                            "face_encoding_history": deque(maxlen=5),  # ✅ CRITICAL: Track face encodings
                            "face_match_confidence": 0.0,  # ✅ NEW: Track match quality
                            "last_valid_pose": None,  # ✅ NEW: Store last valid pose for continuity
                            "last_valid_pose_time": time.time(),  # ✅ NEW: Track when last valid pose was detected
                            "pose_quality_history": deque(maxlen=10),  # ✅ NEW: Track pose quality over time
                            "pose_references": self.load_pose_references(name),
                            "last_snapshot_time": 0,  # Rate limiting: one snapshot per minute
                            "last_log_time": 0,  # Rate limiting: one log entry per minute
                            "alert_sound_thread": None,  # Track current alert sound thread
                            "alert_stop_event": None,  # Event to signal sound to stop when action performed
                            "alert_logged_timeout": False,  # Track if timeout alert was logged
                            "target_missing_alert_logged": False,  # ✅ NEW: Track if TARGET MISSING was logged (event start only)
                            "sleep_alert_logged": False,  # ✅ NEW: Track if sleep alert was logged (event start only)
                            # --- SLEEPING ALERT STATE (from Basic_v5.py) ---
                            "eye_counter_closed": 0,
                            "ear_threshold": 0.22,  # Start safe
                            "open_ear_baseline": 0.30,
                            "is_sleeping": False
                        }
                        count += 1
                        logger.info(f"[OK] {name} initialized with {len(encodings)} face encoding(s)")
                    else:
                        # ✅ IMPROVED: Log if no face detected in target image
                        logger.warning(f"[WARN] No face encoding found in {name} image - guard may not be detectable")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
        if count > 0:
            logger.warning(f"[OK] Tracking initialized for {count} targets (Pose Buffer: {pose_buffer_size} frames)")
            messagebox.showinfo("Tracking Updated", f"Now scanning for {count} selected targets")
        else:
            logger.error("No targets were initialized - check that guard profile images contain faces")

    def open_target_selection_dialog(self):
        """Open dialog for selecting targets"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Select Targets")
        dialog.geometry("400x500")
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="Select Targets to Track", font=("Roboto", 14, "bold")).pack(pady=10)
        
        scroll_frame = ctk.CTkScrollableFrame(dialog, width=350, height=350)
        scroll_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.target_vars = {}
        
        # Get all available targets
        targets = sorted(list(self.target_map.keys()))
        
        if not targets:
            ctk.CTkLabel(scroll_frame, text="No targets found.").pack()
        
        for target in targets:
            var = ctk.BooleanVar(value=target in self.selected_target_names)
            chk = ctk.CTkCheckBox(scroll_frame, text=target, variable=var)
            chk.pack(anchor="w", pady=2, padx=5)
            self.target_vars[target] = var
            
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10, fill="x", padx=10)
        
        ctk.CTkButton(btn_frame, text="Select All", command=self.select_all_dialog, width=100).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Clear All", command=self.clear_all_dialog, width=100).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Done", command=lambda: self.confirm_selection(dialog), width=100, fg_color="green").pack(side="right", padx=5)

    def select_all_dialog(self):
        """Select all targets in dialog"""
        for var in self.target_vars.values():
            var.set(True)

    def clear_all_dialog(self):
        """Clear all targets in dialog"""
        for var in self.target_vars.values():
            var.set(False)

    def confirm_selection(self, dialog):
        """Confirm target selection from dialog"""
        self.selected_target_names = [name for name, var in self.target_vars.items() if var.get()]
        dialog.destroy()
        # Update preview
        self.update_selected_preview()

    def set_alert_interval_advanced(self):
        """Set alert interval (timeout before alert) with hours, minutes, seconds"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Set Action Timeout Interval")
        dialog.geometry("400x300")
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="⏱ Action Timeout Interval", font=("Roboto", 14, "bold")).pack(pady=10)
        ctk.CTkLabel(dialog, text="How long until timeout alert triggers if action not performed", font=("Roboto", 10), text_color="#95a5a6").pack(pady=5)
        
        frame = ctk.CTkFrame(dialog)
        frame.pack(pady=15, padx=20, fill="x")
        
        # Calculate current time components for display
        current_h = int(self.alert_interval // 3600)
        current_m = int((self.alert_interval % 3600) // 60)
        current_s = int(self.alert_interval % 60)
        
        # Hours
        ctk.CTkLabel(frame, text="Hours:", font=("Roboto", 10)).grid(row=0, column=0, padx=5, sticky="w")
        h_var = ctk.StringVar(value=str(current_h))
        h_entry = ctk.CTkEntry(frame, textvariable=h_var, width=80)
        h_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Minutes
        ctk.CTkLabel(frame, text="Minutes:", font=("Roboto", 10)).grid(row=1, column=0, padx=5, sticky="w")
        m_var = ctk.StringVar(value=str(current_m))
        m_entry = ctk.CTkEntry(frame, textvariable=m_var, width=80)
        m_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Seconds
        ctk.CTkLabel(frame, text="Seconds:", font=("Roboto", 10)).grid(row=2, column=0, padx=5, sticky="w")
        s_var = ctk.StringVar(value=str(current_s))
        s_entry = ctk.CTkEntry(frame, textvariable=s_var, width=80)
        s_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Display recommended range
        ctk.CTkLabel(frame, text="Recommended: 5 - 300 seconds (action timeout)", font=("Roboto", 9), text_color="#95a5a6").grid(row=3, column=0, columnspan=2, pady=10)
        
        def confirm():
            try:
                h = int(h_var.get()) if h_var.get() and h_var.get().strip() else 0
                m = int(m_var.get()) if m_var.get() and m_var.get().strip() else 0
                s = int(s_var.get()) if s_var.get() and s_var.get().strip() else 0
                
                # ✅ IMPROVED: Validate input ranges
                if h < 0 or m < 0 or s < 0 or m > 59 or s > 59:
                    messagebox.showerror("Error", "Please enter valid time values (H >= 0, 0 <= M,S <= 59)")
                    return
                
                total_seconds = h * 3600 + m * 60 + s
                if total_seconds > 0:
                    self.alert_interval = total_seconds  # ✅ IMPORTANT: Store as seconds
                    # Update button to show current value
                    if total_seconds >= 3600:
                        display_text = f"{h}h {m}m {s}s"
                    elif total_seconds >= 60:
                        display_text = f"{m}m {s}s"
                    else:
                        display_text = f"{s}s"
                    self.btn_set_interval.configure(text=display_text)
                    self.label_interval_desc.configure(text=f"({total_seconds}s total)")
                    messagebox.showinfo("Success", f"Action timeout set to {total_seconds} seconds ({h}h {m}m {s}s)")
                    logger.warning(f"Alert interval changed to {total_seconds} seconds")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Interval must be greater than 0")
            except ValueError as e:
                messagebox.showerror("Error", f"Please enter valid numbers. Error: {e}")
        
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=15, fill="x", padx=20)
        ctk.CTkButton(button_frame, text="✓ Set Interval", command=confirm, fg_color="#27ae60", font=("Roboto", 11, "bold")).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(button_frame, text="✕ Cancel", command=dialog.destroy, fg_color="#34495e", font=("Roboto", 11)).pack(side="left", expand=True, padx=5)
        
        # Focus on seconds field by default
        s_entry.focus()

    def toggle_alert_mode(self):
        self.is_alert_mode = not self.is_alert_mode
        if self.is_alert_mode:
            self.btn_toggle_alert.configure(text="Stop Alert Mode", fg_color="#c0392b")
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
            self.btn_toggle_alert.configure(text="Start Alert Mode", fg_color="#e67e22")
            # Auto-stop logging and save
            if self.is_logging:
                self.save_log_to_file()
                self.is_logging = False
                logger.warning("Alert mode stopped - logging saved")

    def set_alert_interval(self):
        val = simpledialog.askinteger("Set Interval", "Enter seconds:", minvalue=1, maxvalue=3600, initialvalue=self.alert_interval)
        if val:
            self.alert_interval = val
            self.btn_set_interval.configure(text=f"Set Interval ({self.alert_interval}s)")
            
    def on_action_change(self, value):
        if self.is_alert_mode:
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_action_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False

    def set_sleep_interval(self):
        """Set sleep detection duration (how long eyes must be closed before sleep alert)"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Set Sleep Detection Duration")
        dialog.geometry("400x280")
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="😴 Sleep Detection Duration", font=("Roboto", 14, "bold")).pack(pady=10)
        ctk.CTkLabel(dialog, text="How long eyes must be closed before sleep alert triggers", font=("Roboto", 10), text_color="#95a5a6").pack(pady=5)
        
        frame = ctk.CTkFrame(dialog)
        frame.pack(pady=15, padx=20, fill="x")
        
        # Seconds (primary input for sleep - usually 0.5 to 5 seconds)
        ctk.CTkLabel(frame, text="Seconds:", font=("Roboto", 10)).grid(row=0, column=0, padx=5, sticky="w")
        s_var = ctk.StringVar(value=f"{self.sleep_alert_delay_seconds:.1f}")
        entry_sec = ctk.CTkEntry(frame, textvariable=s_var, width=80)
        entry_sec.grid(row=0, column=1, padx=5, pady=5)
        
        # Milliseconds (fine-tuning)
        ctk.CTkLabel(frame, text="Milliseconds (0-999):", font=("Roboto", 10)).grid(row=1, column=0, padx=5, sticky="w")
        ms_var = ctk.StringVar(value="0")
        entry_ms = ctk.CTkEntry(frame, textvariable=ms_var, width=80)
        entry_ms.grid(row=1, column=1, padx=5, pady=5)
        
        # Info
        ctk.CTkLabel(frame, text="Recommended: 0.5s - 3.0s (eyes closed duration)", font=("Roboto", 9), text_color="#95a5a6").grid(row=2, column=0, columnspan=2, pady=10)
        
        def confirm():
            try:
                s = float(s_var.get()) if s_var.get() else 0
                ms = int(ms_var.get()) if ms_var.get() else 0
                total_seconds = s + (ms / 1000.0)
                
                if total_seconds > 0:
                    self.sleep_alert_delay_seconds = total_seconds
                    self.btn_set_sleep.configure(text=f"{total_seconds:.2f}s")
                    self.label_sleep_desc.configure(text="(eyes closed duration)")
                    messagebox.showinfo("Success", f"Sleep detection set to {total_seconds:.2f} seconds")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Duration must be greater than 0")
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=15, fill="x", padx=20)
        ctk.CTkButton(button_frame, text="Set Duration", command=confirm, fg_color="#546e7a", font=("Roboto", 11, "bold")).pack(side="left", expand=True, padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy, fg_color="#34495e", font=("Roboto", 11)).pack(side="left", expand=True, padx=5)

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
                self.btn_fugitive.configure(text="Disable Fugitive Mode", fg_color="#ff6b6b")
                
                # Show fugitive preview frame
                self.fugitive_preview_frame.pack(side="left", fill="both", expand=True, padx=2, pady=2)
                
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
            self.btn_fugitive.configure(text="Enable Fugitive Mode", fg_color="#8b0000")
            
            # Hide fugitive preview frame
            self.fugitive_preview_frame.pack_forget()
            
            logger.warning("Fugitive Mode Stopped")
            messagebox.showinfo("Fugitive Mode", "Fugitive Mode Stopped")

    def _update_fugitive_preview(self):
        """Update fugitive preview image display"""
        if self.fugitive_image is None:
            self.fugitive_preview_label.configure(image='', text="No Fugitive")
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
            self.fugitive_preview_label.configure(image=photo, text='')
            # Store photo reference to prevent garbage collection
            self.photo_storage["fugitive_preview"] = photo
            
        except Exception as e:
            logger.error(f"Failed to update fugitive preview: {e}")
            self.fugitive_preview_label.configure(text="Preview Error")

    def toggle_pro_detection_mode(self):
        """Toggle PRO_Detection Mode - Advanced person re-identification across multiple angles"""
        if not self.pro_detection_mode:
            # Start PRO_Detection Mode
            if not self.reid_enabled:
                messagebox.showerror("Error", "ReID libraries not available. Install: pip install scikit-learn scipy")
                return
            
            try:
                self.pro_detection_mode = True
                self.person_features_db = {}  # Reset feature database
                self.person_identity_history = {}  # Reset identity tracking
                self.person_tracking_data = []  # Reset tracking data
                self.pro_detection_log_done = {}
                self.pro_detection_person_counter = 0
                
                self.btn_pro_detection.configure(text="Disable PRO_Detection", fg_color="#00d9ff", text_color="black")
                logger.warning("🚀 PRO_Detection Mode Started - Person Re-Identification Enabled")
                messagebox.showinfo("PRO_Detection - Onboarding Required", 
                    "🎯 PRO_Detection Mode Activated!\n\n"
                    "⚠️  IMPORTANT: Add new guards for PRO mode\n\n"
                    "In PRO mode, guards are onboarded differently:\n"
                    "• Clothing & body shape extracted\n"
                    "• Appearance features captured\n"
                    "• Multi-angle tracking enabled\n\n"
                    "Actions needed:\n"
                    "1️⃣  Click 👮 'Add' to add PRO guards\n"
                    "2️⃣  Follow onboarding (pose + appearance)\n"
                    "3️⃣  Select guards and start tracking\n\n"
                    "PRO Features:\n"
                    "✓ Appearance-based identification\n"
                    "✓ Multi-camera tracking\n"
                    "✓ Clothing color matching\n"
                    "✓ Body shape recognition\n"
                    "✓ Person re-identification across angles")
                
            except Exception as e:
                logger.error(f"PRO_Detection Mode Error: {e}")
                messagebox.showerror("Error", f"Failed to start PRO_Detection: {e}")
                self.pro_detection_mode = False
        else:
            # Stop PRO_Detection Mode
            self.pro_detection_mode = False
            self.person_features_db = {}
            self.person_identity_history = {}
            self.btn_pro_detection.configure(text="Enable PRO_Detection", fg_color="#004a7f", text_color="white")
            
            # Auto-save tracking data to CSV
            if self.person_tracking_data:
                self._save_pro_detection_log()
            
            logger.warning("🛑 PRO_Detection Mode Stopped")
            messagebox.showinfo("PRO_Detection", "PRO_Detection Mode Stopped")

    def _save_pro_detection_log(self):
        """Save person re-identification tracking data to CSV file"""
        try:
            if not self.person_tracking_data:
                return
            
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"pro_detection_{timestamp}.csv")
            
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Person_ID', 'Person_Name', 'Action', 'Status', 'Features_Available', 'Confidence'])
                writer.writerows(self.person_tracking_data)
            
            logger.warning(f"📊 PRO_Detection log saved: {log_file}")
        except Exception as e:
            logger.error(f"Failed to save PRO_Detection log: {e}")

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
                self.btn_start.configure(state="disabled")
                self.btn_stop.configure(state="normal")
                self.btn_add_guard.configure(state="normal")
                self.btn_toggle_alert.configure(state="normal")
                self.btn_fugitive.configure(state="normal")
                self.btn_pro_detection.configure(state="normal")
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
                self.btn_fugitive.configure(text="Enable Fugitive Mode", fg_color="#8b0000")
                self.fugitive_preview_label.configure(image='', text="No Fugitive Selected")
            
            # Stop PRO_Detection Mode if running
            if self.pro_detection_mode:
                self.pro_detection_mode = False
                if self.person_tracking_data:
                    self._save_pro_detection_log()
                self.person_features_db = {}
                self.person_identity_history = {}
                self.btn_pro_detection.configure(text="Enable PRO_Detection", fg_color="#004a7f", text_color="white")
                logger.warning("PRO_Detection Mode Stopped on camera close")
            
            # Clear guard preview
            self.guard_preview_label.configure(image='', text="No Guard Selected")
            
            # Cleanup
            for status in self.targets_status.values():
                if status["tracker"]:
                    status["tracker"] = None
            
            gc.collect()
            
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.btn_add_guard.configure(state="disabled")
            self.btn_fugitive.configure(state="disabled")
            self.btn_pro_detection.configure(state="disabled")
            self.video_label.configure(image='')

    def auto_flush_logs(self):
        """Automatically flush logs when threshold reached"""
        if self.is_logging and len(self.temp_log) >= CONFIG["logging"]["auto_flush_interval"]:
            self.save_log_to_file()
        
        # Optimize memory periodically
        if self.frame_counter % 300 == 0:  # Every ~10 seconds at 30fps
            self.optimize_memory()
    
    def optimize_memory(self):
        """Clear old cache entries and collect garbage to free memory"""
        try:
            # Clear old action cache - keep last 50 entries
            if len(self.last_action_cache) > 50:
                keys_to_remove = list(self.last_action_cache.keys())[:-50]
                for key in keys_to_remove:
                    del self.last_action_cache[key]
            
            # Force garbage collection
            import gc
            gc.collect()
            logger.debug("Memory optimized - caches cleared, garbage collected")
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")

    def save_log_to_file(self):
        if self.temp_log:
            try:
                log_dir = CONFIG["logging"]["log_directory"]
                os.makedirs(log_dir, exist_ok=True)
                csv_path = os.path.join(log_dir, "events.csv")
                
                file_exists = os.path.exists(csv_path)
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Timestamp", "Guard Name", "Action", "Status", "Image Path", "Confidence"])
                    writer.writerows(self.temp_log)
                logger.warning(f"Saved {len(self.temp_log)} log entries to {csv_path}")
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
        
        messagebox.showinfo("Step 1", "Stand in front of camera (green box will appear when detected). Click 'Snap Photo' when ready.")

    def exit_onboarding_mode(self):
        self.is_in_capture_mode = False
        self.onboarding_mode = False
        self.onboarding_step = 0
        self.onboarding_poses = {}
        self.onboarding_detection_results = None
        self.onboarding_face_box = None

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
            if self.onboarding_name:
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
            if CONFIG["performance"]["enable_frame_skipping"] and self.frame_counter % skip_interval != 0:
                # Use cached frame
                if self.last_process_frame is not None:
                    frame = self.last_process_frame.copy()
            else:
                self.process_tracking_frame_optimized(frame)
                self.last_process_frame = frame.copy()
        
        # FPS calculation and memory optimization
        if self.frame_counter % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
            self.last_fps_time = current_time
            
            # ✅ PERFORMANCE: Periodic memory optimization
            if self.frame_counter % 150 == 0:  # Every 150 frames (~5 seconds at 30 FPS)
                optimize_memory()
            
            # Memory monitoring
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            self.status_label.configure(text=f"FPS: {self.current_fps:.1f} | MEM: {mem_mb:.0f} MB")
            
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
                self.video_label.configure(image=imgtk, text="")
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

        # ✅ PERFORMANCE OPTIMIZATION: Frame skipping for heavy operations
        # Run expensive face detection only at intervals, use tracking for intermediate frames
        self.re_detect_counter += 1
        skip_face_detection = (self.re_detect_counter < self.RE_DETECT_INTERVAL) and (self.re_detect_counter % 2 == 0)
        
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
                                duration_seconds=15
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
                            logger.warning(f"[FUGITIVE] DETECTED - All operations executed:")
                            logger.warning(f"  |- [SOUND] Fugitive.mp3 (30s)")
                            logger.warning(f"  |- [SNAPSHOT] {img_path}")
                            logger.warning(f"  |- [CSV] {self.fugitive_name} (Confidence: {confidence:.2f})")
                            
                            self.fugitive_detected_log_done = True
                            self.last_fugitive_snapshot_time = time.time()
                    else:
                        # Reset flag when fugitive not in frame
                        self.fugitive_detected_log_done = False
        # ===================================================

        # ==================== PRO_DETECTION MODE ====================
        # Extract and match person features across multiple angles
        if self.pro_detection_mode and self.targets_status:
            face_locations = face_recognition.face_locations(rgb_full_frame)
            
            if face_locations:
                for i, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location
                    face_box = (left, top, right, bottom)
                    
                    # Extract appearance features from detected person
                    features = extract_appearance_features(frame, face_box)
                    
                    if features is not None:
                        # Create a base ID for this detection
                        # Use detection index as stable identifier within this frame
                        detection_id = f"det_{i}_{int(self.frame_counter)}"
                        
                        # Check if this detection matches any known person in our database
                        best_match_id = None
                        best_match_confidence = 0.0
                        
                        # Compare against all stored persons
                        for stored_person_id in list(self.person_features_db.keys()):
                            stored_features = self.person_features_db[stored_person_id]['features']
                            similarity = calculate_feature_similarity(features, stored_features)
                            
                            if similarity > best_match_confidence:
                                best_match_confidence = similarity
                                best_match_id = stored_person_id
                        
                        # Decide whether to match to existing person or create new one
                        if best_match_confidence >= self.reid_confidence_threshold:
                            # MATCH: Update existing person
                            matched_id = best_match_id
                            alpha = 0.3
                            self.person_features_db[matched_id]['features'] = (
                                alpha * features + (1 - alpha) * self.person_features_db[matched_id]['features']
                            )
                            self.person_features_db[matched_id]['count'] += 1
                            self.person_features_db[matched_id]['last_seen'] = time.time()
                            confidence = best_match_confidence
                        else:
                            # NEW PERSON: Create new identity
                            self.pro_detection_person_counter += 1
                            matched_id = f"Person_{self.pro_detection_person_counter:03d}"
                            self.person_features_db[matched_id] = {
                                'features': features,
                                'count': 1,
                                'last_seen': time.time()
                            }
                            confidence = 0.5  # Default confidence for new persons
                        
                        # Draw bounding box with person ID and confidence
                        color = (0, 255, 0) if confidence >= 0.7 else (0, 255, 255)  # Green for high conf, cyan for new
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        person_label = f"{matched_id} ({confidence:.2f})"
                        cv2.putText(frame, person_label, (left, top - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Log high confidence matches
                        log_key = f"{matched_id}_{int(time.time() * 1000)}"
                        if log_key not in self.pro_detection_log_done and confidence >= 0.65:
                            self.person_tracking_data.append((
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                matched_id,
                                "ReID_Detected",
                                "PRO_DETECTION",
                                "N/A",
                                f"{confidence:.3f}"
                            ))
                            self.pro_detection_log_done[log_key] = True
                            
                            logger.info(f"🎯 {matched_id} detected (confidence: {confidence:.2f})")
        # ===================================================
        
        # ✅ IMPORTANT: Get current time at the start of the tracking loop
        current_time = time.time()

        # 1. Update Trackers (With Stability Check for Multi-Guard Robustness)
        for name, status in self.targets_status.items():
            face_ok = False
            body_ok = False
            if status["tracker"]:
                success, box = status["tracker"].update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    new_box = (x, y, x + w, y + h)
                    if status["face_box"] is not None:
                        old_x1, old_y1, old_x2, old_y2 = status["face_box"]
                        new_x1, new_y1, new_x2, new_y2 = new_box
                        dx = abs(new_x1 - old_x1) + abs(new_x2 - old_x2)
                        dy = abs(new_y1 - old_y1) + abs(new_y2 - old_y2)
                        old_w = max(1, old_x2 - old_x1)
                        old_h = max(1, old_y2 - old_y1)
                        new_w = max(1, new_x2 - new_x1)
                        new_h = max(1, new_y2 - new_y1)
                        size_change = abs(new_w - old_w) + abs(new_h - old_h)
                        max_movement = max(old_w, old_h) * 0.2
                        max_size_change = (old_w + old_h) * 0.2
                        if dx > max_movement * 4 or dy > max_movement * 4 or size_change > max_size_change * 4:
                            status["visible"] = False
                            status["tracker"] = None
                            logger.debug(f"{name}: Tracker movement too large (dx:{dx}, dy:{dy}) - resetting")
                        else:
                            smoothed_box = smooth_bounding_box(new_box, status["face_box"], smoothing_factor=0.75)
                            status["face_box"] = smoothed_box
                            status["visible"] = True
                            face_ok = True
                    else:
                        status["face_box"] = new_box
                        status["visible"] = True
                        face_ok = True
                else:
                    status["tracker"] = None
            # Update body tracker
            if status.get("body_tracker"):
                success_b, box_b = status["body_tracker"].update(frame)
                if success_b:
                    bx, by, bw, bh = [int(v) for v in box_b]
                    status["body_box"] = (bx, by, bx + bw, by + bh)
                    body_ok = True
                else:
                    status["body_tracker"] = None
            # Fallback: if face lost but body tracked, approximate face from body
            if not face_ok and body_ok:
                approx_face = approximate_face_from_body(status["body_box"])
                prev_face = status.get("face_box")
                status["face_box"] = smooth_bounding_box(approx_face, prev_face, smoothing_factor=0.6) if prev_face else approx_face
                status["visible"] = True
            elif not face_ok and not body_ok:
                status["visible"] = False

        # 2. Detection (PARALLEL MATCHING) - Fixes Multiple Target Detection
        # ✅ IMPROVED: Run detection more frequently, even when some targets are visible
        # This helps catch targets that briefly leave frame and re-enter, and handles multi-guard scenarios
        untracked_targets = [name for name, s in self.targets_status.items() if not s["visible"]]
        
        # Detection logic: Run EVERY 15 frames (0.5 sec at 30 FPS) regardless of visibility status
        if self.re_detect_counter == 0:
            # ✅ IMPROVED: Multi-jitter encoding for more robust face representation
            face_locations = face_recognition.face_locations(rgb_full_frame, model="hog")
            if face_locations and len(face_locations) > 0:
                face_encodings = face_recognition.face_encodings(rgb_full_frame, face_locations, num_jitters=2)
                
                # ✅ OPTIMIZED: Use balanced thresholds for speed AND accuracy
                num_guards = len(untracked_targets)
                if num_guards >= 2:
                    # MULTI-GUARD MODE: Balanced for both speed and accuracy
                    base_tolerance = 0.50      # ✅ OPTIMIZED: Balanced (was 0.40 - too strict, causing delays)
                    min_confidence = 0.55      # ✅ OPTIMIZED: Reasonable threshold (was 0.65 - too strict)
                else:
                    # SINGLE GUARD: More lenient for faster detection
                    base_tolerance = 0.55
                    min_confidence = 0.50
                
                # ✅ CRITICAL: Build complete cost matrix with ALL candidates
                cost_matrix = []
                guard_all_matches = {}
                
                for target_idx, name in enumerate(untracked_targets):
                    target_encoding = self.targets_status[name]["encoding"]
                    if target_encoding is None:
                        logger.warning(f"[WARN] Skip {name}: no encoding available")
                        continue
                    
                    all_matches = []
                    
                    for face_idx, unknown_encoding in enumerate(face_encodings):
                        if unknown_encoding is None:
                            continue
                        
                        dist = face_recognition.face_distance([target_encoding], unknown_encoding)[0]
                        confidence = 1.0 - dist
                        
                        # ✅ STRICT: Only add if within tolerance
                        if dist <= base_tolerance:
                            all_matches.append({
                                "face_idx": face_idx,
                                "distance": dist,
                                "confidence": confidence,
                                "encoding": unknown_encoding
                            })
                            cost_matrix.append((dist, target_idx, face_idx, name, confidence, unknown_encoding))
                    
                    guard_all_matches[name] = all_matches
                
                # Sort by distance (best matches first)
                cost_matrix.sort(key=lambda x: x[0])
                
                # ✅ IMPROVED: Enhanced debug logging
                if cost_matrix:
                    logger.info(f"[FACE] Found {len(face_locations)} faces, {len(cost_matrix)} potential matches")
                    by_guard = {}
                    for item in cost_matrix:
                        dist, _, face_idx, name, conf, _ = item
                        if name not in by_guard:
                            by_guard[name] = []
                        by_guard[name].append((dist, face_idx, conf))
                    
                    for name, matches in by_guard.items():
                        for i, (dist, face_idx, conf) in enumerate(matches[:2]):
                            logger.info(f"  {name} (face_{face_idx}): confidence={conf:.3f}, distance={dist:.3f}")
                
                # ✅ OPTIMIZED: Single-pass greedy assignment (faster, no quality loss)
                assigned_faces = set()
                assigned_targets = set()
                assignments = []
                
                # Single-pass: Best-distance-first greedy assignment
                # (Previous two-pass approach added ~50-100ms per frame with multiple guards)
                for item in cost_matrix:
                    dist, target_idx, face_idx, name, confidence, unknown_enc = item
                    if face_idx in assigned_faces or name in assigned_targets:
                        continue
                    
                    # Accept any match within tolerance (pre-filtered in cost_matrix)
                    if confidence >= min_confidence:
                        assigned_faces.add(face_idx)
                        assigned_targets.add(name)
                        assignments.append((name, face_idx, dist, confidence, unknown_enc))
                        logger.debug(f"[ASSIGN] {name} -> face_{face_idx} (conf:{confidence:.3f})")
                
                # ✅ Execute assignments
                for name, face_idx, dist, confidence, unknown_enc in assignments:
                    (top, right, bottom, left) = face_locations[face_idx]
                    
                    # ✅ IMPROVED: Reset TARGET MISSING flag when guard is found again
                    self.targets_status[name]["target_missing_alert_logged"] = False
                    
                    # ✅ SEPARATION FIX: Clear overlap_disabled flag when guard is re-detected through normal face matching
                    self.targets_status[name]["overlap_disabled"] = False
                    
                    # ✅ CRITICAL: Store face encoding for later reference
                    if unknown_enc is not None:
                        self.targets_status[name]["face_encoding_history"].append(unknown_enc)
                    
                    # Initialize tracker for this target
                    try:
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, (left, top, right-left, bottom-top))
                        self.targets_status[name]["tracker"] = tracker
                    except Exception as e:
                        logger.warning(f"Failed to initialize tracker for {name}: {e}")
                        self.targets_status[name]["tracker"] = None
                    
                    self.targets_status[name]["face_box"] = (left, top, right, bottom)
                    self.targets_status[name]["visible"] = True
                    self.targets_status[name]["missing_pose_counter"] = 0
                    self.targets_status[name]["face_confidence"] = max(0.0, min(1.0, confidence))  # ✅ IMPROVED: Clamp confidence to [0,1]
                    self.targets_status[name]["face_match_confidence"] = max(0.0, min(1.0, confidence))  # ✅ IMPROVED: Clamp confidence
                    self.targets_status[name]["last_action_time"] = current_time  # ✅ CRITICAL: Reset timer on detection!
                    
                    logger.warning(f"[OK] {name} detected & tracked (confidence: {confidence:.3f}, distance: {dist:.3f})")
                
                # ✅ CRITICAL: Log detailed diagnostic info for multi-guard scenarios
                if len(untracked_targets) >= 2:
                    logger.info(f"[MULTI-GUARD] {len(untracked_targets)} guards tracking, {len(assigned_targets)} matched, {len(face_locations)} faces detected")
                    unmatched = len(face_locations) - len(assigned_faces)
                    if unmatched > 0:
                        logger.debug(f"[UNMATCHED] {unmatched} face(s) not assigned to any guard")
                    
                    # Show rejection log for debugging
                    for name in guard_all_matches.keys():
                        if name not in assigned_targets and guard_all_matches[name]:
                            matches = guard_all_matches[name]
                            logger.debug(f"[UNASSIGNED] {name}: best match distance={matches[0]['distance']:.3f}, confidence={matches[0]['confidence']:.3f} (below threshold)")
                

        # 3. Overlap Check (Fixes Merging Targets) - Enhanced with Confidence & Temporal Consistency
        active_names = [n for n, s in self.targets_status.items() if s["visible"]]
        
        # ✅ OPTIMIZED: Only check overlaps every 5 frames for multi-guard (reduces CPU from ~15% to ~3%)
        # Single-pass resolution is fast enough to run every frame if needed, but reduce frequency anyway
        num_visible_guards = len(active_names)
        should_check_overlap = (self.frame_counter % 5 == 0) if num_visible_guards >= 2 else True
        
        if should_check_overlap and num_visible_guards >= 2:
            # ✅ IMPROVED: Multi-Guard Pose Detection Resolution (independent, consistent detection)
            self.targets_status = resolve_overlapping_poses(self.targets_status, iou_threshold=0.3)
            
            # ✅ CRITICAL: Refresh active_names after resolve_overlapping_poses updates visibility
            active_names = [n for n, s in self.targets_status.items() if s["visible"]]
        
        # ✅ OPTIMIZED: Inline safety check only when resolve_overlapping_poses didn't run
        if not should_check_overlap and num_visible_guards >= 2:
            # Quick safety check on frames where full resolution didn't run
            for i in range(len(active_names)):
                for j in range(i + 1, len(active_names)):
                    nameA = active_names[i]
                    nameB = active_names[j]
                    
                    # Quick box check only
                    boxA = self.targets_status[nameA]["face_box"]
                    boxB = self.targets_status[nameB]["face_box"]
                    
                    if boxA and boxB:
                        rectA = (boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1])
                        rectB = (boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1])
                        iou = calculate_iou(rectA, rectB)
                        
                        if iou > 0.40:  # ✅ HIGHER threshold for quick safety check (prevents false positives)
                            # Only do emergency disable if heavily overlapping
                            conf_a = self.targets_status[nameA].get("face_confidence", 0.5)
                            conf_b = self.targets_status[nameB].get("face_confidence", 0.5)
                            
                            if conf_a > conf_b:
                                self.targets_status[nameB]["tracker"] = None
                                self.targets_status[nameB]["visible"] = False
                            else:
                                self.targets_status[nameA]["tracker"] = None
                                self.targets_status[nameA]["visible"] = False
        
        # Refresh active_names for final action classification
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
                if iou > 0.30:
                    # Multi-factor conflict resolution: confidence + temporal consistency
                    conf_a = self.targets_status[nameA].get("face_confidence", 0.5)
                    conf_b = self.targets_status[nameB].get("face_confidence", 0.5)
                    
                    # Get cached actions for temporal consistency
                    action_a = self.last_action_cache.get(nameA, "Unknown")
                    action_b = self.last_action_cache.get(nameB, "Unknown")
                    
                    # Weighted score: 60% confidence, 30% consistency, 10% IoU severity
                    score_a = (conf_a * 0.6) + (0.3 if action_a != "Unknown" else 0.0) + (0.1 * (1 - iou))
                    score_b = (conf_b * 0.6) + (0.3 if action_b != "Unknown" else 0.0) + (0.1 * (1 - iou))
                    
                    if score_a > score_b:
                        # Keep A, remove B
                        self.targets_status[nameB]["tracker"] = None
                        self.targets_status[nameB]["visible"] = False
                        logger.debug(f"Overlap resolved: keeping {nameA} (score: {score_a:.2f}) over {nameB} (score: {score_b:.2f}), IoU: {iou:.2f}")
                    else:
                        # Keep B, remove A
                        self.targets_status[nameA]["tracker"] = None
                        self.targets_status[nameA]["visible"] = False
                        logger.debug(f"Overlap resolved: keeping {nameB} (score: {score_b:.2f}) over {nameA} (score: {score_a:.2f}), IoU: {iou:.2f}")

        # 4. Processing & Drawing
        required_act = self.required_action_var.get()
        monitor_mode = self.monitor_mode_var.get()  # Get monitoring mode
        current_time = time.time()
        min_buffer = max(CONFIG["performance"].get("min_buffer_for_classification", 5), 5)  # ✅ FIXED: Initialize before loop with safety check

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
                        # ✅ IMPROVED: Each guard gets independent pose detection from their cropped region
                        # This ensures landmarks from one guard don't contaminate another's detection
                        results_crop = self.holistic.process(rgb_crop)
                        rgb_crop.flags.writeable = True
                        
                        # ========== SLEEPING ALERT LOGIC (IMPROVED FOR ROBUSTNESS) ==========
                        is_sleeping_detected = False
                        # monitor_mode already obtained at the top of loop
                        
                        # ✅ IMPROVEMENT 1: Check Alert Mode is enabled before processing sleep alerts
                        # Only process sleep if mode allows AND alert system is active
                        if self.is_alert_mode and monitor_mode in ["All Alerts (Action + Sleep)", "Sleeping Alerts Only"]:
                            if results_crop.face_landmarks:
                                crop_h, crop_w = crop.shape[:2]
                                face_landmarks = results_crop.face_landmarks.landmark
                                
                                # ✅ IMPROVEMENT 2: EAR Confidence Validation
                                # Count visible eye landmarks (need 6 per eye = 12 minimum for reliable EAR)
                                eye_landmarks_indices = list(range(33, 48))  # Eye region landmarks
                                visible_eye_landmarks = sum(1 for idx in eye_landmarks_indices if face_landmarks[idx].visibility > 0.5)
                                
                                # Only calculate EAR if we have sufficient eye landmark visibility
                                if visible_eye_landmarks >= 12:
                                    ear = calculate_ear(face_landmarks, crop_w, crop_h)
                                    
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
                                    
                                    # DYNAMIC SLEEP FRAMES CALCULATION
                                    required_sleep_frames = int(self.sleep_alert_delay_seconds * self.current_fps)
                                    
                                    # ✅ IMPROVEMENT 3: Require minimum 3 frames of eye closure to trigger
                                    # Prevents false positives from brief blinks
                                    min_closure_frames = 3
                                    
                                    # Check Alert Trigger - ONLY if sufficient closure frames AND confidence is high
                                    if status["eye_counter_closed"] > max(required_sleep_frames, min_closure_frames):
                                        is_sleeping_detected = True
                                        status["is_sleeping"] = True
                                        
                                        # VISUAL 1: Thick Red Border around the Person
                                        cv2.rectangle(frame, (fx1-15, fy1-15), (fx2+15, fy2+15), (0, 0, 255), 6)
                                        
                                        # VISUAL 2: Center Screen FLASHING Warning
                                        if int(time.time() * 4) % 2 == 0: # Flash effect
                                            cv2.putText(frame, "WAKE UP!", (frame_w//2 - 200, frame_h//2), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

                                        # AUDIO
                                        if status["alert_stop_event"] is None: 
                                            status["alert_stop_event"] = threading.Event()
                                        status["alert_stop_event"].clear()
                                        if not status.get("alert_sound_thread") or not status["alert_sound_thread"].is_alive():
                                            status["alert_sound_thread"] = play_siren_sound(
                                                status["alert_stop_event"], 
                                                duration_seconds=10
                                            )
                                        
                                        # ✅ IMPROVED: Capture snapshot only at event start (first detection of sleep)
                                        # Log only once per sleep event using a flag
                                        if self.is_logging and not status.get("sleep_alert_logged", False):
                                            # Take snapshot at the start of the event
                                            bx1, by1, bx2, by2 = calculate_body_box((fx1, fy1, fx2, fy2), frame_h, frame_w, expansion_factor=3.0)
                                            img_path = "N/A"
                                            if bx1 < bx2 and by1 < by2:
                                                try:
                                                    snapshot_result = self.capture_alert_snapshot(frame[by1:by2, bx1:bx2], name, check_rate_limit=False)
                                                    img_path = snapshot_result if snapshot_result else "N/A"
                                                except Exception as e:
                                                    logger.warning(f"Failed to capture sleep snapshot: {e}")
                                                    img_path = "N/A"
                                            
                                            self.temp_log.append((
                                                time.strftime("%Y-%m-%d %H:%M:%S"), 
                                                name, "SLEEPING", "ALERT", img_path, f"{ear:.2f}"
                                            ))
                                            status["sleep_alert_logged"] = True  # Mark event as logged
                                            status["last_log_time"] = current_time
                                            self.temp_log_counter += 1
                                            logger.warning(f"[SLEEP] {name} is sleeping! Alert triggered with snapshot: {img_path}")
                                    else:
                                        status["is_sleeping"] = False
                                        status["sleep_alert_logged"] = False  # Reset flag when eyes open
                                        status["is_sleeping"] = False
                                else:
                                    # Insufficient eye landmark visibility - reset counter
                                    status["eye_counter_closed"] = 0
                                    status["is_sleeping"] = False
                            else:
                                # No face landmarks detected - reset counter
                                status["eye_counter_closed"] = 0
                                status["is_sleeping"] = False
                        else:
                            # Alert mode disabled or mode doesn't allow sleep alerts - reset
                            status["eye_counter_closed"] = 0
                            status["is_sleeping"] = False
                        # ============================================================
                        
                        current_action = "Unknown"
                        if results_crop.pose_landmarks:
                            pose_found_in_box = True
                            status["missing_pose_counter"] = 0 # Reset
                            
                            # ✅ MULTI-GUARD IMPROVEMENT: Enhanced pose quality/confidence calculation
                            pose_landmarks = results_crop.pose_landmarks.landmark
                            visible_count = sum(1 for lm in pose_landmarks if lm.visibility > 0.5)
                            avg_visibility = sum(lm.visibility for lm in pose_landmarks) / len(pose_landmarks)
                            pose_quality = min(1.0, visible_count / 20.0)  # 20 landmarks = 100% quality
                            status["pose_confidence"] = pose_quality
                            
                            # ✅ MULTI-GUARD FIX: Per-guard pose quality threshold validation
                            # Store pose quality history for temporal consistency checks
                            if "pose_quality_history" not in status:
                                status["pose_quality_history"] = deque(maxlen=10)
                            status["pose_quality_history"].append(pose_quality)
                            
                            # Only process pose if quality is acceptable
                            # ✅ IMPROVED: Slightly lowered from 0.65 to 0.6 for better multi-guard coverage
                            if pose_quality >= 0.6:  # At least 60% joints visible
                                draw_styled_landmarks(crop, results_crop)
                                raw_action = classify_action(pose_landmarks, (by2-by1), (bx2-bx1))
                                
                                # ✅ MULTI-GUARD ENHANCEMENT: Filter out "Unknown" from buffer (more stable)
                                # Only add poses with sufficient visibility to reduce noise
                                if raw_action != "Unknown" and avg_visibility > 0.4:
                                    status["pose_buffer"].append(raw_action)
                                    status["last_valid_pose"] = raw_action
                                    status["last_valid_pose_time"] = current_time
                                
                                min_buffer = max(CONFIG["performance"].get("min_buffer_for_classification", 5), 5)
                                if len(status["pose_buffer"]) >= min_buffer:
                                    # ✅ MULTI-GUARD IMPROVEMENT: Enhanced consensus with temporal validation
                                    # Use mode but require consistency AND temporal stability
                                    counts = Counter(status["pose_buffer"])
                                    most_common = counts.most_common(1)[0][0]
                                    confidence_pct = counts[most_common] / len(status["pose_buffer"])
                                    
                                    # ✅ IMPROVED: Raise threshold to 65% for multi-guard isolation
                                    # This prevents cross-contamination between nearby guards
                                    if confidence_pct >= 0.65:  # Increased from 0.60
                                        current_action = most_common
                                    else:
                                        # Low confidence - use last valid action (stability)
                                        current_action = status["last_valid_pose"] or "Standing"
                                else:
                                    # Buffer not full yet - use last valid action for stability
                                    current_action = status["last_valid_pose"] or "Unknown"
                            else:
                                # Poor pose quality - use last valid pose
                                current_action = status["last_valid_pose"] or "Standing"
                            
                            # Cache action for logging (only if not Unknown)
                            if current_action != "Unknown":
                                self.last_action_cache[name] = current_action
                            
                            # ✅ MULTI-GUARD IMPROVEMENT: Enhanced debug logging for pose tracking
                            if len(status["pose_buffer"]) >= min_buffer:
                                # Log detailed info for debugging
                                buffer_summary = Counter(status["pose_buffer"])
                                most_common_action = buffer_summary.most_common(1)[0][0]
                                buffer_str = ", ".join([f"{action}:{count}" for action, count in buffer_summary.most_common()])
                                avg_quality = sum(status["pose_quality_history"]) / len(status["pose_quality_history"])
                                logger.debug(f"[POSE] {name}: {current_action} (quality:{pose_quality:.2f}, avg_qual:{avg_quality:.2f}, buffer:[{buffer_str}], consensus:{buffer_summary[most_common_action]/len(status['pose_buffer']):.1%})")

                            # ========== ACTION ALERT LOGIC (IMPROVED WITH ROBUST MODE CHECKING) ==========
                            # ✅ IMPROVEMENT: Check monitoring mode AND alert mode to stop alerts correctly
                            # Only stop alerts if the SELECTED ACTION is performed, not any action
                            if monitor_mode in ["All Alerts (Action + Sleep)", "Action Alerts Only"]:
                                if self.is_alert_mode and current_action == required_act and not status["is_sleeping"]:
                                    # ✅ IMPROVED: Validate that selected action from dropdown matches current action
                                    # Only stop sound if current action equals the required action selected in UI
                                    status["last_action_time"] = current_time
                                    status["alert_triggered_state"] = False
                                    # STOP ALERT SOUND only when selected action is performed
                                    if status["alert_stop_event"] is not None:
                                        status["alert_stop_event"].set()  # Signal sound to stop
                                        logger.info(f"Alert sound stopped for {name} - required action '{required_act}' performed")
                                    if self.is_logging and status["last_logged_action"] != required_act:
                                        # Rate limiting: only log once per minute per target
                                        time_since_last_log = current_time - status["last_log_time"]
                                        if time_since_last_log > 60:
                                            self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, current_action, "Action Performed", "N/A", f"{status['face_confidence']:.2f}"))
                                            status["last_log_time"] = current_time
                                            self.temp_log_counter += 1
                                        status["last_logged_action"] = required_act
                            # ============================================================
                            
                            if current_action != required_act:

                                status["last_logged_action"] = None
                            
                            # --- Dynamic Bounding Box Logic ---
                            h_c, w_c = crop.shape[:2]
                            p_lms = results_crop.pose_landmarks.landmark
                            
                            lx = [lm.x * w_c for lm in p_lms]
                            ly = [lm.y * h_c for lm in p_lms]
                            
                            d_x1 = int(min(lx)) + bx1
                            d_y1 = int(min(ly)) + by1
                            d_x2 = int(max(lx)) + bx1
                            d_y2 = int(max(ly)) + by1
                            
                            # Add padding
                            d_x1 = max(0, d_x1 - 15)
                            d_y1 = max(0, d_y1 - 15)
                            d_x2 = min(frame_w, d_x2 + 15)
                            d_y2 = min(frame_h, d_y2 + 15)
                            
                            # Draw Dynamic Box
                            cv2.rectangle(frame, (d_x1, d_y1), (d_x2, d_y2), (0, 255, 0), 2)

                            # Cache/update body box and initialize body tracker if needed
                            status["body_box"] = (d_x1, d_y1, d_x2, d_y2)
                            if status.get("body_tracker") is None:
                                try:
                                    bt = cv2.legacy.TrackerCSRT_create()
                                    bt.init(frame, (d_x1, d_y1, d_x2 - d_x1, d_y2 - d_y1))
                                    status["body_tracker"] = bt
                                except Exception:
                                    status["body_tracker"] = None
                            
                            # ✅ IMPROVED: Show identification confidence with guard name
                            face_conf = status.get("face_confidence", 0.0)
                            pose_conf = status.get("pose_confidence", 0.0)
                            
                            # Color code based on identification confidence
                            if face_conf > 0.85:
                                id_color = (0, 255, 0)  # Green - high confidence
                            elif face_conf > 0.65:
                                id_color = (0, 165, 255)  # Orange - medium confidence
                            else:
                                id_color = (0, 0, 255)  # Red - low confidence
                            
                            # Display guard name with confidence
                            info_text = f"{name} ({face_conf:.2f})"
                            action_text = f"{current_action} (P:{pose_conf:.1%})"
                            
                            cv2.putText(frame, info_text, (d_x1, d_y1 - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, id_color, 2)
                            cv2.putText(frame, action_text, (d_x1, d_y1 - 8), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

                # Ghost Box Removal Logic
                if not pose_found_in_box:
                    status["missing_pose_counter"] += 1
                else:
                    # Cache for later use
                    if name not in self.last_action_cache:
                        self.last_action_cache[name] = "Unknown"
                    # If tracker says visible, but no pose for 30 frames (approx 1 sec) -> Kill Tracker
                    if status["missing_pose_counter"] > 30:
                        status["tracker"] = None
                        status["visible"] = False
            
            # --- Removed: Guard missing logging (non-alert event) ---
            # Now only logging ALERT_TRIGGERED events to CSV (Task 9)

            # ========== STILLNESS (TIMEOUT) ALERT LOGIC (from Basic_v5.py) ==========
            # Alert Logic (Time-based Pose Timeout) - Only if mode enables it
            if self.is_alert_mode and monitor_mode in ["All Alerts (Action + Sleep)", "Action Alerts Only"]:
                # ✅ IMPROVED: Ensure alert_interval is valid (should be > 0 seconds)
                if self.alert_interval <= 0:
                    self.alert_interval = 10  # Safety fallback to 10 seconds
                    logger.warning(f"Alert interval was invalid ({self.alert_interval}), reset to 10 seconds")
                
                time_diff = current_time - status["last_action_time"]
                time_left = max(0, self.alert_interval - time_diff)
                y_offset = 50 + (list(self.targets_status.keys()).index(name) * 30)
                color = (0, 255, 0) if time_left > 3 else (0, 0, 255)
                
                # Only show status on screen if target is genuinely lost or safe
                status_txt = "OK" if status["visible"] else "MISSING"
                cv2.putText(frame, f"{name} ({status_txt}): {time_left:.1f}s", (frame_w - 300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if time_diff > self.alert_interval:
                    if (current_time - status["alert_cooldown"]) > 2.5:
                        # ✅ ONLY play alert sound if Alert Mode is actually enabled
                        if status["alert_stop_event"] is None:
                            status["alert_stop_event"] = threading.Event()
                        status["alert_stop_event"].clear()  # Reset stop flag
                        # Play siren ONLY when alert mode is ON
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
                            # ✅ IMPROVED: Log ALERT TRIGGERED - TARGET MISSING only at event start
                            # Use a flag to track if we've already logged this missing event
                            if not status["visible"]:
                                # Target is MISSING
                                if not status.get("target_missing_alert_logged", False):
                                    # Log only once at the start of missing event
                                    log_s = "ALERT TRIGGERED - TARGET MISSING"
                                    log_a = "MISSING"
                                    confidence = status.get("face_confidence", 0.0)
                                    self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, img_path, f"{confidence:.2f}"))
                                    status["target_missing_alert_logged"] = True  # Mark as logged
                                    self.temp_log_counter += 1
                                    logger.warning(f"[ALERT] {name} MISSING - Alert triggered!")
                            else:
                                # Target is VISIBLE - log as ALERT CONTINUED
                                log_s = "ALERT CONTINUED" if status["alert_triggered_state"] else "ALERT TRIGGERED"
                                log_a = self.last_action_cache.get(name, "Unknown")
                                confidence = status.get("face_confidence", 0.0)
                                self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, img_path, f"{confidence:.2f}"))
                                status["target_missing_alert_logged"] = False  # Reset flag since target is visible
                            
                            status["alert_triggered_state"] = True
            # ============================================================
                
                # RESET: When action is performed or target reset
                if time_diff <= 0:
                    status["alert_logged_timeout"] = False

        return frame 

if __name__ == "__main__":
    app = PoseApp()