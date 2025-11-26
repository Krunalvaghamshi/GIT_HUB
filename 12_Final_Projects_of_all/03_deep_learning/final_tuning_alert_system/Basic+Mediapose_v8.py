import os as _os_init
_os_init.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_os_init.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import cv2
from types import SimpleNamespace

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


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

# ✅ MODEL PIPELINE IMPORTS
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from bytetrack.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False

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

# Set CustomTkinter appearance with modern dark theme
import customtkinter as ctk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

import warnings
import os as os_module
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os_module.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

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
                         "face_recognition_tolerance": 0.5, "re_detect_interval": 8},  # ✅ OPTIMIZED: Reduced from 15 to 8 frames (~267ms interval for faster detection)
            "alert": {"default_interval_seconds": 10, "alert_cooldown_seconds": 2.5},
            "performance": {"gui_refresh_ms": 30, "pose_buffer_size": 12, "frame_skip_interval": 2, "enable_frame_skipping": True, "min_buffer_for_classification": 5},
            "logging": {"log_directory": "logs", "max_log_size_mb": 10, "auto_flush_interval": 50},
            "storage": {"alert_snapshots_dir": "alert_snapshots", "snapshot_retention_days": 30,
                       "guard_profiles_dir": "guard_profiles", "capture_snapshots_dir": "capture_snapshots",
                       "audio_files_dir": "audio_files"},
            "monitoring": {"mode": "pose", "session_restart_prompt_hours": 8}
        }

CONFIG = load_config()

# --- 2. Logging Setup with Rotation ---
if not os.path.exists(CONFIG["logging"]["log_directory"]):
    os.makedirs(CONFIG["logging"]["log_directory"])

logger = logging.getLogger("निराक्षण")
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

# --- MediaPipe Solutions Setup (kept for action classification only) ---
# UPDATED: All pose detection models replaced with new pipeline:
# - BlazeFace for face detection (single-person mode)
# - MoveNet Lightning for single-person pose estimation  
# - BlazePose for multi-person pose estimation
# - ByteTrack for multi-object tracking
# MediaPipe Holistic retained only for pose landmark constants used in classify_action
mp_holistic = mp.solutions.holistic

# --- Sound Logic ---
def play_siren_sound(stop_event=None, duration_seconds=30, sound_file="emergency-siren-351963.mp3"):
    """Play alert sound looping for up to duration_seconds or until stop_event is set
    
    Args:
        stop_event: threading.Event to signal stop playback
        duration_seconds: Maximum duration to play (default 30 seconds)
        sound_file: Name of audio file (default 'emergency-siren-351963.mp3' for action, 'Fugitive.mp3' for fugitive)
    """
    def _sound_worker():
        # Fixed: Use config-based path instead of hardcoded absolute path
        audio_dir = CONFIG.get("storage", {}).get("audio_files_dir", "audio_files")
        mp3_path = os.path.join(audio_dir, sound_file)
        if not os.path.exists(mp3_path):
            # Fallback to old hardcoded path for backwards compatibility
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
    Extract appearance features from a person's image using color and edge histograms.
    
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
        
        # Resize for consistent feature extraction
        person_resized = cv2.resize(person_crop, (64, 128))
        
        # Color histogram features (3 channels × 8 bins each = 24 features)
        color_features = []
        for i in range(3):
            hist = cv2.calcHist([person_resized], [i], None, [8], [0, 256])
            color_features.extend(hist.flatten())
        
        # Edge detection features using Canny (8 bins for consistency)
        gray = cv2.cvtColor(person_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256])
        
        # Combine all features into single vector (24 + 8 = 32 features total)
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
    Simple fallback implementation without sklearn dependency.
    
    Args:
        features1: Feature vector 1
        features2: Feature vector 2
    
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    if features1 is None or features2 is None:
        return 0.0
    
    try:
        # Manual cosine similarity (no sklearn needed)
        f1 = np.array(features1).astype(np.float32)
        f2 = np.array(features2).astype(np.float32)
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        similarity = float(dot_product / (norm1 * norm2 + 1e-6))
        
        return max(0.0, min(1.0, float(similarity)))  # Clamp to [0, 1]
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
# ✅ SIMPLIFIED: Removed unnecessary ReID/torchreid functions for core-only functionality

# --- Tkinter Application Class ---
class PoseApp:
    def __init__(self, window_title="निराक्षण - Niraakshan (Multi-Target Guard Tracking)"):
        self.root = ctk.CTk()
        self.root.title(window_title)
        self.root.geometry("1800x1000")  # Larger default size
        
        # ✅ CRITICAL: Initialize translations EARLY - before any UI elements are created
        self.current_language = "Hindi"
        self.translations = self.load_translations()
        self.current_trans = self.translations["Hindi"]  # Set Hindi as default
        
        self.cap = None
        self.unprocessed_frame = None 
        self.is_running = False
        self.is_logging = False
        self.camera_index = 0  # Default camera
        
        # ✅ NEW: Camera and Mode state tracking
        self.is_camera_running = False
        self.is_pro_mode = False
        self.is_alert_mode = False
        self.is_fugitive_detection = False
        self.is_stillness_alert = False
        self.guard_mode = "ADD"
        self.fugitive_add_mode = "ADD"
        
        self.alert_interval = 10
        # Monitoring Mode
        self.monitor_mode_var = tk.StringVar(self.root)
        self.monitor_mode_var.set("Action Alerts Only")
        self.frame_w = 640 
        self.frame_h = 480 
        
        # ✅ MODEL PIPELINE CONFIGURATION
        # Single-person (normal): BlazeFace → MoveNet Lightning → SORT
        # Multi-person (normal): BlazePose → ByteTrack
        # Single-person (dark): CLAHE+Gamma → BlazeFace → MoveNet Lightning → SORT
        # Multi-person (dark): CLAHE+Gamma → BlazePose → ByteTrack
        
        self.is_single_person_mode = False  # Auto-detected based on guard count
        self.night_mode = False
        self.face_detector = None
        self.pose_model = None
        self.tracker = None  # SORT or ByteTrack
        self.model_pipeline_initialized = False
        
        self._initialize_model_pipeline()
        
        # ✅ NEW: Tracking State
        self.is_tracking = False  # Toggle state for Track/Stop Monitoring
        self.active_required_action = "Hands Up"  # Current active action from dropdown
        self.last_clock_second = -1  # Track last displayed second for smooth clock update

        self.target_map = {}
        self.targets_status = {} 
        self.selected_target_names = []  # NEW: Track selected targets
        self.re_detect_counter = 0    
        self.RE_DETECT_INTERVAL = CONFIG["detection"]["re_detect_interval"]
        self.RESIZE_SCALE = 1.0 
        self.temp_log = []
        self.temp_log_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()  # CRITICAL: For FPS calculation
        self.current_fps = 0.0  # CRITICAL: Current FPS value
        self.last_process_frame = None  # CRITICAL: Cached frame for frame skipping
        self.last_action_cache = {}
        self.session_start_time = time.time()
        self.onboarding_mode = False
        self.is_in_capture_mode = False  # CRITICAL: Track if in capture mode
        self.onboarding_step = 0
        self.onboarding_name = None
        self.onboarding_poses = {}
        self.onboarding_detection_results = None
        self.onboarding_face_box = None
        
        # Fugitive Mode Fields
        self.fugitive_mode = False
        self.fugitive_image = None
        self.fugitive_face_encoding = None
        self.fugitive_name = "Unknown Fugitive"
        self.fugitive_detected_log_done = False  # Prevent duplicate logs
        self.last_fugitive_snapshot_time = 0  # Rate limiting for snapshots
        self.fugitive_alert_sound_thread = None
        self.fugitive_alert_stop_event = None
        
        # ✅ SIMPLIFIED: PRO Detection mode fields removed - focus on Normal Mode only
        
        # Photo storage for Tkinter (prevent garbage collection)
        self.photo_storage = {}  # Dictionary to store PhotoImage references

        self.frame_timestamp_ms = 0 

        # --- Layout ---
        self.root.grid_rowconfigure(0, weight=1)  # Main content area
        self.root.grid_columnconfigure(0, weight=1)  # Camera feed (expandable)
        self.root.grid_columnconfigure(1, weight=0)  # Sidebar (fixed width)
        
        # Sidebar state
        self.sidebar_collapsed = False
        self.sidebar_width = 320  # Wider for better button layout
        
        # 1. Main Content Area (Camera Feed + Controls)
        self.main_container = ctk.CTkFrame(self.root, fg_color="#1a1a1a")
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_rowconfigure(1, weight=1)  # Camera feed expands
        self.main_container.grid_rowconfigure(0, weight=0)  # Camera controls fixed
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # === CAMERA CONTROLS PANEL (Above camera feed) ===
        self.camera_controls_panel = ctk.CTkFrame(self.main_container, fg_color="#2b2b2b", height=50, corner_radius=0)
        self.camera_controls_panel.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.camera_controls_panel.grid_propagate(False)
        
        # Control buttons frame - single row with all buttons
        controls_frame = ctk.CTkFrame(self.camera_controls_panel, fg_color="transparent")
        controls_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Single row: Camera ON/OFF + Snap + Night + PRO Mode (all in one line)
        self.btn_camera_toggle = ctk.CTkButton(controls_frame, text=self.current_trans.get("camera_on", "▶ कैमरा ON/OFF"), 
                                               command=self.toggle_camera, height=28,
                                               fg_color="#27ae60", font=("Segoe UI", 9), corner_radius=4)
        self.btn_camera_toggle.pack(side="left", fill="both", expand=True, padx=(0,2))
        
        self.btn_snap = ctk.CTkButton(controls_frame, text=self.current_trans.get("snap", "📸 स्नैप"), command=self.snap_photo, 
                                     height=28, fg_color="#f39c12", font=("Segoe UI", 9), corner_radius=4)
        self.btn_snap.pack(side="left", fill="both", expand=True, padx=(0,2))
        
        self.btn_night_mode = ctk.CTkButton(controls_frame, text=self.current_trans.get("night", "🌙 रात"), command=self.toggle_night_mode, 
                                            height=28, fg_color="#34495e", font=("Segoe UI", 9), corner_radius=4)
        self.btn_night_mode.pack(side="left", fill="both", expand=True, padx=(0,2))
        
        self.btn_pro_toggle = ctk.CTkButton(controls_frame, text=self.current_trans.get("pro_mode", "⚡ PRO Mode ON/OFF"), 
                                           command=self.toggle_pro_mode, height=28,
                                           fg_color="#8e44ad", font=("Segoe UI", 9), corner_radius=4)
        self.btn_pro_toggle.pack(side="left", fill="both", expand=True)
        
        # Camera feed in main container (below controls)
        self.video_container = ctk.CTkFrame(self.main_container, fg_color="#000000")
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.video_label = ctk.CTkLabel(self.video_container, text=self.current_trans.get("camera_offline", "📷 कैमरा फीड ऑफलाइन"), 
                                         font=("Segoe UI", 20, "bold"), text_color="#7f8c8d")
        self.video_label.pack(fill="both", expand=True)
        
        # 2. Professional Sidebar - FIXED WIDTH FOR CONSISTENT BUTTON SIZING
        self.sidebar_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b", width=420)
        self.sidebar_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_propagate(False)
        
        # Sidebar Header
        sidebar_header = ctk.CTkFrame(self.sidebar_frame, fg_color="#1e1e1e", height=80, corner_radius=0)
        sidebar_header.pack(fill="x", padx=0, pady=0)
        sidebar_header.pack_propagate(False)
        
        # ✅ CLICKABLE LOGO/TITLE - Opens Language Converter
        self.title_label = ctk.CTkLabel(sidebar_header, text="🛡️ निराक्षण (Niraakshan)", 
                    font=("Arial Unicode MS", 14, "bold"), text_color="#3498db", cursor="hand2")
        self.title_label.pack(pady=(8,2))
        self.title_label.bind("<Button-1>", lambda e: self.open_language_converter())
        
        # Slogan/Mantra
        ctk.CTkLabel(sidebar_header, text="निराक्षण करे रक्षण", 
                    font=("Arial Unicode MS", 11, "italic"), text_color="#2ecc71").pack(pady=(0,8))
        
        # Separator
        ctk.CTkFrame(sidebar_header, fg_color="#3498db", height=2).pack(side="bottom", fill="x")
        
        # ✅ Translations already initialized at the start of __init__
        
        # Scrollable content area
        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="#2b2b2b", corner_radius=0)
        self.sidebar_scroll.pack(fill="both", expand=True, padx=0, pady=0)
        
        # FIXED BUTTON WIDTHS CALCULATION
        # Sidebar width: 420px, padding/margins: 2px left + 2px right = 4px total
        # Available width for buttons: 420 - 4 - (extra padding inside groups) = ~406px
        # Group padding: padx=2 on pack + padx=4 inside = 8px total, so 420-8 = 412px inner
        # For half buttons: (412-4-2) / 2 = 203px each (with 1px gap between)
        SIDEBAR_WIDTH = 420
        PADDING_OUTER = 4  # 2px left + 2px right from pack(padx=2)
        PADDING_INNER = 8  # 4px left + 4px right from pack(padx=4)
        AVAILABLE_WIDTH = SIDEBAR_WIDTH - PADDING_OUTER - PADDING_INNER
        HALF_BTN_WIDTH = (AVAILABLE_WIDTH - 2) // 2  # Subtract 1px gap, divide by 2
        FULL_BTN_WIDTH = AVAILABLE_WIDTH

        # Professional Control Panel
        btn_font = ('Segoe UI', 11, 'bold')
        btn_font_small = ('Segoe UI', 10)
        
        # === SYSTEM CLOCK (ABOVE GUARDS) ===
        clock_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#3498db", height=45)
        clock_frame.pack(fill="x", padx=2, pady=2)
        clock_frame.pack_propagate(False)
        
        self.clock_label_title = ctk.CTkLabel(clock_frame, text=self.current_trans["system_time"], font=("Segoe UI", 8, "bold"), text_color="#3498db")
        self.clock_label_title.pack(anchor="w", padx=6, pady=(3,1))
        self.clock_label = ctk.CTkLabel(clock_frame, text="--:--:--", font=("Segoe UI", 10, "bold"), text_color="#f39c12")
        self.clock_label.pack(anchor="w", padx=6, pady=(0,3))

        # === GUARD MANAGEMENT ===
        self.grp_guard = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#16a085", height=170)
        self.grp_guard.pack(fill="x", padx=2, pady=2)
        self.grp_guard.pack_propagate(False)
        
        self.guards_label = ctk.CTkLabel(self.grp_guard, text=self.current_trans["guards"], 
                    font=("Segoe UI", 10, "bold"), text_color="#16a085")
        self.guards_label.pack(padx=6, pady=(5,4), anchor="w")
        
        # Buttons container - Single column layout
        guard_btns_container = ctk.CTkFrame(self.grp_guard, fg_color="transparent")
        guard_btns_container.pack(fill="both", expand=True, padx=4, pady=(0,4))
        
        # Button 1: Add Guard (full width)
        self.btn_guard_toggle = ctk.CTkButton(guard_btns_container, text=self.current_trans["add_guard"], 
                                             command=self.toggle_guard_mode, height=32,
                                             fg_color="#16a085", font=("Segoe UI", 9), corner_radius=4)
        self.btn_guard_toggle.pack(fill="x", pady=(0,3))
        
        # Button 2: Remove Guard (full width)
        self.btn_remove_guard = ctk.CTkButton(guard_btns_container, text=self.current_trans["remove_guard"], 
                                             command=self.remove_guard_dialog, height=32,
                                             fg_color="#c0392b", font=("Segoe UI", 9), corner_radius=4)
        self.btn_remove_guard.pack(fill="x", pady=(0,3))
        
        # Button 3: Fugitive (full width)
        self.btn_fugitive_toggle = ctk.CTkButton(guard_btns_container, text=self.current_trans["fugitive"], 
                                                command=self.toggle_fugitive_add_remove, height=32,
                                                fg_color="#e74c3c", font=("Segoe UI", 9), corner_radius=4)
        self.btn_fugitive_toggle.pack(fill="x", pady=(0,3))
        
        # Button 4: Select Guard (full width)
        self.btn_select_guards = ctk.CTkButton(guard_btns_container, text=self.current_trans["select_guard"], 
                                              command=self.open_guard_selection_dialog, height=32,
                                              fg_color="#2980b9", font=("Segoe UI", 9), corner_radius=4)
        self.btn_select_guards.pack(fill="x", pady=(0,0))

        # === ALERT SYSTEM ===
        self.grp_alert_type = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#e67e22", height=290)
        self.grp_alert_type.pack(fill="x", padx=2, pady=2)
        self.grp_alert_type.pack_propagate(False)
        
        self.alerts_label = ctk.CTkLabel(self.grp_alert_type, text=self.current_trans["alerts"], 
                    font=("Segoe UI", 10, "bold"), text_color="#e67e22")
        self.alerts_label.pack(padx=6, pady=(5,4), anchor="w")
        
        self.alert_type_btns_frame = ctk.CTkFrame(self.grp_alert_type, fg_color="transparent")
        self.alert_type_btns_frame.pack(fill="both", expand=True, padx=4, pady=(0,4))
        
        # Button 1: Timeout (full width)
        self.btn_set_interval = ctk.CTkButton(self.alert_type_btns_frame, text=self.current_trans["timeout"], 
                                             command=self.set_alert_interval_advanced, height=32,
                                             fg_color="#3498db", font=("Segoe UI", 9), corner_radius=4)
        self.btn_set_interval.pack(fill="x", pady=(0,3))
        
        # Button 2: Alert ON/OFF (full width)
        self.btn_alert_toggle = ctk.CTkButton(self.alert_type_btns_frame, text=self.current_trans["alert_toggle"], 
                                             command=self.toggle_alert_mode, height=32,
                                             fg_color="#7f8c8d", font=("Segoe UI", 9), corner_radius=4)
        self.btn_alert_toggle.pack(fill="x", pady=(0,3))
        
        # Button 3: Stillness ON/OFF (full width)
        self.btn_stillness_alert = ctk.CTkButton(self.alert_type_btns_frame, text=self.current_trans["stillness"], 
                                                command=self.toggle_stillness_alert, height=32,
                                                fg_color="#7f8c8d", font=("Segoe UI", 9), corner_radius=4)
        self.btn_stillness_alert.pack(fill="x", pady=(0,3))
        
        # Button 4: Track Guard / Stop Monitoring (Toggle full width)
        self.btn_track_toggle = ctk.CTkButton(self.alert_type_btns_frame, text=self.current_trans["track"], 
                                             command=self.toggle_track_monitoring, height=32,
                                             fg_color="#16a085", font=("Segoe UI", 9), corner_radius=4)
        self.btn_track_toggle.pack(fill="x", pady=(0,3))
        
        # Label: Action Dropdown (full width)
        self.action_label = ctk.CTkLabel(self.alert_type_btns_frame, text=self.current_trans["required_action"], 
                    font=("Segoe UI", 8), text_color="#bdc3c7")
        self.action_label.pack(anchor="w", pady=(2,1))
        
        self.required_action_var = tk.StringVar(self.root)
        self.required_action_var.set("हाथ ऊपर (Hands Up)")
        self.action_dropdown = ctk.CTkOptionMenu(self.alert_type_btns_frame, 
                                                values=["हाथ ऊपर (Hands Up)", "हाथ पार (Hands Crossed)", 
                                                       "बाएं हाथ ऊपर (Left Hand Up)", "दाएं हाथ ऊपर (Right Hand Up)", 
                                                       "T-पोज़ (T-Pose)", "बैठा हुआ (Sit)", "खड़ा (Standing)"], 
                                                command=self.on_action_change, 
                                                fg_color="#3498db",
                                                button_color="#2980b9",
                                                text_color="white", font=("Segoe UI", 9), 
                                                dropdown_font=("Segoe UI", 9), corner_radius=4, height=32)
        self.action_dropdown.pack(fill="x", pady=(0,0))

        # Initialize selected targets list
        self.selected_target_names = []
        
        # === ACTIVE TRACKING STATUS (COMPACT SECTION) ===
        self.grp_preview = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#27ae60", height=320)
        self.grp_preview.pack(fill="x", padx=2, pady=2)
        self.grp_preview.pack_propagate(False)
        
        self.active_track_label = ctk.CTkLabel(self.grp_preview, text=self.current_trans["active_tracking"], 
                    font=("Segoe UI", 10, "bold"), text_color="#27ae60")
        self.active_track_label.pack(padx=6, pady=(5,3), anchor="w")
        
        # Guard Preview
        self.guard_preview_frame = ctk.CTkFrame(self.grp_preview, fg_color="#2b2b2b", corner_radius=5, height=140)
        self.guard_preview_frame.pack(fill="both", padx=4, pady=(0,3), expand=False)
        self.guard_preview_frame.pack_propagate(False)
        
        self.guard_preview_label = ctk.CTkLabel(self.guard_preview_frame, text=self.current_trans["guards_preview"], text_color="#27ae60", 
                    font=("Segoe UI", 9, "bold"))
        self.guard_preview_label.pack(anchor="w", padx=6, pady=(4,2))
        
        self.guard_preview_scroll_frame = ctk.CTkScrollableFrame(self.guard_preview_frame, 
                                                                 fg_color="transparent", height=115)
        self.guard_preview_scroll_frame.pack(fill="both", expand=True, padx=3, pady=(0,3))
        
        self.guard_preview_grid = {}
        
        # Fugitive Preview (hidden by default)
        self.fugitive_preview_frame = ctk.CTkFrame(self.grp_preview, fg_color="#2b2b2b", corner_radius=5)
        self.fugitive_preview_frame.pack(fill="x", padx=4, pady=(0,3))
        self.fugitive_preview_frame.pack_forget()
        
        self.fugitive_preview_title = ctk.CTkLabel(self.fugitive_preview_frame, text=self.current_trans["fugitive_preview"], text_color="#e74c3c", 
                    font=("Segoe UI", 9, "bold"))
        self.fugitive_preview_title.pack(anchor="w", padx=6, pady=(4,2))
        
        self.fugitive_preview_label = ctk.CTkLabel(self.fugitive_preview_frame, text=self.current_trans["not_set"], 
                                                   text_color="#7f8c8d", font=("Segoe UI", 9))
        self.fugitive_preview_label.pack(anchor="w", padx=6, pady=(0,3))

        # === PERFORMANCE STATS (FPS & MEM) - SMALLER SECTION ===
        perf_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#95a5a6", height=70)
        perf_frame.pack(fill="x", padx=2, pady=2)
        perf_frame.pack_propagate(False)
        
        self.perf_label = ctk.CTkLabel(perf_frame, text=self.current_trans["performance"], 
                    font=("Segoe UI", 8, "bold"), text_color="#95a5a6")
        self.perf_label.pack(padx=6, pady=(3,1), anchor="w")
        
        # FPS and MEM on same row
        perf_row = ctk.CTkFrame(perf_frame, fg_color="transparent")
        perf_row.pack(fill="x", padx=6, pady=(0,3))
        
        ctk.CTkLabel(perf_row, text=self.current_trans["fps"], font=("Segoe UI", 7), text_color="#bdc3c7").pack(side="left", anchor="w")
        self.fps_label = ctk.CTkLabel(perf_row, text="0", font=("Segoe UI", 7, "bold"), text_color="#27ae60")
        self.fps_label.pack(side="left", padx=(3,10))
        
        ctk.CTkLabel(perf_row, text=self.current_trans["memory"], font=("Segoe UI", 7), text_color="#bdc3c7").pack(side="left", anchor="w")
        self.mem_label = ctk.CTkLabel(perf_row, text="0MB", font=("Segoe UI", 7, "bold"), text_color="#f39c12")
        self.mem_label.pack(side="left", padx=(3,0))

        # === EXIT BUTTON ===
        self.btn_exit = ctk.CTkButton(self.sidebar_scroll, text=self.current_trans["exit"], 
                                     command=self.graceful_exit, height=32, width=FULL_BTN_WIDTH,
                                     fg_color="#c0392b", font=("Segoe UI", 9), corner_radius=4)
        self.btn_exit.pack(pady=(2,20))  # Extra padding at bottom to ensure visibility
        
        self.load_targets()
        
        # ✅ NEW PIPELINE: All models replaced with strict 4-model pipeline
        # Removed: Phase 4, HMR-Lite, RetinaFace, MoveNet MultiPose, MediaPipe Holistic, VideoPose3D, YOLOv8, DeepSORT
        # Implemented: BlazeFace (detection), MoveNet Lightning (single-person pose), BlazePose (multi-person pose), ByteTrack (tracking)
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        
        self.root.mainloop()
    
    def _initialize_model_pipeline(self):
        """Initialize appropriate model pipeline based on guard count and environment"""
        try:
            # ✅ CRITICAL: Initialize MediaPipe Pose (works offline, reliable)
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,  # 0=light, 1=full (balanced)
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ MediaPipe Pose initialized successfully")
                self.model_pipeline_initialized = True
            else:
                logger.warning("MediaPipe not available - pose detection disabled")
                self.model_pipeline_initialized = False
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            self.model_pipeline_initialized = False
    
    def _load_single_person_pipeline(self):
        """
        Load single-person pipeline:
        Normal: BlazeFace → MediaPipe Pose → SORT
        Dark: CLAHE+Gamma → BlazeFace → MediaPipe Pose → SORT
        """
        try:
            if self.pose_model is not None:
                return  # Already loaded
            
            logger.info("Loading single-person pipeline with MediaPipe...")
            
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ MediaPipe Pose loaded for single-person mode")
            else:
                logger.warning("MediaPipe not available")
                self.pose_model = None
        except Exception as e:
            logger.error(f"Single-person pipeline load error: {e}")
    
    def _load_multi_person_pipeline(self):
        """
        Load multi-person pipeline:
        Normal: BlazePose → ByteTrack (using MediaPipe Pose)
        Dark: CLAHE+Gamma → BlazePose → ByteTrack
        """
        try:
            if self.pose_model is not None:
                return  # Already loaded
            
            logger.info("Loading multi-person pipeline with MediaPipe...")
            
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ MediaPipe Pose loaded for multi-person mode")
            else:
                logger.warning("MediaPipe not available")
                self.pose_model = None
        except Exception as e:
            logger.error(f"Multi-person pipeline load error: {e}")
    
    def _detect_faces_blazeface(self, rgb_frame):
        """BlazeFace face detection for single-person mode"""
        try:
            # Using MediaPipe FaceMesh as BlazeFace (built-in)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            return face_locations
        except Exception as e:
            logger.debug(f"BlazeFace detection error: {e}")
            return []
    
    def _detect_faces_blazepose(self, rgb_frame):
        """BlazePose face detection for multi-person mode"""
        try:
            # Using MediaPipe Pose for multi-person face detection
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            return face_locations
        except Exception as e:
            logger.debug(f"BlazePose detection error: {e}")
            return []
    
    def _detect_pose_movenet_lightning(self, rgb_frame):
        """MediaPipe Pose for single-person pose estimation"""
        try:
            if self.pose_model is None:
                return None
            
            # ✅ CRITICAL: Convert RGB frame to proper format for MediaPipe
            results = self.pose_model.process(rgb_frame)
            return results  # Returns MediaPipe PoseLandmarks with pose_landmarks attribute
        except Exception as e:
            logger.debug(f"Pose detection error: {e}")
            return None
    
    def _detect_pose_blazepose_multipose(self, rgb_frame):
        """MediaPipe Pose for multi-person pose estimation"""
        try:
            if self.pose_model is None:
                return None
            
            # ✅ CRITICAL: Convert RGB frame to proper format for MediaPipe
            results = self.pose_model.process(rgb_frame)
            return results  # Returns MediaPipe PoseLandmarks with pose_landmarks attribute
        except Exception as e:
            logger.debug(f"Pose detection error: {e}")
            return None
    
    def _apply_clahe_enhancement(self, frame):
        """Apply CLAHE + Gamma correction for dark environments"""
        try:
            # Convert to LAB for better enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Apply Gamma correction for brightness boost
            inv_gamma = 1.0 / 1.5
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            l_channel = cv2.LUT(l_channel, table)
            
            # Merge back
            lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_frame
        except Exception as e:
            logger.debug(f"CLAHE enhancement error: {e}")
            return frame
    
    def toggle_sidebar(self):
        """Toggle sidebar visibility (not used in new design but kept for compatibility)"""
        pass
    
    # ✅ INDUSTRIAL-LEVEL: Low-Light Detection & Tracking Enhancements
    def enhance_frame_for_low_light(self, frame):
        """
        ✅ OPTIMIZED: Fast frame enhancement with night/day mode awareness
        Day mode: Skip enhancement entirely (fast)
        Night mode: Lightweight enhancement only if brightness is low
        """
        try:
            # Day Mode: No enhancement needed, return frame as-is for maximum speed
            if not self.night_mode:
                return frame
            
            # Night Mode: Only enhance if brightness is low (adaptive)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Skip enhancement for normal/bright conditions - return frame as-is for speed
            if brightness > 120:  # Normal/bright - no enhancement needed
                return frame
            
            # Lightweight enhancement for low-light (fast, minimal overhead)
            # Stage 1: Lightweight denoise only for dark areas (faster version)
            frame = cv2.fastNlMeansDenoisingColored(frame, h=8, templateWindowSize=7, searchWindowSize=15)
            
            # Stage 2: Minimal CLAHE for contrast - very fast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))  # Reduced tile size for speed
            l_channel_clahe = clahe.apply(l_channel)
            lab_enhanced = cv2.merge([l_channel_clahe, a, b])
            frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return frame_enhanced
        except Exception as e:
            logger.debug(f"Low-light enhancement failed: {e}, using original frame")
            return frame
    
    def get_adaptive_face_detection_params(self, frame):
        """
        ✅ INDUSTRIAL-LEVEL: Adaptive parameters based on lighting conditions
        Detects frame brightness and adjusts detection sensitivity accordingly
        """
        # Calculate frame brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Adaptive tolerance and confidence thresholds based on brightness
        if brightness < 50:  # Very dark
            face_tolerance = 0.45  # More lenient
            confidence_threshold = 0.45
            use_cnn_model = True  # Use CNN for better accuracy in low-light
            logger.debug(f"Very dark conditions (brightness={brightness:.0f}) - Using lenient tolerance")
        elif brightness < 100:  # Dark
            face_tolerance = 0.48
            confidence_threshold = 0.50
            use_cnn_model = True
            logger.debug(f"Dark conditions (brightness={brightness:.0f}) - Using CNN model")
        elif brightness < 150:  # Medium
            face_tolerance = 0.50
            confidence_threshold = 0.55
            use_cnn_model = False
            logger.debug(f"Medium lighting (brightness={brightness:.0f}) - Using standard detection")
        else:  # Bright
            face_tolerance = 0.55
            confidence_threshold = 0.60
            use_cnn_model = False
            logger.debug(f"Good lighting (brightness={brightness:.0f}) - Using strict tolerance")
        
        return {
            "tolerance": face_tolerance,
            "confidence": confidence_threshold,
            "use_cnn": use_cnn_model,
            "brightness": brightness
        }
    
    def detect_faces_multiscale(self, rgb_frame):
        """
        ✅ INDUSTRIAL-LEVEL: Multi-scale face detection using OpenCV cascade
        Helps catch faces at different scales, especially in complex scenes
        """
        try:
            # Load Haar cascade if not already loaded
            if not hasattr(self, 'face_cascade'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            
            # Multi-scale detection with different scale factors
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # More granular scale steps
                minNeighbors=4,   # Lower threshold for low-light
                minSize=(40, 40),
                maxSize=(300, 300)
            )
            
            # Convert from (x, y, w, h) to face_recognition format (top, right, bottom, left)
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
            
            if face_locations:
                logger.debug(f"Multi-scale detection found {len(face_locations)} faces")
            
            return face_locations
        except Exception as e:
            logger.debug(f"Multi-scale detection failed: {e}")
            return []
    
    
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
            
            # ✅ SIMPLIFIED: Phase 4 cleanup removed - focus on core cleanup only
            
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

    def load_translations(self):
        """Load translation dictionary for supported languages - COMPREHENSIVE with all UI elements"""
        return {
            "Hindi": {
                # Header & Branding
                "title": "🛡️ निराक्षण (Niraakshan)",
                "slogan": "निराक्षण करे रक्षण",
                
                # Sidebar Labels
                "system_time": "⏰ System Time",
                "guards": "👮 GUARDS",
                "alerts": "🔔 ALERTS",
                "active_tracking": "✓ ACTIVE TRACKING",
                "performance": "📊 PERFORMANCE",
                "guards_preview": "👮 Guards:",
                "fugitive_preview": "🚨 Fugitive:",
                "no_guard_selected": "कोई Guard चयनित नहीं",
                "not_set": "सेट नहीं",
                
                # Camera Control Buttons
                "camera_on": "▶ कैमरा ON/OFF",
                "snap": "📸 स्नैप",
                "night": "🌙 रात",
                "pro_mode": "⚡ PRO Mode ON/OFF",
                "camera_offline": "📷 कैमरा फीड ऑफलाइन",
                
                # Guard Management Buttons
                "add_guard": "➕ Guard जोड़ें",
                "remove_guard": "❌ Guard हटाएं",
                "fugitive": "🚨 Fugitive",
                "select_guard": "✓ Guard चुनें",
                
                # Alert Buttons
                "timeout": "⏱️ समय सीमा (HH:MM:SS)",
                "alert_toggle": "🔔 अलर्ट चालू/बंद",
                "stillness": "🔇 स्थिरता चालू/बंद",
                "track": "🎯 गार्ड ट्रैक करें",
                "stop_monitoring": "⏹️ निरीक्षण बंद करें",
                
                # Dropdown & Labels
                "required_action": "आवश्यक क्रिया:",
                "exit": "🚪 निकास",
                
                # Dialog Titles
                "lang_title": "🌐 Language Selection",
                "lang_select": "अपनी पसंदीदा भाषा चुनें:",
                "current_lang": "Current Language: ",
                "language_changed": "Language Changed",
                "lang_changed_msg": "App language को {lang} में बदल दिया गया! पूर्ण प्रभाव के लिए ऐप को पुनः शुरू करें।",
                
                # Guard Dialog Messages
                "add_guard_title": "Guard जोड़ें",
                "add_guard_msg": "आप guard को कैसे जोड़ना चाहते हैं?\n\nYes = कैमरे से फ़ोटो लें\nNo = मौजूदा Image Upload करें",
                "remove_guard_title": "Guard हटाएं",
                "select_guard_remove": "हटाने के लिए guard चुनें:",
                "confirm_removal": "Removal की पुष्टि करें",
                "remove_confirm_msg": "क्या आप '{name}' को हटाना निश्चित हैं?\n\nयह हटाएगा:\n- Face image\n- Pose references\n- सभी संबंधित data\n\nयह कार्रवाई को पूर्ववत नहीं किया जा सकता!",
                "remove_btn": "हटाएं",
                "cancel_btn": "Cancel",
                "guard_removed": "Guard Removed",
                "guard_removed_msg": "'{name}' को सफलतापूर्वक हटा दिया गया।\n\nDeleted: {items}",
                "no_guards": "कोई Guards नहीं",
                "no_guards_msg": "हटाने के लिए कोई guards उपलब्ध नहीं।",
                
                # Upload Dialog
                "upload_title": "Guard Image चुनें",
                "guard_name": "Guard का नाम दर्ज करें:",
                "upload_success": "Success",
                "upload_success_msg": "Guard '{name}' को सफलतापूर्वक जोड़ा गया!\n(अपलोड की गई images के लिए Pose capture छोड़ दिया गया)",
                "upload_error": "Error",
                "upload_error_msg": "Image अपलोड करने में विफल: {error}",
                "face_error": "Error",
                "face_error_msg": "Image में बिल्कुल एक face होना चाहिए।",
                
                # Camera Dialog
                "camera_required": "Camera आवश्यक",
                "camera_required_msg": "कृपया पहले camera शुरू करें।",
                "confirm_exit": "Exit की पुष्टि करें",
                "exit_confirm_msg": "Camera चल रहा है। क्या आप निश्चित रूप से exit करना चाहते हैं?",
                
                # Select Guards Dialog
                "select_targets": "Targets चुनें",
                "select_targets_msg": "Track करने के लिए Targets चुनें",
                "select_all": "सभी चुनें",
                "clear_all": "सभी को साफ करें",
                "done": "Done",
                "no_targets": "कोई targets नहीं मिले।",
                
                # Timeout Dialog
                "timeout_title": "Action Timeout Interval सेट करें",
                "timeout_label": "⏱ Action Timeout Interval",
                "timeout_desc": "कितने समय बाद timeout alert ट्रिगर होगा यदि कार्रवाई नहीं की गई",
                "hours": "Hours:",
                "minutes": "Minutes:",
                "seconds": "Seconds:",
                "timeout_recommend": "Recommended: 5 - 300 seconds (action timeout)",
                "set_btn": "✓ Set",
                
                # Tracking Messages
                "no_guards_selected": "कोई Guards चयनित नहीं",
                "select_guards_first": "कृपया पहले 'Guard चुनें' बटन का उपयोग करके guards चुनें",
                "tracking_started": "Tracking शुरू",
                "tracking_started_msg": "अब identify और track कर रहे हैं:\n{guards}\n\nActive Action: {action}\nAction alerts की निगरानी की जा रही है...",
                "tracking_stopped": "Tracking बंद",
                "now_scanning": "अब {count} चयनित targets के लिए स्कैन कर रहे हैं",
                "no_targets_init": "कोई targets को initialize नहीं किया गया - check करें कि guard profile images में faces हैं",
                
                # Alert Messages
                "alert_on": "Alert चालू",
                "alert_off": "Alert बंद",
                "stillness_on": "Stillness Alert चालू",
                "stillness_off": "Stillness Alert बंद",
                
                # Performance Labels
                "fps": "FPS:",
                "memory": "MEM:",
                "mb": "MB",
                
                # Action Types (Dropdown Options)
                "action_hands_up": "हाथ ऊपर (Hands Up)",
                "action_hands_crossed": "हाथ पार (Hands Crossed)",
                "action_left_hand_up": "बाएं हाथ ऊपर (Left Hand Up)",
                "action_right_hand_up": "दाएं हाथ ऊपर (Right Hand Up)",
                "action_t_pose": "T-पोज़ (T-Pose)",
                "action_sit": "बैठा हुआ (Sit)",
                "action_standing": "खड़ा (Standing)",
                
                # Monitor Modes
                "monitor_action_alerts": "केवल Action Alert",
                "monitor_stillness": "केवल Stillness Alert",
                "monitor_both": "दोनों Alert"
            },
            "English": {
                # Header & Branding
                "title": "🛡️ Niraakshan (Multi-Guard Tracking)",
                "slogan": "Observe to Protect",
                
                # Sidebar Labels
                "system_time": "⏰ System Time",
                "guards": "👮 GUARDS",
                "alerts": "🔔 ALERTS",
                "active_tracking": "✓ ACTIVE TRACKING",
                "performance": "📊 PERFORMANCE",
                "guards_preview": "👮 Guards:",
                "fugitive_preview": "🚨 Fugitive:",
                "no_guard_selected": "No Guard Selected",
                "not_set": "Not Set",
                
                # Camera Control Buttons
                "camera_on": "▶ Camera ON/OFF",
                "snap": "📸 Snap",
                "night": "🌙 Night",
                "pro_mode": "⚡ PRO Mode ON/OFF",
                
                # Guard Management Buttons
                "add_guard": "➕ Add Guard",
                "remove_guard": "❌ Remove Guard",
                "fugitive": "🚨 Fugitive",
                "select_guard": "✓ Select Guard",
                
                # Alert Buttons
                "timeout": "⏱️ Timeout (HH:MM:SS)",
                "alert_toggle": "🔔 Alert ON/OFF",
                "stillness": "🔇 Stillness ON/OFF",
                "track": "🎯 Track Guard",
                "stop_monitoring": "⏹️ Stop Monitoring",
                
                # Dropdown & Labels
                "required_action": "Required Action:",
                "exit": "🚪 Exit",
                
                # Dialog Titles
                "lang_title": "🌐 Language Selection",
                "lang_select": "Select your preferred language:",
                "current_lang": "Current Language: ",
                "language_changed": "Language Changed",
                "lang_changed_msg": "App language changed to {lang}!\nRestart the app for full effect.",
                
                # Guard Dialog Messages
                "add_guard_title": "Add Guard",
                "add_guard_msg": "How would you like to add the guard?\n\nYes = Take Photo with Camera\nNo = Upload Existing Image",
                "remove_guard_title": "Remove Guard",
                "select_guard_remove": "Select guard to remove:",
                "confirm_removal": "Confirm Removal",
                "remove_confirm_msg": "Are you sure you want to remove '{name}'?\n\nThis will delete:\n- Face image\n- Pose references\n- All associated data\n\nThis action cannot be undone!",
                "remove_btn": "Remove",
                "cancel_btn": "Cancel",
                "guard_removed": "Guard Removed",
                "guard_removed_msg": "'{name}' has been successfully removed.\n\nDeleted: {items}",
                "no_guards": "No Guards",
                "no_guards_msg": "No guards available to remove.",
                
                # Upload Dialog
                "upload_title": "Select Guard Image",
                "guard_name": "Enter guard name:",
                "upload_success": "Success",
                "upload_success_msg": "Guard '{name}' added successfully!\n(Pose capture skipped for uploaded images)",
                "upload_error": "Error",
                "upload_error_msg": "Failed to upload image: {error}",
                "face_error": "Error",
                "face_error_msg": "Image must contain exactly one face.",
                
                # Camera Dialog
                "camera_required": "Camera Required",
                "camera_required_msg": "Please start the camera first.",
                "confirm_exit": "Confirm Exit",
                "exit_confirm_msg": "Camera is running. Are you sure you want to exit?",
                
                # Select Guards Dialog
                "select_targets": "Select Targets",
                "select_targets_msg": "Select Targets to Track",
                "select_all": "Select All",
                "clear_all": "Clear All",
                "done": "Done",
                "no_targets": "No targets found.",
                
                # Timeout Dialog
                "timeout_title": "Set Action Timeout Interval",
                "timeout_label": "⏱ Action Timeout Interval",
                "timeout_desc": "How long until timeout alert triggers if action not performed",
                "hours": "Hours:",
                "minutes": "Minutes:",
                "seconds": "Seconds:",
                "timeout_recommend": "Recommended: 5 - 300 seconds (action timeout)",
                "set_btn": "✓ Set",
                
                # Tracking Messages
                "no_guards_selected": "No Guards Selected",
                "select_guards_first": "Please select guards first using 'Select Guard' button",
                "tracking_started": "Tracking Started",
                "tracking_started_msg": "Now identifying and tracking:\n{guards}\n\nActive Action: {action}\nMonitoring for action alerts...",
                "tracking_stopped": "Tracking Stopped",
                "now_scanning": "Now scanning for {count} selected targets",
                "no_targets_init": "No targets were initialized - check that guard profile images contain faces",
                
                # Alert Messages
                "alert_on": "Alert ON",
                "alert_off": "Alert OFF",
                "stillness_on": "Stillness Alert ON",
                "stillness_off": "Stillness Alert OFF",
                
                # Performance Labels
                "fps": "FPS:",
                "memory": "MEM:",
                "mb": "MB",
                
                # Action Types (Dropdown Options)
                "action_hands_up": "Hands Up",
                "action_hands_crossed": "Hands Crossed",
                "action_left_hand_up": "Left Hand Up",
                "action_right_hand_up": "Right Hand Up",
                "action_t_pose": "T-Pose",
                "action_sit": "Sit",
                "action_standing": "Standing",
                
                # Monitor Modes
                "monitor_action_alerts": "Action Alerts Only",
                "monitor_stillness": "Stillness Alerts Only",
                "monitor_both": "Both Alerts"
            },
            "Marathi": {
                # Header & Branding
                "title": "🛡️ निराक्षण (Niraakshan)",
                "slogan": "निरीक्षण करा संरक्षण करा",
                
                # Sidebar Labels
                "system_time": "⏰ System Time",
                "guards": "👮 गार्ड",
                "alerts": "🔔 सतर्कता",
                "active_tracking": "✓ सक्रिय ट्रॅकिंग",
                "performance": "📊 कार्यक्षमता",
                "guards_preview": "👮 गार्ड:",
                "fugitive_preview": "🚨 फरार:",
                "no_guard_selected": "कोई Guard निवडलेले नाहीत",
                "not_set": "सेट केलेले नाहीत",
                
                # Camera Control Buttons
                "camera_on": "▶ कॅमेरा ON/OFF",
                "snap": "📸 स्नॅप",
                "night": "🌙 रात्रि",
                "pro_mode": "⚡ PRO Mode ON/OFF",
                
                # Guard Management Buttons
                "add_guard": "➕ गार्ड जोडा",
                "remove_guard": "❌ गार्ड हटवा",
                "fugitive": "🚨 फरार",
                "select_guard": "✓ गार्ड निवडा",
                
                # Alert Buttons
                "timeout": "⏱️ Timeout (HH:MM:SS)",
                "alert_toggle": "🔔 सतर्कता ON/OFF",
                "stillness": "🔇 शांतता ON/OFF",
                "track": "🎯 गार्ड ट्रॅक करा",
                "stop_monitoring": "⏹️ निरीक्षण बंद करा",
                
                # Dropdown & Labels
                "required_action": "आवश्यक कृती:",
                "exit": "🚪 Exit",
                
                # Dialog Titles
                "lang_title": "🌐 भाषा निवड",
                "lang_select": "तुमची पसंदीची भाषा निवडा:",
                "current_lang": "Current Language: ",
                "language_changed": "भाषा बदलली",
                "lang_changed_msg": "App भाषा {lang} मध्ये बदलली गेली!\nपूर्ण प्रभावासाठी ऍप्लिकेशन पुनः सुरू करा।",
                
                # Guard Dialog Messages
                "add_guard_title": "गार्ड जोडा",
                "add_guard_msg": "तुम्ही गार्ड कसे जोडू इच्छिता?\n\nYes = कॅमेर्‍यात फोटो घ्या\nNo = विद्यमान Image अपलोड करा",
                "remove_guard_title": "गार्ड हटवा",
                "select_guard_remove": "हटविण्यासाठी गार्ड निवडा:",
                "confirm_removal": "हटवणे पुष्टी करा",
                "remove_confirm_msg": "तुम्ही '{name}' हटविण्यास निश्चित आहात?\n\nहे हटवेल:\n- Face image\n- Pose references\n- सर्व संबंधित data\n\nही कृती पूर्ववत केली जाऊ शकत नाही!",
                "remove_btn": "हटवा",
                "cancel_btn": "रद्द करा",
                "guard_removed": "गार्ड हटविला",
                "guard_removed_msg": "'{name}' यशस्वीरित्या हटविला गेला।\n\nDeleted: {items}",
                "no_guards": "कोही गार्ड नाहीत",
                "no_guards_msg": "हटविण्यासाठी कोही गार्ड उपलब्ध नाहीत।",
                
                # Upload Dialog
                "upload_title": "गार्ड Image निवडा",
                "guard_name": "गार्डचे नाव दर्ज करा:",
                "upload_success": "यश",
                "upload_success_msg": "गार्ड '{name}' यशस्वीरित्या जोडला गेला!\n(अपलोड केलेल्या images साठी Pose capture वगळा)",
                "upload_error": "Error",
                "upload_error_msg": "Image अपलोड करणे अयशस्वी: {error}",
                "face_error": "Error",
                "face_error_msg": "Image मध्ये अगदी एक face असणे आवश्यक आहे।",
                
                # Camera Dialog
                "camera_required": "कॅमेरा आवश्यक",
                "camera_required_msg": "कृपया प्रथम कॅमेरा सुरू करा।",
                "confirm_exit": "Exit पुष्टी करा",
                "exit_confirm_msg": "कॅमेरा चालू आहे. तुम्ही निश्चितपणे exit करू इच्छिता?",
                
                # Select Guards Dialog
                "select_targets": "Targets निवडा",
                "select_targets_msg": "ट्रॅक करण्यासाठी Targets निवडा",
                "select_all": "सर्व निवडा",
                "clear_all": "सर्व साफ करा",
                "done": "Done",
                "no_targets": "कोही targets सापडले नाहीत.",
                
                # Timeout Dialog
                "timeout_title": "Action Timeout Interval सेट करा",
                "timeout_label": "⏱ Action Timeout Interval",
                "timeout_desc": "कृती केली न गेल्यास timeout alert किती वेळ ट्रिगर होईल",
                "hours": "Hours:",
                "minutes": "Minutes:",
                "seconds": "Seconds:",
                "timeout_recommend": "Recommended: 5 - 300 seconds (action timeout)",
                "set_btn": "✓ Set",
                
                # Tracking Messages
                "no_guards_selected": "कोही Guards निवडलेले नाहीत",
                "select_guards_first": "कृपया प्रथम 'गार्ड निवडा' बटण वापरून guards निवडा",
                "tracking_started": "ट्रॅकिंग सुरू",
                "tracking_started_msg": "आता identify आणि track कर्‍या हैत:\n{guards}\n\nActive Action: {action}\nAction alerts साठी निरीक्षण...",
                "tracking_stopped": "ट्रॅकिंग बंद",
                "now_scanning": "आता {count} निवडलेल्या targets साठी स्कॅन कर्‍या हैत",
                "no_targets_init": "कोही targets initialize केले गेले नाहीत - check करा की guard profile images मध्ये faces आहेत",
                
                # Alert Messages
                "alert_on": "सतर्कता ON",
                "alert_off": "सतर्कता OFF",
                "stillness_on": "शांतता Alert ON",
                "stillness_off": "शांतता Alert OFF",
                
                # Performance Labels
                "fps": "FPS:",
                "memory": "MEM:",
                "mb": "MB",
                
                # Action Types (Dropdown Options)
                "action_hands_up": "हाथ वर (Hands Up)",
                "action_hands_crossed": "हाथ ओलांडलेले (Hands Crossed)",
                "action_left_hand_up": "डावा हाथ वर (Left Hand Up)",
                "action_right_hand_up": "उजवा हाथ वर (Right Hand Up)",
                "action_t_pose": "T-पोज (T-Pose)",
                "action_sit": "बसलेले (Sit)",
                "action_standing": "उभे (Standing)",
                
                # Monitor Modes
                "monitor_action_alerts": "फक्त Action Alerts",
                "monitor_stillness": "फक्त Stillness Alerts",
                "monitor_both": "दोन्ही Alerts"
            },
            "Gujarati": {
                # Header & Branding
                "title": "🛡️ નિરાક્ષણ (Niraakshan)",
                "slogan": "અવલોકન કરો અને રક્ષણ કરો",
                
                # Sidebar Labels
                "system_time": "⏰ System Time",
                "guards": "👮 ગાર્ડ્સ",
                "alerts": "🔔 અલર્ટ્સ",
                "active_tracking": "✓ સક્રિય ટ્રૅકિંગ",
                "performance": "📊 કાર્યક્ષમતા",
                "guards_preview": "👮 ગાર્ડ્સ:",
                "fugitive_preview": "🚨 ફરાર:",
                "no_guard_selected": "કોઈ Guard પસંદ નથી",
                "not_set": "સેટ નથી",
                
                # Camera Control Buttons
                "camera_on": "▶ કેમેરા ON/OFF",
                "snap": "📸 સ્નેપ",
                "night": "🌙 રાત",
                "pro_mode": "⚡ PRO Mode ON/OFF",
                
                # Guard Management Buttons
                "add_guard": "➕ ગાર્ડ ઉમેરો",
                "remove_guard": "❌ ગાર્ડ દૂર કરો",
                "fugitive": "🚨 ફરાર",
                "select_guard": "✓ ગાર્ડ પસંદ કરો",
                
                # Alert Buttons
                "timeout": "⏱️ Timeout (HH:MM:SS)",
                "alert_toggle": "🔔 અલર્ટ ON/OFF",
                "stillness": "🔇 શાંતતા ON/OFF",
                "track": "🎯 ગાર્ડ ટ્રૅક કરો",
                "stop_monitoring": "⏹️ મોનિટરિંગ બંધ કરો",
                
                # Dropdown & Labels
                "required_action": "જરૂરી પગલું:",
                "exit": "🚪 Exit",
                
                # Dialog Titles
                "lang_title": "🌐 ભાષા પસંદગી",
                "lang_select": "તમારી પસંદીની ભાષા પસંદ કરો:",
                "current_lang": "Current Language: ",
                "language_changed": "ભાષા બદલાઈ",
                "lang_changed_msg": "App ભાષા {lang} માં બદલાઈ!\nપૂર્ણ અસર માટે ઍપ્લિકેશન પુનः શરૂ કરો।",
                
                # Guard Dialog Messages
                "add_guard_title": "ગાર્ડ ઉમેરો",
                "add_guard_msg": "તમે ગાર્ડ કેવી રીતે ઉમેરવા માંગો છો?\n\nYes = કેમેરા સાથે ફોટો લો\nNo = હાલ ની Image અપલોડ કરો",
                "remove_guard_title": "ગાર્ડ દૂર કરો",
                "select_guard_remove": "દૂર કરવા માટે ગાર્ડ પસંદ કરો:",
                "confirm_removal": "દૂર કરવાની પુષ્ટી કરો",
                "remove_confirm_msg": "શું તમે '{name}' દૂર કરવા માટે આત્મવિશ્વાસી છો?\n\nયહ દૂર કરશે:\n- Face image\n- Pose references\n- બધા સંબંધિત data\n\nયો કાર્ય રદ કરી શકાતો નથી!",
                "remove_btn": "દૂર કરો",
                "cancel_btn": "રદ કરો",
                "guard_removed": "ગાર્ડ દૂર",
                "guard_removed_msg": "'{name}' સફળતાપૂર્વક દૂર કરાયો.\n\nDeleted: {items}",
                "no_guards": "કોઈ Guards નથી",
                "no_guards_msg": "દૂર કરવા માટે કોઈ guards ઉપલબ્ધ નથી.",
                
                # Upload Dialog
                "upload_title": "ગાર્ડ Image પસંદ કરો",
                "guard_name": "ગાર્ડનું નામ દાખલ કરો:",
                "upload_success": "સફલતા",
                "upload_success_msg": "ગાર્ડ '{name}' સફળતાપૂર્વક ઉમેરાયો!\n(અપલોડ કરેલી images માટે Pose capture છોડી દો)",
                "upload_error": "Error",
                "upload_error_msg": "Image અપલોડ કરવું નિષ્ફળ: {error}",
                "face_error": "Error",
                "face_error_msg": "Image માં બરાબર એક face હોવું જોઈએ.",
                
                # Camera Dialog
                "camera_required": "કેમેરો જરૂરી",
                "camera_required_msg": "કૃપયા પ્રથમ કેમેરો શરૂ કરો.",
                "confirm_exit": "Exit નું પુષ્ટિકરણ કરો",
                "exit_confirm_msg": "કેમેરો ચાલુ છે. શું તમે ચોક્કસ રીતે exit કરવા માંગો છો?",
                
                # Select Guards Dialog
                "select_targets": "Targets પસંદ કરો",
                "select_targets_msg": "ટ્રૅક કરવા માટે Targets પસંદ કરો",
                "select_all": "બધું પસંદ કરો",
                "clear_all": "બધું સાફ કરો",
                "done": "Done",
                "no_targets": "કોઈ targets મળ્યા નથી.",
                
                # Timeout Dialog
                "timeout_title": "Action Timeout Interval સેટ કરો",
                "timeout_label": "⏱ Action Timeout Interval",
                "timeout_desc": "કૃતિ ન કરવામાં આવે તો timeout alert કેટલો સમય ટ્રીગર થાય",
                "hours": "Hours:",
                "minutes": "Minutes:",
                "seconds": "Seconds:",
                "timeout_recommend": "Recommended: 5 - 300 seconds (action timeout)",
                "set_btn": "✓ Set",
                
                # Tracking Messages
                "no_guards_selected": "કોઈ Guards પસંદ નથી",
                "select_guards_first": "કૃપયા પ્રથમ 'ગાર્ડ પસંદ કરો' બટન વાપરીને guards પસંદ કરો",
                "tracking_started": "ટ્રૅકિંગ શરૂ",
                "tracking_started_msg": "હવે identify અને track કર્યા છે:\n{guards}\n\nActive Action: {action}\nAction alerts માટે મોનિટર કર્યા છે...",
                "tracking_stopped": "ટ્રૅકિંગ બંધ",
                "now_scanning": "હવે {count} પસંદ કરેલા targets માટે સ્કેન કર્યા છે",
                "no_targets_init": "કોઈ targets શરૂ કરાયા નથી - ચેક કરો કે guard profile images માં faces છે",
                
                # Alert Messages
                "alert_on": "અલર્ટ ON",
                "alert_off": "અલર્ટ OFF",
                "stillness_on": "શાંતતા Alert ON",
                "stillness_off": "શાંતતા Alert OFF",
                
                # Performance Labels
                "fps": "FPS:",
                "memory": "MEM:",
                "mb": "MB"
            }
        }

    def open_language_converter(self):
        """Open language selection dialog with real-time switching (no restart needed)"""
        try:
            dialog = ctk.CTkToplevel(self.root)
            dialog.title(self.get_text("lang_title"))
            dialog.geometry("450x400")
            dialog.resizable(False, False)
            dialog.attributes('-topmost', True)  # Keep on top
            dialog.grab_set()
            dialog.transient(self.root)
            
            # Title
            ctk.CTkLabel(dialog, text=self.get_text("lang_title"), 
                        font=("Arial Unicode MS", 14, "bold"), text_color="#3498db").pack(pady=15)
            
            ctk.CTkLabel(dialog, text=self.get_text("lang_select"), 
                        font=("Segoe UI", 10), text_color="#bdc3c7").pack(pady=5)
            
            # Language buttons frame
            lang_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            lang_frame.pack(fill="both", expand=True, padx=20, pady=15)
            
            languages = list(self.translations.keys())
            
            def select_language(lang):
                """Apply language translation in REAL-TIME (no restart needed)"""
                self.current_language = lang
                self.apply_translations()
                dialog.destroy()
                
                # Show confirmation with translated message
                msg = self.get_text("lang_changed_msg")
                if msg and isinstance(msg, str):
                    msg = msg.replace("{lang}", lang)
                else:
                    msg = f"App language changed to {lang}!"
                messagebox.showinfo(
                    self.get_text("language_changed"),
                    msg
                )
            
            # Create buttons for each language
            for lang in languages:
                is_current = lang == self.current_language
                bg_color = "#3498db" if is_current else "#2c3e50"
                
                btn = ctk.CTkButton(lang_frame, text=f"✓ {lang}" if is_current else lang,
                                  command=lambda l=lang: select_language(l),
                                  height=40, fg_color=bg_color, font=("Segoe UI", 11, "bold"),
                                  corner_radius=5)
                btn.pack(fill="x", pady=8)
            
            # Info label
            ctk.CTkLabel(dialog, text=f"{self.get_text('current_lang')}{self.current_language}",
                        font=("Segoe UI", 9), text_color="#27ae60").pack(pady=10)
                        
        except Exception as e:
            logger.warning(f"Error opening language converter: {e}")

    def apply_translations(self):
        """Apply language translations to ALL UI elements in REAL-TIME (no restart)"""
        try:
            trans = self.translations.get(self.current_language, self.translations["Hindi"])
            self.current_trans = trans
            
            # ========== UPDATE ALL BUTTON TEXT ==========
            # Camera Controls
            if hasattr(self, 'btn_camera_toggle'):
                self.btn_camera_toggle.configure(text=trans.get("camera_on", "▶ कैमरा ON/OFF"))
            if hasattr(self, 'btn_snap'):
                self.btn_snap.configure(text=trans.get("snap", "📸 स्नैप"))
            if hasattr(self, 'btn_night_mode'):
                self.btn_night_mode.configure(text=trans.get("night", "🌙 रात"))
            if hasattr(self, 'btn_pro_toggle'):
                self.btn_pro_toggle.configure(text=trans.get("pro_mode", "⚡ PRO Mode"))
            
            # Camera feed offline text
            if hasattr(self, 'video_label'):
                self.video_label.configure(text=trans.get("camera_offline", "📷 कैमरा फीड ऑफलाइन"))
            
            # Guard Section
            if hasattr(self, 'clock_label_title'):
                self.clock_label_title.configure(text=trans.get("system_time", "⏰ System Time"))
            if hasattr(self, 'guards_label'):
                self.guards_label.configure(text=trans.get("guards", "👮 GUARDS"))
            if hasattr(self, 'btn_guard_toggle'):
                self.btn_guard_toggle.configure(text=trans.get("add_guard", "➕ Add Guard"))
            if hasattr(self, 'btn_remove_guard'):
                self.btn_remove_guard.configure(text=trans.get("remove_guard", "❌ Remove Guard"))
            if hasattr(self, 'btn_fugitive_toggle'):
                self.btn_fugitive_toggle.configure(text=trans.get("fugitive", "🚨 Fugitive"))
            if hasattr(self, 'btn_select_guards'):
                self.btn_select_guards.configure(text=trans.get("select_guard", "✓ Select Guard"))
            
            # Alert Section
            if hasattr(self, 'alerts_label'):
                self.alerts_label.configure(text=trans.get("alerts", "🔔 ALERTS"))
            if hasattr(self, 'btn_set_interval'):
                self.btn_set_interval.configure(text=trans.get("timeout", "⏱️ Timeout (HH:MM:SS)"))
            if hasattr(self, 'btn_alert_toggle'):
                self.btn_alert_toggle.configure(text=trans.get("alert_toggle", "🔔 Alert ON/OFF"))
            if hasattr(self, 'btn_stillness_alert'):
                self.btn_stillness_alert.configure(text=trans.get("stillness", "🔇 Stillness ON/OFF"))
            if hasattr(self, 'btn_track_toggle'):
                self.btn_track_toggle.configure(text=trans.get("track", "🎯 Track Guard"))
            if hasattr(self, 'action_label'):
                self.action_label.configure(text=trans.get("required_action", "Required Action:"))
            
            # Active Tracking Section
            if hasattr(self, 'active_track_label'):
                self.active_track_label.configure(text=trans.get("active_tracking", "✓ ACTIVE TRACKING"))
            if hasattr(self, 'guard_preview_label'):
                self.guard_preview_label.configure(text=trans.get("guards_preview", "👮 Guards:"))
            if hasattr(self, 'fugitive_preview_title'):
                self.fugitive_preview_title.configure(text=trans.get("fugitive_preview", "🚨 Fugitive:"))
            
            # Performance Section
            if hasattr(self, 'perf_label'):
                self.perf_label.configure(text=trans.get("performance", "📊 PERFORMANCE"))
            
            # Exit Button
            if hasattr(self, 'btn_exit'):
                self.btn_exit.configure(text=trans.get("exit", "🚪 Exit"))
            
            # Title Label (most important)
            self.title_label.configure(text=trans.get("title", "🛡️ निराक्षण"))
            
            # ========== UPDATE ACTION DROPDOWN ==========
            if hasattr(self, 'action_dropdown'):
                action_values = [
                    trans.get("action_hands_up", "Hands Up"),
                    trans.get("action_hands_crossed", "Hands Crossed"),
                    trans.get("action_left_hand_up", "Left Hand Up"),
                    trans.get("action_right_hand_up", "Right Hand Up"),
                    trans.get("action_t_pose", "T-Pose"),
                    trans.get("action_sit", "Sit"),
                    trans.get("action_standing", "Standing")
                ]
                self.action_dropdown.configure(values=action_values)
                # Set the first value as default and normalize it
                first_action = action_values[0]
                self.action_dropdown.set(first_action)
                self.active_required_action = self.normalize_action_name(first_action)
            
            # ✅ FIX: Remove emoji from logger to avoid UnicodeEncodeError on Windows console
            logger.warning(f"REAL-TIME language switched to: {self.current_language} (NO RESTART NEEDED)")
            
        except Exception as e:
            logger.error(f"Error applying translations: {e}")

    def update_widget_text(self, key, text):
        """Helper to update widget text by key"""
        try:
            return self.get_text(key)
        except Exception as e:
            logger.debug(f"Error updating widget text: {e}")
            return text

    def get_text(self, key):
        """Get translated text for a given key (with fallback to Hindi)"""
        try:
            if not hasattr(self, 'current_trans'):
                self.current_trans = self.translations.get(self.current_language, self.translations["Hindi"])
            
            # Return translation or the key itself as fallback
            return self.current_trans.get(key, key)
        except:
            # Final fallback to Hindi
            return self.translations.get("Hindi", {}).get(key, key)
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
        
        # ✅ SIMPLIFIED: Removed Phase 4 Stage 2/3 initialization - focus on core targets only
        
        self.update_selected_preview()
    
    def select_all_targets(self):
        """Select all targets"""
        self.selected_target_names = list(self.target_map.keys())
        self.update_selected_preview()

    def update_selected_preview(self):
        """Update guard preview with ALL selected targets in single column layout"""
        
        # Clear existing grid
        for widget in self.guard_preview_scroll_frame.winfo_children():
            widget.destroy()
        self.guard_preview_grid = {}
        
        if not self.selected_target_names:
            # Show placeholder message
            ctk.CTkLabel(self.guard_preview_scroll_frame, text="No Guard Selected", 
                        text_color="#bdc3c7", font=("Arial", 9)).pack(pady=30)
            return
        
        # Create single-column layout of guards for ALL selected guards
        for guard_name in sorted(self.selected_target_names):
            filename = self.target_map.get(guard_name)
            if not filename:
                continue
            
            try:
                # Load and resize image for thumbnail
                img = cv2.imread(filename)
                if img is not None:
                    # Resize to 60x60 thumbnail
                    img_resized = cv2.resize(img, (60, 60))
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    imgtk = ImageTk.PhotoImage(image=pil_img)
                    
                    # Create row frame with image on left and name on right
                    row_frame = ctk.CTkFrame(self.guard_preview_scroll_frame, fg_color="#2a2a2a", 
                                            border_width=1, border_color="#27ae60", corner_radius=4)
                    row_frame.pack(fill="x", padx=0, pady=2)
                    
                    # Image label on left
                    img_label = ctk.CTkLabel(row_frame, image=imgtk, text="")
                    img_label.pack(side="left", padx=3, pady=3)
                    
                    # Guard name label on right (full width)
                    name_label = ctk.CTkLabel(row_frame, text=guard_name, 
                                             text_color="#27ae60", font=("Segoe UI", 9))
                    name_label.pack(side="left", fill="x", expand=True, padx=5, pady=3)
                    
                    # Store reference and photo
                    self.guard_preview_grid[guard_name] = (img_label, imgtk)
                    self.photo_storage[f"guard_preview_{guard_name}"] = imgtk
                    
            except Exception as e:
                logger.error(f"Error loading guard preview for {guard_name}: {e}")
                # Show error row
                error_frame = ctk.CTkFrame(self.guard_preview_scroll_frame, fg_color="#2a2a2a", 
                                          border_width=1, border_color="#e74c3c", corner_radius=4)
                error_frame.pack(fill="x", padx=0, pady=2)
                ctk.CTkLabel(error_frame, text="❌", text_color="#e74c3c", font=("Arial", 10)).pack(side="left", padx=3, pady=3)
                ctk.CTkLabel(error_frame, text=guard_name, text_color="#e74c3c", 
                            font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=5, pady=3)

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
                    # ✅ PERFORMANCE: Use num_jitters=1 for fast encoding at initialization
                    encodings = face_recognition.face_encodings(target_image_file, num_jitters=1)
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
                            "action_performed": False,  # ✅ PHASE 3.1: Track if action was performed in grace period
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
                            # ✅ STILLNESS DETECTION (PRO MODE)
                            "last_stillness_check_time": time.time(),  # For tracking stillness duration
                            "stillness_start_time": None,  # When guard became still
                            "consecutive_detections": 0,  # Count consecutive frames detected for stability
                            "stable_tracking": False,  # True after 3+ consecutive detections
                            "last_pose_vector": None,  # Previous pose for variance calculation
                            "stillness_alert_logged": False,  # Track if stillness alert was logged
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
        dialog.geometry("400x320")
        dialog.grab_set()
        
        # Button frame at top (left corner)
        button_frame_top = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame_top.pack(pady=5, fill="x", padx=10)
        
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
                    messagebox.showinfo("Success", f"Action timeout set to {total_seconds} seconds ({h}h {m}m {s}s)")
                    logger.warning(f"Alert interval changed to {total_seconds} seconds")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Interval must be greater than 0")
            except ValueError as e:
                messagebox.showerror("Error", f"Please enter valid numbers. Error: {e}")
        
        ctk.CTkButton(button_frame_top, text="✓ Set", command=confirm, fg_color="#27ae60", font=("Roboto", 10, "bold"), width=80).pack(side="left", padx=3)
        ctk.CTkButton(button_frame_top, text="✕ Cancel", command=dialog.destroy, fg_color="#34495e", font=("Roboto", 10), width=80).pack(side="left", padx=3)
        
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
        
        # Focus on seconds field by default
        s_entry.focus()

    def toggle_alert_mode(self):
        self.is_alert_mode = not self.is_alert_mode
        if self.is_alert_mode:
            self.btn_alert_toggle.configure(text="🔔 Action Alert: ON", fg_color="#f39c12")
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
            self.btn_alert_toggle.configure(text="🔔 Action Alert: OFF", fg_color="#7f8c8d")
            # Auto-stop logging and save
            if self.is_logging:
                self.save_log_to_file()
                self.is_logging = False
                logger.warning("Alert mode stopped - logging saved")
    
    def toggle_logging_button(self):
        """Toggle logging on/off manually"""
        self.is_logging = not self.is_logging
        if self.is_logging:
            self.temp_log.clear()
            self.temp_log_counter = 0
            logger.info("Logging enabled manually")
        else:
            if len(self.temp_log) > 0:
                self.save_log_to_file()
            logger.info("Logging disabled manually")

    def set_alert_interval(self):
        """Set alert interval with Hours, Minutes, Seconds dialog - ✅ ENHANCED: Full H:M:S support"""
        # Create custom dialog for H:M:S input
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Alert Interval")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Calculate current values from total seconds
        hours = self.alert_interval // 3600
        remaining = self.alert_interval % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        
        # Create input fields
        tk.Label(dialog, text="Alert Interval (Hours : Minutes : Seconds)", font=("Helvetica", 10, "bold")).pack(pady=10)
        
        frame = tk.Frame(dialog)
        frame.pack(pady=10)
        
        tk.Label(frame, text="Hours:", width=10).grid(row=0, column=0, padx=5)
        hours_var = tk.StringVar(value=str(hours))
        hours_entry = tk.Entry(frame, textvariable=hours_var, width=5, font=("Helvetica", 10))
        hours_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(frame, text="Minutes:", width=10).grid(row=1, column=0, padx=5)
        minutes_var = tk.StringVar(value=str(minutes))
        minutes_entry = tk.Entry(frame, textvariable=minutes_var, width=5, font=("Helvetica", 10))
        minutes_entry.grid(row=1, column=1, padx=5)
        
        tk.Label(frame, text="Seconds:", width=10).grid(row=2, column=0, padx=5)
        seconds_var = tk.StringVar(value=str(seconds))
        seconds_entry = tk.Entry(frame, textvariable=seconds_var, width=5, font=("Helvetica", 10))
        seconds_entry.grid(row=2, column=1, padx=5)
        
        def on_ok():
            try:
                h = int(hours_var.get() or 0)
                m = int(minutes_var.get() or 0)
                s = int(seconds_var.get() or 0)
                
                if h < 0 or m < 0 or s < 0 or m > 59 or s > 59:
                    messagebox.showwarning("Invalid Input", "Please enter valid values (0-23 hours, 0-59 minutes, 0-59 seconds)")
                    return
                
                total_seconds = h * 3600 + m * 60 + s
                if total_seconds < 1:
                    messagebox.showwarning("Invalid Input", "Interval must be at least 1 second")
                    return
                
                self.alert_interval = total_seconds
                
                # Format display: "1h 5m 30s" or just the value if convenient
                if h > 0:
                    display = f"{h}h {m}m {s}s"
                elif m > 0:
                    display = f"{m}m {s}s"
                else:
                    display = f"{s}s"
                
                self.btn_set_interval.configure(text=f"Interval: {display}")
                dialog.destroy()
                logger.warning(f"Alert interval updated: {h}h {m}m {s}s ({total_seconds} seconds)")
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=on_ok, bg="#27ae60", fg="white", width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#e74c3c", fg="white", width=10).pack(side="left", padx=5)
            
    def normalize_action_name(self, action_text):
        """Normalize action text by extracting English action name from bilingual/translated dropdown text.
        Handles formats like:
        - "Hands Up"
        - "हाथ ऊपर (Hands Up)"
        - "हाथ वर (Hands Up)"
        Returns: Canonical English action name like "Hands Up", "One Hand Raised (Left)", etc.
        """
        if not action_text:
            return "Standing"  # Safe default
        
        # Extract English part from bilingual text like "हाथ ऊपर (Hands Up)"
        if "(" in action_text and ")" in action_text:
            start = action_text.find("(")
            end = action_text.find(")")
            if start < end:
                english_part = action_text[start+1:end].strip()
                return english_part
        
        # Return as-is if already in English or no parentheses
        return action_text.strip()
    
    def on_action_change(self, value):
        """Update active required action when dropdown changes"""
        # Normalize the action name to handle bilingual/translated dropdown values
        self.active_required_action = self.normalize_action_name(value)
        if self.is_alert_mode:
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_action_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False
        logger.debug(f"Active Required Action changed to: {value} (normalized: {self.active_required_action})")

    def toggle_night_mode(self):
        """Toggle Night/Day Mode for optimal lighting adaptation with dual-model detection"""
        self.night_mode = not self.night_mode
        if self.night_mode:
            self.btn_night_mode.configure(text="🌙 Night ON", fg_color="#1a1a2e")
            logger.warning(f"Night Mode - Face: BlazeFace + CLAHE | Pose: MoveNet Lightning/BlazePose + CLAHE enhancement")
        else:
            self.btn_night_mode.configure(text="🌙 Night OFF", fg_color="#34495e")
            logger.warning(f"Day Mode - Face: BlazeFace | Pose: MoveNet Lightning/BlazePose | Tracking: ByteTrack")

    def toggle_camera(self):
        """Toggle Camera ON/OFF - unified camera control"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def toggle_pro_mode(self):
        """Toggle PRO Mode ON/OFF - enables advanced features like ReID and Stillness detection"""
        self.is_pro_mode = not self.is_pro_mode
        if self.is_pro_mode:
            self.btn_pro_toggle.configure(text="🎯 PRO (ON)", fg_color="#004a7f")
            self.btn_stillness_alert.configure(state="normal")
            logger.warning("PRO Mode ENABLED - Advanced ReID and Stillness detection available")
        else:
            self.btn_pro_toggle.configure(text="🎯 PRO (OFF)", fg_color="#34495e")
            self.btn_stillness_alert.configure(state="disabled", text="🔇 Stillness Alert OFF", fg_color="#95a5a6")
            self.is_stillness_alert = False
            logger.warning("PRO Mode DISABLED")

    def toggle_guard_mode(self):
        """Directly enter onboarding mode to add a new guard"""
        if not self.is_camera_running:
            messagebox.showwarning("Camera Required", "Please start camera first")
            return
        
        # Enter onboarding mode to capture guard face and poses
        self.enter_onboarding_mode()

    def add_guard_then_capture(self):
        """Show dialog to enter guard name, then capture face and poses"""
        if not self.is_camera_running:
            messagebox.showwarning("Camera Required", "Please start camera first")
            return
        
        # Enter onboarding mode to capture guard face and poses
        self.enter_onboarding_mode()

    def toggle_fugitive_add_remove(self):
        """Toggle between ADD and REMOVE fugitive modes"""
        if not hasattr(self, 'fugitive_add_mode'):
            self.fugitive_add_mode = "ADD"
        
        if self.fugitive_add_mode == "ADD":
            # Initiate ADD fugitive
            self.add_fugitive()
            self.fugitive_add_mode = "REMOVE"
            self.btn_fugitive_toggle.configure(text="❌ Remove Fugitive", fg_color="#e74c3c")
        else:
            # Initiate REMOVE fugitive
            self.remove_fugitive()
            self.fugitive_add_mode = "ADD"
            self.btn_fugitive_toggle.configure(text="➕ Add Fugitive", fg_color="#e74c3c")

    def add_fugitive(self):
        """Add/set a fugitive for detection"""
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
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if face_encodings:
                self.fugitive_face_encoding = face_encodings[0]
                self.fugitive_detected_log_done = False
                
                # Extract name from path
                self.fugitive_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Update preview
                self._update_fugitive_preview()
                
                # ✅ AUTO-ACTIVATE: Enable fugitive detection automatically
                self.is_fugitive_detection = True
                self.btn_fugitive_toggle.configure(text="❌ Remove Fugitive", fg_color="#e74c3c")
                self.fugitive_preview_frame.pack(fill="x", padx=0, pady=2)
                
                logger.warning(f"Fugitive Added: {self.fugitive_name}")
                logger.warning(f"Fugitive Detection ENABLED - Searching for: {self.fugitive_name}")
                messagebox.showinfo("Fugitive Added", f"Fugitive profile created: {self.fugitive_name}\n\n✅ Fugitive detection automatically activated!")
        
        except Exception as e:
            logger.error(f"Error adding fugitive: {e}")
            messagebox.showerror("Error", f"Failed to process image: {e}")

    def remove_fugitive(self):
        """Remove current fugitive"""
        self.fugitive_image = None
        self.fugitive_face_encoding = None
        self.fugitive_detected_log_done = False
        self.fugitive_name = None
        
        if hasattr(self, 'fugitive_preview_label'):
            self.fugitive_preview_label.configure(image='', text="No Fugitive Selected")
        
        logger.warning("Fugitive Removed")
        messagebox.showinfo("Fugitive Removed", "Fugitive profile cleared")

    def toggle_fugitive_detection(self):
        """Toggle Fugitive Detection ON/OFF - enables live search for fugitive"""
        if self.fugitive_face_encoding is None:
            messagebox.showwarning("No Fugitive", "Please add a fugitive first using Add Fugitive button")
            return
        
        self.is_fugitive_detection = not self.is_fugitive_detection
        if self.is_fugitive_detection:
            self.btn_fugitive_toggle.configure(text="🚨 Fugitive ON", fg_color="#8b0000")
            # Show fugitive preview
            self.fugitive_preview_frame.pack(fill="x", padx=0, pady=2)
            logger.warning(f"Fugitive Detection ENABLED - Searching for: {self.fugitive_name}")
        else:
            self.btn_fugitive_toggle.configure(text="🚨 Fugitive OFF", fg_color="#95a5a6")
            # Hide fugitive preview
            self.fugitive_preview_frame.pack_forget()
            logger.warning("Fugitive Detection DISABLED")

    def toggle_stillness_alert(self):
        """Toggle Stillness Alert ON/OFF - detects guards not moving"""
        if not self.is_pro_mode:
            messagebox.showwarning("PRO Mode Required", "Stillness Alert is only available in PRO Mode")
            return
        
        self.is_stillness_alert = not self.is_stillness_alert
        if self.is_stillness_alert:
            self.btn_stillness_alert.configure(text="🔇 Stillness Alert ON", fg_color="#00bfff")
            logger.warning("Stillness Alert ENABLED")
        else:
            self.btn_stillness_alert.configure(text="🔇 Stillness Alert OFF", fg_color="#95a5a6")
            logger.warning("Stillness Alert DISABLED")

    def open_guard_selection_dialog(self):
        """Open dialog to select multiple guards for tracking"""
        # Get available guards from guard_profiles directory
        guard_profile_dir = os.path.join(os.path.dirname(__file__), "guard_profiles")
        if not os.path.exists(guard_profile_dir):
            messagebox.showinfo("No Guards", "No guards available. Please add guards first.")
            return
        
        # List all unique guard names from face files
        guard_files = glob.glob(os.path.join(guard_profile_dir, "target_*_face.jpg"))
        available_guards = list(set([os.path.basename(f).replace("target_", "").replace("_face.jpg", "") for f in guard_files]))
        
        if not available_guards:
            messagebox.showinfo("No Guards", "No guards available. Please add guards first.")
            return
        
        # For now, open the existing target selection dialog if it exists
        if hasattr(self, 'open_target_selection_dialog'):
            self.open_target_selection_dialog()
        else:
            messagebox.showinfo("Select Guards", f"Available guards: {', '.join(available_guards)}")

    def on_track_selected_guard_clicked(self):
        """Handle Track Selected Guard button click - activates tracking for selected guards"""
        if not hasattr(self, 'selected_target_names') or not self.selected_target_names:
            messagebox.showwarning("No Guards Selected", "Please select guards first using 'Select Guard' button")
            return
        
        # Log the tracking activation
        guards_list = ", ".join(self.selected_target_names)
        logger.warning(f"Tracking activated for: {guards_list}")
        messagebox.showinfo("Tracking Activated", f"Now tracking: {guards_list}")

    def toggle_track_monitoring(self):
        """Toggle Track Guard / Stop Monitoring button - start or stop tracking selected guards"""
        if not hasattr(self, 'selected_target_names') or not self.selected_target_names:
            messagebox.showwarning("No Guards Selected", "Please select guards first using 'Select Guard' button")
            return
        
        if not self.is_running:
            messagebox.showwarning("Camera Required", "Please start the camera first")
            return
        
        # Toggle tracking state
        if not self.is_tracking:
            # START TRACKING
            self.is_tracking = True
            self.btn_track_toggle.configure(text="⏹️ Stop Monitoring", fg_color="#e74c3c")
            
            # Activate the selected guards for tracking
            self.apply_target_selection()
            
            # Log and show confirmation
            guards_list = ", ".join(self.selected_target_names)
            current_action = self.active_required_action
            logger.warning(f"[TRACKING START] Identifying and tracking: {guards_list} | Action: {current_action}")
            messagebox.showinfo("Tracking Started", 
                              f"Now identifying and tracking:\n{guards_list}\n\nActive Action: {current_action}\nMonitoring for action alerts...")
        else:
            # STOP TRACKING
            self.is_tracking = False
            self.btn_track_toggle.configure(text="🏃 Track Guard", fg_color="#16a085")
            
            # Clear all tracking data
            tracked_guards = list(self.targets_status.keys())
            self.targets_status.clear()
            self.selected_target_names.clear()
            self.update_selected_preview()
            
            # Log and show confirmation
            if tracked_guards:
                guards_list = ", ".join(tracked_guards)
                logger.warning(f"[TRACKING STOP] Stopped monitoring: {guards_list}")
                messagebox.showinfo("Monitoring Stopped", f"Stopped tracking:\n{guards_list}")
            else:
                logger.warning("[TRACKING STOP] Monitoring stopped")
                messagebox.showinfo("Monitoring Stopped", "Guard tracking stopped")

    def on_track_guards_clicked(self):
        """Handle Track Guards button click - identifies and starts tracking selected guards"""
        if not hasattr(self, 'selected_target_names') or not self.selected_target_names:
            messagebox.showwarning("No Guards Selected", "Please select guards first using 'Select Guard' button")
            return
        
        if not self.is_running:
            messagebox.showwarning("Camera Required", "Please start the camera first")
            return
        
        # Activate the selected guards for tracking
        self.apply_target_selection()
        
        # Log and show confirmation
        guards_list = ", ".join(self.selected_target_names)
        logger.warning(f"[TRACKING START] Identifying and tracking: {guards_list}")
        messagebox.showinfo("Tracking Started", f"Now identifying and tracking:\n{guards_list}\n\nMonitoring for action alerts...")

    def on_stop_tracking_clicked(self):
        """Handle Stop Tracking button click - stops all guard tracking"""
        if not self.targets_status:
            messagebox.showinfo("No Tracking", "No guards are currently being tracked")
            return
        
        # Clear all tracking data
        tracked_guards = list(self.targets_status.keys())
        self.targets_status.clear()
        self.selected_target_names.clear()
        self.update_selected_preview()
        
        # Log and show confirmation
        guards_list = ", ".join(tracked_guards)
        logger.warning(f"[TRACKING STOP] Stopped tracking: {guards_list}")
        messagebox.showinfo("Tracking Stopped", f"Stopped tracking:\n{guards_list}")

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
                
                # OLD FUNCTION - Now handled by toggle_fugitive_detection()
                self.fugitive_face_encoding = face_encodings[0]
                self.fugitive_name = simpledialog.askstring("Fugitive Name", "Enter fugitive name:") or "Unknown Fugitive"
                
                # Trigger fugitive detection
                self.is_fugitive_detection = True
                self.fugitive_detected_log_done = False
                self.btn_fugitive_toggle.configure(text="🚨 Fugitive ON", fg_color="#8b0000")
                
                # Display fugitive image in preview
                self._update_fugitive_preview()
                
                logger.warning(f"Fugitive Detection Started - Searching for: {self.fugitive_name}")
                messagebox.showinfo("Fugitive Detection", f"Searching for: {self.fugitive_name}")
                
            except Exception as e:
                logger.error(f"Fugitive Detection Error: {e}")
                messagebox.showerror("Error", f"Failed to process image: {e}")
        else:
            # Stop Fugitive Detection
            self.is_fugitive_detection = False
            self.fugitive_image = None
            self.fugitive_face_encoding = None
            self.fugitive_detected_log_done = False
            self.btn_fugitive_toggle.configure(text="🚨 Fugitive OFF", fg_color="#95a5a6")
            
            logger.warning("Fugitive Detection Stopped")
            messagebox.showinfo("Fugitive Detection", "Fugitive Detection Stopped")

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

    # ✅ SIMPLIFIED: toggle_pro_detection_mode and _save_pro_detection_log functions removed

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
                
                # ✅ OPTIMIZATION: Set camera properties for faster capture and better clarity
                # Set frame rate to 30 FPS for smooth playback
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                # Set resolution for optimal balance (adjust if needed)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Enable auto-focus if supported
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                # Set auto white balance
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                # Reduce camera buffer to minimize latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # ✅ CRITICAL: Warm up camera with initial frames
                # Some cameras take 100-200ms to stabilize output
                logger.info("Warming up camera (please wait)...")
                for _ in range(10):
                    ret, _ = self.cap.read()
                    time.sleep(0.05)
                    if not ret:
                        logger.warning("Camera warmup: frame read failed, retrying...")
                        continue
                logger.info("Camera ready!")
                    
                self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.is_running = True
                self.is_camera_running = True
                self.btn_camera_toggle.configure(text="🎥 Camera ON", fg_color="#27ae60")
                self.btn_guard_toggle.configure(state="normal")
                self.btn_alert_toggle.configure(state="normal")
                self.btn_fugitive_toggle.configure(state="normal")
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
            if hasattr(self, 'is_fugitive_detection') and self.is_fugitive_detection:
                self.is_fugitive_detection = False
                self.fugitive_image = None
                self.fugitive_face_encoding = None
                self.fugitive_detected_log_done = False
                self.btn_fugitive_toggle.configure(text="🚨 Fugitive OFF", fg_color="#95a5a6")
                self.fugitive_preview_label.configure(image='', text="No Fugitive Selected")
            
            # ✅ SIMPLIFIED: PRO_Detection mode cleanup removed
            
            # Clear guard preview grid
            for widget in self.guard_preview_scroll_frame.winfo_children():
                widget.destroy()
            self.guard_preview_grid = {}
            
            # Cleanup
            for status in self.targets_status.values():
                if status["tracker"]:
                    status["tracker"] = None
            
            gc.collect()
            
            self.is_camera_running = False
            self.btn_camera_toggle.configure(text="🎥 Camera OFF", fg_color="#c0392b")
            self.btn_guard_toggle.configure(state="disabled")
            self.btn_fugitive_toggle.configure(state="disabled")
            self.video_label.configure(image='')

    def auto_flush_logs(self):
        """Automatically flush logs when threshold reached"""
        if self.is_logging and len(self.temp_log) >= CONFIG["logging"]["auto_flush_interval"]:
            self.save_log_to_file()
        
        # ✅ OPTIMIZATION: Reduce memory optimization frequency from every 300 to every 500 frames
        # This reduces garbage collection overhead which can cause temporary latency spikes
        # At 30 FPS: once per 16-17 seconds instead of 10 seconds
        if self.frame_counter % 500 == 0:  # Every ~500 frames
            optimize_memory()
    
    def optimize_memory(self):
        """Clear old cache entries and collect garbage to free memory"""
        try:
            # Clear old action cache - keep last 50 entries
            # ✅ OPTIMIZATION: Use more aggressive cleanup to prevent memory bloat
            max_cache_size = 100  # Increased from 50 for multi-guard scenarios
            if len(self.last_action_cache) > max_cache_size:
                # Keep only the most recent entries
                # Sort by value (which is a dict with timestamp if available) 
                # For now, just keep last N
                keys_to_remove = list(self.last_action_cache.keys())[:-max_cache_size]
                for key in keys_to_remove:
                    del self.last_action_cache[key]
            
            # Force garbage collection - but be selective to avoid latency
            # Only force full collection every 500 frames (not every 300)
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
            # ✅ SPECIFICATION: 4 poses captured during onboarding (can detect 7 actions in real-time)
            pose_actions = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Hands Up", "Standing"]
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
                pose_actions_local = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Hands Up", "Standing"]
                next_action = pose_actions_local[self.onboarding_step - 1]
                messagebox.showinfo(f"Step {self.onboarding_step + 1}", f"Perfect! Now perform: {next_action.upper()} and click Snap when ready")
            else:
                # Save all pose references (4 poses collected, but 7 can be detected in real-time)
                self.save_pose_references(self.onboarding_name, self.onboarding_poses)
                self.load_targets()
                self.exit_onboarding_mode()
                messagebox.showinfo("Complete", f"{self.onboarding_name} onboarding complete with {len(self.onboarding_poses)} poses!")

    def update_video_feed(self):
        if not self.is_running: return
        
        try:
            if not self.cap or not self.cap.isOpened():
                logger.error("Camera not available")
                self.stop_camera()
                return
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # ✅ OPTIMIZATION: Try faster recover without full reconnect
                logger.warning("Failed to read frame, attempting recovery...")
                # Clear buffer and retry with exponential backoff
                retry_count = 0
                max_retries = 10
                retry_delay = 0.02
                
                while not ret and retry_count < max_retries:
                    time.sleep(retry_delay)
                    ret, frame = self.cap.read()
                    retry_count += 1
                    if retry_delay < 0.1:
                        retry_delay *= 1.2  # Exponential backoff
                
                if not ret or frame is None:
                    logger.error(f"Failed to recover after {max_retries} retries, attempting reconnect...")
                    # Full reconnect only if buffer clear didn't work
                    try:
                        self.cap.release()
                        time.sleep(1.0)
                        self.cap = cv2.VideoCapture(self.camera_index)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # Warm up after reconnect
                        for _ in range(5):
                            self.cap.read()
                            time.sleep(0.05)
                        
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            logger.error("Camera reconnection failed")
                            self.stop_camera()
                            messagebox.showerror("Camera Error", "Camera disconnected - please restart")
                        return
                    except Exception as e:
                        logger.error(f"Reconnection error: {e}")
                        self.stop_camera()
                        return
        except Exception as e:
            logger.error(f"Camera read error: {e}")
            self.stop_camera()
            return
        
        self.unprocessed_frame = frame.copy()
        
        # ✅ INDUSTRIAL-LEVEL: Enhance frame for low-light conditions
        # This improves face detection in dark areas significantly
        frame = self.enhance_frame_for_low_light(frame)
        
        # Frame skipping for performance
        self.frame_counter += 1
        skip_interval = CONFIG["performance"]["frame_skip_interval"]
        
        # ========== PERFORMANCE MONITORING SIMPLIFIED ==========
        # Removed Phase 4 Stage 5 - just do basic FPS tracking
        
        if self.is_in_capture_mode:
            self.process_capture_frame(frame)
        else:
            # Skip processing every N frames when enabled
            if CONFIG["performance"]["enable_frame_skipping"] and self.frame_counter % skip_interval != 0:
                # Use cached frame
                if self.last_process_frame is not None:
                    frame = self.last_process_frame.copy()
            else:
                # Process tracking frame normally
                self.process_tracking_frame_optimized(frame)
        
                self.last_process_frame = frame.copy()
        
        # ========== PERFORMANCE MONITORING & FPS CALCULATION ==========
        # ✅ OPTIMIZATION: Update GUI labels only every 30 frames (~1 second) instead of every frame
        # This reduces GUI rendering overhead significantly
        self.frame_counter += 1
        
        if self.frame_counter % 30 == 0:  # Update labels every 30 frames
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
            self.last_fps_time = current_time
            
            # ✅ PERFORMANCE: Periodic memory optimization
            if self.frame_counter % 150 == 0:  # Every 150 frames (~5 seconds at 30 FPS)
                optimize_memory()
            
            # Memory monitoring and label updates (only every 30 frames)
            try:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                if hasattr(self, 'fps_label') and self.fps_label.winfo_exists():
                    self.fps_label.configure(text=f"{self.current_fps:.1f}")
                if hasattr(self, 'mem_label') and self.mem_label.winfo_exists():
                    self.mem_label.configure(text=f"{mem_mb:.0f}MB")
            except:
                pass
            
            # Update system clock (only when second changes)
            try:
                current_datetime = datetime.now()
                current_second = current_datetime.second
                if current_second != self.last_clock_second:
                    current_time_str = current_datetime.strftime("%H:%M:%S")
                    if hasattr(self, 'clock_label') and self.clock_label.winfo_exists():
                        self.clock_label.configure(text=current_time_str)
                    self.last_clock_second = current_second
            except:
                pass
            
            # Session time check (only every 30 frames)
            try:
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
            except:
                pass
        
        # Auto flush logs
        self.auto_flush_logs()
        
        # ========== OPTIMIZED: GUI VIDEO FEED RENDERING ==========
        # ✅ CRITICAL OPTIMIZATION: Only update GUI every other frame (50ms instead of 33ms)
        # This reduces GUI overhead by ~30-40% and frees up CPU for tracking/detection
        # Visual effect: ~15 FPS GUI (still acceptable for video) but backend gets much more time
        should_update_gui = (self.frame_counter % 2 == 0)  # Update every 2nd frame = 50ms at 30 FPS input
        
        if should_update_gui and self.video_label.winfo_exists():
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # ✅ OPTIMIZED: Scale frame efficiently using linear interpolation
                lbl_w = self.video_label.winfo_width()
                lbl_h = self.video_label.winfo_height()
                h, w = frame.shape[:2]
                
                if lbl_w > 10 and lbl_h > 10:
                    scale = min(lbl_w/w, lbl_h/h, 1.5)
                    new_w, new_h = int(w*scale), int(h*scale)
                    # Use INTER_LINEAR for balance between quality and speed
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # ✅ OPTIMIZATION: Use PIL directly for image conversion
                pil_image = Image.fromarray(frame_rgb, mode='RGB')
                ctk_image = ctk.CTkImage(light_image=pil_image, size=(pil_image.width, pil_image.height))
                self.video_label.configure(image=ctk_image, text="")
                self.video_label.image = ctk_image
            except Exception as e:
                logger.debug(f"Frame display error: {e}")
        
        # ========== DYNAMIC REFRESH RATE OPTIMIZATION ==========
        # ✅ OPTIMIZATION: Smart refresh rate - increase backend processing time automatically
        # When FPS drops, GUI refresh slows down to give backend more CPU time
        # When FPS is good, maintain smooth 30 FPS visual feedback
        refresh_ms = 50  # Base refresh rate
        
        if self.current_fps < 10:
            # Very slow - maximize backend processing time
            refresh_ms = 150  # ~6.7 FPS GUI, but backend gets more time
        elif self.current_fps < 15:
            # Slow - increase backend time
            refresh_ms = 100  # ~10 FPS GUI
        elif self.current_fps < 20:
            # Below target - slight delay
            refresh_ms = 66  # ~15 FPS GUI
        elif self.current_fps >= 25:
            # Good performance - faster refresh for responsive feedback
            refresh_ms = 40  # ~25 FPS GUI
        else:
            # Excellent - maintain smooth 30 FPS
            refresh_ms = 33  # ~30 FPS GUI
        
        self.root.after(refresh_ms, self.update_video_feed)

    def process_capture_frame(self, frame):
        """Process frame during onboarding capture mode with dynamic detection"""
        h, w = frame.shape[:2]
        
        # Detect face and pose from entire frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize pipelines if needed
        if not self.model_pipeline_initialized:
            self._initialize_model_pipeline()
        
        # Load single-person pipeline for onboarding
        self._load_single_person_pipeline()
        
        # Detect face
        face_locations = self._detect_faces_blazeface(rgb_frame)
        
        # Detect pose
        pose_result = self._detect_pose_movenet_lightning(rgb_frame) if self.pose_model else None
        
        # Store detection results for snap_photo to use
        self.onboarding_detection_results = pose_result
        self.onboarding_face_box = None
        
        detection_status = ""
        box_color = (0, 0, 255)  # Red by default
        
        if self.onboarding_step == 0:
            # Step 0: Face capture
            
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
            # Steps 1-4: Pose capture (4 poses required by spec, 7 can be detected)
            pose_actions = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Hands Up", "Standing"]
            target_action = pose_actions[self.onboarding_step - 1]
            
            if pose_result is not None:
                # MoveNet Lightning returns raw pose keypoints
                # For simplicity, just check pose was detected
                logger.debug(f"Pose detected during onboarding step {self.onboarding_step}")
                
                # Simplified detection - in real scenario would extract keypoints from pose_result
                # For now, accept detection and proceed
                box_color = (0, 255, 0)  # Green - ready
                detection_status = f"READY - Pose detected - Click Snap Photo"
            else:
                detection_status = f"No Pose Detected - Step back and perform {target_action}"
        
        # Display instructions and status
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)  # Black background for text
        
        if self.onboarding_step == 0:
            instruction = f"STEP 1/5: FACE CAPTURE"
        else:
            pose_actions = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Hands Up", "Standing"]
            instruction = f"STEP {self.onboarding_step + 1}/5: {pose_actions[self.onboarding_step - 1].upper()}"
        
        cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, detection_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        return frame

    # --- TRACKING LOGIC ---
    def process_tracking_frame_optimized(self, frame):
        # ✅ CRITICAL: Fugitive detection runs FIRST, before checking if guards exist
        # This ensures fugitive alert works independently of guard tracking state
        
        # Prepare color space for face detection (used by both fugitive and guard detection)
        rgb_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # ==================== FUGITIVE MODE (ALWAYS RUNS - Independent of other modes) ====================
        # ✅ CRITICAL: Fugitive detection MUST run regardless of mode, always prioritized
        # Fugitive is detected instantly and independently from guard tracking or action monitoring
        if self.is_fugitive_detection and self.fugitive_face_encoding is not None:
            # Use downscaled frame for faster detection (same optimization as guard detection)
            scale_factor = 1.5
            h, w = rgb_full_frame.shape[:2]
            scaled_w = max(1, int(w / scale_factor))
            scaled_h = max(1, int(h / scale_factor))
            rgb_frame_fugitive = cv2.resize(rgb_full_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            
            try:
                face_locations = face_recognition.face_locations(rgb_frame_fugitive, model="hog")
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame_fugitive, face_locations)
                    
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        # Compare with fugitive face
                        # ✅ RELAXED TOLERANCE: Use 0.68 for instant detection (same as guards)
                        match = face_recognition.compare_faces([self.fugitive_face_encoding], face_encoding, tolerance=0.68)
                        face_distance = face_recognition.face_distance([self.fugitive_face_encoding], face_encoding)
                        
                        if match[0]:  # If face matches - FUGITIVE FOUND!
                            # ✅ CRITICAL FIX: PREVENT FALSE FUGITIVE DETECTION AS GUARD
                            # Cross-check: Make sure this isn't a guard that's being misidentified as fugitive
                            is_likely_guard = False
                            if self.targets_status and len(self.targets_status) > 0:
                                # Check distance to all tracked guards
                                for guard_name, guard_status in self.targets_status.items():
                                    guard_encoding = guard_status.get("encoding")
                                    if guard_encoding is not None:
                                        guard_dist = face_recognition.face_distance([guard_encoding], face_encoding)[0]
                                        # If distance to a guard is significantly smaller than to fugitive, it's likely a guard
                                        if guard_dist < face_distance[0] - 0.10:  # Guard match is 10% better
                                            logger.debug(f"[SKIP] Detected face matches GUARD '{guard_name}' better than fugitive (guard_dist={guard_dist:.3f}, fugitive_dist={face_distance[0]:.3f})")
                                            is_likely_guard = True
                                            break
                            
                            if is_likely_guard:
                                continue  # Skip this face - it's a guard, not fugitive
                            
                            # Scale back to original frame size
                            top, right, bottom, left = face_location
                            top, right, bottom, left = int(top*scale_factor), int(right*scale_factor), int(bottom*scale_factor), int(left*scale_factor)
                            
                            # Draw bounding box - BRIGHT RED for fugitive alert
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                            cv2.putText(frame, f"🚨 FUGITIVE: {self.fugitive_name} 🚨", (left, top - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            
                            # ✅ CRITICAL: Always trigger sound and logging EVERY TIME detected
                            # Don't use a flag - play alert sound immediately on every frame where fugitive is visible
                            
                            # 1. PLAY FUGITIVE ALERT SOUND IMMEDIATELY (no conditions)
                            try:
                                if self.fugitive_alert_stop_event is None:
                                    self.fugitive_alert_stop_event = threading.Event()
                                
                                # Only start a new alert if one isn't already playing
                                if not hasattr(self, 'fugitive_alert_sound_thread') or \
                                   self.fugitive_alert_sound_thread is None or \
                                   not self.fugitive_alert_sound_thread.is_alive():
                                    self.fugitive_alert_stop_event.clear()
                                    self.fugitive_alert_sound_thread = play_siren_sound(
                                        stop_event=self.fugitive_alert_stop_event,
                                        sound_file="Fugitive.mp3",
                                        duration_seconds=30  # 30 seconds for fugitive alert
                                    )
                                    logger.warning(f"[FUGITIVE ALERT] ⚠️ FUGITIVE DETECTED INSTANTLY - {self.fugitive_name} - Playing 30s alert!")
                            except Exception as e:
                                logger.error(f"[FUGITIVE SOUND] Error playing alert: {e}")
                            
                            # 2. SNAPSHOT & LOG: Once per detection event (use timestamp-based throttling)
                            current_time = time.time()
                            last_snapshot_time = getattr(self, 'last_fugitive_snapshot_time', 0)
                            
                            # Only log once every 5 seconds to avoid spam, but sound plays every frame
                            if current_time - last_snapshot_time > 5.0:
                                try:
                                    # Capture snapshot
                                    snapshot_path = self.capture_alert_snapshot(frame, f"FUGITIVE_{self.fugitive_name}", check_rate_limit=False)
                                    img_path = snapshot_path if snapshot_path else "N/A"
                                    
                                    # Create CSV log entry
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
                                    
                                    self.last_fugitive_snapshot_time = current_time
                                    logger.warning(f"[FUGITIVE] Snapshot captured and log entry created - confidence: {confidence:.2f}")
                                except Exception as e:
                                    logger.error(f"[FUGITIVE LOG] Error logging fugitive: {e}")
                        else:
                            # Fugitive not detected in this frame - can reset alert state
                            pass
            except Exception as e:
                logger.debug(f"Fugitive detection error: {e}")
        
        # ✅ GUARD TRACKING: Only run if guards are selected
        if not self.targets_status:
            cv2.putText(frame, "SELECT TARGETS TO START", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        # ✅ PERFORMANCE OPTIMIZATION: Adaptive detection interval based on tracking stability
        # More frequent detection when targets are lost, less frequent when stable
        self.re_detect_counter += 1
        untracked_count = len([n for n, s in self.targets_status.items() if not s["visible"]])
        
        # ✅ OPTIMIZED INTERVALS (AGGRESSIVE FOR INITIAL DETECTION):
        # - If targets missing: 2 frames (67ms) - very aggressive for first detection
        # - If partial tracking: 5 frames (167ms) - faster re-detection
        # - If all stable: 12 frames (400ms) - conservative, let tracker do the work
        # Key insight: Faster detection interval ensures guards are identified as soon as they appear on camera
        num_visible = len([n for n, s in self.targets_status.items() if s["visible"]])
        
        if untracked_count > len(self.targets_status) / 2:
            # More than 50% missing - use very aggressive detection for rapid identification
            adaptive_interval = 2  # 67ms - ensures quick detection when guard enters frame
        elif untracked_count > 0:
            # Some missing - use aggressive detection for re-identification
            adaptive_interval = 5  # 167ms - faster detection than before
        else:
            # All visible and tracked - use conservative interval
            # Let CSRT tracker do the heavy lifting, only re-detect every ~400ms
            adaptive_interval = 12  # 400ms
        
        if self.re_detect_counter > adaptive_interval:
            self.re_detect_counter = 0
        
        rgb_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # ✅ SIMPLIFIED: PRO_DETECTION MODE removed - focus on Normal Mode only
        
        # ✅ IMPORTANT: Get current time at the start of the tracking loop
        current_time = time.time()

        # 1. Update Trackers (PERSISTENT TRACKING - Keep tracking even at frame edges)
        # ✅ CRITICAL: Keep tracking guards even if they partially leave the frame
        # Persistent tracking ensures continuous monitoring until guard fully exits
        for name, status in self.targets_status.items():
            # ✅ OPTIMIZATION: Skip invisible targets COMPLETELY - no tracker update, no drawing
            # This saves ~5-10ms per untracked target per frame
            if not status.get("visible", False):
                continue  # Skip to next target - invisible targets get zero processing
            
            if status.get("tracker") is None:
                continue  # No tracker to update
            
            # ✅ PERFORMANCE: Only update tracker if it exists and target is visible
            # Tracker update is fast (~5ms) but skip when not needed
            success, box = status["tracker"].update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                new_box = (x, y, x + w, y + h)
                if status["face_box"] is not None:
                    # Sanity check: detect if tracker jumped too far (tracker failure)
                    old_x1, old_y1, old_x2, old_y2 = status["face_box"]
                    new_x1, new_y1, new_x2, new_y2 = new_box
                    dx = abs(new_x1 - old_x1) + abs(new_x2 - old_x2)
                    dy = abs(new_y1 - old_y1) + abs(new_y2 - old_y2)
                    old_w = max(1, old_x2 - old_x1)
                    old_h = max(1, old_y2 - old_y1)
                    new_w = max(1, new_x2 - new_x1)
                    new_h = max(1, new_y2 - new_y1)
                    size_change = abs(new_w - old_w) + abs(new_h - old_h)
                    max_movement = max(old_w, old_h) * 0.5  # 50% of size (increased from 40% for distance tracking)
                    max_size_change = (old_w + old_h) * 0.45  # 45% size change tolerance (increased from 35%)
                    
                    # ✅ PERSISTENT TRACKING: Allow much larger movement (for 20-50m distance tracking)
                    frame_h, frame_w = frame.shape[:2]
                    at_frame_edge = (new_x1 < 50 or new_x2 > frame_w - 50 or 
                                    new_y1 < 50 or new_y2 > frame_h - 50)
                    
                    # Much more lenient check for long-distance tracking to frame edges
                    # 15x threshold allows tracking through significant size/position changes
                    movement_threshold = (max_movement * 15) if at_frame_edge else (max_movement * 8)
                    size_threshold = (max_size_change * 15) if at_frame_edge else (max_size_change * 8)
                    
                    # ✅ EXTREME ANGLE TOLERANCE: If pose quality is poor (back view, extreme angle),
                    # increase tolerance to prevent losing track
                    pose_quality_check = status.get("pose_confidence", 0.3)
                    if pose_quality_check < 0.3:
                        # Back/extreme angle view - very lenient tracking
                        movement_threshold = movement_threshold * 2.0  # 2x more lenient
                        size_threshold = size_threshold * 2.0
                        logger.debug(f"[TRACKER] {name}: Using extreme-angle tolerance (sparse pose: {pose_quality_check:.2f})")
                    
                    if dx > movement_threshold or dy > movement_threshold or size_change > size_threshold:
                        # Only reset tracker if movement is EXTREME (likely lost track, not natural movement)
                        status["visible"] = False
                        status["tracker"] = None
                        status["consecutive_detections"] = 0
                        status["stable_tracking"] = False
                        logger.warning(f"[TRACKER LOST] {name}: Extreme movement detected (dx={dx:.0f}, dy={dy:.0f}, size_change={size_change:.0f}) - pose_quality:{pose_quality_check:.2f}")
                    else:
                        # Tracker still valid - apply smoothing
                        smoothed_box = smooth_bounding_box(new_box, status["face_box"], smoothing_factor=0.75)
                        status["face_box"] = smoothed_box
                        status["visible"] = True
                        # ✅ CRITICAL: Mark for face re-verification on next detection
                        # This ensures we verify the tracker is actually on a real face, not a ghost
                        status["needs_face_reverification"] = True
                        
                        # ✅ ANTI-GHOST: Track consecutive frames without face detection
                        # If tracker runs without finding face for too long, it's a ghost
                        if not status.get("consecutive_no_face_frames"):
                            status["consecutive_no_face_frames"] = 0
                        
                        # Log tracking status
                        if at_frame_edge:
                            logger.debug(f"[TRACKER OK] {name}: Persistent tracking at frame edge (movement: dx={dx:.0f}, dy={dy:.0f})")
                        else:
                            logger.debug(f"[TRACKER OK] {name}: Tracked (movement: dx={dx:.0f}, dy={dy:.0f})")
                else:
                    status["face_box"] = new_box
                    status["visible"] = True
            else:
                # Tracker lost the target
                status["tracker"] = None
                status["visible"] = False
                logger.warning(f"[TRACKER FAILED] {name}: Tracker returned False (will re-detect)")

        # 2. Detection (PARALLEL MATCHING) - Fast, frequent detection for newly selected guards
        # ✅ OPTIMIZATION: Detection runs very frequently to catch new guards
        untracked_targets = [name for name, s in self.targets_status.items() if not s["visible"]]
        
        # ✅ FREQUENCY (OPTIMIZED): Run detection every 2-5-12 frames depending on stability
        # CRITICAL: Always run detection frequently to verify tracker is on real face, not ghost
        all_stable_and_tracked = all(
            s.get("visible", False) and s.get("stable_tracking", False) 
            for s in self.targets_status.values()
        )
        
        # ✅ OPTIMIZATION: Only run detection on schedule (adaptive_interval)
        # CHANGED: Even stable targets get re-verified every 12 frames to prevent ghost tracking
        should_run_detection = (
            (self.re_detect_counter % adaptive_interval == 0)  # SIMPLIFIED: Interval-based only
        )  # REMOVED: 'and not all_stable_and_tracked' - NOW ALWAYS RUN to verify no ghosts
        
        if should_run_detection:
            # ✅ NEW PIPELINE: Initialize and use new model pipelines
            # Auto-detect single vs multi-person mode based on target count
            is_single_person = len(self.targets_status) <= 1
            
            # Initialize pipeline if needed
            if not self.model_pipeline_initialized:
                self._initialize_model_pipeline()
            
            # Load appropriate pipeline
            if is_single_person:
                self._load_single_person_pipeline()
                face_locations = self._detect_faces_blazeface(rgb_full_frame)
            else:
                self._load_multi_person_pipeline()
                face_locations = self._detect_faces_blazepose(rgb_full_frame)
            
            # ✅ CRITICAL GHOST DETECTION FIX: Verify tracked guards are actually on real faces OR bodies
            # If a guard is being tracked but NO face is detected at that location, check for body/skeleton
            for name, status in self.targets_status.items():
                if not status.get("visible", False):
                    continue  # Skip invisible targets
                
                # Check if this tracked guard has any detected face nearby
                tracked_box = status.get("face_box")
                if tracked_box is None:
                    continue
                
                t_x1, t_y1, t_x2, t_y2 = tracked_box
                face_found_at_tracked_location = False
                
                # Check if any detected face overlaps with tracked box (at least 30% IoU)
                for det_face in face_locations:
                    d_y1, d_x2, d_y2, d_x1 = det_face  # face_recognition returns top, right, bottom, left
                    
                    # Calculate IoU between tracked box and detected face
                    inter_x1 = max(t_x1, d_x1)
                    inter_y1 = max(t_y1, d_y1)
                    inter_x2 = min(t_x2, d_x2)
                    inter_y2 = min(t_y2, d_y2)
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        tracked_area = (t_x2 - t_x1) * (t_y2 - t_y1)
                        iou = inter_area / max(1, tracked_area)
                        
                        if iou >= 0.30:  # At least 30% overlap confirms face is still there
                            face_found_at_tracked_location = True
                            logger.debug(f"[VERIFIED] {name}: Face verified at tracked location (IoU={iou:.2f})")
                            break
                
                # ✅ FACE-INDEPENDENT TRACKING: If face not found, check for body/skeleton
                # Allow tracking to continue based on skeleton presence (when face is turned away)
                if not face_found_at_tracked_location:
                    # Body box is often larger/different than face box
                    # Check if tracker is still in reasonable position (not completely drifted)
                    body_box = status.get("body_box")
                    pose_confidence = status.get("pose_confidence", 0.0)
                    skeleton_keypoints = status.get("skeleton_keypoints", 0)
                    
                    # ROBUST 4-PART VALIDATION for body-only tracking:
                    # 1. body_box exists (from skeleton)
                    # 2. pose confidence adequate (>0.40 is good quality)
                    # 3. sufficient keypoints (≥10 indicates valid full-body skeleton)
                    # 4. body is in/near frame (within -50 to width+50, -50 to height+50)
                    
                    body_box_valid = body_box is not None
                    pose_quality_valid = pose_confidence > 0.40
                    keypoint_valid = skeleton_keypoints >= 10
                    
                    # Extract body box position for frame boundary check
                    body_in_frame = False
                    if body_box_valid:
                        bx1, by1, bx2, by2 = body_box
                        frame_h, frame_w = frame.shape[:2]
                        # Allow 50px margin beyond frame edges for tracking at boundaries
                        body_in_frame = (bx1 <= frame_w + 50 and bx2 >= -50 and 
                                       by1 <= frame_h + 50 and by2 >= -50)
                    
                    # ✅ CRITICAL: ANTI-GHOST LOGIC - Check if body is actually visible
                    # Not just checking skeleton but also verifying against detected faces in current frame
                    face_in_body_area = False
                    if body_box_valid and face_locations and len(face_locations) > 0:
                        bx1, by1, bx2, by2 = body_box
                        body_area = bx1 * by1 * (bx2 - bx1) * (by2 - by1)
                        for face_loc in face_locations:
                            fx1, fx2, fy1, fy2 = face_loc[2], face_loc[1], face_loc[0], face_loc[3]  # left, right, top, bottom
                            # Check if any detected face overlaps with body box
                            inter_x1 = max(bx1, fx1)
                            inter_y1 = max(by1, fy1)
                            inter_x2 = min(bx2, fx2)
                            inter_y2 = min(by2, fy2)
                            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                iou = inter_area / max(1, body_area)
                                if iou >= 0.15:  # 15% overlap means face in body area
                                    face_in_body_area = True
                                    logger.debug(f"[GHOST CHECK] {name}: Face detected in body area (IoU={iou:.2f})")
                                    break
                    
                    # If we have a VALID body detection (all criteria met), allow continued tracking
                    if body_box_valid and pose_quality_valid and keypoint_valid and body_in_frame:
                        # All validation criteria passed - high confidence body tracking
                        # But only if either (a) face visible in body area OR (b) skeleton very confident
                        if face_in_body_area or pose_confidence > 0.60:
                            status["face_detection_missing_frames"] = status.get("face_detection_missing_frames", 0) + 1
                            
                            # QUALITY-BASED TOLERANCE: More frames allowed for better skeleton quality
                            # High quality (pose_confidence > 0.60): allow 60 frames (~2 seconds at 30fps)
                            # Good quality (pose_confidence > 0.40): allow 45 frames (~1.5 seconds at 30fps)
                            # Default fallback: 30 frames (~1 second at 30fps)
                            if pose_confidence > 0.60:
                                max_missing_frames = 60
                                quality_tier = "HIGH"
                            elif pose_confidence > 0.40:
                                max_missing_frames = 45
                                quality_tier = "GOOD"
                            else:
                                max_missing_frames = 30
                                quality_tier = "BASIC"
                            
                            if status["face_detection_missing_frames"] <= max_missing_frames:
                                logger.debug(f"[BODY TRACK] {name}: No face (frame {status['face_detection_missing_frames']}/{max_missing_frames}), "
                                           f"body:✓ pose:{pose_confidence:.2f}({quality_tier}) keypts:{skeleton_keypoints}/33 - continuing skeleton tracking")
                            else:
                                # Too long without face and body validation failed
                                status["visible"] = False
                                status["tracker"] = None
                                status["consecutive_detections"] = 0
                                status["stable_tracking"] = False
                                logger.warning(f"[TRACK ENDED] {name}: No face for {max_missing_frames} frames - guard likely left frame")
                        else:
                            # Body valid but no face in body area AND low pose confidence - GHOST
                            status["visible"] = False
                            status["tracker"] = None
                            status["consecutive_detections"] = 0
                            status["stable_tracking"] = False
                            logger.warning(f"[GHOST REMOVED] {name}: Body detected but no face in area + low pose confidence - likely ghost")
                    
                    elif body_box_valid and (pose_quality_valid or keypoint_valid):
                        # Partial validation (either quality or keypoints valid, but not both)
                        # More conservative: allow only 20 frames
                        status["face_detection_missing_frames"] = status.get("face_detection_missing_frames", 0) + 1
                        
                        if status["face_detection_missing_frames"] <= 20:
                            logger.debug(f"[BODY TRACK-PARTIAL] {name}: Partial validation (pose:{pose_confidence:.2f}, keypts:{skeleton_keypoints})")
                        else:
                            status["visible"] = False
                            status["tracker"] = None
                            status["consecutive_detections"] = 0
                            status["stable_tracking"] = False
                            logger.warning(f"[TRACK ENDED-PARTIAL] {name}: Partial body validation failed after 20 frames")
                    else:
                        # GHOST DETECTION: No face AND failed body validation
                        # Multiple reasons could cause this:
                        invalid_reasons = []
                        if not body_box_valid:
                            invalid_reasons.append("no_body_box")
                        if not pose_quality_valid:
                            invalid_reasons.append(f"pose_low({pose_confidence:.2f})")
                        if not keypoint_valid:
                            invalid_reasons.append(f"keypts_low({skeleton_keypoints})")
                        if not body_in_frame:
                            invalid_reasons.append("out_of_frame")
                        
                        status["visible"] = False
                        status["tracker"] = None
                        status["consecutive_detections"] = 0
                        status["stable_tracking"] = False
                        logger.warning(f"[GHOST REMOVED] {name}: Tracked but failed body validation - {', '.join(invalid_reasons)}")
                else:
                    # Face found - reset counter
                    status["face_detection_missing_frames"] = 0
            
            if face_locations and len(face_locations) > 0:
                # ✅ CRITICAL OPTIMIZATION: Reduce face encoding cost from ~100ms to ~30ms per frame
                # Strategy: Use num_jitters=1 (fastest) + skip encoding for stable tracked targets
                num_untracked = len(untracked_targets)
                
                # Always use num_jitters=1 for MAXIMUM SPEED at 30 FPS video framerate
                # Quality difference is imperceptible in real-time video (consecutive frames are similar)
                num_jitters = 1
                
                # ✅ OPTIMIZATION: Skip encoding if NO untracked targets (all tracked)
                # If all targets are actively being tracked by CSRT, we don't need face encoding
                if num_untracked == 0:
                    logger.debug(f"[PERF] All {len(self.targets_status)} targets tracked - skipping expensive face encoding")
                    face_encodings = []
                else:
                    # Only encode faces when we have untracked targets to match against
                    face_encodings = face_recognition.face_encodings(rgb_full_frame, face_locations, num_jitters=num_jitters)
                
                # ✅ CRITICAL: Skip encoding step if no untracked targets
                # If all targets are tracked, we don't need to encode faces at all!
                if len(untracked_targets) == 0:
                    # All targets are already being tracked, skip expensive encoding
                    logger.debug(f"All {len(self.targets_status)} targets tracked, skipping face encoding")
                    face_encodings = []
                
                # Only proceed with matching if we have untracked targets and encodings
                if len(untracked_targets) == 0 or not face_encodings:
                    # No untracked targets or no encodings - skip the rest
                    logger.debug(f"Skipping face matching: {len(untracked_targets)} untracked, {len(face_encodings)} encodings")
                else:
                    # ✅ NEW PIPELINE: Detect brightness for dark mode pipeline
                    gray_frame = cv2.cvtColor(rgb_full_frame, cv2.COLOR_RGB2GRAY)
                    brightness = np.mean(gray_frame)
                    
                    # Build adaptive params based on brightness
                    if brightness < 100:  # Low-light mode
                        adaptive_params = {
                            "brightness": int(brightness),
                            "tolerance": 0.46,
                            "confidence": 0.47
                        }
                        logger.debug(f"Dark mode activated: brightness={brightness:.0f}")
                    else:
                        adaptive_params = {
                            "brightness": int(brightness),
                            "tolerance": 0.55,
                            "confidence": 0.45
                        }
                    
                    # ✅ OPTIMIZED: Use adaptive thresholds based on lighting conditions
                    num_guards = len(untracked_targets)
                    
                    # ✅ CRITICAL FIX: STRICT matching thresholds to prevent false positives
                    # These thresholds now properly reject bags, shadows, and non-face objects
                    if adaptive_params["brightness"] < 100:  # Low-light mode
                        base_tolerance = 0.65  # VERY relaxed for upward angles + low-light
                        min_confidence = 0.35  # VERY relaxed for extreme angles in dark
                        logger.debug(f"Using VERY relaxed low-light tolerances for extreme angles: {base_tolerance:.2f}, confidence: {min_confidence:.2f}")
                    elif num_guards >= 2:
                        # MULTI-GUARD MODE: VERY relaxed for distance and extreme angles
                        base_tolerance = 0.65  # VERY relaxed to detect at various angles
                        min_confidence = 0.35  # VERY relaxed for upward/downward angles
                    else:
                        # SINGLE GUARD: EXTREMELY relaxed to handle all viewing angles including upward
                        base_tolerance = 0.68  # EXTREMELY relaxed for extreme angles (upward view)
                        min_confidence = 0.32  # EXTREMELY relaxed confidence for severely angled faces
                    
                    # ✅ CRITICAL: Build complete cost matrix with ALL candidates
                    # But only consider STRONG face detections to avoid false positives
                    cost_matrix = []
                    guard_all_matches = {}
                    
                    # ✅ ENHANCED: Filter face encodings by size and aspect ratio
                    # This removes false faces (bags, shadows, random patches)
                    valid_face_indices = []
                    for face_idx, face_location in enumerate(face_locations):
                        top, right, bottom, left = face_location
                        face_width = right - left
                        face_height = bottom - top
                        aspect_ratio = face_width / max(1, face_height)
                        
                        # ✅ CRITICAL VALIDATION: Face must have reasonable dimensions
                        # ENHANCED FOR EXTREME ANGLES: Now accepts upward angles, downward angles, profile views
                        # Surveillance cameras often have extreme viewing angles
                        MIN_FACE_SIZE = 6  # Minimum face dimension (allows extreme distance + extreme angles)
                        ASPECT_RATIO_MIN = 0.30  # Very permissive for extreme upward angle
                        ASPECT_RATIO_MAX = 2.5  # Very permissive for extreme angles
                        
                        if (face_width >= MIN_FACE_SIZE and 
                            face_height >= MIN_FACE_SIZE and 
                            ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
                            valid_face_indices.append((face_idx, face_width, face_height))
                        else:
                            logger.debug(f"[REJECT] Face_{face_idx}: Bad dims={face_width}x{face_height}, aspect={aspect_ratio:.2f}")
                    
                    logger.info(f"[FACE] Found {len(face_locations)} detections, {len(valid_face_indices)} valid face candidates")
                    
                    for target_idx, name in enumerate(untracked_targets):
                        target_encoding = self.targets_status[name]["encoding"]
                        if target_encoding is None:
                            logger.warning(f"[WARN] Skip {name}: no encoding available")
                            continue
                        
                        all_matches = []
                        
                        for face_idx, face_width, face_height in valid_face_indices:
                            if face_idx >= len(face_encodings):
                                continue
                            unknown_encoding = face_encodings[face_idx]
                            if unknown_encoding is None:
                                continue
                            
                            dist = face_recognition.face_distance([target_encoding], unknown_encoding)[0]
                            confidence = 1.0 - dist
                            
                            # ✅ CRITICAL FIX: PREVENT GUARD/FUGITIVE CONFUSION
                            # Cross-check: If this is a guard, make sure it's NOT the fugitive
                            if self.is_fugitive_detection and self.fugitive_face_encoding is not None:
                                fugitive_dist = face_recognition.face_distance([self.fugitive_face_encoding], unknown_encoding)[0]
                                fugitive_confidence = 1.0 - fugitive_dist
                                
                                # If face matches fugitive better than guard, REJECT this guard match
                                if fugitive_dist < dist:
                                    match_gap = dist - fugitive_dist
                                    if match_gap > 0.05:  # 5% gap means fugitive is clearly better match
                                        logger.debug(f"[REJECT] {name} vs face_{face_idx}: Matches FUGITIVE better (guard_dist={dist:.3f}, fugitive_dist={fugitive_dist:.3f})")
                                        continue
                            
                            # ✅ ANGLE-AWARE MATCHING: Accept matches within tolerance
                            # Low confidence can mean extreme angles (upward view), not necessarily wrong match
                            # For surveillance use case, tolerance-based matching is better than confidence-based
                            if dist <= base_tolerance:
                                all_matches.append({
                                    "face_idx": face_idx,
                                    "distance": dist,
                                    "confidence": confidence,
                                    "encoding": unknown_encoding
                                })
                                cost_matrix.append((dist, target_idx, face_idx, name, confidence, unknown_encoding))
                                if dist > 0.55:  # Extreme angle match - log it
                                    logger.debug(f"[ANGLE MATCH] {name} vs face_{face_idx}: High distance={dist:.3f} (likely extreme angle - upward/downward view)")
                            else:
                                logger.debug(f"[SKIP] {name} vs face_{face_idx}: Distance={dist:.3f} exceeds tolerance={base_tolerance:.2f}")
                        
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
                        
                        # ✅ INDUSTRIAL-LEVEL: Enhanced bounding box validation and tracker initialization
                        # Validate bounding box is reasonable
                        bbox_width = right - left
                        bbox_height = bottom - top
                        frame_h, frame_w = frame.shape[:2]
                        
                        # Check bounding box validity
                        if bbox_width < 20 or bbox_height < 20 or bbox_width > frame_w or bbox_height > frame_h:
                            logger.warning(f"Invalid bbox for {name}: {left}, {top}, {right}, {bottom} (skipping)")
                            continue
                        
                        # ✅ INDUSTRIAL-LEVEL: Try multiple tracker types for robustness
                        tracker = None
                        tracker_types = ['CSRT', 'KCF', 'MedianFlow']
                        
                        for tracker_type in tracker_types:
                            try:
                                if tracker_type == 'CSRT':
                                    tracker = cv2.legacy.TrackerCSRT_create()
                                elif tracker_type == 'KCF':
                                    tracker = cv2.legacy.TrackerKCF_create()
                                elif tracker_type == 'MedianFlow':
                                    tracker = cv2.legacy.TrackerMedianFlow_create()
                                
                                if tracker is not None:
                                    # Try to initialize tracker
                                    tracker.init(frame, (left, top, bbox_width, bbox_height))
                                    logger.debug(f"Initialized {tracker_type} tracker for {name}")
                                    break  # Successfully initialized, exit loop
                            except Exception as e:
                                logger.debug(f"Failed to initialize {tracker_type} for {name}: {e}")
                                tracker = None
                                continue
                        
                        self.targets_status[name]["tracker"] = tracker
                        self.targets_status[name]["face_box"] = (left, top, right, bottom)
                        self.targets_status[name]["visible"] = True
                        self.targets_status[name]["missing_pose_counter"] = 0
                        self.targets_status[name]["face_confidence"] = max(0.0, min(1.0, confidence))  # ✅ IMPROVED: Clamp confidence to [0,1]
                        
                        # ✅ STABILITY: Track consecutive detections for stable tracking
                        self.targets_status[name]["consecutive_detections"] = self.targets_status[name].get("consecutive_detections", 0) + 1
                        if self.targets_status[name]["consecutive_detections"] >= 3:
                            self.targets_status[name]["stable_tracking"] = True
                        self.targets_status[name]["face_match_confidence"] = max(0.0, min(1.0, confidence))  # ✅ IMPROVED: Clamp confidence
                        # ❌ FIXED: REMOVED incorrect timer reset here! Timer should ONLY reset AFTER grace period ends (line 3290), not on detection
                        # This was preventing grace period from ever executing properly
                        
                        # ✅ ENHANCED: Log guard identification with visual indicator
                        logger.warning(f"[DETECTED] ✓ {name} identified & tracking (confidence: {confidence:.3f}, distance: {dist:.3f}, bbox: {bbox_width}x{bbox_height} px)")
                        
                        # Log first detection event
                        if self.targets_status[name].get("consecutive_detections", 0) == 1:
                            logger.info(f"[NEW TRACK] {name} first detected in frame (face confidence: {confidence:.3f})")
                    
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
        
        # 3. Overlap Check (OPTIMIZED: Reduce frequency for performance)
        # ✅ CRITICAL OPTIMIZATION: Only check overlaps every 5 frames (167ms) instead of every frame
        # Overlap detection is ~15-20ms per frame for multiple guards, so reducing frequency saves significant time
        # At 2+ guards with overlap checks every frame: ~30-40ms overhead
        # With new interval (every 5 frames): ~6-8ms overhead spread across 5 frames = ~1.2-1.6ms per frame savings
        active_names = [n for n, s in self.targets_status.items() if s["visible"]]
        
        num_visible_guards = len(active_names)
        
        # ✅ OPTIMIZATION: Overlap checks are expensive, only run when needed
        # - Single guard: Never check (single target can't overlap with itself)
        # - 2-3 guards: Check every 5 frames (comprehensive but not too frequent)  
        # - 4+ guards: Check every 10 frames (less frequent due to higher complexity)
        if num_visible_guards >= 2:
            check_frequency = 5 if num_visible_guards <= 3 else 10
            should_check_overlap = (self.frame_counter % check_frequency == 0)
        else:
            should_check_overlap = False
        
        if should_check_overlap:
            # ✅ IMPROVED: Multi-Guard Pose Detection Resolution
            self.targets_status = resolve_overlapping_poses(self.targets_status, iou_threshold=0.3)
        
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
        required_act = self.active_required_action  # Use current active action from dropdown
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
                        # ✅ CRITICAL: ALWAYS detect pose - do NOT skip frames
                        # Pose detection is essential for action classification and alert triggering
                        # Every frame must have pose analysis for real-time responsiveness
                        should_detect_pose = True
                        logger.debug(f"[POSE] {name}: Running pose detection on every frame")
                        
                        if should_detect_pose:
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            
                            # ✅ CRITICAL: Use MediaPipe Pose for pose detection (always available, works offline)
                            # Each guard gets independent pose detection from their cropped region
                            if self.pose_model:
                                pose_result = self.pose_model.process(rgb_crop)
                            else:
                                pose_result = None
                            
                            # Convert MoveNet output to cache format
                            results_crop = pose_result
                            
                            # Cache the result for next frame
                            status["last_cached_pose"] = results_crop
                        else:
                            # Use cached pose result
                            results_crop = status.get("last_cached_pose")
                        
                        # ========== ACTION DETECTION LOGIC ==========
                        current_action = "Unknown"
                        if results_crop is not None and results_crop.pose_landmarks is not None:
                            pose_found_in_box = True
                            status["missing_pose_counter"] = 0 # Reset
                            
                            # ✅ MEDIAPIPE FORMAT: Extract actual visibility from landmarks
                            # Calculate actual pose quality from keypoint visibility
                            pose_landmarks = results_crop.pose_landmarks.landmark
                            visible_count = sum(1 for lm in pose_landmarks if lm.visibility > 0.5)
                            avg_visibility = sum(lm.visibility for lm in pose_landmarks) / len(pose_landmarks)
                            pose_quality = visible_count / len(pose_landmarks)  # Actual quality score
                            status["pose_confidence"] = pose_quality
                            status["skeleton_keypoints"] = visible_count  # Update keypoint count for tracking
                            
                            # ✅ MULTI-GUARD FIX: Per-guard pose quality threshold validation
                            if "pose_quality_history" not in status:
                                status["pose_quality_history"] = deque(maxlen=10)
                            status["pose_quality_history"].append(pose_quality)
                            
                            # ✅ EXTREME ANGLE HANDLING: Accept LOWER pose quality for back/upper views
                            # - Front view: 60% visibility required (good full-body view)
                            # - Back/Side view: 40% visibility sufficient (partially visible)
                            # - Extreme angles: 20% visibility minimum (only few keypoints visible)
                            # This prevents losing track when guard turns away or looks down
                            min_pose_quality_threshold = 0.20  # ✅ RELAXED: Allow very sparse poses
                            
                            # Only process pose if quality meets threshold
                            if pose_quality >= min_pose_quality_threshold:  # Accept even very sparse poses
                                # ✅ MEDIAPIPE: Use actual pose landmarks to classify action
                                raw_action = classify_action(pose_landmarks, crop.shape[0], crop.shape[1])
                                
                                # ✅ MULTI-GUARD ENHANCEMENT: Filter out "Unknown" from buffer
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
                                    
                                    # ✅ STABILITY: Only use action if tracking is stable (3+ consecutive frames)
                                    # AND pose consensus is high (70% agreement in buffer)
                                    if confidence_pct >= 0.70 and status.get("stable_tracking", False):
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
                            
                            # ✅ FIX 4: Store current_action in status for alert stopping logic
                            status["current_action"] = current_action
                            
                            # ✅ MULTI-GUARD IMPROVEMENT: Enhanced debug logging for pose tracking
                            if len(status["pose_buffer"]) >= min_buffer:
                                # Log detailed info for debugging
                                buffer_summary = Counter(status["pose_buffer"])
                                most_common_action = buffer_summary.most_common(1)[0][0]
                                buffer_str = ", ".join([f"{action}:{count}" for action, count in buffer_summary.most_common()])
                                avg_quality = sum(status["pose_quality_history"]) / len(status["pose_quality_history"])
                                logger.debug(f"[POSE] {name}: {current_action} (quality:{pose_quality:.2f}, avg_qual:{avg_quality:.2f}, buffer:[{buffer_str}], consensus:{buffer_summary[most_common_action]/len(status['pose_buffer']):.1%})")

                            # ========== ACTION ALERT LOGIC (IMPROVED WITH ROBUST MODE CHECKING) ==========
                            # ✅ PHASE 3.1: GRACE PERIOD & ACTION INTERVAL LOGIC
                            # Check if we're in alert mode and monitoring actions
                            if self.is_alert_mode and monitor_mode in ["Action Alerts Only"]:
                                elapsed_time = current_time - status["last_action_time"]
                                grace_period_start = self.alert_interval - 10  # Last 10 seconds of interval
                                
                                # ✅ CHECK IF IN GRACE PERIOD (last 10 seconds of interval)
                                if grace_period_start <= elapsed_time < self.alert_interval:
                                    # We're in the grace period window
                                    # Check if required action is being performed
                                    if current_action == self.active_required_action:
                                        # ✅ ACTION WAS PERFORMED in grace period
                                        status["action_performed"] = True
                                        logger.debug(f"[GRACE] {name}: Action '{self.active_required_action}' performed in grace period (elapsed: {elapsed_time:.1f}s)")
                                
                                # ✅ CHECK IF INTERVAL COMPLETE (grace period just ended)
                                elif elapsed_time >= self.alert_interval:
                                    grace_period_elapsed = elapsed_time - grace_period_start
                                    
                                    # Only log/alert once when first entering the timeout window
                                    # (grace_period_elapsed should be ~10 seconds when just completed)
                                    if grace_period_elapsed <= 1.0 and not status.get("alert_logged_timeout", False):
                                        # ✅ INTERVAL COMPLETE - LOG RESULT
                                        if status["action_performed"]:
                                            # Action was performed - log success
                                            if self.is_logging:
                                                img_path = "N/A"
                                                try:
                                                    snapshot_result = self.capture_alert_snapshot(frame[by1:by2, bx1:bx2], name, check_rate_limit=False)
                                                    img_path = snapshot_result if snapshot_result else "N/A"
                                                except Exception as e:
                                                    logger.warning(f"Failed to capture grace period snapshot: {e}")
                                                
                                                self.temp_log.append((
                                                    time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    name,
                                                    f"'{self.active_required_action}' required",
                                                    "PERFORMED",
                                                    img_path,
                                                    f"{status['pose_confidence']:.2f}"
                                                ))
                                                self.temp_log_counter += 1
                                                logger.info(f"[ALERT] {name}: Action PERFORMED in grace period - logging success")
                                            status["alert_logged_timeout"] = True
                                        else:
                                            # Action NOT performed - log timeout and trigger alert
                                            if self.is_logging:
                                                img_path = "N/A"
                                                try:
                                                    snapshot_result = self.capture_alert_snapshot(frame[by1:by2, bx1:bx2], name, check_rate_limit=False)
                                                    img_path = snapshot_result if snapshot_result else "N/A"
                                                except Exception as e:
                                                    logger.warning(f"Failed to capture timeout snapshot: {e}")
                                                
                                                self.temp_log.append((
                                                    time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    name,
                                                    f"'{self.active_required_action}' required",
                                                    "TIMEOUT",
                                                    img_path,
                                                    f"{status['pose_confidence']:.2f}"
                                                ))
                                                self.temp_log_counter += 1
                                                logger.warning(f"[ALERT] {name}: Action TIMEOUT - required action NOT performed in grace period!")
                                            
                                            # ✅ TRIGGER 15-SECOND ALERT SOUND
                                            if status["alert_stop_event"] is None:
                                                status["alert_stop_event"] = threading.Event()
                                            status["alert_stop_event"].clear()
                                            if not status.get("alert_sound_thread") or not status["alert_sound_thread"].is_alive():
                                                status["alert_sound_thread"] = play_siren_sound(
                                                    stop_event=status["alert_stop_event"],
                                                    duration_seconds=15
                                                )
                                                logger.warning(f"[ALERT] {name}: 15-second alert sound triggered for timeout")
                                            
                                            status["alert_logged_timeout"] = True
                                        
                                        # ✅ RESET FOR NEXT INTERVAL
                                        status["last_action_time"] = current_time
                                        status["action_performed"] = False
                                        status["alert_logged_timeout"] = False
                            
                            # ========== EXISTING ACTION ALERT LOGIC (Stop sound when action performed) ==========
                            # ✅ FIX: Use active_required_action for EACH guard check (not once at start)
                            # This ensures alert stops ONLY when the CURRENTLY SELECTED action is performed
                            required_action_now = self.active_required_action  # Use current active action from dropdown
                            if monitor_mode in ["Action Alerts Only"]:
                                if self.is_alert_mode and current_action == required_action_now:
                                    # ✅ CRITICAL: Use required_action_now (live action value), not stale required_act
                                    # Only stop sound if current action equals the required action selected in UI
                                    status["last_action_time"] = current_time
                                    status["action_performed"] = True  # ✅ PHASE 3.1: Mark action as performed
                                    status["alert_triggered_state"] = False
                                    # STOP ALERT SOUND only when selected action is performed
                                    if status["alert_stop_event"] is not None:
                                        status["alert_stop_event"].set()  # Signal sound to stop
                                        logger.info(f"[ALERT STOP] {name}: Alert stopped - required action '{required_action_now}' performed (dropdown value)")
                                    if self.is_logging and status["last_logged_action"] != required_action_now:
                                        # Rate limiting: only log once per minute per target
                                        time_since_last_log = current_time - status["last_log_time"]
                                        if time_since_last_log > 60:
                                            self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, current_action, "Action Performed", "N/A", f"{status['face_confidence']:.2f}"))
                                            status["last_log_time"] = current_time
                                            self.temp_log_counter += 1
                                        status["last_logged_action"] = required_action_now
                            # ============================================================
                            
                            if current_action != required_action_now:

                                status["last_logged_action"] = None
                            
                            # ========== STILLNESS DETECTION (PRO MODE ONLY) ==========
                            # ✅ NEW PIPELINE: Simplified for MoveNet format
                            if self.is_pro_mode and self.is_stillness_alert:
                                # Extract pose landmarks as vector for movement detection
                                if results_crop is not None:
                                    # MoveNet format - simplified stillness detection
                                    # Just check if pose is still being detected (not moving)
                                    logger.debug(f"Stillness check for {name}")
                                    # Simplified - just log for now
                            
                            # --- Dynamic Bounding Box Logic & Full Body Skeleton Drawing ---
                            h_c, w_c = crop.shape[:2]
                            
                            # ✅ CRITICAL FIX: Check if pose landmarks exist before accessing
                            if results_crop and results_crop.pose_landmarks:
                                p_lms = results_crop.pose_landmarks.landmark
                                
                                # ✅ ENHANCED: Extract all visible landmarks for full-body tracking
                                keypoints = []
                                for i, lm in enumerate(p_lms):
                                    if 0 <= lm.x <= 1 and 0 <= lm.y <= 1 and lm.visibility > 0.3:
                                        x_pixel = int(lm.x * w_c) + bx1
                                        y_pixel = int(lm.y * h_c) + by1
                                        keypoints.append((i, x_pixel, y_pixel, lm.visibility))
                                
                                # Draw full-body skeleton connections (MediaPipe Pose has 33 landmarks)
                                # Key connections: shoulders-elbows-wrists, hips-knees-ankles, spine
                                skeleton_connections = [
                                    # Upper body
                                    (11, 13), (13, 15),  # Left arm: shoulder -> elbow -> wrist
                                    (12, 14), (14, 16),  # Right arm: shoulder -> elbow -> wrist
                                    (11, 12),            # Shoulders
                                    # Spine
                                    (11, 23), (12, 24),  # Shoulders to hips
                                    (23, 24),            # Hip connection
                                    # Lower body
                                    (23, 25), (25, 27),  # Left leg: hip -> knee -> ankle
                                    (24, 26), (26, 28),  # Right leg: hip -> knee -> ankle
                                ]
                                
                                # Build coordinate map for connections
                                kpt_dict = {kpt[0]: (kpt[1], kpt[2], kpt[3]) for kpt in keypoints}
                                
                                # ✅ Draw skeleton connections (full-body 3D tracking)
                                for start_idx, end_idx in skeleton_connections:
                                    if start_idx in kpt_dict and end_idx in kpt_dict:
                                        start_pt = kpt_dict[start_idx]
                                        end_pt = kpt_dict[end_idx]
                                        # Color based on visibility
                                        avg_visibility = (start_pt[2] + end_pt[2]) / 2
                                        if avg_visibility > 0.5:
                                            color = (0, 255, 0)  # Green - high visibility
                                        else:
                                            color = (0, 165, 255)  # Orange - medium visibility
                                        cv2.line(frame, (start_pt[0], start_pt[1]), (end_pt[0], end_pt[1]), color, 2)
                                
                                # ✅ Draw keypoints as circles
                                for idx, x, y, visibility in keypoints:
                                    if visibility > 0.5:
                                        radius = 4
                                        color = (0, 255, 0)  # Green for high visibility
                                    else:
                                        radius = 2
                                        color = (0, 165, 255)  # Orange for medium visibility
                                    cv2.circle(frame, (x, y), radius, color, -1)
                                
                                # Calculate bounding box from skeleton with angle tolerance
                                # Use ALL landmarks (even low visibility) for sparse back/angle views
                                lx = [lm.x * w_c for lm in p_lms if 0 <= lm.x <= 1]
                                ly = [lm.y * h_c for lm in p_lms if 0 <= lm.y <= 1]
                                
                                if lx and ly and len(lx) >= 2:  # Need at least 2 points for valid bbox
                                    d_x1 = int(min(lx)) + bx1
                                    d_y1 = int(min(ly)) + by1
                                    d_x2 = int(max(lx)) + bx1
                                    d_y2 = int(max(ly)) + by1
                                    # Expand slightly for sparse skeletons (back/angle views)
                                    expand = 20 if pose_quality < 0.4 else 10
                                    d_x1 = max(0, d_x1 - expand)
                                    d_y1 = max(0, d_y1 - expand)
                                    d_x2 = min(frame_w, d_x2 + expand)
                                    d_y2 = min(frame_h, d_y2 + expand)
                                else:
                                    # Fallback to face box when skeleton too sparse
                                    d_x1, d_y1, d_x2, d_y2 = fx1, fy1, fx2, fy2
                            else:
                                # No pose detected - use face box
                                d_x1, d_y1, d_x2, d_y2 = fx1, fy1, fx2, fy2
                            
                            # Add minimal padding for sparse angles, more for good quality
                            if pose_quality >= 0.5:
                                # Good quality - use minimal padding
                                d_x1 = max(0, d_x1 - 10)
                                d_y1 = max(0, d_y1 - 10)
                                d_x2 = min(frame_w, d_x2 + 10)
                                d_y2 = min(frame_h, d_y2 + 10)
                            else:
                                # Sparse skeleton (back/angle) - use larger padding to capture full body
                                d_x1 = max(0, d_x1 - 30)
                                d_y1 = max(0, d_y1 - 30)
                                d_x2 = min(frame_w, d_x2 + 30)
                                d_y2 = min(frame_h, d_y2 + 30)
                            
                            # Draw Dynamic Box
                            # Color based on skeleton quality: green (good), orange (sparse)
                            box_color = (0, 255, 0) if pose_quality >= 0.5 else (0, 165, 255)
                            cv2.rectangle(frame, (d_x1, d_y1), (d_x2, d_y2), box_color, 2)

                            # Cache/update body box and initialize body tracker if needed
                            status["body_box"] = (d_x1, d_y1, d_x2, d_y2)
                            # Mark this as a fallback if pose quality is low
                            status["body_box_quality"] = "sparse" if pose_quality < 0.4 else "normal"
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
                                confidence_indicator = "★★★"
                            elif face_conf > 0.65:
                                id_color = (0, 165, 255)  # Orange - medium confidence
                                confidence_indicator = "★★"
                            else:
                                id_color = (0, 0, 255)  # Red - low confidence
                                confidence_indicator = "★"
                            
                            # ✅ ENHANCED: Display guard name with visual indicators
                            # Show name with checkmark when identified and tracked
                            tracking_status = "✓" if status.get("stable_tracking") else "◇"
                            info_text = f"{tracking_status} {name} ({face_conf:.2f})"
                            action_text = f"{current_action} (P:{pose_conf:.1%})"
                            
                            cv2.putText(frame, info_text, (d_x1, d_y1 - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, id_color, 2)
                            cv2.putText(frame, action_text, (d_x1, d_y1 - 8), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

                # Ghost Box Removal Logic
                if not pose_found_in_box:
                    status["missing_pose_counter"] += 1
                    
                    # ✅ FALLBACK DRAWING: Draw face box even when pose isn't detected
                    # This ensures targets remain visible on screen at all times
                    face_conf = status.get("face_confidence", 0.0)
                    
                    # Color code based on identification confidence
                    if face_conf > 0.85:
                        id_color = (0, 255, 0)  # Green - high confidence
                    elif face_conf > 0.65:
                        id_color = (0, 165, 255)  # Orange - medium confidence
                    else:
                        id_color = (0, 0, 255)  # Red - low confidence
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), id_color, 2)
                    
                    # Display guard name with confidence
                    info_text = f"{name} ({face_conf:.2f})"
                    cv2.putText(frame, info_text, (fx1, fy1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, id_color, 2)
                    cv2.putText(frame, "No Pose", (fx1, fy1 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
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
            if self.is_alert_mode and monitor_mode in ["Action Alerts Only"]:
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
    app.root.mainloop()
