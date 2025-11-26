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
import sys

# --- 0. Critical Dependency Check ---
try:
    # This requires 'opencv-contrib-python'
    _test_tracker = cv2.legacy.TrackerCSRT_create()
except AttributeError:
    print("CRITICAL ERROR: 'cv2.legacy' is missing.")
    print("Please run: pip install opencv-contrib-python")
    # We continue, but tracking will likely fail if not fixed.

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

# --- 1. Configuration Loading (Robust Merge Strategy) ---
def load_config():
    # These defaults cover ALL settings used in the code to prevent KeyErrors
    defaults = {
        "detection": {
            "min_detection_confidence": 0.5, 
            "min_tracking_confidence": 0.5, 
            "face_recognition_tolerance": 0.5, 
            "re_detect_interval": 60
        },
        "alert": {
            "default_interval_seconds": 10, 
            "alert_cooldown_seconds": 2.5
        },
        "performance": {
            "gui_refresh_ms": 30, 
            "pose_buffer_size": 12, 
            "frame_skip_interval": 2,
            "enable_frame_skipping": True,          # Fixed: Ensure this key exists
            "min_buffer_for_classification": 5      # Fixed: Ensure this key exists
        },
        "logging": {
            "log_directory": "logs", 
            "max_log_size_mb": 10, 
            "auto_flush_interval": 50
        },
        "storage": {
            "alert_snapshots_dir": "alert_snapshots", 
            "snapshot_retention_days": 30,
            "guard_profiles_dir": "guard_profiles", 
            "capture_snapshots_dir": "capture_snapshots",
            "pose_references_dir": "pose_references"
        },
        "monitoring": {
            "mode": "pose", 
            "session_restart_prompt_hours": 8
        }
    }

    try:
        # Use script directory to locate config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Recursive merge: Fill in missing keys in user_config with defaults
            # This fixes the crash where old config files lack new settings
            for section, keys in defaults.items():
                if section not in user_config:
                    user_config[section] = keys
                else:
                    for key, val in keys.items():
                        if key not in user_config[section]:
                            user_config[section][key] = val
            
            # Optional: Save back the merged config so the file is updated
            try:
                with open(config_path, 'w') as f:
                    json.dump(user_config, f, indent=4)
            except: pass
            
            return user_config
        else:
            # Create fresh config file
            with open(config_path, 'w') as f:
                json.dump(defaults, f, indent=4)
            return defaults
    except Exception as e:
        print(f"Config load warning: {e}. Using internal defaults.")
        return defaults

CONFIG = load_config()

# --- 2. Logging Setup ---
# Ensure logs go to script directory/logs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, CONFIG["logging"]["log_directory"])

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logger = logging.getLogger("PoseGuard")
logger.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "session.log"),
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
    # Helper to ensure all paths are absolute relative to script
    def resolve(path_name):
        return os.path.join(SCRIPT_DIR, CONFIG["storage"].get(path_name, path_name))

    paths = {
        "guard_profiles": resolve("guard_profiles_dir"),
        "pose_references": resolve("pose_references_dir"),
        "capture_snapshots": resolve("capture_snapshots_dir"),
        "alert_snapshots": resolve("alert_snapshots_dir"),
        "logs": LOG_DIR
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

# Initialize directories
PATHS = get_storage_paths()

csv_file = os.path.join(LOG_DIR, "events.csv")
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name", "Action", "Status", "Image_Path", "Confidence"])

# --- 4. Cleanup Old Snapshots ---
def cleanup_old_snapshots():
    try:
        retention_days = CONFIG["storage"]["snapshot_retention_days"]
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        snapshot_dir = PATHS["alert_snapshots"]
        
        if os.path.exists(snapshot_dir):
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
    def _sound_worker():
        # Look for sound file in script directory
        mp3_path = os.path.join(SCRIPT_DIR, sound_file)
        start_time = time.time()
        
        if PYGAME_AVAILABLE and os.path.exists(mp3_path):
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                pygame.mixer.music.load(mp3_path)
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.play(-1)
                
                while True:
                    elapsed = time.time() - start_time
                    if stop_event and stop_event.is_set(): break
                    if elapsed >= duration_seconds: break
                    time.sleep(0.1)
                    
                pygame.mixer.music.stop()
                return
            except Exception as e:
                logger.warning(f"Pygame playback error: {e}")
        
        # Fallback to Beep
        try:
            while True:
                elapsed = time.time() - start_time
                if stop_event and stop_event.is_set(): break
                if elapsed >= duration_seconds: break
                
                if platform.system() == "Windows":
                    import winsound
                    winsound.Beep(2500, 150)
                    time.sleep(0.05)
                    winsound.Beep(1800, 150)
                    time.sleep(0.05)
                else:
                    print('\a')
                    time.sleep(0.5)
        except Exception:
            pass

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

# --- classify_action ---
def classify_action(landmarks, h, w):
    try:
        lm = mp_holistic.PoseLandmark
        
        nose = landmarks[lm.NOSE.value]
        l_wrist, r_wrist = landmarks[lm.LEFT_WRIST.value], landmarks[lm.RIGHT_WRIST.value]
        l_elbow, r_elbow = landmarks[lm.LEFT_ELBOW.value], landmarks[lm.RIGHT_ELBOW.value]
        l_shoulder, r_shoulder = landmarks[lm.LEFT_SHOULDER.value], landmarks[lm.RIGHT_SHOULDER.value]
        l_hip, r_hip = landmarks[lm.LEFT_HIP.value], landmarks[lm.RIGHT_HIP.value]
        l_knee, r_knee = landmarks[lm.LEFT_KNEE.value], landmarks[lm.RIGHT_KNEE.value]

        nose_y = nose.y * h
        lw_y, rw_y = l_wrist.y * h, r_wrist.y * h
        lw_x, rw_x = l_wrist.x * w, r_wrist.x * w
        ls_y, rs_y = l_shoulder.y * h, r_shoulder.y * h
        ls_x, rs_x = l_shoulder.x * w, r_shoulder.x * w
        
        # Visibility
        l_vis, r_vis = l_wrist.visibility > 0.6, r_wrist.visibility > 0.6
        lk_vis, rk_vis = l_knee.visibility > 0.6, r_knee.visibility > 0.6
        
        # 1. Hands Up
        if (l_vis and r_vis and lw_y < (nose_y - 0.1 * h) and rw_y < (nose_y - 0.1 * h)):
            return "Hands Up"
        
        # 2. Hands Crossed
        if (l_vis and r_vis):
            chest_y = (ls_y + rs_y) / 2
            center_x = (ls_x + rs_x) / 2
            if (abs(lw_y - chest_y) < 0.2 * h and abs(rw_y - chest_y) < 0.2 * h):
                if ((lw_x > center_x and rw_x < center_x) or (lw_x < center_x and rw_x > center_x)):
                    return "Hands Crossed"
        
        # 3. T-Pose
        if (l_vis and r_vis and l_elbow.visibility > 0.6 and r_elbow.visibility > 0.6):
            if (abs(lw_y - ls_y) < 0.15 * h and abs(rw_y - rs_y) < 0.15 * h):
                if (lw_x < (ls_x - 0.2 * w) and rw_x > (rs_x + 0.2 * w)):
                    return "T-Pose"
        
        # 4. One Hand Raised
        if l_vis and lw_y < (nose_y - 0.1 * h) and (not r_vis or rw_y > ls_y):
            return "One Hand Raised (Left)"
        if r_vis and rw_y < (nose_y - 0.1 * h) and (not l_vis or lw_y > rs_y):
            return "One Hand Raised (Right)"
        
        # 5. Sit/Stand
        if lk_vis and rk_vis:
            angle_l = abs(l_knee.y - l_hip.y)
            angle_r = abs(r_knee.y - r_hip.y)
            if ((angle_l + angle_r) / 2) < 0.15:
                return "Sit"
        
        return "Standing" 
    except Exception:
        return "Unknown"

# --- Helper: Body Box ---
def calculate_body_box(face_box, frame_h, frame_w):
    x1, y1, x2, y2 = face_box
    fw, fh = x2 - x1, y2 - y1
    cx = x1 + (fw // 2)
    
    bx1 = max(0, int(cx - (fw * 3.0)))
    bx2 = min(frame_w, int(cx + (fw * 3.0)))
    by1 = max(0, int(y1 - (fh * 0.5)))
    by2 = frame_h
    return (bx1, by1, bx2, by2)

# --- Helper: Cameras ---
def detect_available_cameras(max_cameras=3):
    available = []
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: available.append(i)
                cap.release()
        except: pass
    return available

# --- Main Application ---
class PoseApp:
    def __init__(self, window_title="Pose Guard (Safe Version)"):
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1600x900")
        self.root.configure(bg="black")
        try: self.root.state('zoomed') 
        except: pass
        
        self.cap = None
        self.is_running = False
        self.is_logging = False
        self.is_alert_mode = False
        self.is_in_capture_mode = False
        self.fugitive_mode = False
        
        # State
        self.targets_status = {}
        self.re_detect_counter = 0
        self.temp_log = []
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.last_process_frame = None
        self.alert_interval = CONFIG["alert"]["default_interval_seconds"]
        
        # Fugitive
        self.fugitive_enc = None
        self.fugitive_name = "Unknown"
        self.fugitive_alert_done = False
        self.fugitive_stop_event = None

        try:
            self.holistic = mp_holistic.Holistic(
                min_detection_confidence=CONFIG["detection"]["min_detection_confidence"],
                min_tracking_confidence=CONFIG["detection"]["min_tracking_confidence"],
                static_image_mode=False
            )
        except Exception as e:
            messagebox.showerror("Error", f"MediaPipe Error: {e}")
            self.root.destroy()
            return

        self.setup_ui()
        self.load_targets()
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        self.root.mainloop()

    def setup_ui(self):
        self.root.grid_rowconfigure(0, weight=10)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Video Area
        self.video_frame = tk.Frame(self.root, bg="red")
        self.video_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.video_label = tk.Label(self.video_frame, bg="black", text="Camera Off", fg="white")
        self.video_label.pack(fill="both", expand=True)
        
        # Previews
        self.prev_guard = tk.Label(self.video_label, bg="black", fg="white", width=20, height=8)
        self.prev_guard.place(relx=0.02, rely=0.02, anchor="nw")
        self.prev_fugitive = tk.Label(self.video_label, bg="black", fg="white", width=20, height=8)
        self.prev_fugitive.place(relx=0.98, rely=0.02, anchor="ne")

        # Controls
        self.controls = tk.Frame(self.root, bg="gold")
        self.controls.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        btn_opts = {'bg': '#333', 'fg': 'white', 'font': ('Arial', 10, 'bold'), 'padx': 10, 'pady': 5}
        
        self.btn_start = tk.Button(self.controls, text="Start Camera", command=self.start_camera, bg="green", fg="white")
        self.btn_start.pack(side="left", padx=5, pady=5)
        
        self.btn_stop = tk.Button(self.controls, text="Stop Camera", command=self.stop_camera, state="disabled", bg="red", fg="white")
        self.btn_stop.pack(side="left", padx=5, pady=5)
        
        self.btn_add = tk.Button(self.controls, text="Add Guard", command=self.add_guard_dialog, state="disabled", **btn_opts)
        self.btn_add.pack(side="left", padx=5, pady=5)
        
        self.req_action = tk.StringVar(value="Hands Up")
        tk.OptionMenu(self.controls, self.req_action, "Hands Up", "Hands Crossed", "One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing", command=self.reset_alert_timers).pack(side="left", padx=5)
        
        self.btn_alert = tk.Button(self.controls, text="Start Alert", command=self.toggle_alert, state="disabled", bg="orange", fg="white")
        self.btn_alert.pack(side="left", padx=5)
        
        self.btn_fugitive = tk.Button(self.controls, text="Fugitive Mode", command=self.toggle_fugitive, state="disabled", bg="darkred", fg="white")
        self.btn_fugitive.pack(side="left", padx=5)
        
        self.status_lbl = tk.Label(self.controls, text="FPS: 0", bg="gold")
        self.status_lbl.pack(side="right", padx=10)
        
        # Target List
        self.list_frame = tk.Frame(self.root, bg="silver", width=200)
        self.list_frame.grid(row=0, column=2, rowspan=2, sticky="ns")
        tk.Label(self.list_frame, text="Targets").pack()
        self.target_list = tk.Listbox(self.list_frame, selectmode=tk.MULTIPLE)
        self.target_list.pack(fill="both", expand=True)
        tk.Button(self.list_frame, text="Track Selected", command=self.apply_tracking).pack(fill="x")
        
        # Capture buttons
        self.cap_frame = tk.Frame(self.root, bg="black")
        self.btn_snap = tk.Button(self.cap_frame, text="SNAP", command=self.snap_photo, bg="blue", fg="white", font=("Arial", 14))
        self.btn_snap.pack(side="left", padx=20)
        self.btn_cancel = tk.Button(self.cap_frame, text="CANCEL", command=self.end_capture, bg="gray", fg="white")
        self.btn_cancel.pack(side="left")

    def graceful_exit(self):
        if self.is_running:
            if not messagebox.askyesno("Exit", "Stop camera and exit?"): return
        self.is_running = False
        if self.cap: self.cap.release()
        if self.is_logging: self.flush_logs()
        self.root.destroy()
        sys.exit(0)

    # --- Logic ---
    def load_targets(self):
        self.target_map = {}
        self.target_list.delete(0, tk.END)
        # Load from profiles dir
        files = glob.glob(os.path.join(PATHS["guard_profiles"], "target_*_face.jpg"))
        # Also check root for legacy
        files += glob.glob(os.path.join(SCRIPT_DIR, "target_*_face.jpg"))
        
        seen = set()
        for f in files:
            name = os.path.basename(f).replace("target_", "").replace("_face.jpg", "").replace("_", " ")
            if name not in seen:
                self.target_map[name] = f
                self.target_list.insert(tk.END, name)
                seen.add(name)

    def apply_tracking(self):
        self.targets_status = {}
        selection = self.target_list.curselection()
        for idx in selection:
            name = self.target_list.get(idx)
            path = self.target_map.get(name)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    self.targets_status[name] = {
                        "encoding": encs[0],
                        "tracker": None,
                        "face_box": None,
                        "visible": False,
                        "last_action_time": time.time(),
                        "alert_cooldown": 0,
                        "alert_trig": False,
                        "buffer": deque(maxlen=CONFIG["performance"].get("pose_buffer_size", 10)),
                        "miss_count": 0,
                        "last_snap": 0,
                        "stop_event": None
                    }
            except Exception as e:
                logger.error(f"Load error {name}: {e}")
        messagebox.showinfo("Info", f"Tracking {len(self.targets_status)} targets")

    def start_camera(self):
        if self.is_running: return
        cams = detect_available_cameras()
        if not cams:
            messagebox.showerror("Error", "No camera found")
            return
        self.cap = cv2.VideoCapture(cams[0])
        if self.cap.isOpened():
            self.is_running = True
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.btn_add.config(state="normal")
            self.btn_alert.config(state="normal")
            self.btn_fugitive.config(state="normal")
            self.update_loop()

    def stop_camera(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.video_label.config(image='')
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    def toggle_alert(self):
        self.is_alert_mode = not self.is_alert_mode
        if self.is_alert_mode:
            self.is_logging = True
            self.btn_alert.config(text="Stop Alert", bg="red")
            self.reset_alert_timers()
        else:
            self.flush_logs()
            self.is_logging = False
            self.btn_alert.config(text="Start Alert", bg="orange")

    def reset_alert_timers(self, _=None):
        t = time.time()
        for s in self.targets_status.values(): s["last_action_time"] = t

    def toggle_fugitive(self):
        if not self.fugitive_mode:
            path = filedialog.askopenfilename()
            if not path: return
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if not encs: raise Exception("No face found")
                self.fugitive_enc = encs[0]
                self.fugitive_name = simpledialog.askstring("Name", "Fugitive Name:") or "Unknown"
                self.fugitive_mode = True
                self.btn_fugitive.config(bg="green")
                
                # Preview
                pil = Image.fromarray(img)
                pil.thumbnail((100,100))
                ph = ImageTk.PhotoImage(pil)
                self.prev_fugitive.config(image=ph, text=self.fugitive_name)
                self.prev_fugitive.image = ph
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            self.fugitive_mode = False
            self.btn_fugitive.config(bg="darkred")
            self.prev_fugitive.config(image='', text='')

    # --- Capture/Onboarding ---
    def add_guard_dialog(self):
        if messagebox.askyesno("Add", "Use Camera?"):
            self.start_capture()
        else:
            path = filedialog.askopenfilename()
            if path:
                name = simpledialog.askstring("Name", "Guard Name:")
                if name:
                    try:
                        img = face_recognition.load_image_file(path)
                        if len(face_recognition.face_locations(img)) == 1:
                            import shutil
                            dest = os.path.join(PATHS["guard_profiles"], f"target_{name.replace(' ','_')}_face.jpg")
                            shutil.copy(path, dest)
                            self.load_targets()
                    except Exception as e:
                        logger.error(f"Upload failed: {e}")

    def start_capture(self):
        self.is_in_capture_mode = True
        self.cap_step = 0
        self.cap_name = simpledialog.askstring("Name", "Guard Name:")
        if not self.cap_name:
            self.end_capture()
            return
        self.cap_poses = {}
        self.controls.grid_remove()
        self.cap_frame.place(relx=0.5, rely=0.9, anchor="s")

    def end_capture(self):
        self.is_in_capture_mode = False
        self.cap_frame.place_forget()
        self.controls.grid()

    def snap_photo(self):
        if self.last_process_frame is None: return
        frame = self.last_process_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.cap_step == 0: # Face
            locs = face_recognition.face_locations(rgb)
            if len(locs) == 1:
                t,r,b,l = locs[0]
                face = frame[t:b, l:r]
                save_guard_face(face, self.cap_name)
                self.cap_step = 1
                messagebox.showinfo("Next", "Perform: One Hand Raised (Left)")
            else:
                messagebox.showwarning("Error", "Need exactly 1 face")
        else: # Poses
            res = self.holistic.process(rgb)
            if res.pose_landmarks:
                acts = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing"]
                act = acts[self.cap_step - 1]
                lms = [{"x":l.x, "y":l.y, "z":l.z, "vis":l.visibility} for l in res.pose_landmarks.landmark]
                self.cap_poses[act] = lms
                
                self.cap_step += 1
                if self.cap_step > 4:
                    save_pose_landmarks_json(self.cap_name, self.cap_poses)
                    self.load_targets()
                    self.end_capture()
                    messagebox.showinfo("Done", "Guard Saved")
                else:
                    messagebox.showinfo("Next", f"Perform: {acts[self.cap_step-1]}")
            else:
                messagebox.showwarning("Error", "No pose detected")

    # --- Main Loop ---
    def update_loop(self):
        if not self.is_running: return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return
        
        self.frame_counter += 1
        self.last_process_frame = frame.copy()
        
        # Safe Config Access
        skip = CONFIG["performance"].get("enable_frame_skipping", True)
        interval = CONFIG["performance"].get("frame_skip_interval", 2)
        
        if self.is_in_capture_mode:
            cv2.putText(frame, f"CAPTURE: {self.cap_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        else:
            if not skip or (self.frame_counter % interval == 0):
                self.process_frame(frame)

        # Display
        try:
            # Resize for UI
            disp_w = self.video_label.winfo_width()
            disp_h = self.video_label.winfo_height()
            if disp_w > 10:
                h, w = frame.shape[:2]
                scale = min(disp_w/w, disp_h/h)
                resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                self.video_label.config(image=img)
                self.video_label.image = img
        except: pass
        
        # FPS & Logs
        if self.frame_counter % 30 == 0:
            fps = 30 / (time.time() - self.last_fps_time + 1e-5)
            self.last_fps_time = time.time()
            self.status_lbl.config(text=f"FPS: {int(fps)}")
            if self.is_logging: self.flush_logs(limit=50)

        self.root.after(CONFIG["performance"].get("gui_refresh_ms", 30), self.update_loop)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Fugitive Check
        if self.fugitive_mode and self.fugitive_enc is not None:
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            found = False
            for loc, enc in zip(locs, encs):
                if face_recognition.compare_faces([self.fugitive_enc], enc, 0.5)[0]:
                    found = True
                    t,r,b,l = loc
                    cv2.rectangle(frame, (l,t), (r,b), (0,0,255), 4)
                    cv2.putText(frame, "FUGITIVE", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
                    if not self.fugitive_alert_done:
                        self.fugitive_stop_event = threading.Event()
                        play_siren_sound(self.fugitive_stop_event, 30, "Fugitive.mp3")
                        path = save_capture_snapshot(frame, f"FUGITIVE_{self.fugitive_name}")
                        self.log_event(f"FUGITIVE_{self.fugitive_name}", "DETECTED", "ALERT", path, 1.0)
                        self.fugitive_alert_done = True
            if not found: self.fugitive_alert_done = False

        # 2. Guard Tracking
        if not self.targets_status: return
        
        self.re_detect_counter += 1
        if self.re_detect_counter > CONFIG["detection"].get("re_detect_interval", 60):
            self.re_detect_counter = 0

        # Tracker Update
        for name, s in self.targets_status.items():
            if s["tracker"]:
                ok, box = s["tracker"].update(frame)
                if ok:
                    x,y,w_box,h_box = [int(v) for v in box]
                    s["face_box"] = (x,y,x+w_box,y+h_box)
                    s["visible"] = True
                else:
                    s["visible"] = False
                    s["tracker"] = None

        # Re-detection
        untracked = [n for n,s in self.targets_status.items() if not s["visible"]]
        if untracked and self.re_detect_counter == 0:
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            for i, enc in enumerate(encs):
                for name in untracked:
                    # Safe tolerance access
                    tol = CONFIG["detection"].get("face_recognition_tolerance", 0.5)
                    if face_recognition.face_distance([self.targets_status[name]["encoding"]], enc)[0] < tol:
                        t,r,b,l = locs[i]
                        try:
                            # This requires opencv-contrib-python
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(frame, (l,t,r-l,b-t))
                            self.targets_status[name]["tracker"] = tracker
                            self.targets_status[name]["face_box"] = (l,t,r,b)
                            self.targets_status[name]["visible"] = True
                        except Exception:
                            logger.error("Tracker init failed. Check opencv-contrib-python.")

        # Pose Analysis
        req_act = self.req_action.get()
        now = time.time()
        
        for name, s in self.targets_status.items():
            if s["visible"]:
                bx1, by1, bx2, by2 = calculate_body_box(s["face_box"], h, w)
                if bx1 < bx2 and by1 < by2:
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size > 0:
                        # Optimization: Read-only
                        crop.flags.writeable = False
                        res = self.holistic.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        crop.flags.writeable = True
                        
                        if res.pose_landmarks:
                            s["miss_count"] = 0
                            draw_styled_landmarks(crop, res)
                            act = classify_action(res.pose_landmarks.landmark, by2-by1, bx2-bx1)
                            
                            s["buffer"].append(act)
                            # Safe Config Access
                            min_buf = CONFIG["performance"].get("min_buffer_for_classification", 5)
                            final_act = act
                            if len(s["buffer"]) >= min_buf:
                                final_act = Counter(s["buffer"]).most_common(1)[0][0]
                            
                            cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,255,0), 2)
                            cv2.putText(frame, f"{name}: {final_act}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                            
                            if final_act == req_act:
                                s["last_action_time"] = now
                                s["alert_trig"] = False
                                if s["stop_event"]: s["stop_event"].set()
                        else:
                            s["miss_count"] += 1
                            if s["miss_count"] > 15:
                                s["visible"] = False
                                s["tracker"] = None
            
            # Alert Logic
            if self.is_alert_mode:
                diff = now - s["last_action_time"]
                left = max(0, self.alert_interval - diff)
                col = (0,255,0) if left > 3 else (0,0,255)
                
                y_pos = 50 + (list(self.targets_status.keys()).index(name) * 30)
                cv2.putText(frame, f"{name}: {left:.1f}s", (w-200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                
                if diff > self.alert_interval and (now - s["alert_cooldown"] > 3.0):
                    s["alert_cooldown"] = now
                    if not s["stop_event"]: s["stop_event"] = threading.Event()
                    s["stop_event"].clear()
                    play_siren_sound(s["stop_event"])
                    
                    # Rate limited snapshot
                    path = "N/A"
                    if now - s["last_snap"] > 60:
                        path = save_capture_snapshot(frame, name)
                        s["last_snap"] = now
                    
                    if self.is_logging:
                        self.log_event(name, "MISSING_ACTION", "ALERT", path, 1.0)
                        s["alert_trig"] = True

    def log_event(self, name, act, status, path, conf):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.temp_log.append((ts, name, act, status, path, str(conf)))

    def flush_logs(self, limit=0):
        if self.temp_log and len(self.temp_log) >= limit:
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerows(self.temp_log)
            self.temp_log = []

if __name__ == "__main__":
    app = PoseApp()