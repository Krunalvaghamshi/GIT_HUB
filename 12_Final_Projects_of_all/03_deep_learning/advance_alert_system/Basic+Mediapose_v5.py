import os
# Suppress TensorFlow/MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import csv
import time
import tkinter as tk
from tkinter import font, simpledialog, messagebox, filedialog
from PIL import Image, ImageTk
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
import math

# --- 0. SOUND CONFIGURATION (USER PATHS) ---
SOUND_PATH_SIREN = r"D:\GIT_HUB\12_Final_Projects_of_all\03_deep_learning\advance_alert_system\emergency-siren-351963.mp3"
SOUND_PATH_FUGITIVE = r"D:\GIT_HUB\12_Final_Projects_of_all\03_deep_learning\advance_alert_system\Fugitive.mp3"

# --- 1. CRITICAL DEPENDENCY CHECK ---
try:
    _test_tracker = cv2.legacy.TrackerCSRT_create()
    TRACKER_AVAILABLE = True
except AttributeError:
    print("WARNING: 'cv2.legacy' is missing. Install: pip install opencv-contrib-python")
    TRACKER_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# --- 2. Configuration ---
def load_config():
    defaults = {
        "detection": {
            "min_detection_confidence": 0.5, 
            "min_tracking_confidence": 0.5, 
            "face_recognition_tolerance": 0.5, 
            "re_detect_interval": 40 
        },
        "alert": {
            "default_interval_seconds": 10, 
            "alert_cooldown_seconds": 3.0
        },
        "performance": {
            "gui_refresh_ms": 20, 
            "pose_buffer_size": 10,
            "frame_skip_interval": 2,
            "enable_frame_skipping": True,
            "min_buffer_for_classification": 6
        },
        "logging": {
            "log_directory": "logs", 
            "max_log_size_mb": 10, 
            "auto_flush_interval": 20
        },
        "storage": {
            "alert_snapshots_dir": "alert_snapshots", 
            "snapshot_retention_days": 30,
            "guard_profiles_dir": "guard_profiles", 
            "capture_snapshots_dir": "capture_snapshots",
            "pose_references_dir": "pose_references"
        }
    }
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            for section, keys in defaults.items():
                if section not in user_config: user_config[section] = keys
                else:
                    for k, v in keys.items():
                        if k not in user_config[section]: user_config[section][k] = v
            return user_config
        return defaults
    except: return defaults

CONFIG = load_config()

# --- 3. Setup Directories & Logging ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, CONFIG["logging"]["log_directory"])
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

logger = logging.getLogger("PoseGuard")
logger.setLevel(logging.WARNING)
handler = RotatingFileHandler(os.path.join(LOG_DIR, "session.log"), maxBytes=5*1024*1024, backupCount=2)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def get_storage_paths():
    def resolve(p): return os.path.join(SCRIPT_DIR, CONFIG["storage"].get(p, p))
    paths = {
        "guard_profiles": resolve("guard_profiles_dir"),
        "pose_references": resolve("pose_references_dir"),
        "capture_snapshots": resolve("capture_snapshots_dir"),
        "alert_snapshots": resolve("alert_snapshots_dir"),
        "logs": LOG_DIR
    }
    for p in paths.values(): 
        if not os.path.exists(p): os.makedirs(p)
    return paths

PATHS = get_storage_paths()
csv_file = os.path.join(LOG_DIR, "events.csv")
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Name", "Action", "Status", "Image_Path", "Confidence"])

# --- 4. Sound Logic ---
def play_sound_once(sound_path, duration=5):
    """Plays a sound file for a fixed duration or once, then stops."""
    def _worker():
        if not os.path.exists(sound_path):
            return
        if PYGAME_AVAILABLE:
            try:
                if not pygame.mixer.get_init(): pygame.mixer.init()
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play(0) # Play once
                start = time.time()
                while pygame.mixer.music.get_busy() and (time.time() - start) < duration:
                    time.sleep(0.1)
                pygame.mixer.music.stop()
            except: pass
        else:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(2000, 500)
    threading.Thread(target=_worker, daemon=True).start()

# --- 5. MediaPipe & Drawing ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 

# --- 6. Detection Logic ---
def classify_action(landmarks, h, w):
    try:
        lm = mp_holistic.PoseLandmark
        nose = landmarks[lm.NOSE.value]
        lw, rw = landmarks[lm.LEFT_WRIST.value], landmarks[lm.RIGHT_WRIST.value]
        ls, rs = landmarks[lm.LEFT_SHOULDER.value], landmarks[lm.RIGHT_SHOULDER.value]
        lk, rk = landmarks[lm.LEFT_KNEE.value], landmarks[lm.RIGHT_KNEE.value]
        lhip, rhip = landmarks[lm.LEFT_HIP.value], landmarks[lm.RIGHT_HIP.value]

        nose_y = nose.y * h
        hands_vis = (lw.visibility > 0.5 and rw.visibility > 0.5)
        
        # 1. Hands Up
        if hands_vis:
            if lw.y * h < (nose_y - 0.05 * h) and rw.y * h < (nose_y - 0.05 * h):
                return "Hands Up"

        # 2. Hands Crossed
        if hands_vis:
            chest_y = (ls.y * h + rs.y * h) / 2
            if abs(lw.y * h - chest_y) < 0.2 * h and abs(rw.y * h - chest_y) < 0.2 * h:
                center_x = (ls.x * w + rs.x * w) / 2
                if (lw.x * w > center_x and rw.x * w < center_x):
                    return "Hands Crossed"

        # 3. T-Pose
        if hands_vis:
            shoulder_y = (ls.y * h + rs.y * h) / 2
            if abs(lw.y * h - shoulder_y) < 0.15 * h and abs(rw.y * h - shoulder_y) < 0.15 * h:
                if lw.x < ls.x and rw.x > rs.x:
                    return "T-Pose"

        # 4. One Hand Raised
        if lw.visibility > 0.6 and lw.y * h < nose_y: return "One Hand Raised (Left)"
        if rw.visibility > 0.6 and rw.y * h < nose_y: return "One Hand Raised (Right)"

        # 5. Sit/Stand
        if lk.visibility > 0.6 and lhip.visibility > 0.6:
            vertical_dist = abs(lhip.y - lk.y)
            total_len = math.hypot(lhip.x - lk.x, lhip.y - lk.y)
            if total_len > 0:
                if (vertical_dist / total_len) < 0.45: return "Sit"

        return "Standing"
    except: return "Unknown"

def calculate_body_box(face_box, h, w):
    x1, y1, x2, y2 = face_box
    fw = x2 - x1
    cx = x1 + fw//2
    bx1 = max(0, int(cx - fw * 3.0))
    bx2 = min(w, int(cx + fw * 3.0))
    by1 = max(0, int(y1 - (y2-y1) * 0.5))
    return (bx1, by1, bx2, h)

# --- RESTORED: IoU Helper ---
def calculate_iou(boxA, boxB):
    # box = (x, y, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if float(boxAArea + boxBArea - interArea) == 0: return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def detect_cameras(max=3):
    arr = []
    for i in range(max):
        try:
            c = cv2.VideoCapture(i)
            if c.isOpened(): arr.append(i); c.release()
        except: pass
    return arr

# --- Main App ---
class PoseApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pose Guard v9 (Full Restoration)")
        self.root.geometry("1280x720")
        self.root.configure(bg="black")
        
        self.cap = None
        self.is_running = False
        self.is_logging = False
        self.is_alert_mode = False
        self.is_in_capture_mode = False
        self.fullscreen_state = False
        self.cap_name = ""
        
        self.targets_status = {}
        self.target_map = {}
        self.frame_counter = 0
        self.re_detect_counter = 0
        self.last_fps_time = time.time()
        self.temp_log = []
        self.last_process_frame = None
        self.alert_interval = CONFIG["alert"]["default_interval_seconds"]
        
        self.fugitive_mode = False
        self.fugitive_enc = None
        self.fugitive_name = "Unknown"
        self.fugitive_alert_done = False
        
        try:
            self.holistic = mp_holistic.Holistic(
                min_detection_confidence=CONFIG["detection"]["min_detection_confidence"],
                min_tracking_confidence=CONFIG["detection"]["min_tracking_confidence"]
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.root.destroy(); sys.exit()

        self.setup_gui()
        self.load_targets()
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        self.root.mainloop()

    def setup_gui(self):
        # V3 Style Layout
        self.root.grid_rowconfigure(0, weight=10) 
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # 1. Video Zone
        self.red_zone = tk.Frame(self.root, bg="red", bd=2)
        self.red_zone.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.video_lbl = tk.Label(self.red_zone, bg="black", text="Camera Off", fg="white")
        self.video_lbl.pack(fill="both", expand=True)
        
        self.fug_prev_frame = tk.Frame(self.video_lbl, bg="darkred", bd=1)
        self.fug_prev_frame.place(relx=0.98, rely=0.02, anchor="ne")
        self.fug_prev_lbl = tk.Label(self.fug_prev_frame, bg="black", width=12, height=5)
        self.fug_prev_lbl.pack()

        # Bottom Container
        self.bot_zone = tk.Frame(self.root, bg="black")
        self.bot_zone.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.bot_zone.grid_columnconfigure(0, weight=7)
        self.bot_zone.grid_columnconfigure(1, weight=3)

        # 2. Controls
        self.yel_zone = tk.Frame(self.bot_zone, bg="gold", bd=2)
        self.yel_zone.grid(row=0, column=0, sticky="nsew")
        
        self.ctrl_frame = tk.Frame(self.yel_zone, bg="gold")
        self.ctrl_frame.pack(fill="x", padx=5, pady=5)
        
        bf = font.Font(family='Arial', size=9, weight='bold')
        
        # Buttons
        self.btn_start = tk.Button(self.ctrl_frame, text="Start Camera", command=self.start_camera, bg="green", fg="white", font=bf)
        self.btn_start.grid(row=0, column=0, padx=2)
        self.btn_stop = tk.Button(self.ctrl_frame, text="Stop", command=self.stop_camera, bg="red", fg="white", font=bf, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=2)
        self.btn_add = tk.Button(self.ctrl_frame, text="Add Guard", command=self.add_guard_dialog, bg="#333", fg="white", font=bf, state="disabled")
        self.btn_add.grid(row=0, column=2, padx=2)
        self.btn_rem = tk.Button(self.ctrl_frame, text="Remove", command=self.rem_guard_diag, bg="#c0392b", fg="white", font=bf)
        self.btn_rem.grid(row=0, column=3, padx=2)
        self.btn_exit = tk.Button(self.ctrl_frame, text="Exit", command=self.graceful_exit, bg="black", fg="white", font=bf)
        self.btn_exit.grid(row=0, column=5, padx=2)
        
        # Maximize Screen
        self.btn_max = tk.Button(self.ctrl_frame, text="[ ] Maximize", command=self.toggle_screen, bg="#555", fg="white", font=bf)
        self.btn_max.grid(row=0, column=4, padx=2)

        # Row 2
        tk.Label(self.ctrl_frame, text="Action:", bg="gold").grid(row=1, column=0, sticky="e")
        self.act_var = tk.StringVar(value="Hands Up")
        tk.OptionMenu(self.ctrl_frame, self.act_var, "Hands Up", "Hands Crossed", "One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing", command=self.reset_timer).grid(row=1, column=1, columnspan=2, sticky="ew")
        self.btn_alert = tk.Button(self.ctrl_frame, text="Start Alert Mode", command=self.toggle_alert, bg="orange", fg="white", font=bf, state="disabled")
        self.btn_alert.grid(row=1, column=3, columnspan=2, sticky="ew")

        # Row 3
        self.btn_fug = tk.Button(self.ctrl_frame, text="Fugitive Mode", command=self.toggle_fug, bg="darkred", fg="white", font=bf, state="disabled")
        self.btn_fug.grid(row=2, column=0, columnspan=5, sticky="ew", pady=3)

        # Listbox
        self.list_frame = tk.Frame(self.yel_zone, bg="gold")
        self.list_frame.pack(fill="both", expand=True, padx=5)
        tk.Label(self.list_frame, text="Targets:", bg="gold").pack(anchor="w")
        self.lb = tk.Listbox(self.list_frame, selectmode=tk.MULTIPLE, height=4)
        self.lb.pack(side="left", fill="both", expand=True)
        self.lb.bind('<<ListboxSelect>>', self.on_select)
        tk.Button(self.list_frame, text="TRACK", command=self.apply_track, bg="black", fg="gold", font=bf).pack(side="right", fill="y")

        # 3. Green Zone
        self.grn_zone = tk.Frame(self.bot_zone, bg="#00FF00", bd=2)
        self.grn_zone.grid(row=0, column=1, sticky="nsew", padx=2)
        tk.Label(self.grn_zone, text="Preview", bg="#00AA00", fg="white").pack(fill="x")
        self.prev_disp = tk.Frame(self.grn_zone, bg="black")
        self.prev_disp.pack(fill="both", expand=True)

        # Capture Overlay
        self.btn_snap = tk.Button(self.ctrl_frame, text="SNAP", command=self.snap_photo, bg="blue", fg="white")
        self.btn_can = tk.Button(self.ctrl_frame, text="Cancel", command=self.exit_onboarding)
        self.stat_lbl = tk.Label(self.ctrl_frame, text="FPS: 0", bg="gold")
        self.stat_lbl.grid(row=3, column=0, columnspan=6)

    # --- Logic ---
    def toggle_screen(self):
        self.fullscreen_state = not self.fullscreen_state
        try:
            if platform.system() == "Windows":
                self.root.state('zoomed' if self.fullscreen_state else 'normal')
            else:
                self.root.attributes('-fullscreen', self.fullscreen_state)
        except: pass

    def graceful_exit(self):
        if self.is_running and not messagebox.askyesno("Exit", "Stop?"): return
        self.is_running = False
        if self.cap: self.cap.release()
        self.flush_log()
        self.root.destroy()
        sys.exit(0)

    def load_targets(self):
        self.target_map = {}
        self.lb.delete(0, tk.END)
        files = glob.glob(os.path.join(PATHS["guard_profiles"], "target_*_face.jpg"))
        files += glob.glob(os.path.join(SCRIPT_DIR, "target_*_face.jpg"))
        seen = set()
        for f in files:
            name = os.path.basename(f).replace("target_", "").replace("_face.jpg", "").replace("_", " ")
            if name not in seen:
                self.target_map[name] = f
                self.lb.insert(tk.END, name)
                seen.add(name)

    def on_select(self, event):
        for w in self.prev_disp.winfo_children(): w.destroy()
        sel = self.lb.curselection()
        if not sel: return
        cols = 2 if len(sel) > 1 else 1
        for i, idx in enumerate(sel):
            name = self.lb.get(idx)
            path = self.target_map.get(name)
            if path:
                try:
                    img = Image.open(path)
                    img.thumbnail((80, 80))
                    ph = ImageTk.PhotoImage(img)
                    f = tk.Frame(self.prev_disp, bg="black")
                    f.grid(row=i//cols, column=i%cols, padx=2, pady=2)
                    l = tk.Label(f, image=ph, bg="black")
                    l.image = ph
                    l.pack()
                    tk.Label(f, text=name, bg="black", fg="white", font=("Arial",7)).pack()
                except: pass

    def apply_track(self):
        self.targets_status = {}
        sel = self.lb.curselection()
        for idx in sel:
            name = self.lb.get(idx)
            path = self.target_map.get(name)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    self.targets_status[name] = {
                        "encoding": encs[0], "tracker": None, "face_box": None, "visible": False,
                        "last_action_time": time.time(), "alert_cooldown": 0,
                        "buffer": deque(maxlen=CONFIG["performance"]["pose_buffer_size"]),
                        "miss_count": 0, "last_snap": 0, "miss_pose_count": 0
                    }
            except Exception as e: logger.error(str(e))
        messagebox.showinfo("Info", f"Tracking {len(self.targets_status)} targets")

    def toggle_alert(self):
        self.is_alert_mode = not self.is_alert_mode
        if self.is_alert_mode:
            self.is_logging = True
            self.btn_alert.config(text="STOP ALERT", bg="red")
            self.reset_timer()
            # Auto-Maximize
            if not self.fullscreen_state: self.toggle_screen()
        else:
            self.flush_log()
            self.is_logging = False
            self.btn_alert.config(text="Start Alert Mode", bg="orange")
            if self.fullscreen_state: self.toggle_screen()

    def reset_timer(self, _=None):
        t = time.time()
        for s in self.targets_status.values(): s["last_action_time"] = t

    def start_camera(self):
        if self.is_running: return
        cams = detect_cameras()
        if not cams: messagebox.showerror("Error", "No Camera"); return
        self.cap = cv2.VideoCapture(cams[0])
        if self.cap.isOpened():
            self.is_running = True
            self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
            self.btn_add.config(state="normal"); self.btn_alert.config(state="normal")
            self.btn_fug.config(state="normal")
            self.update_loop()

    def stop_camera(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.video_lbl.config(image='')
        self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.btn_add.config(state="disabled"); self.btn_alert.config(state="disabled")

    def toggle_fug(self):
        if not self.fugitive_mode:
            path = filedialog.askopenfilename()
            if not path: return
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if not encs: raise Exception("No face")
                self.fugitive_enc = encs[0]
                self.fugitive_name = simpledialog.askstring("Name", "Name:") or "Unknown"
                self.fugitive_mode = True
                self.btn_fug.config(bg="green", text="Disable Fugitive")
                pil = Image.fromarray(img); pil.thumbnail((100,100))
                ph = ImageTk.PhotoImage(pil)
                self.fug_prev_lbl.config(image=ph, text=self.fugitive_name); self.fug_prev_lbl.image=ph
            except Exception as e: messagebox.showerror("Error", str(e))
        else:
            self.fugitive_mode = False
            self.btn_fug.config(bg="darkred", text="Fugitive Mode")
            self.fug_prev_lbl.config(image='', text='')

    def add_guard_dialog(self):
        if not self.is_running: return
        if messagebox.askyesno("Add", "Use Camera?"):
            self.cap_name = simpledialog.askstring("Name", "Name:")
            if self.cap_name:
                self.is_in_capture_mode = True
                self.onboarding_step = 0
                self.btn_start.grid_remove(); self.btn_stop.grid_remove()
                self.btn_snap.grid(row=0, column=1); self.btn_can.grid(row=0, column=2)
        else:
            path = filedialog.askopenfilename()
            if path:
                name = simpledialog.askstring("Name", "Name:")
                if name:
                    dest = os.path.join(PATHS["guard_profiles"], f"target_{name.replace(' ','_')}_face.jpg")
                    try:
                        with open(path, 'rb') as s, open(dest, 'wb') as d: d.write(s.read())
                        self.load_targets()
                    except: pass

    def exit_onboarding(self):
        self.is_in_capture_mode = False
        self.btn_snap.grid_remove(); self.btn_can.grid_remove()
        self.btn_start.grid(); self.btn_stop.grid()

    def snap_photo(self):
        if self.last_process_frame is None: return
        frame = self.last_process_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.onboarding_step == 0:
            locs = face_recognition.face_locations(rgb)
            if len(locs) == 1:
                t,r,b,l = locs[0]
                face = frame[t:b, l:r]
                save_guard_face(face, self.cap_name)
                self.onboarding_step = 1
                messagebox.showinfo("Next", "Perform: One Hand Raised (Left)")
            else: messagebox.showwarning("Error", "Need 1 Face")
        else:
            res = self.holistic.process(rgb)
            if res.pose_landmarks:
                acts = ["One Hand Raised (Left)", "One Hand Raised (Right)", "Sit", "Standing"]
                if self.onboarding_step <= 4:
                    save_pose_landmarks_json(self.cap_name, {}) # Placeholder
                    self.onboarding_step += 1
                    if self.onboarding_step > 4:
                        self.load_targets()
                        self.exit_onboarding()
                        messagebox.showinfo("Done", "Saved")
                    else: messagebox.showinfo("Next", f"Perform: {acts[self.onboarding_step-1]}")
            else: messagebox.showwarning("Error", "No Pose")

    def rem_guard_diag(self):
        sel = self.lb.curselection()
        if not sel: return
        name = self.lb.get(sel[0])
        if messagebox.askyesno("Delete", f"Remove {name}?"):
            p = self.target_map.get(name)
            if p and os.path.exists(p): os.remove(p)
            self.load_targets()

    def update_loop(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if not ret: self.stop_camera(); return
        self.frame_counter += 1
        self.last_process_frame = frame.copy()
        
        if self.is_in_capture_mode:
            cv2.putText(frame, f"CAPTURE: {self.cap_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        else:
            skip = CONFIG["performance"]["enable_frame_skipping"]
            if not skip or (self.frame_counter % 2 == 0):
                self.process_frame(frame)

        try:
            dw = self.video_lbl.winfo_width(); dh = self.video_lbl.winfo_height()
            if dw > 10:
                h,w = frame.shape[:2]
                scale = min(dw/w, dh/h)
                res = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)))
                self.video_lbl.config(image=img); self.video_lbl.image = img
        except: pass

        if self.frame_counter % 30 == 0:
            fps = 30 / (time.time() - self.last_fps_time + 1e-5)
            self.last_fps_time = time.time()
            self.stat_lbl.config(text=f"FPS: {int(fps)}")
            if self.is_logging: self.flush_log(50)

        self.root.after(CONFIG["performance"]["gui_refresh_ms"], self.update_loop)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Fugitive (Sound Logic Fixed)
        if self.fugitive_mode and self.fugitive_enc is not None:
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            found = False
            for loc, enc in zip(locs, encs):
                if face_recognition.compare_faces([self.fugitive_enc], enc, 0.5)[0]:
                    found = True; t,r,b,l = loc
                    cv2.rectangle(frame, (l,t), (r,b), (0,0,255), 4)
                    cv2.putText(frame, "FUGITIVE", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
                    if not self.fugitive_alert_done:
                        play_sound_once(SOUND_PATH_FUGITIVE, duration=5)
                        path = save_capture_snapshot(frame, f"FUGITIVE_{self.fugitive_name}")
                        self.log_ev(f"FUGITIVE_{self.fugitive_name}", "DETECTED", "ALERT", path, 1.0)
                        self.fugitive_alert_done = True
            if not found: self.fugitive_alert_done = False

        if not self.targets_status: return
        
        self.re_detect_counter += 1
        if self.re_detect_counter > CONFIG["detection"]["re_detect_interval"]:
            self.re_detect_counter = 0

        # 2. Tracker Update
        for name, s in self.targets_status.items():
            if s["tracker"]:
                ok, box = s["tracker"].update(frame)
                if ok:
                    x,y,wb,hb = [int(v) for v in box]
                    s["face_box"] = (x,y,x+wb,y+hb); s["visible"] = True
                else:
                    s["visible"] = False; s["tracker"] = None
        
        # 3. Re-detection
        untracked = [n for n,s in self.targets_status.items() if not s["visible"]]
        if untracked and self.re_detect_counter == 0:
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            for i, enc in enumerate(encs):
                for name in untracked:
                    if face_recognition.face_distance([self.targets_status[name]["encoding"]], enc)[0] < 0.5:
                        t,r,b,l = locs[i]
                        if TRACKER_AVAILABLE:
                            try:
                                tr = cv2.legacy.TrackerCSRT_create()
                                tr.init(frame, (l,t,r-l,b-t))
                                self.targets_status[name]["tracker"] = tr
                            except: pass
                        self.targets_status[name]["face_box"] = (l,t,r,b)
                        self.targets_status[name]["visible"] = True
                        self.targets_status[name]["miss_pose_count"] = 0

        # 4. RESTORED: Overlap Check
        active = [n for n,s in self.targets_status.items() if s["visible"]]
        for i in range(len(active)):
            for j in range(i+1, len(active)):
                nA, nB = active[i], active[j]
                bA = self.targets_status[nA]["face_box"]
                bB = self.targets_status[nB]["face_box"]
                if calculate_iou(bA, bB) > 0.5:
                    # Force re-detect if overlapped
                    self.targets_status[nA]["tracker"] = None
                    self.targets_status[nB]["tracker"] = None
                    self.targets_status[nA]["visible"] = False
                    self.targets_status[nB]["visible"] = False

        # 5. Pose & Logic
        req = self.act_var.get()
        now = time.time()
        for name, s in self.targets_status.items():
            if s["visible"]:
                bx1, by1, bx2, by2 = calculate_body_box(s["face_box"], h, w)
                if bx1 < bx2 and by1 < by2:
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size > 0:
                        crop.flags.writeable = False
                        res = self.holistic.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        crop.flags.writeable = True
                        
                        if res.pose_landmarks:
                            s["miss_pose_count"] = 0
                            draw_styled_landmarks(crop, res)
                            act = classify_action(res.pose_landmarks.landmark, by2-by1, bx2-bx1)
                            s["buffer"].append(act)
                            
                            fin = act
                            if len(s["buffer"]) >= CONFIG["performance"]["min_buffer_for_classification"]:
                                fin = Counter(s["buffer"]).most_common(1)[0][0]
                            
                            cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,255,0), 2)
                            cv2.putText(frame, f"{name}: {fin}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                            
                            if fin == req:
                                s["last_action_time"] = now
                        else:
                            # RESTORED: Ghost Box Logic
                            s["miss_pose_count"] += 1
                            if s["miss_pose_count"] > 5:
                                s["visible"] = False; s["tracker"] = None

            # Alerts
            if self.is_alert_mode:
                diff = now - s["last_action_time"]
                rem = max(0, self.alert_interval - diff)
                col = (0,255,0) if rem > 3 else (0,0,255)
                yp = 50 + list(self.targets_status.keys()).index(name)*30
                cv2.putText(frame, f"{name}: {rem:.1f}s", (w-250, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                
                if diff > self.alert_interval and (now - s["alert_cooldown"] > 3.0):
                    s["alert_cooldown"] = now
                    play_sound_once(SOUND_PATH_SIREN, 3)
                    path = save_capture_snapshot(frame, name)
                    if self.is_logging:
                        self.log_ev(name, "MISSING", "ALERT", path, 1.0)

    def log_ev(self, n, a, s, p, c):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.temp_log.append((ts, n, a, s, p, str(c)))

    def flush_log(self, limit=0):
        if self.temp_log and len(self.temp_log) >= limit:
            with open(csv_file, "a", newline="") as f: csv.writer(f).writerows(self.temp_log)
            self.temp_log = []

if __name__ == "__main__":
    app = PoseApp()