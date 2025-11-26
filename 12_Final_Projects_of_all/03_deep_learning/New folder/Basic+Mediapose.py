import cv2
import mediapipe as mp
import csv
import time
import tkinter as tk
from tkinter import font
from tkinter import simpledialog
from tkinter import messagebox 
from PIL import Image, ImageTk
import os 
import glob
import face_recognition
import numpy as np 
import threading 
import platform
import logging
from datetime import datetime
from collections import deque, Counter

# --- 1. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PoseGuard")

if not os.path.exists("alert_snapshots"):
    os.makedirs("alert_snapshots")

csv_file = "activity_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name", "Action", "Status", "Image_Path"])

# --- MediaPipe Solutions Setup ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Sound Logic ---
def play_siren_sound():
    def _sound_worker():
        sys_plat = platform.system()
        try:
            if sys_plat == "Windows":
                import winsound
                for _ in range(3):
                    winsound.Beep(2000, 300) 
                    winsound.Beep(1000, 300) 
            else:
                for _ in range(3):
                    print('\a')
                    time.sleep(0.3)
                    print('\a')
                    time.sleep(0.3)
        except Exception as e:
            logger.error(f"Sound Error: {e}")

    t = threading.Thread(target=_sound_worker, daemon=True)
    t.start()

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
        NOSE = mp_holistic.PoseLandmark.NOSE.value
        L_WRIST = mp_holistic.PoseLandmark.LEFT_WRIST.value
        R_WRIST = mp_holistic.PoseLandmark.RIGHT_WRIST.value
        L_HIP = mp_holistic.PoseLandmark.LEFT_HIP.value
        L_KNEE = mp_holistic.PoseLandmark.LEFT_KNEE.value
        L_ANKLE = mp_holistic.PoseLandmark.LEFT_ANKLE.value
        
        nose = landmarks[NOSE]
        l_wrist = landmarks[L_WRIST]
        r_wrist = landmarks[R_WRIST]
        l_hip = landmarks[L_HIP]
        l_knee = landmarks[L_KNEE]
        l_ankle = landmarks[L_ANKLE]

        nose_y = nose.y * h
        lw_y = l_wrist.y * h
        rw_y = r_wrist.y * h
        
        # 1. Wave Detection
        if l_wrist.visibility > 0.5 and lw_y < nose_y:
            return "Wave Left"
        if r_wrist.visibility > 0.5 and rw_y < nose_y:
            return "Wave Right"
            
        # 2. Sit/Stand Detection
        if l_hip.visibility > 0.5 and l_knee.visibility > 0.5:
            if abs(l_knee.y - l_hip.y) < 0.15: # Thigh is horizontal
                return "Sit"
            else:
                return "Standing"

        return "Standing" 

    except Exception as e:
        return "Unknown"

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

# --- Tkinter Application Class ---
class PoseApp:
    def __init__(self, window_title="Pose Guard (Multi-Target)"):
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1400x950")
        self.root.configure(bg="black") 
        
        self.cap = None
        self.unprocessed_frame = None 
        self.is_running = False
        self.is_logging = False
        
        self.is_alert_mode = False
        self.alert_interval = 10  
        self.is_in_capture_mode = False
        self.frame_w = 640 
        self.frame_h = 480 

        self.target_map = {}
        self.targets_status = {} 
        self.re_detect_counter = 0    
        self.RE_DETECT_INTERVAL = 30  
        self.RESIZE_SCALE = 1.0 
        self.temp_log = [] 
        
        try:
            self.holistic_full = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.holistic_crop = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            logger.info("MediaPipe Holistic Loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Holistic Model: {e}")
            self.root.destroy()
            return

        self.frame_timestamp_ms = 0 

        # --- Layout ---
        self.root.grid_rowconfigure(0, weight=3) 
        self.root.grid_rowconfigure(1, weight=1) 
        self.root.grid_columnconfigure(0, weight=1)

        # 1. Red Zone
        self.red_zone = tk.Frame(self.root, bg="red", bd=4)
        self.red_zone.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.video_container = tk.Frame(self.red_zone, bg="black")
        self.video_container.pack(fill="both", expand=True, padx=2, pady=2)
        self.video_label = tk.Label(self.video_container, bg="black", text="Camera Feed Off", fg="white")
        self.video_label.pack(fill="both", expand=True)

        # Bottom Container
        self.bottom_container = tk.Frame(self.root, bg="black")
        self.bottom_container.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.bottom_container.grid_columnconfigure(0, weight=7) 
        self.bottom_container.grid_columnconfigure(1, weight=3) 
        self.bottom_container.grid_rowconfigure(0, weight=1)

        # 2. Yellow Zone
        self.yellow_zone = tk.Frame(self.bottom_container, bg="gold", bd=4)
        self.yellow_zone.grid(row=0, column=0, sticky="nsew", padx=2)
        self.controls_frame = tk.Frame(self.yellow_zone, bg="gold")
        self.controls_frame.pack(side="top", fill="x", padx=5, pady=5)
        self.listbox_frame = tk.Frame(self.yellow_zone, bg="gold")
        self.listbox_frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # 3. Green Zone
        self.green_zone = tk.Frame(self.bottom_container, bg="#00FF00", bd=4)
        self.green_zone.grid(row=0, column=1, sticky="nsew", padx=2)
        self.preview_container = tk.Frame(self.green_zone, bg="black")
        self.preview_container.pack(fill="both", expand=True, padx=2, pady=2)
        self.preview_display = tk.Frame(self.preview_container, bg="black")
        self.preview_display.pack(fill="both", expand=True)

        # Widgets
        btn_font = font.Font(family='Helvetica', size=10, weight='bold')

        self.btn_start = tk.Button(self.controls_frame, text="Start Camera", command=self.start_camera, font=btn_font, bg="#27ae60", fg="white", width=12)
        self.btn_start.grid(row=0, column=0, padx=3, pady=3)
        self.btn_stop = tk.Button(self.controls_frame, text="Stop Camera", command=self.stop_camera, font=btn_font, bg="#c0392b", fg="white", width=12, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=3, pady=3)
        self.btn_toggle_log = tk.Button(self.controls_frame, text="Start Logging", command=self.toggle_logging, font=btn_font, bg="#2980b9", fg="white", width=12, state="disabled")
        self.btn_toggle_log.grid(row=0, column=2, padx=3, pady=3)
        self.btn_capture_target = tk.Button(self.controls_frame, text="Capture New", command=self.enter_capture_mode, font=btn_font, bg="#8e44ad", fg="white", width=12, state="disabled")
        self.btn_capture_target.grid(row=0, column=3, padx=3, pady=3)

        tk.Label(self.controls_frame, text="Action:", bg="gold", font=btn_font).grid(row=1, column=0, sticky="e")
        self.required_action_var = tk.StringVar(self.root)
        self.required_action_var.set("Wave Right")
        self.action_dropdown = tk.OptionMenu(self.controls_frame, self.required_action_var, "Wave Right", "Wave Left", "Jump", "Sit", command=self.on_action_change)
        self.action_dropdown.grid(row=1, column=1, sticky="ew")
        self.btn_set_interval = tk.Button(self.controls_frame, text=f"Set Interval ({self.alert_interval}s)", command=self.set_alert_interval, font=btn_font, bg="#7f8c8d", fg="white")
        self.btn_set_interval.grid(row=1, column=2, padx=3, pady=3)
        self.btn_toggle_alert = tk.Button(self.controls_frame, text="Start Alert Mode", command=self.toggle_alert_mode, font=btn_font, bg="#e67e22", fg="white", width=12, state="disabled")
        self.btn_toggle_alert.grid(row=1, column=3, padx=3, pady=3)

        tk.Label(self.listbox_frame, text="Select Targets to Track (Multi-Select):", bg="gold", font=btn_font).pack(anchor="w")
        self.target_listbox = tk.Listbox(self.listbox_frame, selectmode=tk.MULTIPLE, height=8, font=('Helvetica', 10))
        self.target_listbox.pack(side="left", fill="both", expand=True)
        self.target_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        scrollbar = tk.Scrollbar(self.listbox_frame)
        scrollbar.pack(side="right", fill="y")
        self.target_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.target_listbox.yview)
        self.btn_apply_targets = tk.Button(self.listbox_frame, text="TRACK SELECTED", command=self.apply_target_selection, font=btn_font, bg="black", fg="gold")
        self.btn_apply_targets.pack(side="bottom", fill="x", pady=2)
        self.btn_refresh = tk.Button(self.listbox_frame, text="Refresh List", command=self.load_targets, font=btn_font, bg="#e67e22", fg="white")
        self.btn_refresh.pack(side="bottom", fill="x", pady=2)

        self.btn_snap = tk.Button(self.controls_frame, text="Snap Photo", command=self.snap_photo, font=btn_font, bg="#d35400", fg="white")
        self.btn_cancel_capture = tk.Button(self.controls_frame, text="Cancel", command=self.exit_capture_mode, font=btn_font, bg="#7f8c8d", fg="white")

        self.load_targets()
        self.root.mainloop()

    def load_targets(self):
        logger.info("Loading targets...")
        self.target_map = {}
        target_files = glob.glob("target_*.jpg")
        display_names = []
        for f in target_files:
            try:
                base_name = f.replace(".jpg", "")
                parts = base_name.split('_')
                if len(parts) >= 4:
                    display_name = " ".join(parts[1:-2])
                    self.target_map[display_name] = f
                    display_names.append(display_name)
            except Exception as e:
                logger.error(f"Error parsing {f}: {e}")

        self.target_listbox.delete(0, tk.END)
        if not display_names:
             self.target_listbox.insert(tk.END, "No targets found")
             self.target_listbox.config(state=tk.DISABLED)
        else:
             self.target_listbox.config(state=tk.NORMAL)
             for name in sorted(list(set(display_names))):
                 self.target_listbox.insert(tk.END, name)

    def on_listbox_select(self, event):
        for widget in self.preview_display.winfo_children():
            widget.destroy()
        selections = self.target_listbox.curselection()
        if not selections:
            tk.Label(self.preview_display, text="No Selection", bg="black", fg="white").pack(expand=True)
            return
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
                            "last_wave_time": time.time(),
                            "alert_cooldown": 0,
                            "alert_triggered_state": False,
                            "last_logged_action": None,
                            "pose_buffer": deque(maxlen=12),
                            "missing_pose_counter": 0 # NEW: Counter for missing body
                        }
                        count += 1
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
        if count > 0:
            logger.info(f"Tracking initialized for {count} targets.")
            messagebox.showinfo("Tracking Updated", f"Now scanning for {count} selected targets.")
            if not self.is_alert_mode:
                 self.is_logging = False
                 self.btn_toggle_log.config(text="Start Logging", bg="#2980b9")

    def toggle_alert_mode(self):
        self.is_alert_mode = not self.is_alert_mode
        if self.is_alert_mode:
            self.btn_toggle_alert.config(text="Stop Alert Mode", bg="#c0392b")
            if not self.is_logging:
                self.toggle_logging()
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_wave_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False
        else:
            self.btn_toggle_alert.config(text="Start Alert Mode", bg="#e67e22")

    def set_alert_interval(self):
        val = simpledialog.askinteger("Set Interval", "Enter seconds:", minvalue=1, maxvalue=3600, initialvalue=self.alert_interval)
        if val:
            self.alert_interval = val
            self.btn_set_interval.config(text=f"Set Interval ({self.alert_interval}s)")
            
    def on_action_change(self, value):
        if self.is_alert_mode:
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_wave_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False

    def start_camera(self):
        if not self.is_running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened(): return
                self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.is_running = True
                self.btn_start.config(state="disabled")
                self.btn_stop.config(state="normal")
                self.btn_toggle_log.config(state="normal")
                self.btn_capture_target.config(state="normal")
                self.btn_toggle_alert.config(state="normal")
                self.update_video_feed()
            except Exception: pass

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap: self.cap.release()
            if self.is_logging: self.save_log_to_file()
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.video_label.config(image='')

    def toggle_logging(self):
        self.is_logging = not self.is_logging
        if self.is_logging:
            self.temp_log.clear()
            self.btn_toggle_log.config(text="Stop Logging", bg="#c0392b")
        else:
            self.btn_toggle_log.config(text="Start Logging", bg="#2980b9")
            self.save_log_to_file()

    def save_log_to_file(self):
        if self.temp_log:
            try:
                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(self.temp_log)
                self.temp_log.clear()
                logger.info("Logs saved.")
            except: pass
            
    def capture_alert_snapshot(self, frame, target_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = target_name.replace(" ", "_")
        filename = f"alert_snapshots/alert_{safe_name}_{timestamp}.jpg"
        try:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr_frame)
            return filename
        except: return "Error"

    def enter_capture_mode(self):
        if not self.is_running: return
        self.is_in_capture_mode = True
        self.btn_start.grid_remove()
        self.btn_stop.grid_remove()
        self.btn_toggle_log.grid_remove()
        self.btn_capture_target.grid_remove()
        self.btn_snap.grid(row=0, column=0)
        self.btn_cancel_capture.grid(row=0, column=1)

    def exit_capture_mode(self):
        self.is_in_capture_mode = False
        self.btn_snap.grid_remove()
        self.btn_cancel_capture.grid_remove()
        self.btn_start.grid()
        self.btn_stop.grid()
        self.btn_toggle_log.grid()
        self.btn_capture_target.grid()

    def snap_photo(self):
        if self.unprocessed_frame is None: return
        rgb_frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 1:
             name = simpledialog.askstring("Name", "Enter Name:")
             if name:
                 safe_name = name.strip().replace(" ", "_")
                 cv2.imwrite(f"target_{safe_name}_face.jpg", self.unprocessed_frame)
                 self.load_targets()
                 self.exit_capture_mode()
        else:
            messagebox.showwarning("Error", "Ensure exactly one face is visible.")

    def update_video_feed(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if not ret: 
            self.stop_camera()
            return
        self.unprocessed_frame = frame.copy()
        if self.is_in_capture_mode:
            self.process_capture_frame(frame)
        else:
            self.process_tracking_frame_optimized(frame)
        
        if self.video_label.winfo_exists():
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
        self.root.after(10, self.update_video_feed)

    def process_capture_frame(self, frame):
        h, w = frame.shape[:2]
        cv2.ellipse(frame, (w//2, h//2), (100, 130), 0, 0, 360, (0, 255, 255), 2)
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
                        if dist < 0.55:
                            possible_matches.append((dist, i, name))
                
                possible_matches.sort(key=lambda x: x[0])
                assigned_faces = set()
                assigned_targets = set()
                
                for dist, face_idx, name in possible_matches:
                    if face_idx in assigned_faces or name in assigned_targets: continue
                    
                    assigned_faces.add(face_idx)
                    assigned_targets.add(name)
                    (top, right, bottom, left) = face_locations[face_idx]
                    
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, (left, top, right-left, bottom-top))
                    self.targets_status[name]["tracker"] = tracker
                    self.targets_status[name]["face_box"] = (left, top, right, bottom)
                    self.targets_status[name]["visible"] = True
                    self.targets_status[name]["missing_pose_counter"] = 0

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
                
                # --- CALCULATE BODY BOX (Torso-Centric) ---
                face_w = fx2 - fx1
                face_cx = fx1 + (face_w // 2)
                bx1 = max(0, int(face_cx - (face_w * 3)))
                bx2 = min(frame_w, int(face_cx + (face_w * 3)))
                by1 = max(0, int(fy1 - (face_w * 0.5)))
                by2 = frame_h 

                # Ghost Box Check: Only draw if tracker is confident AND pose is found
                pose_found_in_box = False
                
                if bx1 < bx2 and by1 < by2:
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size != 0:
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        rgb_crop.flags.writeable = False
                        results_crop = self.holistic_crop.process(rgb_crop)
                        rgb_crop.flags.writeable = True
                        
                        current_action = "Unknown"
                        if results_crop.pose_landmarks:
                            pose_found_in_box = True
                            status["missing_pose_counter"] = 0 # Reset
                            
                            draw_styled_landmarks(crop, results_crop)
                            raw_action = classify_action(results_crop.pose_landmarks.landmark, (by2-by1), (bx2-bx1))
                            
                            status["pose_buffer"].append(raw_action)
                            if len(status["pose_buffer"]) >= 8:
                                most_common = Counter(status["pose_buffer"]).most_common(1)[0][0]
                                current_action = most_common
                            else:
                                current_action = raw_action

                            if current_action == required_act:
                                if self.is_alert_mode:
                                    status["last_wave_time"] = current_time
                                    status["alert_triggered_state"] = False
                                if self.is_logging and status["last_logged_action"] != required_act:
                                    self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, current_action, "SAFE (Reset)", "N/A"))
                                    status["last_logged_action"] = required_act
                            elif status["last_logged_action"] == required_act:
                                status["last_logged_action"] = None
                            
                            # Draw Box ONLY if pose found
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{name}: {current_action}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Ghost Box Removal Logic
                if not pose_found_in_box:
                    status["missing_pose_counter"] += 1
                    # If tracker says visible, but no pose for 5 frames -> Kill Tracker
                    if status["missing_pose_counter"] > 5:
                        status["tracker"] = None
                        status["visible"] = False

            # Alert Logic
            if self.is_alert_mode:
                time_diff = current_time - status["last_wave_time"]
                time_left = max(0, self.alert_interval - time_diff)
                y_offset = 50 + (list(self.targets_status.keys()).index(name) * 30)
                color = (0, 255, 0) if time_left > 3 else (0, 0, 255)
                
                # Only show status on screen if target is genuinely lost or safe
                status_txt = "OK" if status["visible"] else "MISSING"
                cv2.putText(frame, f"{name} ({status_txt}): {time_left:.1f}s", (frame_w - 300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if time_diff > self.alert_interval:
                    if (current_time - status["alert_cooldown"]) > 2.5:
                        play_siren_sound()
                        status["alert_cooldown"] = current_time
                        
                        img_path = "N/A"
                        if status["visible"]:
                            # Snapshot logic (Body box re-calc)
                            fx1, fy1, fx2, fy2 = status["face_box"]
                            face_w = fx2 - fx1
                            bx1 = max(0, int(fx1 + (face_w//2) - (face_w * 3)))
                            bx2 = min(frame_w, int(fx1 + (face_w//2) + (face_w * 3)))
                            by1 = max(0, int(fy1 - (face_w * 0.5)))
                            by2 = frame_h
                            if bx1 < bx2:
                                img_path = self.capture_alert_snapshot(frame[by1:by2, bx1:bx2], name)
                        else:
                            img_path = self.capture_alert_snapshot(frame, name)

                        if self.is_logging:
                            log_s = "ALERT CONTINUED" if status["alert_triggered_state"] else "ALERT TRIGGERED"
                            log_a = current_action if status["visible"] else "MISSING"
                            self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, img_path))
                            status["alert_triggered_state"] = True

        return frame 

if __name__ == "__main__":
    app = PoseApp()