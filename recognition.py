import tkinter as tk
import cv2
import face_recognition
import numpy as np
import os
import mss
import time
import threading
from PIL import Image, ImageTk
from queue import Queue

# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6

MONITOR_REGION = {"top": 100, "left": 300, "width": 1000, "height": 600}

HUD_CONFIG = {
    "width": 500,
    "height": 300,
    "top": 50,
    "left": 50
}

# ============================================================
# 2. PARAMÈTRES DE DÉTECTION (LÉGERS POUR LE CPU)
# ============================================================
DETECTION_MODEL = "hog"       # Modèle rapide (CPU)
UPSAMPLE = 1                  # 1 = rapide, 2 = plus précis mais lourd
ANALYSIS_SCALE = 0.75          # 0.5 = rapide, 1.0 = précis mais très lourd
DETECTION_INTERVAL = 4        # Détecte 1 frame sur 4
MAX_FPS = 30                  # Affichage

# ============================================================
# 3. CHARGEMENT DES VISAGES
# ============================================================
def load_known_faces():
    known_encodings = []
    known_names = []
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[ERREUR] Le dossier '{KNOWN_FACES_DIR}' n'existe pas.")
        return known_encodings, known_names

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                for encoding in encodings:
                    known_encodings.append(encoding)
                    base_name = os.path.splitext(filename)[0].split('_')[0]
                    known_names.append(base_name)
                print(f"[INFO] Visage chargé : {filename}")
            else:
                print(f"[ATTENTION] Aucun visage dans {filename}")
    return known_encodings, known_names

# ============================================================
# 4. CLASSE HUD MULTI-THREADS
# ============================================================
class FaceOverlay:
    def __init__(self, root, known_encodings, known_names):
        self.root = root
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.running = True
        
        self._drag_data = {"x": 0, "y": 0}
        self._resizing = False
        
        self.frame_queue = Queue(maxsize=2)
        self.latest_detections = []
        
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.95)
        self.root.configure(bg='black')
        
        self.hud_width = HUD_CONFIG['width']
        self.hud_height = HUD_CONFIG['height']
        
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.update_geometry(HUD_CONFIG['left'], HUD_CONFIG['top'])
        
        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.do_move)
        self.canvas.bind("<MouseWheel>", self.do_zoom)
        self.canvas.bind("<Button-4>", self.do_zoom)
        self.canvas.bind("<Button-5>", self.do_zoom)
        
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.capture_thread.start()
        self.detection_thread.start()
        
        self.display_loop()

    def update_geometry(self, x, y):
        self.root.geometry(f"{self.hud_width}x{self.hud_height}+{x}+{y}")
        self.canvas.config(width=self.hud_width, height=self.hud_height)

    def start_move(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_move(self, event):
        deltax = event.x - self._drag_data["x"]
        deltay = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def do_zoom(self, event):
        if self._resizing:
            return
        self._resizing = True
        scale = 1.1 if (event.num == 4 or event.delta > 0) else 0.9
        self.hud_width = max(100, int(self.hud_width * scale))
        self.hud_height = max(60, int(self.hud_height * scale))
        self.update_geometry(self.root.winfo_x(), self.root.winfo_y())
        self.canvas.delete("all")
        self._resizing = False

    def capture_loop(self):
        with mss.MSS() as sct:
            while self.running:
                try:
                    screenshot = sct.grab(MONITOR_REGION)
                    frame = np.array(screenshot)[:, :, :3].copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put(frame)
                except Exception as e:
                    print(f"[ERREUR capture] {e}")
                time.sleep(1 / MAX_FPS)

    def detection_loop(self):
        frame_counter = 0
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except:
                continue
            
            frame_counter += 1
            if frame_counter % DETECTION_INTERVAL != 0:
                continue
            
            if ANALYSIS_SCALE != 1.0:
                analyzed = cv2.resize(frame, (0, 0), fx=ANALYSIS_SCALE, fy=ANALYSIS_SCALE)
            else:
                analyzed = frame
            
            rgb_analyzed = cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(
                rgb_analyzed,
                number_of_times_to_upsample=UPSAMPLE,
                model=DETECTION_MODEL
            )
            face_encodings = face_recognition.face_encodings(rgb_analyzed, face_locations)
            
            detections = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    if best_distance < TOLERANCE:
                        name = self.known_names[best_match_index].upper()
                        color = (0, 255, 0)
                    else:
                        name = f"Inconnu ({best_distance:.2f})"
                        color = (0, 0, 255)
                else:
                    name = "Inconnu"
                    color = (0, 0, 255)
                
                inv = 1.0 / ANALYSIS_SCALE
                detections.append((
                    int(top * inv), int(right * inv),
                    int(bottom * inv), int(left * inv),
                    name, color
                ))
            
            self.latest_detections = detections

    def display_loop(self):
        if not self.running:
            return
        
        try:
            frame = self.frame_queue.get_nowait()
        except:
            self.root.after(int(1000 / MAX_FPS), self.display_loop)
            return
        
        display_frame = cv2.resize(frame, (self.hud_width, self.hud_height))
        
        orig_h, orig_w = frame.shape[0], frame.shape[1]
        scale_x = self.hud_width / orig_w
        scale_y = self.hud_height / orig_h
        
        for (top, right, bottom, left, name, color) in self.latest_detections:
            top_d = int(top * scale_y)
            bottom_d = int(bottom * scale_y)
            left_d = int(left * scale_x)
            right_d = int(right * scale_x)
            
            cv2.rectangle(display_frame, (left_d, top_d), (right_d, bottom_d), color, 2)
            cv2.putText(display_frame, name, (left_d + 5, top_d - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk
        
        self.root.after(int(1000 / MAX_FPS), self.display_loop)

    def close(self):
        self.running = False
        self.root.destroy()

# ============================================================
# 5. POINT D'ENTRÉE
# ============================================================
def main():
    print("[INFO] Chargement de la base de données...")
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("[ERREUR] Aucun visage connu.")
        return

    print("[INFO] HUD en cours d'ouverture (mode léger)...")
    print(f"[INFO] Paramètres : modèle={DETECTION_MODEL}, upsample={UPSAMPLE}, scale={ANALYSIS_SCALE}, interval={DETECTION_INTERVAL}")
    print("[INFO] Cliquez-glissez pour déplacer. Molette pour redimensionner. Echap pour quitter.")
    
    root = tk.Tk()
    overlay = FaceOverlay(root, known_encodings, known_names)
    
    root.bind("<Escape>", lambda e: overlay.close())
    try:
        root.mainloop()
    except KeyboardInterrupt:
        overlay.close()

if __name__ == "__main__":
    main()