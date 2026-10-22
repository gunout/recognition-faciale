import tkinter as tk
import cv2
import face_recognition
import numpy as np
import os
import mss
import time
import threading
from PIL import Image, ImageTk, ImageDraw, ImageFont
from queue import Queue

# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6

# ⚠️ Zone de capture de la vidéo sur votre écran
MONITOR_REGION = {"top": 100, "left": 300, "width": 1000, "height": 600}

HUD_CONFIG = {
    "width": 600,
    "height": 360,
    "top": 50,
    "left": 50
}

# ============================================================
# 2. PARAMÈTRES DE DÉTECTION
# ============================================================
DETECTION_MODEL = "hog"
UPSAMPLE = 1
ANALYSIS_SCALE = 0.5
DETECTION_INTERVAL = 4
MAX_FPS = 30

# ============================================================
# 3. COULEURS DU THÈME MILITAIRE
# ============================================================
COLOR_TARGET = (0, 255, 0)      # Vert néon
COLOR_UNKNOWN = (0, 0, 255)     # Rouge
COLOR_BG = (0, 0, 0)            # Noir
COLOR_HUD = (0, 200, 0)         # Vert sombre pour l'interface

# ============================================================
# 4. CHARGEMENT DES VISAGES
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
# 5. CLASSE HUD MILITAIRE
# ============================================================
class MilitaryHUD:
    def __init__(self, root, known_encodings, known_names):
        self.root = root
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.running = True
        
        self._drag_data = {"x": 0, "y": 0}
        self._resizing = False
        
        self.frame_queue = Queue(maxsize=2)
        self.latest_detections = []
        self.scan_line_y = 0
        
        # Fenêtre
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.85)  # Semi-transparent
        self.root.configure(bg='black')
        
        self.hud_width = HUD_CONFIG['width']
        self.hud_height = HUD_CONFIG['height']
        
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.update_geometry(HUD_CONFIG['left'], HUD_CONFIG['top'])
        
        # Événements souris
        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.do_move)
        self.canvas.bind("<MouseWheel>", self.do_zoom)
        self.canvas.bind("<Button-4>", self.do_zoom)
        self.canvas.bind("<Button-5>", self.do_zoom)
        
        # Lancer les threads
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
        self.hud_width = max(150, int(self.hud_width * scale))
        self.hud_height = max(90, int(self.hud_height * scale))
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
                        color = COLOR_TARGET
                    else:
                        name = "INCONNU"
                        color = COLOR_UNKNOWN
                else:
                    name = "INCONNU"
                    color = COLOR_UNKNOWN
                
                inv = 1.0 / ANALYSIS_SCALE
                detections.append((
                    int(top * inv), int(right * inv),
                    int(bottom * inv), int(left * inv),
                    name, color
                ))
            
            self.latest_detections = detections

    def draw_military_box(self, img, left, top, right, bottom, color, name):
        """Dessine un cadre de visée militaire (coins + réticule)."""
        h, w = img.shape[:2]
        thickness = 2
        corner_len = min(20, int((right - left) * 0.2))
        
        # 4 coins
        cv2.line(img, (left, top), (left + corner_len, top), color, thickness)
        cv2.line(img, (left, top), (left, top + corner_len), color, thickness)
        cv2.line(img, (right, top), (right - corner_len, top), color, thickness)
        cv2.line(img, (right, top), (right, top + corner_len), color, thickness)
        cv2.line(img, (left, bottom), (left + corner_len, bottom), color, thickness)
        cv2.line(img, (left, bottom), (left, bottom - corner_len), color, thickness)
        cv2.line(img, (right, bottom), (right - corner_len, bottom), color, thickness)
        cv2.line(img, (right, bottom), (right, bottom - corner_len), color, thickness)
        
        # Réticule central (croix)
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        cross = 8
        cv2.line(img, (cx - cross, cy), (cx + cross, cy), color, 1)
        cv2.line(img, (cx, cy - cross), (cx, cy + cross), color, 1)
        
        # Texte avec ombre portée
        text_pos = (left, top - 10)
        if text_pos[1] < 15:
            text_pos = (left, bottom + 20)
        
        # Ombre
        cv2.putText(img, name, (text_pos[0] + 1, text_pos[1] + 1), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        # Texte
        cv2.putText(img, name, text_pos, 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)
        
        # Distance
        dist_text = f"{name}"
        cv2.putText(img, dist_text, (left, bottom + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def draw_hud_overlay(self, img):
        """Dessine les éléments fixes du HUD (barre de statut, scanline)."""
        h, w = img.shape[:2]
        
        # Barre de statut en haut
        cv2.rectangle(img, (0, 0), (w, 25), (0, 0, 0), -1)
        cv2.putText(img, "SYS://SURVEILLANCE_MODE", (10, 17), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD, 1, cv2.LINE_AA)
        
        # Statut à droite
        status = "SCAN..."
        if self.latest_detections:
            status = "TARGET LOCKED"
        status_color = COLOR_TARGET if self.latest_detections else COLOR_HUD
        cv2.putText(img, status, (w - 120, 17), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)
        
        # Ligne de balayage (scanline) animée
        self.scan_line_y = (self.scan_line_y + 5) % h
        cv2.line(img, (0, self.scan_line_y), (w, self.scan_line_y), COLOR_HUD, 1)
        
        # Bords du HUD
        cv2.rectangle(img, (1, 1), (w-2, h-2), COLOR_HUD, 1)

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
        
        # Dessiner les rectangles de visée
        for (top, right, bottom, left, name, color) in self.latest_detections:
            top_d = int(top * scale_y)
            bottom_d = int(bottom * scale_y)
            left_d = int(left * scale_x)
            right_d = int(right * scale_x)
            self.draw_military_box(display_frame, left_d, top_d, right_d, bottom_d, color, name)
        
        # Dessiner l'overlay HUD
        self.draw_hud_overlay(display_frame)
        
        # Convertir et afficher
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
# 6. POINT D'ENTRÉE
# ============================================================
def main():
    print("[INFO] Chargement de la base de données...")
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("[ERREUR] Aucun visage connu.")
        return

    print("[INFO] Initialisation du HUD MILITAIRE...")
    print("[INFO] Cliquez-glissez pour déplacer. Molette pour redimensionner. Echap pour quitter.")
    
    root = tk.Tk()
    overlay = MilitaryHUD(root, known_encodings, known_names)
    
    root.bind("<Escape>", lambda e: overlay.close())
    try:
        root.mainloop()
    except KeyboardInterrupt:
        overlay.close()

if __name__ == "__main__":
    main()
