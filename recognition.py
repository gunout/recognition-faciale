import tkinter as tk
import cv2
import face_recognition
import numpy as np
import os
import mss
import time
import threading
import pickle
import faiss
from PIL import Image, ImageTk
from queue import Queue
from datetime import datetime

# ⚠️ Désactiver le multithreading interne d'OpenCV (évite les crashs)
cv2.setNumThreads(0)

# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================
KNOWN_FACES_DIR = "known_faces"
SNAPSHOT_DIR = "snapshots_inconnus"
INDEX_FILE = "faiss_index.index"
NAMES_FILE = "faiss_names.pkl"

# Seuil de tolérance FAISS (distance L2 au carré).
# Pour face_recognition, une tolérance de 0.6 correspond environ à une distance L2 de 0.36.
TOLERANCE = 0.45 

MONITOR_REGION = {"top": 100, "left": 300, "width": 1000, "height": 600}

HUD_CONFIG = {
    "width": 700,
    "height": 420,
    "top": 50,
    "left": 50
}

# ============================================================
# 2. PARAMÈTRES DE DÉTECTION
# ============================================================
DETECTION_MODEL = "hog"
UPSAMPLE = 1
ANALYSIS_SCALE = 0.5
DETECTION_INTERVAL = 3
MAX_FPS = 30
SNAPSHOT_COOLDOWN = 30

# ============================================================
# 3. COULEURS MILITAIRES
# ============================================================
COLOR_TARGET = (0, 255, 0)      # Vert néon (reconnu)
COLOR_UNKNOWN = (0, 0, 255)     # Rouge (inconnu)
COLOR_HUD = (0, 180, 0)         # Vert sombre (interface)

# ============================================================
# 4. VERROU GLOBAL POUR FACE_RECOGNITION (ANTI-CRASH)
# ============================================================
face_lock = threading.Lock()

# ============================================================
# 5. GESTIONNAIRE FAISS (HOT RELOAD SÉCURISÉ)
# ============================================================
class FaceDatabase:
    def __init__(self):
        self.index = None
        self.names = []
        self.lock = threading.Lock()
        self.running = True
        self.last_mtime = 0

    def load_index(self):
        """Charge l'index FAISS et les noms depuis le disque."""
        if os.path.exists(INDEX_FILE) and os.path.exists(NAMES_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(NAMES_FILE, 'rb') as f:
                self.names = pickle.load(f)
            print(f"[DB] ✅ Index FAISS chargé : {self.index.ntotal} visages.")
        else:
            print("[DB] ⚠️ Index non trouvé. Création d'un index vide.")
            self.index = None
            self.names = []

    def add_face(self, encoding, name):
        """Ajoute un nouveau visage à l'index et sauvegarde."""
        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatL2(encoding.shape[0])
            
            # Ajout à l'index
            self.index.add(np.array([encoding]).astype('float32'))
            self.names.append(name)
            
            # Sauvegarde
            faiss.write_index(self.index, INDEX_FILE)
            with open(NAMES_FILE, 'wb') as f:
                pickle.dump(self.names, f)
            
            print(f"[HOT RELOAD] ✅ Visage ajouté : {name} (total: {self.index.ntotal})")

    def start_watcher(self):
        thread = threading.Thread(target=self._watch_loop, daemon=True)
        thread.start()
        print("[DB] Hot-reload activé (surveillance de known_faces/).")

    def _watch_loop(self):
        while self.running:
            time.sleep(3) # Vérifie toutes les 3 secondes
            self.reload_if_needed()

    def reload_if_needed(self):
        """Détecte les nouveaux fichiers et les ajoute à l'index."""
        if not os.path.exists(KNOWN_FACES_DIR):
            return

        current_mtime = os.path.getmtime(KNOWN_FACES_DIR)
        if current_mtime <= self.last_mtime:
            return
        self.last_mtime = current_mtime

        files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        existing_bases = set(self.names)
        
        for filename in files:
            base_name = os.path.splitext(filename)[0].split('_')[0]
            if base_name not in existing_bases:
                filepath = os.path.join(KNOWN_FACES_DIR, filename)
                print(f"[HOT RELOAD] Nouveau visage détecté : {filename}")
                try:
                    # ⬇️ VERROU : empêche dlib de planter
                    with face_lock:
                        image = face_recognition.load_image_file(filepath)
                        encs = face_recognition.face_encodings(image)
                    if len(encs) > 0:
                        self.add_face(encs[0], base_name)
                except Exception as e:
                    print(f"[HOT RELOAD] ⚠️ Erreur sur {filename} : {e}")

    def search(self, encoding):
        """Recherche les visages les plus proches dans l'index."""
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return [], []
            
            query = np.array([encoding]).astype('float32')
            distances, indices = self.index.search(query, k=3) # Top 3
            
            valid_distances = []
            valid_names = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.names) and idx >= 0:
                    valid_distances.append(dist)
                    valid_names.append(self.names[idx])
            
            return valid_distances, valid_names

# ============================================================
# 6. CLASSE HUD MILITAIRE (FUSIONNÉE)
# ============================================================
class MilitaryHUD:
    def __init__(self, root, face_db):
        self.root = root
        self.face_db = face_db
        self.running = True
        
        self._drag_data = {"x": 0, "y": 0}
        self._resizing = False
        
        self.frame_queue = Queue(maxsize=2)
        self.latest_detections = []
        self.last_snapshot = {}
        
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.90)
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
        
        print("[HUD] Démarrage des threads...")
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.capture_thread.start()
        self.detection_thread.start()
        
        print("[HUD] Démarrage de la boucle d'affichage...")
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
        if self._resizing: return
        self._resizing = True
        scale = 1.1 if (event.num == 4 or event.delta > 0) else 0.9
        self.hud_width = max(200, int(self.hud_width * scale))
        self.hud_height = max(120, int(self.hud_height * scale))
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
                        try: self.frame_queue.get_nowait()
                        except: pass
                    self.frame_queue.put(frame)
                except Exception as e:
                    print(f"[ERREUR capture] {e}")
                time.sleep(1 / MAX_FPS)

    def save_unknown_snapshot(self, frame, top, right, bottom, left, face_idx):
        """Sauvegarde une capture UNIQUEMENT pour les visages inconnus."""
        now = time.time()
        signature = f"unk_{int(left/100)}_{int(top/100)}"
        
        if signature in self.last_snapshot:
            if now - self.last_snapshot[signature] < SNAPSHOT_COOLDOWN:
                return
        self.last_snapshot[signature] = now
        
        margin = 40
        h, w = frame.shape[:2]
        y1, y2 = max(0, top - margin), min(h, bottom + margin)
        x1, x2 = max(0, left - margin), min(w, right + margin)
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0: return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"INCONNU_{timestamp}_{face_idx}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        
        cv2.imwrite(filepath, face_img)
        print(f"[SNAPSHOT INCONNU] ✅ {filepath}")

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
            
            analyzed = cv2.resize(frame, (0, 0), fx=ANALYSIS_SCALE, fy=ANALYSIS_SCALE) if ANALYSIS_SCALE != 1.0 else frame
            rgb_analyzed = cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB)
            
            # ⬇️ VERROU : empêche dlib de planter pendant la détection
            with face_lock:
                face_locations = face_recognition.face_locations(
                    rgb_analyzed, number_of_times_to_upsample=UPSAMPLE, model=DETECTION_MODEL
                )
                face_encodings = face_recognition.face_encodings(rgb_analyzed, face_locations)
            
            detections = []
            for idx, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # ⬇️ RECHERCHE FAISS
                distances, names = self.face_db.search(face_encoding)
                
                name = "INCONNU"
                color = COLOR_UNKNOWN
                best_dist = 1.0
                
                if len(distances) > 0:
                    best_dist = distances[0]
                    if best_dist < TOLERANCE:
                        name = names[0].upper()
                        color = COLOR_TARGET
                    else:
                        # Snapshot pour inconnu
                        inv = 1.0 / ANALYSIS_SCALE
                        threading.Thread(
                            target=self.save_unknown_snapshot,
                            args=(frame.copy(), int(top*inv), int(right*inv), int(bottom*inv), int(left*inv), idx+1),
                            daemon=True
                        ).start()
                
                inv = 1.0 / ANALYSIS_SCALE
                target_id = f"T-{idx+1:02d}"
                detections.append((
                    int(top * inv), int(right * inv), int(bottom * inv), int(left * inv),
                    name, color, target_id, best_dist
                ))
            
            self.latest_detections = detections

    def draw_military_box(self, img, left, top, right, bottom, color, name, target_id, distance):
        thickness = 2
        corner_len = min(25, int((right - left) * 0.25))
        
        cv2.line(img, (left, top), (left + corner_len, top), color, thickness)
        cv2.line(img, (left, top), (left, top + corner_len), color, thickness)
        cv2.line(img, (right, top), (right - corner_len, top), color, thickness)
        cv2.line(img, (right, top), (right, top + corner_len), color, thickness)
        cv2.line(img, (left, bottom), (left + corner_len, bottom), color, thickness)
        cv2.line(img, (left, bottom), (left, bottom - corner_len), color, thickness)
        cv2.line(img, (right, bottom), (right - corner_len, bottom), color, thickness)
        cv2.line(img, (right, bottom), (right, bottom - corner_len), color, thickness)
        
        cx, cy = (left + right) // 2, (top + bottom) // 2
        cv2.line(img, (cx - 6, cy), (cx + 6, cy), color, 1)
        cv2.line(img, (cx, cy - 6), (cx, cy + 6), color, 1)
        
        label = f"{target_id} | {name}"
        text_y = top - 10 if top > 20 else bottom + 20
        
        cv2.putText(img, label, (left + 1, text_y + 1), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, label, (left, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"DIST: {distance:.2f}", (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def draw_hud_overlay(self, img):
        """Dessine les éléments fixes du HUD (barre de statut, bords) - SANS SCANLINE."""
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(img, "SYS://SURVEILLANCE_MODE", (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_HUD, 1, cv2.LINE_AA)
        
        n = len(self.latest_detections)
        status = f"TARGET LOCKED [{n}]" if n > 0 else "SCANNING..."
        status_color = COLOR_TARGET if n > 0 else COLOR_HUD
        cv2.putText(img, status, (w - 160, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1, cv2.LINE_AA)
        
        cv2.rectangle(img, (1, 1), (w-2, h-2), COLOR_HUD, 1)

    def display_loop(self):
        if not self.running: return
        
        try:
            frame = self.frame_queue.get_nowait()
        except:
            self.root.after(int(1000 / MAX_FPS), self.display_loop)
            return
        
        display_frame = cv2.resize(frame, (self.hud_width, self.hud_height))
        orig_h, orig_w = frame.shape[0], frame.shape[1]
        scale_x, scale_y = self.hud_width / orig_w, self.hud_height / orig_h
        
        for (top, right, bottom, left, name, color, target_id, distance) in self.latest_detections:
            self.draw_military_box(display_frame, int(left*scale_x), int(top*scale_y), int(right*scale_x), int(bottom*scale_y), color, name, target_id, distance)
        
        self.draw_hud_overlay(display_frame)
        
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
# 7. POINT D'ENTRÉE
# ============================================================
def main():
    print("=== DEMARRAGE DU SYSTEME (MODE FAISS) ===")
    
    face_db = FaceDatabase()
    face_db.load_index()
    
    # Si l'index est vide, on le construit
    if face_db.index is None or face_db.index.ntotal == 0:
        print("[DB] Index vide ou inexistant. Veuillez lancer 'python build_index.py' d'abord.")
        return
    
    face_db.start_watcher()
    
    print("[HUD] Création de la fenêtre Tkinter...")
    root = tk.Tk()
    overlay = MilitaryHUD(root, face_db)
    
    root.bind("<Escape>", lambda e: overlay.close())
    
    try:
        print("[HUD] Fenêtre ouverte. En attente de la fermeture (Echap)...")
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] Arrêt demandé par l'utilisateur.")
        face_db.running = False
        overlay.close()

if __name__ == "__main__":
    main()
