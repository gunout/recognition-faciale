"""
Système de surveillance avec reconnaissance faciale - Version refactorisée

Corrections apportées :
  - Tolérance L2² corrigée (0.36 au lieu de 0.6)
  - Queues séparées capture / détection / affichage (plus de race condition)
  - Hot-reload robuste par hash de fichiers (mtime + size)
  - Détection dynamique du moniteur
  - Arrêt propre via threading.Event
  - Logging structuré
  - Configuration centralisée (dataclass)
  - Snapshot inconnu : clustering + cooldowns + rotation (anti-spam)
  - Noms normalisés en interne (lowercase) + affichage Title Case
  - Hot-reload avec normalisation identique à build_index.py
  - Police du HUD réduite + échelle adaptative au zoom
"""

import tkinter as tk
import cv2
import face_recognition
import numpy as np
import os
import mss
import time
import threading
import pickle
import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageTk
from queue import Queue, Empty, Full
from datetime import datetime
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss non installé - le système ne fonctionnera pas")

# ⚠️ Désactiver le multithreading interne d'OpenCV (évite les crashs)
cv2.setNumThreads(0)


# ============================================================
# 0. LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SURV")


# ============================================================
# 1. NORMALISATION DES NOMS (identique à build_index.py)
# ============================================================
def normalize_name(filename: str) -> str:
    """
    Clé interne normalisée pour identifier une personne.
    Doit être IDENTIQUE à la fonction de build_index.py.
    Ex: 'jEAN-PHILLIPE tANGUY.jpg' → 'jean-phillipe tanguy'
    """
    stem = Path(filename).stem
    stem = stem.split("_")[0]
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    stem = stem.lower().strip()
    stem = " ".join(stem.split())
    return stem


def display_name(normalized: str) -> str:
    """
    Version lisible pour affichage : 'emmanuel macron' → 'Emmanuel Macron'.
    Gère les traits d'union : 'jean-phillipe' → 'Jean-Phillipe'.
    """
    if not normalized:
        return ""
    words = []
    for word in normalized.split():
        parts = word.split("-")
        parts = [p.capitalize() if p else "" for p in parts]
        words.append("-".join(parts))
    return " ".join(words)


# ============================================================
# 2. CONFIGURATION
# ============================================================
@dataclass
class Config:
    known_faces_dir: str = "known_faces"
    snapshot_dir: str = "snapshots_inconnus"
    index_file: str = "faiss_index.index"
    names_file: str = "faiss_names.pkl"

    # ⚠️ distance L2² (≈ 0.6 euclidien → 0.36 au carré)
    tolerance_l2_sq: float = 0.36

    monitor_region: Optional[dict] = None

    hud_width: int = 700
    hud_height: int = 420
    hud_top: int = 50
    hud_left: int = 50

    detection_model: str = "hog"
    upsample: int = 1
    analysis_scale: float = 0.4
    detection_interval: int = 3
    max_fps: int = 30
    hot_reload_interval: float = 3.0

    # ---------- Snapshots (anti-spam) ----------
    snapshot_enabled: bool = True
    snapshot_cluster_threshold: float = 0.50
    snapshot_cluster_cooldown: float = 300.0
    snapshot_max_per_minute: int = 5
    snapshot_max_total: int = 500
    snapshot_face_margin: int = 40
    snapshot_jpeg_quality: int = 85

    # ---------- Affichage ----------
    display_names_titlecase: bool = True
    show_distance: bool = True

    # ---------- Tailles de police (échelle relative) ----------
    font_scale_label: float = 0.38         # nom de la personne
    font_scale_distance: float = 0.30      # "DIST: x.xx"
    font_scale_status: float = 0.30        # barre du haut
    font_scale_stats: float = 0.28         # stats en bas
    font_thickness_label: int = 1
    font_thickness_distance: int = 1

    colors: dict = field(default_factory=lambda: {
        "target": (0, 255, 0),
        "unknown": (0, 0, 255),
        "hud": (0, 180, 0),
    })


CFG = Config()


# ============================================================
# 3. VERROUS
# ============================================================
dlib_lock = threading.Lock()


# ============================================================
# 4. BASE DE DONNÉES FAISS
# ============================================================
class FaceDatabase:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.index = None
        self.names: list[str] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self._file_state: dict[str, tuple[float, int]] = {}
        self._watcher_thread: Optional[threading.Thread] = None

    def load_index(self) -> bool:
        if not FAISS_AVAILABLE:
            log.error("faiss indisponible")
            return False

        idx_path = Path(self.cfg.index_file)
        names_path = Path(self.cfg.names_file)

        if idx_path.exists() and names_path.exists():
            try:
                self.index = faiss.read_index(str(idx_path))
                with open(names_path, "rb") as f:
                    self.names = pickle.load(f)

                non_normalized = [n for n in self.names if n != normalize_name(n + ".jpg")]
                if non_normalized:
                    log.warning(
                        "%d nom(s) non normalisé(s) dans l'index "
                        "(reconstruis avec 'python build_index.py')",
                        len(non_normalized),
                    )

                log.info("Index FAISS chargé : %d visages, %d personnes uniques",
                         self.index.ntotal, len(set(self.names)))
                self._file_state = self._scan_state()
                return True
            except Exception as e:
                log.exception("Échec chargement index : %s", e)

        log.warning("Index introuvable → lance 'python build_index.py'")
        self.index = None
        self.names = []
        return False

    def add_face(self, encoding: np.ndarray, normalized_name: str) -> None:
        """Ajoute un visage. `normalized_name` doit être déjà normalisé."""
        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatL2(encoding.shape[0])
            self.index.add(np.array([encoding], dtype="float32"))
            self.names.append(normalized_name)
            self._save()
            log.info("[ADD] %s (total=%d)",
                     display_name(normalized_name), self.index.ntotal)

    def _save(self) -> None:
        try:
            faiss.write_index(self.index, self.cfg.index_file)
            with open(self.cfg.names_file, "wb") as f:
                pickle.dump(self.names, f)
        except Exception as e:
            log.exception("Échec sauvegarde index : %s", e)

    def search(self, encoding: np.ndarray, k: int = 3):
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return [], []
            query = np.array([encoding], dtype="float32")
            distances, indices = self.index.search(query, k=k)
            valid_d, valid_n = [], []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.names):
                    valid_d.append(float(dist))
                    valid_n.append(self.names[idx])
            return valid_d, valid_n

    def _scan_state(self) -> dict[str, tuple[float, int]]:
        state = {}
        d = Path(self.cfg.known_faces_dir)
        if not d.is_dir():
            return state
        for f in d.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    st = f.stat()
                    state[f.name] = (st.st_mtime, st.st_size)
                except OSError:
                    pass
        return state

    def start_watcher(self) -> None:
        if self._watcher_thread and self._watcher_thread.is_alive():
            return
        self._watcher_thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="FaceDB-Watcher"
        )
        self._watcher_thread.start()
        log.info("Hot-reload activé (intervalle %.1fs)", self.cfg.hot_reload_interval)

    def _watch_loop(self) -> None:
        while not self.stop_event.wait(self.cfg.hot_reload_interval):
            try:
                self.reload_if_needed()
            except Exception as e:
                log.exception("Erreur hot-reload : %s", e)

    def reload_if_needed(self) -> None:
        new_state = self._scan_state()
        if new_state == self._file_state:
            return
        self._file_state = new_state

        existing = set(self.names)
        for filename in new_state.keys():
            key = normalize_name(filename)
            if not key or key in existing:
                continue

            filepath = os.path.join(self.cfg.known_faces_dir, filename)
            log.info("[RELOAD] Nouveau visage : %s → '%s'", filename, key)
            try:
                with dlib_lock:
                    image = face_recognition.load_image_file(filepath)
                    encs = face_recognition.face_encodings(image)
                if len(encs) > 0:
                    self.add_face(encs[0].astype(np.float32), key)
                    existing.add(key)
                else:
                    log.warning("[RELOAD] Aucun visage dans %s", filename)
            except Exception as e:
                log.exception("[RELOAD] Erreur sur %s : %s", filename, e)

    def stop(self) -> None:
        self.stop_event.set()


# ============================================================
# 5. GESTIONNAIRE DE SNAPSHOTS (anti-spam)
# ============================================================
@dataclass
class _Cluster:
    encoding: np.ndarray
    count: int = 1
    last_snapshot: float = 0.0
    last_seen: float = 0.0


class SnapshotManager:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.output_dir = Path(cfg.snapshot_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._clusters: list[_Cluster] = []
        self._recent_timestamps: list[float] = []
        self._total_saved = 0
        self._skipped_cluster = 0
        self._skipped_rate = 0
        self._last_rotation = 0.0

    def maybe_snapshot(
        self,
        frame: np.ndarray,
        encoding: np.ndarray,
        top: int, right: int, bottom: int, left: int,
        face_idx: int,
    ) -> bool:
        if not self.cfg.snapshot_enabled:
            return False

        now = time.time()
        enc = encoding.astype(np.float32)

        with self._lock:
            cluster = self._find_cluster(enc)

            if cluster is None:
                if not self._check_rate_limit(now):
                    self._skipped_rate += 1
                    return False
                self._clusters.append(_Cluster(
                    encoding=enc,
                    count=1,
                    last_snapshot=now,
                    last_seen=now,
                ))
                self._prune_clusters(now)
            else:
                if now - cluster.last_snapshot < self.cfg.snapshot_cluster_cooldown:
                    cluster.last_seen = now
                    cluster.count += 1
                    self._skipped_cluster += 1
                    return False

                if not self._check_rate_limit(now):
                    self._skipped_rate += 1
                    return False

                cluster.encoding = (
                    cluster.encoding * cluster.count + enc
                ) / (cluster.count + 1)
                cluster.count += 1
                cluster.last_snapshot = now
                cluster.last_seen = now

            self._recent_timestamps.append(now)

        self._save(frame, top, right, bottom, left, face_idx)
        self._maybe_rotate(now)
        return True

    def stats(self) -> dict:
        with self._lock:
            return {
                "total": self._total_saved,
                "clusters": len(self._clusters),
                "skip_cooldown": self._skipped_cluster,
                "skip_rate": self._skipped_rate,
            }

    def _find_cluster(self, encoding: np.ndarray) -> Optional[_Cluster]:
        if not self._clusters:
            return None
        best = None
        best_dist = float("inf")
        for c in self._clusters:
            d_sq = float(np.sum((c.encoding - encoding) ** 2))
            if d_sq < best_dist and d_sq < self.cfg.snapshot_cluster_threshold:
                best_dist = d_sq
                best = c
        return best

    def _check_rate_limit(self, now: float) -> bool:
        cutoff = now - 60.0
        self._recent_timestamps = [t for t in self._recent_timestamps if t > cutoff]
        return len(self._recent_timestamps) < self.cfg.snapshot_max_per_minute

    def _prune_clusters(self, now: float) -> None:
        cutoff = now - 3600.0
        self._clusters = [c for c in self._clusters if c.last_seen > cutoff]

    def _save(self, frame, top, right, bottom, left, face_idx):
        margin = self.cfg.snapshot_face_margin
        h, w = frame.shape[:2]
        y1, y2 = max(0, top - margin), min(h, bottom + margin)
        x1, x2 = max(0, left - margin), min(w, right + margin)
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"INCONNU_{ts}_{face_idx}.jpg"
        filepath = self.output_dir / filename
        try:
            cv2.imwrite(
                str(filepath), face_img,
                [cv2.IMWRITE_JPEG_QUALITY, self.cfg.snapshot_jpeg_quality],
            )
            self._total_saved += 1
            log.info("[SNAPSHOT] %s", filepath.name)
        except Exception as e:
            log.exception("Échec snapshot : %s", e)

    def _maybe_rotate(self, now: float) -> None:
        if now - self._last_rotation < 60.0:
            return
        self._last_rotation = now
        try:
            files = sorted(
                self.output_dir.glob("INCONNU_*.jpg"),
                key=lambda f: f.stat().st_mtime,
            )
        except OSError:
            return
        excess = len(files) - self.cfg.snapshot_max_total
        if excess <= 0:
            return
        for f in files[:excess]:
            try:
                f.unlink()
            except OSError:
                pass
        log.info("Rotation : %d ancien(s) snapshot(s) supprimé(s)", excess)


# ============================================================
# 6. CAPTURE D'ÉCRAN
# ============================================================
class ScreenCapture:
    def __init__(self, cfg: Config, on_frame):
        self.cfg = cfg
        self.on_frame = on_frame
        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.region = self._resolve_region()

    def _resolve_region(self) -> dict:
        if self.cfg.monitor_region is not None:
            return self.cfg.monitor_region
        with mss.mss() as sct:
            monitors = sct.monitors
            region = monitors[1] if len(monitors) > 1 else monitors[0]
        log.info("Moniteur sélectionné : %s", region)
        return region

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="Capture"
        )
        self._thread.start()

    def _loop(self) -> None:
        consecutive_errors = 0
        with mss.mss() as sct:
            while not self.stop_event.is_set():
                try:
                    shot = sct.grab(self.region)
                    frame = np.array(shot)[:, :, :3].copy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.on_frame(frame)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    log.warning("Capture erreur (%d) : %s", consecutive_errors, e)
                    if consecutive_errors >= 10:
                        log.error("Trop d'erreurs capture → arrêt")
                        return
                time.sleep(1.0 / self.cfg.max_fps)

    def stop(self) -> None:
        self.stop_event.set()


# ============================================================
# 7. DÉTECTION
# ============================================================
class DetectionService:
    def __init__(self, cfg: Config, face_db: FaceDatabase, on_detections,
                 on_unknown_snapshot):
        self.cfg = cfg
        self.face_db = face_db
        self.on_detections = on_detections
        self.on_unknown_snapshot = on_unknown_snapshot

        self.input_queue: Queue = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_counter = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="Detection"
        )
        self._thread.start()

    def push_frame(self, frame: np.ndarray) -> None:
        try:
            self.input_queue.put_nowait(frame)
        except Full:
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass
            try:
                self.input_queue.put_nowait(frame)
            except Full:
                pass

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=0.5)
            except Empty:
                continue

            self._frame_counter += 1
            if self._frame_counter % self.cfg.detection_interval != 0:
                continue

            try:
                detections = self._process(frame)
                self.on_detections(detections)
            except Exception as e:
                log.exception("Erreur détection : %s", e)

    def _process(self, frame: np.ndarray) -> list[tuple]:
        scale = self.cfg.analysis_scale
        if scale != 1.0:
            analyzed = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            analyzed = frame
        rgb = cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB)

        with dlib_lock:
            locations = face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=self.cfg.upsample,
                model=self.cfg.detection_model,
            )
            encodings = face_recognition.face_encodings(rgb, locations)

        inv = 1.0 / scale
        detections = []

        for idx, ((top, right, bottom, left), enc) in enumerate(zip(locations, encodings)):
            distances, names = self.face_db.search(enc)

            name_display = "INCONNU"
            color = self.cfg.colors["unknown"]
            best_dist = 1.0

            if distances:
                best_dist = distances[0]
                if best_dist < self.cfg.tolerance_l2_sq:
                    raw = names[0]
                    name_display = display_name(raw) if self.cfg.display_names_titlecase else raw.upper()
                    color = self.cfg.colors["target"]
                else:
                    self.on_unknown_snapshot(
                        frame, enc,
                        int(top * inv), int(right * inv),
                        int(bottom * inv), int(left * inv),
                        idx + 1,
                    )

            detections.append((
                int(top * inv), int(right * inv), int(bottom * inv), int(left * inv),
                name_display, color, f"T-{idx+1:02d}", best_dist,
            ))

        return detections

    def stop(self) -> None:
        self.stop_event.set()


# ============================================================
# 8. HUD MILITAIRE
# ============================================================
class MilitaryHUD:
    def __init__(self, root: tk.Tk, face_db: FaceDatabase, cfg: Config = CFG):
        self.root = root
        self.face_db = face_db
        self.cfg = cfg
        self.stop_event = threading.Event()

        self._drag_data = {"x": 0, "y": 0}
        self.hud_width = cfg.hud_width
        self.hud_height = cfg.hud_height
        # hauteur de référence pour l'échelle adaptative des polices
        self._ref_height = 420

        self.latest_detections: list[tuple] = []
        self.detections_lock = threading.Lock()
        self.last_detection_time = 0.0

        self.display_queue: Queue = Queue(maxsize=1)

        Path(cfg.snapshot_dir).mkdir(parents=True, exist_ok=True)

        # fenêtre
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.90)
        self.root.configure(bg="black")
        self.root.geometry(
            f"{self.hud_width}x{self.hud_height}+{cfg.hud_left}+{cfg.hud_top}"
        )

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0,
                                width=self.hud_width, height=self.hud_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._start_move)
        self.canvas.bind("<B1-Motion>", self._do_move)
        self.canvas.bind("<MouseWheel>", self._do_zoom)
        self.canvas.bind("<Button-4>", self._do_zoom)
        self.canvas.bind("<Button-5>", self._do_zoom)
        self.root.bind("<Escape>", lambda e: self.close())

        # services
        self.snapshot_mgr = SnapshotManager(cfg)
        self.capture = ScreenCapture(cfg, on_frame=self._on_frame)
        self.detector = DetectionService(
            cfg, face_db,
            on_detections=self._on_detections,
            on_unknown_snapshot=self._on_unknown_snapshot,
        )

        log.info("Démarrage capture + détection")
        self.capture.start()
        self.detector.start()

        self._schedule_display()

    # ---------- callbacks ----------
    def _on_frame(self, frame: np.ndarray) -> None:
        self.detector.push_frame(frame)
        try:
            self.display_queue.put_nowait(frame)
        except Full:
            try:
                self.display_queue.get_nowait()
            except Empty:
                pass
            try:
                self.display_queue.put_nowait(frame)
            except Full:
                pass

    def _on_detections(self, detections: list[tuple]) -> None:
        with self.detections_lock:
            self.latest_detections = detections
            self.last_detection_time = time.time()

    def _on_unknown_snapshot(self, frame, encoding, top, right, bottom, left, face_idx):
        self.snapshot_mgr.maybe_snapshot(
            frame, encoding, top, right, bottom, left, face_idx
        )

    # ---------- drag / zoom ----------
    def _start_move(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _do_move(self, event):
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

    def _do_zoom(self, event):
        scale = 1.1 if (getattr(event, "num", 0) == 4 or event.delta > 0) else 0.9
        self.hud_width = max(200, int(self.hud_width * scale))
        self.hud_height = max(120, int(self.hud_height * scale))
        self.root.geometry(
            f"{self.hud_width}x{self.hud_height}"
            f"+{self.root.winfo_x()}+{self.root.winfo_y()}"
        )

    # ---------- rendu ----------
    def _schedule_display(self):
        if self.stop_event.is_set():
            return
        try:
            self._render()
        except Exception as e:
            log.exception("Erreur rendu : %s", e)
        delay = int(1000 / self.cfg.max_fps)
        self.root.after(delay, self._schedule_display)

    def _render(self):
        try:
            frame = self.display_queue.get_nowait()
        except Empty:
            return

        display = cv2.resize(frame, (self.hud_width, self.hud_height))
        oh, ow = frame.shape[:2]
        sx, sy = self.hud_width / ow, self.hud_height / oh

        with self.detections_lock:
            dets = list(self.latest_detections)
            age = time.time() - self.last_detection_time

        if age < 1.0:
            for (top, right, bottom, left, name, color, tid, dist) in dets:
                self._draw_box(display,
                               int(left*sx), int(top*sy),
                               int(right*sx), int(bottom*sy),
                               color, name, tid, dist)

        self._draw_overlay(display, len(dets) if age < 1.0 else 0)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def _draw_box(self, img, left, top, right, bottom, color, name, tid, dist):
        thickness = 2
        corner = min(25, int((right - left) * 0.25))

        for (p1, p2) in [
            ((left, top), (left + corner, top)),
            ((left, top), (left, top + corner)),
            ((right, top), (right - corner, top)),
            ((right, top), (right, top + corner)),
            ((left, bottom), (left + corner, bottom)),
            ((left, bottom), (left, bottom - corner)),
            ((right, bottom), (right - corner, bottom)),
            ((right, bottom), (right, bottom - corner)),
        ]:
            cv2.line(img, p1, p2, color, thickness)

        cx, cy = (left + right) // 2, (top + bottom) // 2
        cv2.line(img, (cx - 6, cy), (cx + 6, cy), color, 1)
        cv2.line(img, (cx, cy - 6), (cx, cy + 6), color, 1)

        # --- Échelle adaptative selon la taille du HUD ---
        s = max(0.6, min(1.5, self.hud_height / self._ref_height))
        fs_label = self.cfg.font_scale_label * s
        fs_dist = self.cfg.font_scale_distance * s
        ft_label = max(1, int(self.cfg.font_thickness_label * s))
        ft_dist = max(1, int(self.cfg.font_thickness_distance * s))

        label = f"{tid} | {name}"
        text_y = top - 10 if top > 20 else bottom + 20

        # ombre (contour noir)
        cv2.putText(img, label, (left + 1, text_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_label, (0, 0, 0),
                    ft_label + 1, cv2.LINE_AA)
        cv2.putText(img, label, (left, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_label, color,
                    ft_label, cv2.LINE_AA)

        if self.cfg.show_distance:
            cv2.putText(img, f"DIST: {dist:.2f}", (left, bottom + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_dist, color,
                        ft_dist, cv2.LINE_AA)

    def _draw_overlay(self, img, n_detections):
        h, w = img.shape[:2]
        hud_color = self.cfg.colors["hud"]

        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(img, "SYS://SURVEILLANCE_MODE", (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, self.cfg.font_scale_status,
                    hud_color, 1, cv2.LINE_AA)

        if n_detections > 0:
            status = f"TARGET LOCKED [{n_detections}]"
            status_color = self.cfg.colors["target"]
        else:
            status = "SCANNING..."
            status_color = hud_color
        cv2.putText(img, status, (w - 150, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, self.cfg.font_scale_status,
                    status_color, 1, cv2.LINE_AA)

        s = self.snapshot_mgr.stats()
        stats_text = (
            f"SNAP {s['total']} | CLUSTERS {s['clusters']} "
            f"| SKIP {s['skip_cooldown']}/{s['skip_rate']}"
        )
        cv2.putText(img, stats_text, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, self.cfg.font_scale_stats,
                    hud_color, 1, cv2.LINE_AA)

        cv2.rectangle(img, (1, 1), (w - 2, h - 2), hud_color, 1)

    # ---------- arrêt ----------
    def close(self):
        log.info("Arrêt demandé")
        self.stop_event.set()
        self.capture.stop()
        self.detector.stop()
        s = self.snapshot_mgr.stats()
        log.info(
            "Résumé snapshots : %d sauvegardés, %d clusters, "
            "%d ignorés (cooldown), %d ignorés (rate limit)",
            s["total"], s["clusters"], s["skip_cooldown"], s["skip_rate"],
        )
        self.root.after(50, self.root.destroy)


# ============================================================
# 9. POINT D'ENTRÉE
# ============================================================
def main():
    log.info("=== DÉMARRAGE (FAISS) ===")

    if not FAISS_AVAILABLE:
        log.error("faiss requis : pip install faiss-cpu")
        return

    face_db = FaceDatabase(CFG)
    ok = face_db.load_index()

    if not ok or face_db.index is None or face_db.index.ntotal == 0:
        log.error("Index vide ou introuvable → lance 'python build_index.py'")
        return

    face_db.start_watcher()

    root = tk.Tk()
    overlay = MilitaryHUD(root, face_db, CFG)

    def _on_close():
        face_db.stop()
        overlay.close()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        log.info("Interruption utilisateur")
    finally:
        face_db.stop()


if __name__ == "__main__":
    main()
