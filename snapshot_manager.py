"""
Gestionnaire de snapshots pour visages inconnus.

Stratégie anti-spam :
  1. Clustering par similarité : un nouvel encodage proche d'un cluster
     existant (distance < SEUIL_CLUSTER) ne déclenche pas de snapshot.
  2. Cooldown par cluster : un cluster connu ne peut re-snapshoter
     qu'après CLUSTER_COOLDOWN secondes.
  3. Rate limit global : max MAX_PER_MINUTE snapshots/minute.
  4. Rotation disque : garde au plus MAX_TOTAL fichiers.
"""

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("SNAPSHOT")


@dataclass
class SnapshotConfig:
    output_dir: str = "snapshots_inconnus"

    # Seuil de similarité pour considérer "même personne"
    # 0.36 = même tolérance que la reconnaissance (L2²)
    # 0.50 = plus permissif (regroupe plusieurs angles)
    cluster_threshold: float = 0.50

    # Cooldown par cluster (secondes)
    cluster_cooldown: float = 300.0        # 5 min entre 2 snapshots du même cluster

    # Rate limit global
    max_per_minute: int = 5                # max 5 snapshots/minute

    # Rotation disque
    max_total: int = 500                   # garde les 500 plus récents

    # Marge autour du visage (pixels)
    face_margin: int = 40

    # Qualité JPEG
    jpeg_quality: int = 85


@dataclass
class _Cluster:
    encoding: np.ndarray                   # centroïde
    count: int = 1
    last_snapshot: float = 0.0
    last_seen: float = 0.0


class SnapshotManager:
    def __init__(self, cfg: SnapshotConfig = SnapshotConfig()):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._clusters: list[_Cluster] = []
        self._recent_timestamps: list[float] = []   # pour rate limit
        self._total_saved = 0

        # compteur pour logs
        self._skipped_cluster = 0
        self._skipped_rate = 0

    # ---------- API publique ----------
    def maybe_snapshot(
        self,
        frame: np.ndarray,
        encoding: np.ndarray,
        top: int, right: int, bottom: int, left: int,
        face_idx: int,
    ) -> bool:
        """
        Décide si un snapshot doit être pris. Retourne True si sauvegardé.
        `top/right/bottom/left` sont en coordonnées de la frame ORIGINALE.
        """
        now = time.time()
        enc = encoding.astype(np.float32)

        with self._lock:
            # 1. Chercher un cluster existant proche
            cluster = self._find_cluster(enc)

            # 2. Nouveau cluster → vérifier le rate limit global
            if cluster is None:
                if not self._check_rate_limit(now):
                    self._skipped_rate += 1
                    log.debug("Rate limit atteint → snapshot ignoré")
                    return False

                self._clusters.append(_Cluster(
                    encoding=enc,
                    count=1,
                    last_snapshot=now,
                    last_seen=now,
                ))
                # nettoyer les clusters trop vieux (> 1h)
                self._prune_clusters(now)

            else:
                # Cluster connu → cooldown ?
                if now - cluster.last_snapshot < self.cfg.cluster_cooldown:
                    cluster.last_seen = now
                    cluster.count += 1
                    self._skipped_cluster += 1
                    log.debug(
                        "Cluster déjà vu il y a %.1fs → snapshot ignoré",
                        now - cluster.last_snapshot,
                    )
                    return False

                # cooldown écoulé → vérifier aussi le rate limit
                if not self._check_rate_limit(now):
                    self._skipped_rate += 1
                    return False

                # mise à jour du centroïde (moyenne glissante)
                cluster.encoding = (cluster.encoding * cluster.count + enc) / (cluster.count + 1)
                cluster.count += 1
                cluster.last_snapshot = now
                cluster.last_seen = now

            # 3. Enregistrer
            self._recent_timestamps.append(now)

        # sauvegarde hors lock (I/O disque)
        self._save(frame, top, right, bottom, left, face_idx, now)
        self._maybe_rotate()

        return True

    def stats(self) -> dict:
        with self._lock:
            return {
                "clusters": len(self._clusters),
                "total_saved": self._total_saved,
                "skipped_cluster": self._skipped_cluster,
                "skipped_rate": self._skipped_rate,
            }

    # ---------- interne ----------
    def _find_cluster(self, encoding: np.ndarray):
        """Retourne le cluster le plus proche si distance < seuil."""
        if not self._clusters:
            return None

        best_dist = float("inf")
        best = None
        for c in self._clusters:
            d = float(np.linalg.norm(c.encoding - encoding))
            d_sq = d * d
            if d_sq < best_dist and d_sq < self.cfg.cluster_threshold:
                best_dist = d_sq
                best = c
        return best

    def _check_rate_limit(self, now: float) -> bool:
        """Vérifie et purge le rate limit (fenêtre glissante 60s)."""
        cutoff = now - 60.0
        self._recent_timestamps = [t for t in self._recent_timestamps if t > cutoff]
        return len(self._recent_timestamps) < self.cfg.max_per_minute

    def _prune_clusters(self, now: float) -> None:
        """Retire les clusters non vus depuis > 1h."""
        cutoff = now - 3600.0
        before = len(self._clusters)
        self._clusters = [c for c in self._clusters if c.last_seen > cutoff]
        if before != len(self._clusters):
            log.debug("Pruned %d clusters inactifs", before - len(self._clusters))

    def _save(self, frame, top, right, bottom, left, face_idx, now):
        margin = self.cfg.face_margin
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
                [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality],
            )
            self._total_saved += 1
            log.info("[SNAPSHOT] %s", filepath.name)
        except Exception as e:
            log.exception("Échec sauvegarde snapshot : %s", e)

    def _maybe_rotate(self) -> None:
        """Garde au plus `max_total` fichiers sur disque."""
        files = sorted(
            self.output_dir.glob("INCONNU_*.jpg"),
            key=lambda f: f.stat().st_mtime,
        )
        excess = len(files) - self.cfg.max_total
        if excess <= 0:
            return
        for f in files[:excess]:
            try:
                f.unlink()
            except OSError:
                pass
        log.info("Rotation : %d ancien(s) snapshot(s) supprimé(s)", excess)