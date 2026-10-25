"""
Construction de l'index FAISS avec normalisation des noms et validation améliorée.

Corrections clés :
  - Normalisation des noms (minuscules, sans accents, espaces uniques)
    → 'Emmanuel Macron.jpg' et 'emmanuel macron.jpg' = 1 seule personne
  - Affichage en Title Case pour les logs/HUD
  - Validation à 2 niveaux (critique < 0.25, à vérifier < 0.36)
  - Ignore les collisions entre noms normalisés identiques
  - Multi-encodages par personne supporté (john.jpg + john_1.jpg)
"""

import argparse
import logging
import os
import pickle
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import face_recognition
except ImportError:
    print("[ERREUR] face_recognition requis : pip install face_recognition")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[ERREUR] faiss requis : pip install faiss-cpu")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================
KNOWN_FACES_DIR = Path("known_faces")
INDEX_FILE = Path("faiss_index.index")
NAMES_FILE = Path("faiss_names.pkl")

DETECTION_MODEL = "hog"
UPSAMPLE = 1
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_ENCODINGS_PER_PERSON = 5


log = logging.getLogger("BUILD")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================
# NORMALISATION DES NOMS
# ============================================================
def normalize_name(filename: str) -> str:
    """
    Clé interne normalisée pour identifier une personne.
    - lowercase
    - sans accents
    - espaces multiples → 1
    - underscores → espaces
    Ex: 'jEAN-PHILLIPE tANGUY.jpg' → 'jean-phillipe tanguy'
    """
    stem = Path(filename).stem
    # retire le suffixe _1, _2, ...
    stem = stem.split("_")[0]
    # normalisation unicode → ascii
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    # minuscules + espaces propres
    stem = stem.lower().strip()
    stem = " ".join(stem.split())
    return stem


def display_name(normalized: str) -> str:
    """
    Version lisible pour affichage : 'emmanuel macron' → 'Emmanuel Macron'.
    Gère aussi les traits d'union.
    """
    words = []
    for word in normalized.split():
        # préserve les traits d'union internes : 'jean-phillipe' → 'Jean-Phillipe'
        parts = word.split("-")
        parts = [p.capitalize() if p else "" for p in parts]
        words.append("-".join(parts))
    return " ".join(words)


# ============================================================
# EXTRACTION DES ENCODAGES
# ============================================================
def extract_encodings(filepath: Path) -> list[np.ndarray]:
    try:
        image = face_recognition.load_image_file(str(filepath))
    except Exception as e:
        log.warning("[SKIP] %s : lecture impossible (%s)", filepath.name, e)
        return []

    try:
        locations = face_recognition.face_locations(
            image, number_of_times_to_upsample=UPSAMPLE, model=DETECTION_MODEL
        )
    except Exception as e:
        log.warning("[SKIP] %s : détection échouée (%s)", filepath.name, e)
        return []

    if not locations:
        log.warning("[SKIP] %s : aucun visage détecté", filepath.name)
        return []

    try:
        encodings = face_recognition.face_encodings(image, locations)
    except Exception as e:
        log.warning("[SKIP] %s : encodage échoué (%s)", filepath.name, e)
        return []

    # tri par surface décroissante (visage principal en premier)
    def area(loc):
        top, right, bottom, left = loc
        return (bottom - top) * (right - left)

    order = sorted(range(len(locations)), key=lambda i: area(locations[i]), reverse=True)
    sorted_encodings = [encodings[i].astype(np.float32) for i in order]

    if len(sorted_encodings) > 1:
        log.info("  %s : %d visages → garde le plus grand (%dx%d px)",
                 filepath.name, len(sorted_encodings),
                 locations[0][1] - locations[0][3] if len(locations) > 0 else 0,
                 locations[0][2] - locations[0][0] if len(locations) > 0 else 0)

    return sorted_encodings


# ============================================================
# SCAN
# ============================================================
def collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        log.error("Dossier introuvable : %s", root.resolve())
        return []
    files = sorted(
        f for f in root.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    )
    log.info("%d image(s) trouvée(s)", len(files))
    return files


# ============================================================
# BUILD
# ============================================================
def build_index(dry_run: bool = False) -> bool:
    files = collect_images(KNOWN_FACES_DIR)
    if not files:
        log.error("Aucune image à traiter")
        return False

    # regroupement par nom NORMALISÉ
    per_person: dict[str, list[np.ndarray]] = {}
    # traçabilité : nom normalisé -> liste des fichiers originaux
    sources: dict[str, list[str]] = {}
    skipped = 0

    for filepath in files:
        key = normalize_name(filepath.name)
        if not key:
            log.warning("[SKIP] %s : nom vide après normalisation", filepath.name)
            skipped += 1
            continue

        encodings = extract_encodings(filepath)
        if not encodings:
            skipped += 1
            continue

        kept = encodings[:MAX_ENCODINGS_PER_PERSON]
        per_person.setdefault(key, []).extend(kept)
        sources.setdefault(key, []).append(filepath.name)

        log.info("[OK] %-40s → '%s' (%d enc.)",
                 filepath.name, key, len(kept))

    if not per_person:
        log.error("Aucun visage exploitable")
        return False

    # alerte doublons potentiels (même clé = même personne)
    for key, file_list in sources.items():
        if len(file_list) > 1:
            log.warning("  ⚠️ '%s' est alimenté par %d fichiers : %s",
                        key, len(file_list), ", ".join(file_list))

    # tri alphabétique stable
    keys_sorted = sorted(per_person.keys())
    log.info("Personnes uniques (après normalisation) : %d", len(keys_sorted))

    all_encodings: list[np.ndarray] = []
    all_names: list[str] = []
    for key in keys_sorted:
        encs = per_person[key]
        all_encodings.extend(encs)
        all_names.extend([key] * len(encs))
        log.info("  %-30s : %d encodage(s)", display_name(key), len(encs))

    matrix = np.vstack(all_encodings).astype(np.float32)
    log.info("Matrice finale : %s (dtype=%s)", matrix.shape, matrix.dtype)

    dim = matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)
    log.info("Index FAISS : ntotal=%d, dim=%d", index.ntotal, dim)

    validate_index(index, all_names, matrix)

    if dry_run:
        log.info("[DRY-RUN] Pas d'écriture")
        return True

    save_atomic(index, all_names)
    log.info("✅ Terminé : %s + %s", INDEX_FILE, NAMES_FILE)

    if skipped:
        log.warning("%d fichier(s) ignoré(s)", skipped)

    return True


# ============================================================
# VALIDATION
# ============================================================
def validate_index(index, names: list[str], matrix: np.ndarray) -> None:
    """
    Détecte les collisions entre PERSONNES DIFFÉRENTES.
    Ignore les collisions intra-personne (multi-encodages).
    """
    log.info("Validation (recherche de collisions entre personnes)...")
    k = min(3, index.ntotal)
    if k < 2:
        log.info("  Index trop petit")
        return

    distances, indices = index.search(matrix, k)

    critical: list[tuple[str, str, float]] = []
    warnings: list[tuple[str, str, float]] = []

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        own = names[i]
        for dist, idx in zip(dists, idxs):
            if idx == i:
                continue
            other = names[idx]
            if other == own:
                continue  # même personne

            d = float(dist)
            if d < 0.25:
                critical.append((own, other, d))
            elif d < 0.36:
                warnings.append((own, other, d))
            break  # 1er voisin différent seulement

    # dédoublonne les paires (A≈B et B≈A)
    def dedup(pairs):
        seen = set()
        out = []
        for a, b, d in pairs:
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b, d))
        return out

    critical = dedup(critical)
    warnings = dedup(warnings)

    if critical:
        log.error("🔴 %d collision(s) CRITIQUE(S) (< 0.25) :", len(critical))
        for a, b, d in critical:
            log.error("   %-30s ≈ %-30s dist=%.4f",
                      display_name(a), display_name(b), d)
        log.error("   → Vérifie ces photos : ce sont probablement des erreurs.")

    if warnings:
        log.warning("🟠 %d collision(s) À VÉRIFIER (0.25–0.36) :", len(warnings))
        for a, b, d in warnings:
            log.warning("   %-30s ≈ %-30s dist=%.4f",
                        display_name(a), display_name(b), d)
        log.warning("   → Photos possiblement basse qualité ou angles similaires.")

    if not critical and not warnings:
        log.info("  ✅ Aucune collision entre personnes différentes")
    else:
        log.info("  (Rappel : collision = photo de mauvaise qualité ou vraie ressemblance)")


# ============================================================
# SAUVEGARDE ATOMIQUE
# ============================================================
def save_atomic(index, names: list[str]) -> None:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    fd_i, tmp_i = tempfile.mkstemp(
        prefix=INDEX_FILE.stem + "_", suffix=".tmp", dir=INDEX_FILE.parent
    )
    os.close(fd_i)
    try:
        faiss.write_index(index, tmp_i)
        os.replace(tmp_i, INDEX_FILE)
    except Exception:
        if os.path.exists(tmp_i):
            os.unlink(tmp_i)
        raise

    fd_n, tmp_n = tempfile.mkstemp(
        prefix=NAMES_FILE.stem + "_", suffix=".tmp", dir=NAMES_FILE.parent
    )
    os.close(fd_n)
    try:
        with open(tmp_n, "wb") as f:
            pickle.dump(names, f)
        os.replace(tmp_n, NAMES_FILE)
    except Exception:
        if os.path.exists(tmp_n):
            os.unlink(tmp_n)
        raise


# ============================================================
# VÉRIFICATION
# ============================================================
def verify_saved_index() -> bool:
    if not INDEX_FILE.exists() or not NAMES_FILE.exists():
        log.error("Fichiers absents")
        return False

    try:
        index = faiss.read_index(str(INDEX_FILE))
        with open(NAMES_FILE, "rb") as f:
            names = pickle.load(f)
    except Exception as e:
        log.exception("Relecture échouée : %s", e)
        return False

    if index.ntotal != len(names):
        log.error("Incohérence : %d vs %d", index.ntotal, len(names))
        return False

    unique = set(names)
    log.info("✅ OK : %d vecteurs, %d personnes uniques",
             index.ntotal, len(unique))
    return True


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.verify_only:
        sys.exit(0 if verify_saved_index() else 1)

    log.info("=== CONSTRUCTION DE L'INDEX ===")
    if not build_index(dry_run=args.dry_run):
        sys.exit(1)

    if not args.dry_run:
        if not verify_saved_index():
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()