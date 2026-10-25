<div align="center">

# 🎯 Recognition Faciale

> **Reconnaissance faciale temps réel** — HUD flottant style militaire · FAISS · Hot-reload · 100 % local

[![Security](https://img.shields.io/badge/Security-Local%20Only-brightgreen?style=for-the-badge&logo=shield&logoColor=white)](https://github.com/gunout/recognition-faciale)
[![Privacy](https://img.shields.io/badge/Privacy-Zero%20Cloud-blue?style=for-the-badge&logo=linux&logoColor=white)](https://github.com/gunout/recognition-faciale)
[![No Cloud](https://img.shields.io/badge/No%20Cloud-100%25%20Offline-critical?style=for-the-badge&logo=cloud-off&logoColor=white)](https://github.com/gunout/recognition-faciale)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Enabled-orange?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![dlib](https://img.shields.io/badge/dlib-HOG-008080?style=for-the-badge)](http://dlib.net/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Made in France](https://img.shields.io/badge/Made_in-France-002395?style=flat-square&labelColor=FFFFFF&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNjAwIj48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjYwMCIgZmlsbD0iIzAwMjM5NSIvPjxyZWN0IHdpZHRoPSI5MDAiIGhlaWdodD0iNDAwIiB5PSIxMDAiIGZpbGw9IiNmZmYiLz48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjIwMCIgeT0iNDAwIiBmaWxsPSIjZWQyOTM5Ii8+PC9zdmc+)](https://github.com/gunout/recognition-faciale)
[![GitHub](https://img.shields.io/badge/GitHub-gunout-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/gunout)
[![Year](https://img.shields.io/badge/2026-ED2939?style=flat-square&labelColor=FFFFFF)](https://github.com/gunout/recognition-faciale)

<sub>© 2026 <strong>Gunout</strong> — Tous droits réservés.</sub>

</div>

---

## 🛡️ Sécurité & Vie Privée

Ce projet est conçu en **Privacy by Design**. Aucune image, aucun flux vidéo, aucune donnée biométrique ne quitte votre machine.

| Aspect | Statut |
| :--- | :--- |
| **Traitement** | 100 % local (CPU / GPU) |
| **Stockage visages** | `known_faces/` local, index FAISS chiffrable |
| **Réseau** | Aucune requête sortante (hors `pip`) |
| **Télémétrie** | Aucune |
| **RGPD** | Pas de collecte — l'utilisateur est responsable du consentement des personnes filmées |

> ⚠️ **Avertissement légal** — La reconnaissance faciale est une technologie sensible. L'utilisateur est seul responsable de son usage (consentement préalable, respect de la vie privée, réglementation locale).

---

## ✨ Fonctionnalités

| | |
| :--- | :--- |
| 🎯 **Détection multi-visages** | Modèle HOG optimisé, ou `cnn` pour plus de précision |
| ⚡ **Recherche FAISS** | 10 000+ visages sans ralentissement (index L2 plat) |
| 🎨 **HUD flottant** | Interface style militaire, drag & drop, zoom molette |
| 📸 **Snapshots anti-spam** | Clustering + cooldown 5 min + rate-limit + rotation disque |
| 🔄 **Hot-reload** | Ajout de photos à chaud, sans redémarrage |
| 🏷️ **Noms normalisés** | `Emmanuel Macron` = `emmanuel macron` (dédoublonnage) |
| 🧵 **Multi-thread** | Capture / détection / affichage découplés |
| 🔒 **Anti-crash dlib** | Verrou global sur `face_recognition` |
| 📊 **Stats live** | Snapshots / clusters / skips affichés dans le HUD |
| 🇫🇷 **Noms Title Case** | Affichage propre et uniforme |

---

## 🏗️ Architecture
```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  ScreenCapture  │───▶│ DetectionService │───▶│  MilitaryHUD    │
│  (thread #1)    │    │  (thread #2)     │    │  (Tk mainloop)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                      │                        │
        │                      ▼                        │
        │              ┌───────────────┐                │
        │              │  FaceDatabase │◀───────────────┘
        │              │  (FAISS)      │
        │              └───────────────┘
        │                      │
        │                      ▼
        │              ┌───────────────────┐
        └─────────────▶│  SnapshotManager  │
                       │  (clustering +    │
                       │   cooldown)       │
                       └───────────────────┘
```
**Points clés** : 3 threads + 1 mainloop Tk · 2 queues bornées à 1 · 2 verrous (`dlib_lock`, `FaceDatabase.lock`)

---

## 📦 Installation

**Prérequis** : Python 3.12+, CMake + build-essential (Linux) / VS Build Tools (Windows), une webcam ou flux écran.

    git clone https://github.com/gunout/recognition-faciale.git
    cd recognition-faciale
    python3 -m venv face_env
    source face_env/bin/activate          # Windows: face_env\Scripts\activate
    pip install "setuptools<81" "numpy<2"
    pip install dlib-bin --no-deps
    pip install face_recognition --no-deps
    pip install face-recognition-models
    pip install opencv-python mss faiss-cpu Pillow Click

Vérification :

    python -c "import face_recognition, cv2, faiss, mss, PIL; print('✅ OK')"

---

## 🚀 Démarrage rapide

### 1. Préparer la base

    mkdir -p known_faces
    # 1 à 5 photos par personne, nommées prénom_nom.jpg

- ✅ Visage **de face**, bien éclairé, ≥ 200×200 px
- ✅ **1 à 5 photos** par personne (angles différents = plus robuste)
- ❌ Pas de photos de groupe, lunettes de soleil, ni flou
- ✅ Noms **identiques** pour la même personne (`john.jpg`, `john_2.jpg`)

### 2. Construire l'index

    python build_index.py

Options : `--dry-run` · `-v` · `--verify-only`

### 3. Lancer la surveillance

    python recognition.py

### 4. Utiliser le HUD

| Action | Effet |
| :--- | :--- |
| 🖱️ **Clic-glisser** | Déplace la fenêtre |
| 🖱️ **Molette** | Redimensionne (texte adaptatif) |
| ⌨️ **Échap** | Quitte proprement |
| 🔴 **Cadre rouge** | Visage **inconnu** (snapshot pris) |
| 🟢 **Cadre vert** | Visage **reconnu** (nom affiché) |

---

## ⚙️ Configuration

Toute la config est dans la **dataclass `Config`** (haut de `reco.py`) :

    @dataclass
    class Config:
        tolerance_l2_sq: float = 0.36
        detection_model: str = "hog"
        upsample: int = 1
        analysis_scale: float = 0.4
        detection_interval: int = 3
        hud_width: int = 700
        hud_height: int = 420
        display_names_titlecase: bool = True
        show_distance: bool = True
        font_scale_label: float = 0.38
        font_scale_distance: float = 0.30
        snapshot_enabled: bool = True
        snapshot_cluster_threshold: float = 0.50
        snapshot_cluster_cooldown: float = 300.0
        snapshot_max_per_minute: int = 5
        snapshot_max_total: int = 500

Exemples :

    CFG = Config(snapshot_enabled=False)                                  # discret
    CFG = Config(tolerance_l2_sq=0.30, upsample=2, analysis_scale=0.6)    # strict
    CFG = Config(detection_interval=5, analysis_scale=0.3, max_fps=20)    # CPU-friendly
    CFG = Config(font_scale_label=0.32, font_scale_distance=0.25)         # compact

---

## 📊 Référence des paramètres

| Paramètre | Rôle | Plage |
| :--- | :--- | :--- |
| `tolerance_l2_sq` | Seuil de reconnaissance | `0.30` → `0.40` |
| `detection_model` | Modèle | `"hog"` / `"cnn"` |
| `upsample` | Ré-échantillonnage | `0` / `1` / `2` |
| `analysis_scale` | Échelle d'analyse | `0.3` / `0.4` / `0.6` |
| `detection_interval` | Frames sautées | `1` / `3` / `5` |
| `snapshot_cluster_threshold` | Regroupement inconnus | `0.40` / `0.50` / `0.70` |
| `snapshot_cluster_cooldown` | Délai entre snapshots | `60` → `1800` s |
| `snapshot_max_per_minute` | Rate-limit global | `2` → `15` |

---

## 🧪 Vérification de l'index

    python build_index.py --verify-only

Sortie : `✅ OK : 106 vecteurs, 103 personnes uniques`

| Niveau | Signification | Action |
| :--- | :--- | :--- |
| 🔴 `< 0.25` | Collision critique | Vérifier les photos |
| 🟠 `0.25–0.36` | Collision suspecte | Comparer visuellement |
| ⚠️ Doublons de noms | Même nom normalisé | OK (fusionné auto) |

---

## 🐛 Dépannage

- **`ModuleNotFoundError: face_recognition`** → `source face_env/bin/activate`
- **`free(): invalid size`** → `cv2.setNumThreads(0)` en haut + `dlib_lock` autour de tous les appels
- **Index vide** → `python build_index.py`
- **Aucun visage détecté** → photo trop sombre/floue/profil, ou `UPSAMPLE = 2`
- **HUD ne s'affiche pas** → `echo $DISPLAY` + `python reco.py 2>&1 | tee debug.log`
- **Trop de snapshots** → `snapshot_max_per_minute=2`, `snapshot_cluster_cooldown=900`, `snapshot_cluster_threshold=0.40`

---

## 📁 Structure

    recognition-faciale/
    ├── reco.py
    ├── build_index.py
    ├── faiss_index.index
    ├── faiss_names.pkl
    ├── known_faces/
    ├── snapshots_inconnus/
    ├── LICENSE
    └── README.md

---

## ⚠️ Limitations

- **HOG** : détection de face principalement. Pour les profils, activer `cnn` (GPU).
- **Visages très petits** (< 80 px) → augmenter `analysis_scale`.
- **Snapshots** : un inconnu > 5 min peut générer un 2ᵉ snapshot (voulu).
- **Rotation disque** : 500 derniers snapshots par défaut.
- **Hot-reload** : ajoute mais ne supprime pas les personnes retirées.

---

## 🛠️ Stack technique

| Composant | Rôle | Version |
| :--- | :--- | :--- |
| `face_recognition` | Encodage 128-D | 1.3.0 |
| `dlib-bin` | Backend dlib | 19.24 |
| `faiss-cpu` | Recherche vectorielle | 1.8.0 |
| `opencv-python` | Traitement d'image | 4.10 |
| `mss` | Capture d'écran | 9.0 |
| `Pillow` | Affichage Tk | 10.4 |
| `tkinter` | Fenêtre HUD | builtin |

---

## 📄 Licence

**MIT License** — voir [LICENSE](LICENSE).

---

## 🙏 Remerciements

- [@ageitgey](https://github.com/ageitgey) — `face_recognition`
- [Facebook Research](https://github.com/facebookresearch) — FAISS
- [OpenCV team](https://opencv.org/) — Vision par ordinateur
- [dlib](http://dlib.net/) — Détection de visages

---

<div align="center">

### 🇫🇷 Gunout · 2026

![Made in France](https://img.shields.io/badge/Made_in-France-002395?style=flat-square&labelColor=FFFFFF&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNjAwIj48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjYwMCIgZmlsbD0iIzAwMjM5NSIvPjxyZWN0IHdpZHRoPSI5MDAiIGhlaWdodD0iNDAwIiB5PSIxMDAiIGZpbGw9IiNmZmYiLz48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjIwMCIgeT0iNDAwIiBmaWxsPSIjZWQyOTM5Ii8+PC9zdmc+)
![GitHub](https://img.shields.io/badge/GitHub-gunout-181717?style=flat-square&logo=github&logoColor=white)
![Year](https://img.shields.io/badge/2026-ED2939?style=flat-square&labelColor=FFFFFF)

<sub>© 2026 <strong>Gunout</strong> — Tous droits réservés.</sub>

</div>
