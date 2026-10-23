<div align="center">

# 🎯 Recognition Faciale

> Détection et reconnaissance faciale en temps réel sur flux vidéo, avec interface HUD flottante, snapshots d'inconnus et recherche vectorielle rapide (FAISS).

[![Security Status](https://img.shields.io/badge/Security-Audited%20Local-brightgreen?style=for-the-badge&logo=shield)](https://github.com/gunout/recognition-faciale)
[![Privacy](https://img.shields.io/badge/Privacy-Local%20Only-blue?style=for-the-badge&logo=linux)](https://github.com/gunout/recognition-faciale)
[![No Cloud](https://img.shields.io/badge/No%20Cloud-100%25%20Offline-critical?style=for-the-badge&logo=cloud-off)](https://github.com/gunout/recognition-faciale)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Enabled-orange?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
</div>

---

## 🛡️ Sécurité & Vie Privée

Ce projet est conçu avec une approche **"Privacy by Design"**. Contrairement aux solutions cloud, aucune image, aucun flux vidéo et aucune donnée biométrique ne quitte votre machine.

| Aspect | Statut |
| :--- | :--- |
| **Traitement des données** | 100% Local (CPU/GPU de la machine) |
| **Stockage des visages** | Dossier local `known_faces/` (aucune base distante) |
| **Connexion réseau** | Aucune requête sortante (hormis dépendances pip) |
| **Conformité RGPD** | Le projet ne collecte aucune donnée personnelle |

> ⚠️ **Avertissement** : La reconnaissance faciale est une technologie sensible. L'utilisateur est seul responsable de l'usage qu'il en fait (consentement des personnes filmées, respect de la vie privée, etc.).

---

## 🚀 Fonctionnalités

- **Détection multi-visages** : Analyse les plans larges et détecte plusieurs visages simultanément (modèle HOG optimisé).
- **Recherche vectorielle ultra-rapide (FAISS)** : Capable de gérer **10 000+ visages** sans ralentissement, grâce à un index pré-calculé.
- **HUD Flottant** : Interface militaire transparente, déplaçable et redimensionnable à la souris. Cadres de visée, IDs de cibles et distances affichées.
- **Snapshots automatiques d'inconnus** : Capture et enregistre les visages non reconnus dans un dossier dédié (`snapshots_inconnus/`), avec anti-spam (cooldown).
- **Hot-Reload sécurisé** : Ajoutez de nouvelles photos dans `known_faces/` pendant que le script tourne ; il les indexe automatiquement sans redémarrer (verrou anti-crash).
- **Optimisé CPU** : Fonctionne sans GPU grâce à une architecture multi-threads.

---

## 📦 Installation

### Prérequis

- Python 3.12+
- `cmake` et `build-essential` (pour `dlib`)
- Firefox + `geckodriver` (uniquement pour le script de scraping Facebook optionnel)

### Étapes

    # 1. Cloner le dépôt
    git clone https://github.com/gunout/recognition-faciale.git
    cd recognition-faciale

    # 2. Créer un environnement virtuel
    python3 -m venv face_env
    source face_env/bin/activate

    # 3. Installer les dépendances
    pip install dlib-bin face_recognition --no-deps
    pip install face-recognition-models Click Pillow numpy opencv-python mss faiss-cpu
    pip install "setuptools<81"
    pip install "numpy<2"

> **Note** : L'installation de `dlib-bin` évite la compilation fastidieuse de `dlib`. Si vous rencontrez des erreurs, consultez la section "Dépannage" ci-dessous.

---

## 🖥️ Utilisation

### 1. Préparer la base de visages
Créez un dossier `known_faces/` et placez-y des photos claires (1 à 2 par personne). Nommez les fichiers avec le nom de la personne (ex: `christelle_floricourt.jpg`).

### 2. Construire l'index (première fois ou après ajout massif)

    python build_index.py

Cela génère les fichiers `faiss_index.index` et `faiss_names.pkl`.

### 3. Lancer la surveillance

    python recognition.py

### 4. Utiliser le HUD
- **Déplacer** : Cliquez-glissez sur la vidéo.
- **Redimensionner** : Utilisez la molette de la souris.
- **Quitter** : Appuyez sur `Echap`.

---

## 🛠️ Configuration

Les paramètres sont ajustables en haut du fichier `recognition.py` :

| Paramètre | Description | Valeur par défaut |
| :--- | :--- | :--- |
| `MONITOR_REGION` | Zone de l'écran à capturer | `{top: 100, left: 300, width: 1000, height: 600}` |
| `TOLERANCE` | Seuil de reconnaissance (distance L2 FAISS) | `0.45` |
| `DETECTION_MODEL` | `"hog"` (rapide) ou `"cnn"` (précis, lent) | `"hog"` |
| `UPSAMPLE` | Zoom numérique pour petits visages | `1` |
| `ANALYSIS_SCALE` | Réduction de l'image pour l'analyse | `0.5` |
| `SNAPSHOT_COOLDOWN` | Anti-spam pour les captures d'inconnus (secondes) | `30` |

---

## ⚠️ Limitations connues

- Le modèle `hog` détecte principalement les visages **de face** ou légèrement de trois-quarts.
- Les visages très petits (plans de foule) peuvent être manqués si `ANALYSIS_SCALE` est trop bas.
- Le script de scraping Facebook (`scrap.py`) nécessite une connexion manuelle et peut déclencher les protections anti-robot de Meta.
- **Conseil** : Pour de meilleures performances, réduisez `MONITOR_REGION` à la zone utile de l'écran (là où apparaissent les visages). Les visages occuperont une plus grande proportion de l'image, améliorant ainsi la détection sans surcharger le CPU.

---

## 🐛 Dépannage

### `ModuleNotFoundError: No module named 'face_recognition'`
Vous avez oublié d'activer l'environnement virtuel. Exécutez :

    source ~/face_env/bin/activate

### `free(): invalid size` ou crash mémoire
Assurez-vous que `cv2.setNumThreads(0)` est bien présent au début du script, et que le verrou `face_lock` est utilisé autour de tous les appels à `face_recognition` (c'est le cas dans la version actuelle).

### `Index vide ou inexistant`
Lancez `python build_index.py` pour générer l'index FAISS à partir de vos photos.

---

## 📄 Licence

MIT License. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🔗 Liens utiles

- [face_recognition (GitHub)](https://github.com/ageitgey/face_recognition)
- [OpenCV](https://opencv.org/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [Shields.io - Badges](https://shields.io/)

---

### 🇫🇷 Gunout · 2026

© 2026 **Gunout** — Tous droits réservés.

---

<div align="center">

### 🇫🇷 Gunout · 2026

![Made in France](https://img.shields.io/badge/Made_in-France-002395?style=flat-square&labelColor=FFFFFF&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNjAwIj48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjYwMCIgZmlsbD0iIzAwMjM5NSIvPjxyZWN0IHdpZHRoPSI5MDAiIGhlaWdodD0iNDAwIiB5PSIxMDAiIGZpbGw9IiNmZmYiLz48cmVjdCB3aWR0aD0iOTAwIiBoZWlnaHQ9IjIwMCIgeT0iNDAwIiBmaWxsPSIjZWQyOTM5Ii8+PC9zdmc+)
![GitHub](https://img.shields.io/badge/GitHub-gunout-181717?style=flat-square&logo=github&logoColor=white)
![Year](https://img.shields.io/badge/2026-ED2939?style=flat-square&labelColor=FFFFFF)

<sub>© 2026 <strong>Gunout</strong> — Tous droits réservés.</sub>

</div>
