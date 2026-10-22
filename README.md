# 🎯 Recognition Faciale

> Détection et reconnaissance faciale en temps réel sur flux vidéo, avec interface HUD flottante et personnalisable.

[![Security Status](https://img.shields.io/badge/Security-Audited%20Local-brightgreen?style=for-the-badge&logo=shield)](https://github.com/gunout/recognition-faciale)
[![Privacy](https://img.shields.io/badge/Privacy-Local%20Only-blue?style=for-the-badge&logo=linux)](https://github.com/gunout/recognition-faciale)
[![No Cloud](https://img.shields.io/badge/No%20Cloud-100%25%20Offline-critical?style=for-the-badge&logo=cloud-off)](https://github.com/gunout/recognition-faciale)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

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
- **HUD Flottant** : Interface transparente, déplaçable et redimensionnable à la souris.
- **Reconnaissance en temps réel** : Comparaison des visages capturés avec la base locale (`known_faces/`).
- **Optimisé CPU** : Fonctionne sans GPU (architecture multi-threads pour la capture et l'affichage).

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
    pip install face-recognition-models Click Pillow numpy opencv-python mss
    pip install "setuptools<81"
    pip install "numpy<2"

---

## 🖥️ Utilisation

1. **Préparer la base de visages** :
   Créez un dossier `known_faces/` et placez-y des photos claires (1 à 2 par personne). Nommez les fichiers avec le nom de la personne (ex: `christelle_floricourt.jpg`).

2. **Lancer le script** :

       python recognition.py

3. **Utiliser le HUD** :
   - **Déplacer** : Cliquez-glissez sur la vidéo.
   - **Redimensionner** : Utilisez la molette de la souris.
   - **Quitter** : Appuyez sur `Echap`.

---

## 🛠️ Configuration

Les paramètres sont ajustables en haut du fichier `recognition.py` :

| Paramètre | Description | Valeur par défaut |
| :--- | :--- | :--- |
| `MONITOR_REGION` | Zone de l'écran à capturer | `{top, left, width, height}` |
| `TOLERANCE` | Seuil de reconnaissance (0.4 à 0.6) | `0.6` |
| `DETECTION_MODEL` | `"hog"` (rapide) ou `"cnn"` (précis, lent) | `"hog"` |
| `UPSAMPLE` | Zoom numérique pour petits visages | `1` |
| `ANALYSIS_SCALE` | Réduction de l'image pour l'analyse | `0.5` |

---

## ⚠️ Limitations connues

- Le modèle `hog` détecte principalement les visages **de face** ou légèrement de trois-quarts.
- Les visages très petits (plans de foule) peuvent être manqués si `ANALYSIS_SCALE` est trop bas.
- Le script de scraping Facebook (`scrap.py`) nécessite une connexion manuelle et peut déclencher les protections anti-robot de Meta.

---

## 📄 Licence

MIT License. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🔗 Liens utiles

- [face_recognition (GitHub)](https://github.com/ageitgey/face_recognition)
- [OpenCV](https://opencv.org/)
- [Shields.io - Badges](https://shields.io/)
