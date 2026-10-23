# build_index.py
import face_recognition
import numpy as np
import os
import pickle
import faiss

KNOWN_FACES_DIR = "known_faces"
INDEX_FILE = "faiss_index.index"
NAMES_FILE = "faiss_names.pkl"

def build_index():
    print("[BUILD] Démarrage de la génération de l'index...")
    encodings = []
    names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[ERREUR] Le dossier '{KNOWN_FACES_DIR}' n'existe pas.")
        return

    files = [f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(files)
    
    for i, filename in enumerate(files):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        print(f"[{i+1}/{total}] Encodage de {filename}...")
        try:
            image = face_recognition.load_image_file(filepath)
            encs = face_recognition.face_encodings(image)
            if len(encs) > 0:
                encodings.append(encs[0]) # On ne garde que le premier visage
                base_name = os.path.splitext(filename)[0].split('_')[0]
                names.append(base_name)
        except Exception as e:
            print(f"  ⚠️ Erreur sur {filename} : {e}")
    
    if not encodings:
        print("[ERREUR] Aucun visage valide trouvé.")
        return
    
    # Conversion en float32 pour FAISS
    encodings_np = np.array(encodings).astype('float32')
    
    # Création de l'index (IndexFlatL2 = recherche par distance euclidienne)
    index = faiss.IndexFlatL2(encodings_np.shape[1])
    index.add(encodings_np)
    
    # Sauvegarde
    faiss.write_index(index, INDEX_FILE)
    with open(NAMES_FILE, 'wb') as f:
        pickle.dump(names, f)
    
    print(f"✅ Index FAISS créé avec {len(names)} visages. Fichiers : {INDEX_FILE}, {NAMES_FILE}")

if __name__ == "__main__":
    build_index()