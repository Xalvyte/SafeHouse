import os
import pickle
from typing import List, Tuple, Dict, Any

import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.neighbors import KNeighborsClassifier
import shutil

# -----------------
# Configuration
# -----------------
FACES_DIR = "faces"
CACHE_FILE = os.path.join(FACES_DIR, "embeddings_cache.pkl")
IMG_SIZE = 160
K_NEIGHBORS = 3
DIST_THRESHOLD = 0.85  # smaller -> stricter; tuned lightly for vggface2
VALID_EXTS = (".jpg", ".jpeg", ".png")

# CPU for Pi stability
DEVICE = torch.device("cpu")


class recognize_faces:
    def __init__(self, faces_dir: str = FACES_DIR):
        self.faces_dir = faces_dir
        os.makedirs(self.faces_dir, exist_ok=True)

        # Models
        self.mtcnn = MTCNN(image_size=IMG_SIZE, margin=20)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

        # Data
        self.embeddings: np.ndarray = np.empty((0, 512))
        self.labels: List[str] = []
        self.knn: KNeighborsClassifier | None = None

        # Load cache if present
        self._load_cache()
        self._fit_knn()

    # ---------- Cache utils ----------
    def _load_cache(self) -> None:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    self.embeddings, self.labels = pickle.load(f)
                print(f"[INFO] Loaded cached embeddings: {self.embeddings.shape} vectors, {len(self.labels)} labels")
            except Exception as e:
                print(f"[WARN] Failed to load cache: {e}. Rebuilding...")
                self.rebuild_cache()
        else:
            print("[INFO] No cache found. Building from faces directory...")
            self.rebuild_cache()

    def _save_cache(self) -> None:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((self.embeddings, self.labels), f)
        print("[INFO] Cached embeddings saved.")

    def _fit_knn(self) -> None:
        if self.embeddings.shape[0] >= max(1, K_NEIGHBORS):
            self.knn = KNeighborsClassifier(n_neighbors=min(K_NEIGHBORS, len(set(self.labels)) or 1))
            self.knn.fit(self.embeddings, self.labels)
            print("[INFO] kNN fitted.")
        else:
            self.knn = None
            print("[WARN] Not enough embeddings to fit kNN yet.")

    # ---------- Build/Rebuild ----------
    def rebuild_cache(self) -> None:
        all_embeddings = []
        all_labels: List[str] = []

        for person in sorted(os.listdir(self.faces_dir)):
            person_dir = os.path.join(self.faces_dir, person)
            if not os.path.isdir(person_dir):
                continue

            for fname in sorted(os.listdir(person_dir)):
                if not fname.lower().endswith(VALID_EXTS):
                    continue
                path = os.path.join(person_dir, fname)
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    print(f"[WARN] Could not read image: {path}")
                    continue

                # Convert to RGB and ensure correct dtype
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = img_rgb.astype(np.uint8)  # ensure uint8
                img_pil = Image.fromarray(img_rgb).convert("RGB")

                try:
                    face_tensor = self.mtcnn(img_pil)
                except Exception as e:
                    print(f"[WARN] MTCNN failed on {path}: {e}")
                    face_tensor = None

                if face_tensor is None:
                    # fallback center crop
                    h, w = img_rgb.shape[:2]
                    s = min(h, w)
                    y0 = (h - s) // 2
                    x0 = (w - s) // 2
                    crop = cv2.resize(img_rgb[y0:y0+s, x0:x0+s], (IMG_SIZE, IMG_SIZE))
                    face_tensor = torch.from_numpy(crop).permute(2,0,1).float() / 255.0

                if face_tensor.ndim == 3:
                    face_tensor = face_tensor.unsqueeze(0)

                with torch.no_grad():
                    emb = self.facenet(face_tensor.to(DEVICE)).cpu().numpy()
                all_embeddings.append(emb[0])
                all_labels.append(person)


        self.embeddings = np.array(all_embeddings) if all_embeddings else np.empty((0, 512))
        self.labels = all_labels
        self._save_cache()
        self._fit_knn()

    # ---------- Inference ----------
    def recognize_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        detections: List[Dict[str, Any]] = []
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        boxes, probs = self.mtcnn.detect(img_pil)
        if boxes is None or len(boxes) == 0:
            return frame_bgr, detections

        faces = []
        bboxes = []
        for (x1, y1, x2, y2), p in zip(boxes, probs):
            if p is None:
                continue
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            crop = img_rgb[max(0, y1i):max(0, y2i), max(0, x1i):max(0, x2i)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            faces.append(tensor)
            bboxes.append((x1i, y1i, x2i, y2i))

        if not faces:
            return frame_bgr, detections

        face_batch = torch.stack(faces, dim=0)
        with torch.no_grad():
            embs = self.facenet(face_batch.to(DEVICE)).cpu().numpy()

        for emb, (x1, y1, x2, y2) in zip(embs, bboxes):
            label, conf = self._predict_label(emb)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "confidence": float(conf),
            })
            color = (0, 200, 0) if label != "Unknown" else (50, 50, 200)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, f"{label} ({conf:.2f})", (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        return frame_bgr, detections

    def _predict_label(self, emb: np.ndarray) -> Tuple[str, float]:
        if self.knn is None or self.embeddings.shape[0] == 0:
            return "Unknown", 0.0
        distances, indices = self.knn.kneighbors([emb], n_neighbors=1)
        d = float(distances[0][0])
        pred = self.knn.predict([emb])[0]
        if d < DIST_THRESHOLD:
            conf = max(0.0, min(1.0, 1.0 - d))
            return str(pred), conf
        return "Unknown", max(0.0, 1.0 - d)

    # ---------- Add new user ----------
    def add_new_user(self, username: str, image_paths: list[str]) -> None:
        """Add a new user from a list of image file paths and update embeddings/cache incrementally."""
        new_embeddings = []
        for path in image_paths:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print(f"[WARN] Could not read image: {path}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).convert("RGB")
            face_tensor = self.mtcnn(img_pil)
            if face_tensor is None:
                # fallback center crop
                h, w = img_rgb.shape[:2]
                s = min(h, w)
                y0 = (h - s)//2
                x0 = (w - s)//2
                crop = cv2.resize(img_rgb[y0:y0+s, x0:x0+s], (IMG_SIZE, IMG_SIZE))
                face_tensor = torch.from_numpy(crop).permute(2,0,1).float()/255.0

            if face_tensor.ndim == 3:
                face_tensor = face_tensor.unsqueeze(0)

            with torch.no_grad():
                emb = self.facenet(face_tensor.to(DEVICE)).cpu().numpy()
            new_embeddings.append(emb[0])

        if not new_embeddings:
            print(f"[WARN] No valid embeddings for {username}, skipping.")
            return

        self.embeddings = np.vstack([self.embeddings, new_embeddings]) if self.embeddings.size else np.array(new_embeddings)
        self.labels.extend([username]*len(new_embeddings))

        self._save_cache()
        self._fit_knn()
        print(f"[INFO] Added {username} with {len(new_embeddings)} embeddings.")
        
        # ---------- Utility ----------
    def list_users(self) -> list[str]:
        """
        Returns a sorted list of unique user labels currently known.
        """
        return sorted(set(self.labels))

    def sync_faces_to_static(self):
        static_faces_dir = os.path.join("static", "faces")
        os.makedirs(static_faces_dir, exist_ok=True)

        for person in os.listdir(self.faces_dir):
            person_dir = os.path.join(self.faces_dir, person)
            if not os.path.isdir(person_dir):
                continue

            target_dir = os.path.join(static_faces_dir, person)
            os.makedirs(target_dir, exist_ok=True)

            for fname in os.listdir(person_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    src = os.path.join(person_dir, fname)
                    dst = os.path.join(target_dir, fname)
                    if not os.path.exists(dst):  # don’t overwrite every time
                        shutil.copy2(src, dst)

    def delete_user_from_static(self, username: str, static_dir: str = "static/faces") -> None:
        """
        Deletes a user folder from static when they are removed.
        """
        target_dir = os.path.join(static_dir, username)
        if os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
                print(f"[INFO] Deleted {username} from static.")
            except Exception as e:
                print(f"[WARN] Could not delete {target_dir}: {e}")

