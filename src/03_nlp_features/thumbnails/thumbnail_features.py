#!/usr/bin/env python3
"""
thumbnail_features.py — GCP Batch task entry point.

Each task slices 1/N of all thumbnails in GCS (interleaved), extracts
interpretable visual features, and writes a parquet shard to GCS.
Idempotent: if the output shard already exists the task exits cleanly, so
spot-VM preemptions and retries are safe.

Feature groups:
  colour    : brightness_mean, saturation_mean
  face      : count, area_ratio, position, center_distance
              (MediaPipe FaceDetection)
  face_pose : yaw, pitch, roll, is_frontal
              (MediaPipe FaceMesh + cv2.solvePnP)
  face_expr : mouth_openness, eye_openness_mean, gaze_off_camera
              (MediaPipe FaceMesh landmarks)
  text      : present, area_ratio, word_count, has_number  (EasyOCR it+en)
  clip      : clip_title_align — cosine similarity between the thumbnail
              image embedding and the video title text embedding
              (CLIP ViT-L/14, cpu)

GCS layout:
  INPUT    gs://{GCS_BUCKET}/thumbnails/{video_id}.jpg
  META     gs://{GCS_BUCKET}/metadata/yt_videos_metadata.parquet
  OUTPUT   gs://{GCS_BUCKET}/output/thumbnail_features/part_{N:04d}.parquet

Env vars (GCP Batch injects BATCH_TASK_INDEX / BATCH_TASK_COUNT automatically):
  BATCH_TASK_INDEX   0-based task index        (default 0)
  BATCH_TASK_COUNT   total task count          (default 1)
  GCS_BUCKET         bucket name               (default socialmediaanalyticsproject)
  THUMB_PREFIX       GCS prefix for images     (default thumbnails/)
  OUT_PREFIX         GCS prefix for outputs    (default output/thumbnail_features/)
  META_BLOB          GCS path to metadata      (default metadata/yt_videos_metadata.parquet)
"""
import io
import logging
import os
import sys
import warnings

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from google.cloud import storage
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Task config ───────────────────────────────────────────────────────────────
TASK_INDEX   = int(os.environ.get("CLOUD_RUN_TASK_INDEX",  os.environ.get("BATCH_TASK_INDEX",  "0")))
TASK_COUNT   = int(os.environ.get("CLOUD_RUN_TASK_COUNT", os.environ.get("BATCH_TASK_COUNT", "1")))
GCS_BUCKET   = os.environ.get("GCS_BUCKET",   "socialmediaanalyticsproject")
THUMB_PREFIX = os.environ.get("THUMB_PREFIX", "thumbnails/")
OUT_PREFIX   = os.environ.get("OUT_PREFIX",   "output/thumbnail_features/")
META_BLOB    = os.environ.get("META_BLOB",    "metadata/yt_videos_metadata.parquet")

# ── MediaPipe (module-level: paid once per container) ─────────────────────────
log.info("Initialising MediaPipe models...")
_face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.4
)
_face_mesher = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
)

# ── CLIP (lazy — loaded on first use to keep startup fast) ────────────────────
_clip_model = None
_clip_proc  = None

def _get_clip():
    global _clip_model, _clip_proc
    if _clip_model is None:
        log.info("Loading CLIP ViT-L/14...")
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
        _clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return _clip_model, _clip_proc

# ── EasyOCR (lazy) ─────────────────────────────────────────────────────────────
_ocr_reader = None

def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        log.info("Loading EasyOCR (it, en)...")
        import easyocr
        _ocr_reader = easyocr.Reader(["it", "en"], gpu=False, verbose=False)
    return _ocr_reader

# ── FaceMesh landmark constants ────────────────────────────────────────────────
# 6-point head pose via solvePnP (nose, chin, l-eye-outer, r-eye-outer, l-mouth, r-mouth)
_POSE_IDX = [1, 152, 33, 263, 61, 291]
_FACE_3D  = np.array([
    [  0.0,    0.0,   0.0],   # nose tip
    [  0.0, -330.0, -65.0],   # chin
    [-225.0,  170.0,-135.0],  # left eye outer corner
    [ 225.0,  170.0,-135.0],  # right eye outer corner
    [-150.0, -150.0,-125.0],  # left mouth corner
    [ 150.0, -150.0,-125.0],  # right mouth corner
], dtype=np.float64)

# Eye landmarks
_LE_TOP, _LE_BOT = 159, 145   # left  eyelid top / bottom
_RE_TOP, _RE_BOT = 386, 374   # right eyelid top / bottom
_LE_INN, _LE_OUT = 133, 33    # left  eye inner / outer corner
_RE_INN, _RE_OUT = 362, 263   # right eye inner / outer corner

# Iris centres — available when refine_landmarks=True (indices 468, 473)
_L_IRIS, _R_IRIS = 468, 473

# Mouth
_ULIP, _LLIP = 13, 14    # upper / lower lip centre
_ML,   _MR   = 61, 291   # left / right mouth corner


def _null_face_row() -> dict:
    return {
        "face_count":           0,
        "face_present":         False,
        "face_area_ratio":      float("nan"),
        "face_position_x":      float("nan"),
        "face_position_y":      float("nan"),
        "face_center_distance": float("nan"),
        "face_yaw_deg":         float("nan"),
        "face_pitch_deg":       float("nan"),
        "face_roll_deg":        float("nan"),
        "face_is_frontal":      float("nan"),
        "mouth_openness":       float("nan"),
        "eye_openness_mean":    float("nan"),
        "gaze_off_camera":      float("nan"),
    }


# ── Feature extractors ─────────────────────────────────────────────────────────

def extract_colour(bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return {
        "brightness_mean": float(hsv[:, :, 2].mean() / 255.0),
        "saturation_mean": float(hsv[:, :, 1].mean() / 255.0),
    }


def extract_face(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    det = _face_detector.process(rgb)
    if not det.detections:
        return _null_face_row()

    feats: dict = {"face_count": len(det.detections), "face_present": True}

    # Largest face by bounding-box area
    largest = max(
        det.detections,
        key=lambda d: (d.location_data.relative_bounding_box.width *
                       d.location_data.relative_bounding_box.height),
    )
    bb = largest.location_data.relative_bounding_box
    cx = bb.xmin + bb.width  / 2.0
    cy = bb.ymin + bb.height / 2.0

    feats["face_area_ratio"]       = float(bb.width * bb.height)
    feats["face_position_x"]       = float(cx)
    feats["face_position_y"]       = float(cy)
    feats["face_center_distance"]  = float(np.hypot(cx - 0.5, cy - 0.5))

    # Crop with 15 % padding for more stable FaceMesh
    pad = 0.15
    x1  = max(0, int((bb.xmin - pad * bb.width)          * w))
    y1  = max(0, int((bb.ymin - pad * bb.height)         * h))
    x2  = min(w, int((bb.xmin + (1 + pad) * bb.width)   * w))
    y2  = min(h, int((bb.ymin + (1 + pad) * bb.height)  * h))
    crop = rgb[y1:y2, x1:x2]

    nan_mesh = {k: float("nan") for k in [
        "face_yaw_deg", "face_pitch_deg", "face_roll_deg",
        "face_is_frontal", "mouth_openness", "eye_openness_mean", "gaze_off_camera",
    ]}

    if crop.size == 0:
        feats.update(nan_mesh)
        return feats

    mesh = _face_mesher.process(crop)
    if not mesh.multi_face_landmarks:
        feats.update(nan_mesh)
        return feats

    lm   = mesh.multi_face_landmarks[0].landmark
    ch, cw = crop.shape[:2]

    def pt(i: int) -> np.ndarray:
        return np.array([lm[i].x * cw, lm[i].y * ch], dtype=np.float64)

    # ── Head pose (ZYX Euler from solvePnP) ──────────────────────────────────
    pts2d   = np.array([pt(i) for i in _POSE_IDX])
    focal   = float(cw)
    cam_mat = np.array([[focal, 0, cw / 2],
                        [0, focal, ch / 2],
                        [0,     0,      1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _FACE_3D, pts2d, cam_mat, np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP
    )
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        sy   = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if sy > 1e-6:
            pitch = float(np.degrees(np.arctan2( R[2, 1],  R[2, 2])))
            yaw   = float(np.degrees(np.arctan2(-R[2, 0],  sy)))
            roll  = float(np.degrees(np.arctan2( R[1, 0],  R[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
            yaw   = float(np.degrees(np.arctan2(-R[2, 0], sy)))
            roll  = 0.0
        feats["face_yaw_deg"]    = yaw
        feats["face_pitch_deg"]  = pitch
        feats["face_roll_deg"]   = roll
        feats["face_is_frontal"] = bool(abs(yaw) < 25.0 and abs(pitch) < 20.0)
    else:
        feats.update({
            "face_yaw_deg": float("nan"), "face_pitch_deg": float("nan"),
            "face_roll_deg": float("nan"), "face_is_frontal": float("nan"),
        })

    # ── Mouth openness (vertical gap / mouth width) ───────────────────────────
    mouth_v = float(np.linalg.norm(pt(_ULIP) - pt(_LLIP)))
    mouth_h = float(np.linalg.norm(pt(_ML)   - pt(_MR)))
    feats["mouth_openness"] = mouth_v / mouth_h if mouth_h > 0 else float("nan")

    # ── Eye openness (vertical gap / horizontal span, mean of both eyes) ─────
    def _eye_open(top: int, bot: int, inn: int, out: int) -> float:
        v = float(np.linalg.norm(pt(top) - pt(bot)))
        h = float(np.linalg.norm(pt(inn) - pt(out)))
        return v / h if h > 0 else float("nan")

    feats["eye_openness_mean"] = float(np.nanmean([
        _eye_open(_LE_TOP, _LE_BOT, _LE_INN, _LE_OUT),
        _eye_open(_RE_TOP, _RE_BOT, _RE_INN, _RE_OUT),
    ]))

    # ── Gaze direction (iris displacement from eye centre, normalised) ────────
    # refine_landmarks=True adds iris landmarks 468 (left) and 473 (right).
    # If the iris is displaced >15 % of eye width from the eye midpoint the
    # person is not making direct eye contact with the camera.
    try:
        l_cx = (lm[_LE_INN].x + lm[_LE_OUT].x) / 2.0
        r_cx = (lm[_RE_INN].x + lm[_RE_OUT].x) / 2.0
        l_ew = abs(lm[_LE_INN].x - lm[_LE_OUT].x) or 1e-6
        r_ew = abs(lm[_RE_INN].x - lm[_RE_OUT].x) or 1e-6
        disp = (abs(lm[_L_IRIS].x - l_cx) / l_ew +
                abs(lm[_R_IRIS].x - r_cx) / r_ew) / 2.0
        feats["gaze_off_camera"] = bool(disp > 0.15)
    except (IndexError, AttributeError):
        feats["gaze_off_camera"] = float("nan")

    return feats


def extract_text(bgr: np.ndarray) -> dict:
    h, w     = bgr.shape[:2]
    img_area = h * w
    _null    = {"text_present": False, "text_area_ratio": 0.0,
                "text_word_count": 0,  "text_has_number": False}
    try:
        results = _get_ocr().readtext(bgr, detail=1, paragraph=False)
    except Exception:
        return _null

    if not results:
        return _null

    text_area  = 0
    words: list[str] = []
    has_number = False
    for bbox, text, _conf in results:
        pts  = np.array(bbox, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        text_area += bw * bh
        words.extend(text.split())
        if any(c.isdigit() for c in text):
            has_number = True

    return {
        "text_present":    True,
        "text_area_ratio": float(min(text_area / img_area, 1.0)),
        "text_word_count": len(words),
        "text_has_number": has_number,
    }


@torch.no_grad()
def extract_clip_align(bgr: np.ndarray, title: str) -> float:
    if not title:
        return float("nan")
    model, proc = _get_clip()
    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    img_in  = proc(images=pil_img, return_tensors="pt")
    txt_in  = proc(
        text=[title], return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    )
    img_emb = model.get_image_features(**img_in)
    txt_emb = model.get_text_features(**txt_in)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return float((img_emb * txt_emb).sum().item())


def extract_all(video_id: str, img_bytes: bytes, title: str) -> dict:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"video_id": video_id, "error": "decode_failed"}
    row = {"video_id": video_id}
    row.update(extract_colour(bgr))
    row.update(extract_face(bgr))
    row.update(extract_text(bgr))
    row["clip_title_align"] = extract_clip_align(bgr, title)
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gcs    = storage.Client()
    bucket = gcs.bucket(GCS_BUCKET)

    # Load videoId → title mapping
    meta_bytes = bucket.blob(META_BLOB).download_as_bytes()
    meta_df    = pd.read_parquet(io.BytesIO(meta_bytes), columns=["videoId", "title"])
    title_map  = dict(zip(meta_df["videoId"], meta_df["title"].fillna("")))
    log.info("Loaded %d title mappings from %s", len(title_map), META_BLOB)

    # List all thumbnail blobs
    all_blobs = sorted(
        b.name for b in bucket.list_blobs(prefix=THUMB_PREFIX)
        if b.name.endswith(".jpg")
    )
    if not all_blobs:
        log.error("No .jpg files found under gs://%s/%s", GCS_BUCKET, THUMB_PREFIX)
        sys.exit(1)
    log.info("Total thumbnails: %d", len(all_blobs))

    # Interleaved slice so load is balanced across tasks
    my_blobs = all_blobs[TASK_INDEX::TASK_COUNT]
    log.info("Task %d/%d → %d thumbnails", TASK_INDEX, TASK_COUNT, len(my_blobs))

    # Idempotent: skip if shard already written (covers spot-VM preemptions)
    out_name = f"{OUT_PREFIX}part_{TASK_INDEX:04d}.parquet"
    if bucket.blob(out_name).exists():
        log.info("Shard already exists — skipping: gs://%s/%s", GCS_BUCKET, out_name)
        return

    rows: list[dict] = []
    failed = 0
    for i, blob_name in enumerate(my_blobs):
        video_id = blob_name.removeprefix(THUMB_PREFIX).removesuffix(".jpg")
        title    = title_map.get(video_id, "")
        try:
            img_bytes = bucket.blob(blob_name).download_as_bytes()
            row       = extract_all(video_id, img_bytes, title)
        except Exception as exc:
            log.warning("Failed %s: %s", video_id, exc)
            row = {"video_id": video_id, "error": str(exc)}
            failed += 1
        rows.append(row)
        if (i + 1) % 50 == 0:
            log.info("  %d / %d processed", i + 1, len(my_blobs))

    log.info("Done: %d ok, %d failed", len(rows) - failed, failed)

    df  = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    bucket.blob(out_name).upload_from_file(buf, content_type="application/octet-stream")
    log.info("Wrote %d rows → gs://%s/%s", len(df), GCS_BUCKET, out_name)


if __name__ == "__main__":
    main()
