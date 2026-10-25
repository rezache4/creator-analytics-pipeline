# Thumbnail Feature Extraction Plan

**Scope:** Extract interpretable visual features from YouTube thumbnail images for use as predictors in the engagement regression model.  
**Input:** `thumbnails/{video_id}.jpg` — downloaded at `hqdefault` resolution (480×360), ~11,801 images  
**Output:** `thumbnail_features.parquet` — one row per video, all features as scalars or one-hot dummies

Note: CLIP-based embedding and clustering (Phase 5 of `ANALYSIS_PLAN.md`) is a separate pipeline that produces a different class of feature (latent visual representation). This document covers classical and deep-learning-based feature extraction that yields directly interpretable scalars.

---

## Feature Groups

### 1. Colour

Colour is the fastest signal a thumbnail communicates and has documented effects on click-through rate in the platform literature.

| Feature | Method | Output |
|---------|--------|--------|
| `brightness_mean` | Convert to HSV; mean of V channel | float [0, 1] |
| `saturation_mean` | Mean of S channel in HSV | float [0, 1] |
| `contrast` | Standard deviation of L channel in LAB space | float |
| `warm_cool_ratio` | Ratio of red+yellow pixel mass to blue+green pixel mass in RGB | float |
| `dominant_hue_1/2/3` | K-means (k=3) on H channel; cluster centres | float × 3 |
| `dominant_hue_entropy` | Shannon entropy of the hue histogram (256 bins) | float |

Library: `opencv-python` (`cv2`), `numpy`.

---

### 2. Face Detection

Face presence is one of the strongest predictors of thumbnail click-through rate reported in the YouTube literature. Multiple faces, emotional expressions, and face position all carry independent signal.

| Feature | Method | Output |
|---------|--------|--------|
| `face_count` | MediaPipe Face Detection (model 1, short-range) | int |
| `face_present` | Binary indicator | 0/1 |
| `face_area_ratio` | Sum of bounding box areas / image area | float [0, 1] |
| `face_position_x/y` | Centroid of largest face, normalised to [0, 1] | float × 2 |
| `face_center_distance` | Euclidean distance of largest face centroid from image centre | float |

Library: `mediapipe` (`mp.solutions.face_detection`). Preferred over OpenCV Haar cascades because it handles partial faces, varied angles, and non-frontal poses — common in expressive thumbnails.

---

### 3. Facial Emotion

Emotional valence on a face is a distinct signal from face presence. Surprise, excitement, and disgust are associated with higher click-through in entertainment content; neutral expressions dominate instructional content.

| Feature | Method | Output |
|---------|--------|--------|
| `emotion_dominant` | DeepFace (`DeepFace.analyze`, action=`emotion`) on largest detected face | categorical |
| `emotion_*` (7 dummies) | angry / disgust / fear / happy / sad / surprise / neutral | 0/1 × 7 |
| `valence` | Composite: happy + surprise − angry − disgust − fear − sad | float |

Library: `deepface`. Falls back to `None` for images with no detected face; these receive `emotion_dominant = "no_face"` and all dummies = 0.

**Note:** Emotion detection is applied only to the largest detected face per image to avoid ambiguity in multi-face thumbnails.

---

### 4. Text Presence and Density

Overlaid text on thumbnails (title repetition, emphasis words, numbers) is a deliberate editorial decision with measurable engagement effects.

| Feature | Method | Output |
|---------|--------|--------|
| `text_present` | EasyOCR (`easyocr.Reader(['it', 'en'])`) | 0/1 |
| `text_area_ratio` | Sum of OCR bounding box areas / image area | float [0, 1] |
| `text_word_count` | Number of words detected | int |
| `text_contains_number` | Regex `\d+` over OCR output | 0/1 |

Library: `easyocr`. Italian and English are both loaded since the corpus is Italian-language with frequent English loanwords in thumbnail text.

**Note:** OCR on compressed 480×360 JPEG is imperfect for small text. `text_present` and `text_area_ratio` are more reliable than the exact word count.

---

### 5. Composition

Compositional choices — where the subject sits in the frame, how edges are distributed — are correlates of production professionalism and genre conventions.

| Feature | Method | Output |
|---------|--------|--------|
| `edge_density` | Canny edge detection; fraction of edge pixels | float [0, 1] |
| `visual_complexity` | Standard deviation of pixel intensities in grayscale | float |
| `rule_of_thirds_score` | Sum of edge pixel mass within ±10% of the four rule-of-thirds lines, normalised | float [0, 1] |
| `left_right_asymmetry` | Absolute difference in mean brightness between left and right halves | float |
| `top_bottom_asymmetry` | Absolute difference in mean brightness between top and bottom halves | float |

Library: `cv2` (Canny, grayscale conversion, histogram operations).

---

### 6. Cross-Modal Alignment (CLIP)

Covered in `ANALYSIS_PLAN.md` Phase 5D. Produces `thumbnail_title_alignment` — cosine similarity between the CLIP visual embedding of the thumbnail and the CLIP text embedding of the video title. Recorded here for completeness as it logically belongs alongside the other thumbnail features in the final regression feature matrix.

---

## Processing Pipeline

```python
# thumbnail_features.py  (sketch — full implementation pending)
import cv2, numpy as np, pandas as pd, mediapipe as mp, easyocr, deepface
from pathlib import Path

THUMB_DIR = Path("thumbnails")
reader    = easyocr.Reader(['it', 'en'], gpu=False)
mp_face   = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                       min_detection_confidence=0.4)

rows = []
for img_path in THUMB_DIR.glob("*.jpg"):
    video_id = img_path.stem
    img_bgr  = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]

    # ── Colour ────────────────────────────────────────────────────────────────
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float) / 255
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    row  = {
        "video_id":           video_id,
        "brightness_mean":    hsv[:, :, 2].mean(),
        "saturation_mean":    hsv[:, :, 1].mean(),
        "contrast":           lab[:, :, 0].std(),
        "warm_cool_ratio":    _warm_cool(img_bgr),
        "dominant_hue_entropy": _hue_entropy(hsv[:, :, 0]),
    }

    # ── Faces ─────────────────────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = mp_face.process(img_rgb)
    faces   = result.detections or []
    row["face_count"]   = len(faces)
    row["face_present"] = int(len(faces) > 0)
    if faces:
        bb = faces[0].location_data.relative_bounding_box
        row["face_area_ratio"]      = bb.width * bb.height
        row["face_position_x"]      = bb.xmin + bb.width / 2
        row["face_position_y"]      = bb.ymin + bb.height / 2
        row["face_center_distance"] = np.hypot(row["face_position_x"] - 0.5,
                                               row["face_position_y"] - 0.5)
    else:
        row.update(face_area_ratio=0, face_position_x=None,
                   face_position_y=None, face_center_distance=None)

    # ── Emotion ───────────────────────────────────────────────────────────────
    if row["face_present"]:
        try:
            analysis = deepface.DeepFace.analyze(img_rgb, actions=["emotion"],
                                                 enforce_detection=False, silent=True)
            emotions = analysis[0]["emotion"]   # dict of float scores
            dom      = max(emotions, key=emotions.get)
            row["emotion_dominant"] = dom
            row["valence"] = (emotions.get("happy", 0) + emotions.get("surprise", 0)
                              - emotions.get("angry",   0) - emotions.get("disgust", 0)
                              - emotions.get("fear",    0) - emotions.get("sad",     0))
            for e in ["angry","disgust","fear","happy","sad","surprise","neutral"]:
                row[f"emotion_{e}"] = int(dom == e)
        except Exception:
            row["emotion_dominant"] = "no_face"
            row["valence"] = None
    else:
        row["emotion_dominant"] = "no_face"
        row["valence"] = None

    # ── Text ──────────────────────────────────────────────────────────────────
    ocr_result = reader.readtext(img_bgr)
    text_area  = sum((x2 - x1) * (y2 - y1)
                     for bbox, _, _ in ocr_result
                     for (x1, y1), _, (x2, y2), _ in [(*bbox,)])
    ocr_text   = " ".join(t for _, t, _ in ocr_result)
    row["text_present"]        = int(len(ocr_result) > 0)
    row["text_area_ratio"]     = text_area / (w * h)
    row["text_word_count"]     = len(ocr_text.split())
    row["text_contains_number"]= int(bool(__import__("re").search(r"\d+", ocr_text)))

    # ── Composition ───────────────────────────────────────────────────────────
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    row["edge_density"]         = edges.mean() / 255
    row["visual_complexity"]    = float(gray.std())
    row["rule_of_thirds_score"] = _rule_of_thirds(edges, h, w)
    row["left_right_asymmetry"] = abs(gray[:, :w//2].mean() - gray[:, w//2:].mean()) / 255
    row["top_bottom_asymmetry"] = abs(gray[:h//2].mean() - gray[h//2:].mean()) / 255

    rows.append(row)

pd.DataFrame(rows).to_parquet("thumbnail_features.parquet", index=False)
```

Helper functions (`_warm_cool`, `_hue_entropy`, `_rule_of_thirds`) are omitted from the sketch for brevity but are straightforward numpy operations.

---

## Runtime and Dependencies

| Library | Install | Notes |
|---------|---------|-------|
| `opencv-python` | `pip install opencv-python` | Colour, composition, edge detection |
| `mediapipe` | `pip install mediapipe` | Face detection |
| `deepface` | `pip install deepface` | Emotion; downloads model weights on first run (~200 MB) |
| `easyocr` | `pip install easyocr` | Text detection; downloads models on first run (~500 MB) |

**Estimated runtime:** OCR and emotion recognition dominate. On CPU: ~2–4 seconds per image → ~7–13 hours for 11,801 images. Run overnight or parallelise with `multiprocessing.Pool`. No GPU required, but GPU accelerates EasyOCR significantly.

---

## Output Schema

`thumbnail_features.parquet` — one row per video:

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | |
| `brightness_mean` | float | |
| `saturation_mean` | float | |
| `contrast` | float | Std dev of LAB L channel |
| `warm_cool_ratio` | float | |
| `dominant_hue_entropy` | float | |
| `face_count` | int | |
| `face_present` | 0/1 | |
| `face_area_ratio` | float | |
| `face_position_x/y` | float | Normalised [0, 1]; null if no face |
| `face_center_distance` | float | Null if no face |
| `emotion_dominant` | string | One of 7 emotions or "no_face" |
| `emotion_{angry,…,neutral}` | 0/1 × 7 | |
| `valence` | float | Null if no face |
| `text_present` | 0/1 | |
| `text_area_ratio` | float | |
| `text_word_count` | int | |
| `text_contains_number` | 0/1 | |
| `edge_density` | float | |
| `visual_complexity` | float | |
| `rule_of_thirds_score` | float | |
| `left_right_asymmetry` | float | |
| `top_bottom_asymmetry` | float | |

Join on `video_id` with `thumbnail_clusters.parquet` (CLIP-based clusters, Phase 5C) and `cross_modal.parquet` (Phase 5D) to assemble the full thumbnail feature set for regression.
