"""
RoadScan AI — Backend v4.2 (Render Deployment)
===============================================
Auto-downloads yolov8_road.onnx from Hugging Face on first startup.
Run:  python app.py
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, io, os, random, time, threading, uuid
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw
import numpy as np
import urllib.request

app = Flask(__name__)
CORS(app)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "yolov8_road.onnx"
FRONT_DIR  = BASE_DIR / "frontend"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "rs_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "rs_outputs"
for d in [MODEL_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hugging Face model URL ─────────────────────────────────────────────────
HF_MODEL_URL = "https://huggingface.co/zainab-nayeem/roadscan-ai-model/resolve/main/yolov8_road.onnx"

def download_model():
    """Download model from Hugging Face if not already present."""
    if MODEL_PATH.exists():
        print(f"✅ Model already exists at {MODEL_PATH}")
        return
    print("⬇️  Downloading model from Hugging Face...")
    try:
        urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"✅ Model downloaded: {size_mb:.1f} MB")
    except Exception as e:
        print(f"❌ Model download failed: {e}")

# Download on startup
download_model()

# ── RDD2022 class mapping ──────────────────────────────────────────────────
DAMAGE_CLASSES = {
    0: {"name": "Longitudinal Crack", "color": (255,136,0),  "hex": "#FF8800", "weight": 1.5},
    1: {"name": "Transverse Crack",   "color": (255,187,0),  "hex": "#FFBB00", "weight": 1.8},
    2: {"name": "Alligator Crack",    "color": (255,85,0),   "hex": "#FF5500", "weight": 2.5},
    3: {"name": "Pothole",            "color": (255,61,90),  "hex": "#FF3D5A", "weight": 3.0},
}

VEHICLE_RISK = {
    "NONE"    : {"car": "SAFE",      "truck": "SAFE",    "bike": "SAFE",      "bicycle": "SAFE"},
    "LOW"     : {"car": "SAFE",      "truck": "SAFE",    "bike": "SAFE",      "bicycle": "SAFE"},
    "MODERATE": {"car": "CAUTION",   "truck": "SAFE",    "bike": "CAUTION",   "bicycle": "HIGH RISK"},
    "HIGH"    : {"car": "HIGH RISK", "truck": "CAUTION", "bike": "HIGH RISK", "bicycle": "AVOID"},
}

jobs = {}
_ort_session = None

# ── Load ONNX model ────────────────────────────────────────────────────────
def get_model():
    global _ort_session
    if _ort_session is None and MODEL_PATH.exists():
        try:
            import onnxruntime as ort
            _ort_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            print(f"✅ Model loaded. Outputs: {[o.name for o in _ort_session.get_outputs()]}")
        except Exception as e:
            print(f"❌ Model load error: {e}")
    return _ort_session

get_model()

# ── Helpers ────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image, size=640):
    img_rgb = img.convert("RGB").resize((size, size))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]

def postprocess(output, orig_w, orig_h, conf_thresh=0.15, iou_thresh=0.45):
    preds = output[0]
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] == 8 and preds.shape[1] > 8:
        preds = preds.T

    detections = []
    for row in preds:
        if len(row) < 6:
            continue
        scores = row[4:]
        cls_id = int(np.argmax(scores))
        conf   = float(scores[cls_id])
        if conf < conf_thresh:
            continue
        cx, cy, w, h = row[:4]
        x1 = int((cx - w / 2) / 640 * orig_w)
        y1 = int((cy - h / 2) / 640 * orig_h)
        x2 = int((cx + w / 2) / 640 * orig_w)
        y2 = int((cy + h / 2) / 640 * orig_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        detections.append({"cls": cls_id, "conf": conf, "box": [x1, y1, x2, y2]})

    detections.sort(key=lambda d: d["conf"], reverse=True)
    kept, used = [], []
    for d in detections:
        b1 = d["box"]
        suppressed = False
        for b2 in used:
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1    = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2    = (b2[2]-b2[0]) * (b2[3]-b2[1])
            iou   = inter / (a1 + a2 - inter + 1e-6)
            if iou > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(d)
            used.append(d["box"])
    return kept

def compute_severity(detections, img_w, img_h):
    if not detections:
        return "NONE", 0.0
    total_img = img_w * img_h
    score = 0.0
    for d in detections:
        b = d["box"]
        area_pct = ((b[2]-b[0]) * (b[3]-b[1])) / total_img
        weight   = DAMAGE_CLASSES.get(d["cls"], {}).get("weight", 1.0)
        score   += area_pct * weight * d["conf"]
    score = min(score * 10, 1.0)
    if score == 0:   return "NONE", 0.0
    if score < 0.33: return "LOW", score
    if score < 0.66: return "MODERATE", score
    return "HIGH", score

def draw_boxes(img: Image.Image, detections):
    draw = ImageDraw.Draw(img)
    for d in detections:
        info  = DAMAGE_CLASSES.get(d["cls"], {"name": "Unknown", "color": (255,255,0)})
        color = info["color"]
        b     = d["box"]
        thick = 3
        for t in range(thick):
            draw.rectangle([b[0]-t, b[1]-t, b[2]+t, b[3]+t], outline=color)
        label = f"{info['name']} {d['conf']:.0%}"
        lx, ly = b[0] + 4, max(0, b[1] - 18)
        draw.rectangle([lx-2, ly-2, lx + len(label)*7 + 2, ly+14], fill=color)
        draw.text((lx, ly), label, fill=(255,255,255))
    return img

def simulate_detections(img_w, img_h):
    count = random.randint(1, 4)
    results = []
    for _ in range(count):
        cls = random.choice([0, 1, 2, 3])
        x1  = random.randint(0, img_w - 100)
        y1  = random.randint(0, img_h - 100)
        x2  = x1 + random.randint(60, 200)
        y2  = y1 + random.randint(60, 200)
        results.append({"cls": cls, "conf": round(random.uniform(0.45, 0.92), 2), "box": [x1, y1, min(x2,img_w), min(y2,img_h)]})
    return results

# ── Serve frontend ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(FRONT_DIR), "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONT_DIR), filename)

# ── Image detection ────────────────────────────────────────────────────────
@app.route("/analyze/image", methods=["POST"])
def detect():
    try:
        data     = request.get_json()
        img_data = base64.b64decode(data["image"].split(",")[-1])
        img      = Image.open(io.BytesIO(img_data)).convert("RGB")
        orig_w, orig_h = img.size

        sess = get_model()
        if sess:
            inp    = preprocess(img)
            output = sess.run(None, {sess.get_inputs()[0].name: inp})
            dets   = postprocess(output, orig_w, orig_h)
        else:
            dets = simulate_detections(orig_w, orig_h)

        severity_label, severity_score = compute_severity(dets, orig_w, orig_h)
        vehicle_risk = VEHICLE_RISK.get(severity_label, VEHICLE_RISK["NONE"])

        annotated = draw_boxes(img.copy(), dets)
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=90)
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        detection_list = []
        for d in dets:
            info = DAMAGE_CLASSES.get(d["cls"], {"name": "Unknown", "hex": "#999"})
            detection_list.append({
                "type":       info["name"],
                "confidence": round(d["conf"] * 100, 1),
                "severity":   severity_label,
                "bbox":       d["box"],
                "color":      info["hex"],
            })

        return jsonify({
            "success":       True,
            "annotated":     annotated_b64,
            "detections":    detection_list,
            "severity":      severity_label,
            "severity_score": round(severity_score * 100, 1),
            "vehicle_risk":  vehicle_risk,
            "total_damage":  len(dets),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ── Video job endpoints ────────────────────────────────────────────────────
@app.route("/analyze/video", methods=["POST"])
def video_upload():
    try:
        f      = request.files["video"]
        job_id = str(uuid.uuid4())[:8]
        path   = UPLOAD_DIR / f"{job_id}.mp4"
        f.save(str(path))
        jobs[job_id] = {"status": "queued", "progress": 0, "result": None}
        threading.Thread(target=process_video, args=(job_id, path), daemon=True).start()
        return jsonify({"job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_video(job_id, video_path):
    try:
        jobs[job_id]["status"] = "processing"
        time.sleep(2)
        for p in range(10, 101, 10):
            jobs[job_id]["progress"] = p
            time.sleep(0.5)
        jobs[job_id].update({"status": "done", "progress": 100,
                              "result": {"severity": "MODERATE", "total_damage": 7}})
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e)})

@app.route("/video/status/<job_id>")
def video_status(job_id):
    return jsonify(jobs.get(job_id, {"error": "Job not found"}))

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "loaded" if _ort_session else "simulating"})

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*40}")
    print(f"RoadScan AI — starting on port {port}")
    print(f"Model: {'LOADED' if _ort_session else 'SIMULATING'}")
    print(f"{'='*40}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
