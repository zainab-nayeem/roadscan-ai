from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import base64, io, os, random, time, threading, uuid, webbrowser
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw
import numpy as np

app = Flask(__name__)
CORS(app)

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "yolov8_road.onnx"
FRONT_DIR  = BASE_DIR.parent / "frontend"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "rs_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "rs_outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DC = {
    0: {"name": "Longitudinal Crack", "color": (255,136,0),  "hex": "#FF8800", "weight": 1.5},
    1: {"name": "Transverse Crack",   "color": (255,187,0),  "hex": "#FFBB00", "weight": 1.8},
    2: {"name": "Alligator Crack",    "color": (255,85,0),   "hex": "#FF5500", "weight": 2.5},
    3: {"name": "Pothole",            "color": (255,61,90),  "hex": "#FF3D5A", "weight": 3.0},
}
VR = {
    "NONE":     {"car":"SAFE",      "truck":"SAFE",    "bike":"SAFE",      "bicycle":"SAFE"},
    "LOW":      {"car":"SAFE",      "truck":"SAFE",    "bike":"SAFE",      "bicycle":"SAFE"},
    "MODERATE": {"car":"CAUTION",   "truck":"SAFE",    "bike":"CAUTION",   "bicycle":"HIGH RISK"},
    "HIGH":     {"car":"HIGH RISK", "truck":"CAUTION", "bike":"HIGH RISK", "bicycle":"AVOID"},
}
jobs  = {}
_sess = None


@app.route("/")
def index():
    return send_from_directory(str(FRONT_DIR), "index.html")


@app.route("/health")
def health():
    s = get_model()
    return jsonify({"status": "ok", "model": "real model" if s else "simulation",
                    "exists": MODEL_PATH.exists()})


@app.route("/analyze/image", methods=["POST"])
def analyze_image():
    d = request.get_json()
    if not d or "image" not in d:
        return jsonify({"error": "no image"}), 400
    vehicle = d.get("vehicle", "car")
    try:
        img  = Image.open(io.BytesIO(base64.b64decode(d["image"]))).convert("RGB")
        print("IMG", img.size)
        t    = time.time()
        dets = detect(img)
        ann  = annotate(img.copy(), dets)
        ms   = int((time.time() - t) * 1000)
        print("Detections:", len(dets), "-", ms, "ms")
        return jsonify({
            "success": True,
            "detections": dets,
            "annotated_image": b64(ann),
            "risk_summary": risk(dets, vehicle),
            "processing_ms": ms,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/video", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "no video"}), 400
    vehicle = request.form.get("vehicle", "car")
    f   = request.files["video"]
    ext = Path(f.filename).suffix.lower() or ".mp4"
    jid = str(uuid.uuid4())[:8]
    inp = str(UPLOAD_DIR / (jid + ext))
    out = str(OUTPUT_DIR / (jid + "_annotated.mp4"))
    f.save(inp)
    jobs[jid] = {"status": "queued", "progress": 0, "result": None, "error": None}
    threading.Thread(target=process_video, args=(jid, inp, out, vehicle), daemon=True).start()
    return jsonify({"job_id": jid, "status": "queued"})


def process_video(jid, inp, out, vehicle):
    try:
        import cv2
        jobs[jid]["status"] = "processing"
        cap  = cv2.VideoCapture(inp)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wrt  = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        step = max(1, int(fps / 3))
        all_dets = []
        idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx % step == 0:
                pil  = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                dets = detect(pil, idx)
                if dets:
                    ann = annotate(pil.copy(), dets)
                    bgr = cv2.cvtColor(np.array(ann), cv2.COLOR_RGB2BGR)
                all_dets.extend(dets)
            wrt.write(bgr)
            idx += 1
            jobs[jid]["progress"] = round(idx / tot * 100, 1)
        cap.release()
        wrt.release()
        freq = {}
        for d in all_dets:
            freq[d["class_name"]] = freq.get(d["class_name"], 0) + 1
        rs = risk(all_dets, vehicle)
        rs.update({
            "total_frames": idx,
            "processed_frames": max(1, idx // step),
            "video_fps": round(fps, 1),
            "video_duration_s": round(idx / fps, 1),
        })
        jobs[jid].update({
            "status": "done", "progress": 100,
            "result": {
                "output_video": jid,
                "risk_summary": rs,
                "damage_frequency": freq,
                "total_detections": len(all_dets),
            },
        })
        print("Video done:", len(all_dets), "detections")
    except Exception as e:
        import traceback; traceback.print_exc()
        jobs[jid].update({"status": "error", "error": str(e)})


@app.route("/jobs/<jid>")
def get_job(jid):
    j = jobs.get(jid)
    return jsonify({"job_id": jid, **j}) if j else (jsonify({"error": "not found"}), 404)


@app.route("/jobs/<jid>/download")
def download_video(jid):
    p = OUTPUT_DIR / (jid + "_annotated.mp4")
    if p.exists():
        return send_file(str(p), mimetype="video/mp4", as_attachment=True,
                         download_name="roadscan_" + jid + ".mp4")
    return jsonify({"error": "not ready"}), 404


@app.route("/stats")
def stats():
    return jsonify({
        "total_inspections": 1482, "high_risk_roads": 47,
        "damage_breakdown": {"Pothole": 312, "Longitudinal Crack": 245,
                             "Transverse Crack": 198, "Alligator Crack": 156},
        "severity_distribution": {"LOW": 523, "MODERATE": 621, "HIGH": 338},
    })


def get_model():
    global _sess
    if _sess is None and MODEL_PATH.exists():
        try:
            import onnxruntime as ort
            _sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            print("Model loaded. Outputs:", [o.name for o in _sess.get_outputs()])
        except Exception as e:
            print("Model load error:", e)
    return _sess


def preprocess(img, size=640):
    arr = np.array(img.resize((size, size))).astype("float32") / 255.0
    return np.expand_dims(arr.transpose(2, 0, 1), 0)


def severity(area, cls):
    raw   = min(area * DC[cls]["weight"] * 10, 1.0)
    level = "HIGH" if raw > 0.66 else "MODERATE" if raw > 0.33 else "LOW"
    return {"level": level, "score": round(raw, 3)}


def detect(img, seed=None):
    s = get_model()
    if s:
        w, h  = img.size
        out   = s.run(None, {s.get_inputs()[0].name: preprocess(img)})
        raw   = out[0]
        print("raw shape:", raw.shape)
        if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
            preds = raw[0].T
        elif raw.ndim == 3:
            preds = raw[0]
        else:
            preds = raw
        dets = []
        for p in preds:
            if len(p) < 8:
                continue
            scores = p[4:8]
            cls    = int(np.argmax(scores))
            conf   = float(scores[cls])
            if conf < 0.01 or cls not in DC:
                continue
            x, y, bw, bh = p[:4]
            x1 = int((x - bw / 2) / 640 * w)
            y1 = int((y - bh / 2) / 640 * h)
            x2 = int((x + bw / 2) / 640 * w)
            y2 = int((y + bh / 2) / 640 * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area = ((x2 - x1) * (y2 - y1)) / (w * h)
            dets.append({
                "class_id":   cls,
                "class_name": DC[cls]["name"],
                "color":      DC[cls]["hex"],
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(conf, 2),
                "area_ratio": round(area, 4),
                "severity":   severity(area, cls),
            })
        print("Detections:", len(dets))
        return dets
    rng  = random.Random(seed)
    dets = []
    for _ in range(rng.randint(1, 3)):
        cls  = rng.randint(0, 3)
        x1   = rng.uniform(0.1, 0.5) * img.width
        y1   = rng.uniform(0.1, 0.5) * img.height
        x2   = min(x1 + rng.uniform(0.1, 0.3) * img.width,  img.width)
        y2   = min(y1 + rng.uniform(0.1, 0.2) * img.height, img.height)
        area = ((x2 - x1) * (y2 - y1)) / (img.width * img.height)
        dets.append({
            "class_id":   cls,
            "class_name": DC[cls]["name"],
            "color":      DC[cls]["hex"],
            "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(rng.uniform(0.4, 0.85), 2),
            "area_ratio": round(area, 4),
            "severity":   severity(area, cls),
        })
    return dets


def annotate(img, dets):
    draw = ImageDraw.Draw(img)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        rgb   = DC[d["class_id"]]["color"]
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)
        label = d["class_name"] + "  " + str(int(d["confidence"] * 100)) + "%  " + d["severity"]["level"]
        lw    = len(label) * 6 + 8
        draw.rectangle([x1, max(0, y1 - 18), x1 + lw, y1], fill=rgb)
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill="white")
    return img


def b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def risk(dets, vehicle="car"):
    if not dets:
        return {"overall_severity": "NONE", "overall_score": 0,
                "vehicle_risk": "SAFE", "damage_count": 0, "vehicle_type": vehicle}
    scores = [d["severity"]["score"] * DC[d["class_id"]]["weight"] for d in dets]
    avg    = min(sum(scores) / len(scores) / 3, 1.0)
    level  = "HIGH" if avg > 0.66 else "MODERATE" if avg > 0.33 else "LOW" if avg > 0 else "NONE"
    return {
        "overall_severity": level,
        "overall_score":    round(avg, 3),
        "vehicle_risk":     VR[level].get(vehicle, "CAUTION"),
        "damage_count":     len(dets),
        "vehicle_type":     vehicle,
    }


if __name__ == "__main__":
    print("=" * 40)
    print("RoadScan AI v4.2")
    print("Model   :", "FOUND" if MODEL_PATH.exists() else "NOT FOUND - place yolov8_road.onnx in backend/models/")
    print("Frontend:", "FOUND" if (FRONT_DIR / "index.html").exists() else "NOT FOUND")
    print("=" * 40)
    get_model()
    webbrowser.open("http://localhost:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")
