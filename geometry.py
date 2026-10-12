"""Deterministic face measurements from MediaPipe FaceMesh landmarks (all local).
Same photo -> same numbers, which a vision-language model can't promise."""
import base64, math, threading
try:
    import numpy as np
    import cv2, mediapipe as mp
    AVAILABLE = True
except Exception:
    AVAILABLE = False
_lock, _mesh = threading.Lock(), None


def _s(v, lo, hi, tol):  # 1.0 inside the ideal band, falling linearly to 0 at `tol` outside it
    return max(0.0, 1 - max(lo - v, v - hi, 0) / tol)


def _angle(a, b, c):
    v, w = a - b, c - b
    return math.degrees(math.acos(max(-1, min(1, float(v @ w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-9)))))


def metrics(P, sex="male"):
    """P: (N,2) landmark pixels in MediaPipe order. Returns raw values, 0-1 sub-scores, 0-8 category scores."""
    d = lambda i, j: float(np.linalg.norm(P[i] - P[j]))
    fw = d(234, 454)                                   # bizygomatic width
    axis = (P[152] - P[10]) / d(10, 152)
    nrm, mid = np.array([-axis[1], axis[0]]), (P[10] + P[152]) / 2
    pairs = [(33, 263), (133, 362), (61, 291), (234, 454), (172, 397), (129, 358), (70, 300), (105, 334)]
    asym = np.mean([abs((P[a] - mid) @ nrm + (P[b] - mid) @ nrm) + abs((P[a] - P[b]) @ axis) for a, b in pairs]) / fw
    u, m, l = d(10, 168), d(168, 2), d(2, 152)
    thirds_dev = max(abs(x - (u + m + l) / 3) for x in (u, m, l)) / ((u + m + l) / 3)
    tilt = np.mean([math.degrees(math.atan2(P[i][1] - P[o][1], abs(P[i][0] - P[o][0]))) for i, o in ((133, 33), (362, 263))])
    ic, ew = d(133, 362), (d(33, 133) + d(263, 362)) / 2
    jaw = d(172, 397) / fw
    gonial = (_angle(P[234], P[172], P[152]) + _angle(P[454], P[397], P[152])) / 2
    fwhr, lips, nose = fw / d(168, 0), d(0, 13) / max(d(14, 17), 1e-6), d(129, 358) / ic
    f = sex == "female"
    raw = {"symmetry_asym": asym, "thirds_dev": thirds_dev, "canthal_tilt": tilt, "eye_spacing": ic / ew, "nose_to_ic": nose,
           "lip_ratio": lips, "fwhr": fwhr, "jaw_to_cheek": jaw, "gonial_angle": gonial, "lower_third": l / (u + m + l)}
    sc = {"symmetry": 1 - asym / 0.06, "thirds": 1 - thirds_dev / 0.35, "canthal": _s(tilt, 3, 8, 8), "eyes": _s(ic / ew, .9, 1.1, .3),
          "nose": _s(nose, .9, 1.1, .35), "lips": _s(lips, .5, .8, .4), "chin": _s(l / (u + m + l), .31, .37, .1),
          "fwhr": _s(fwhr, 1.7, 2.0, .4) if f else _s(fwhr, 1.85, 2.15, .4),
          "jaw": _s(jaw, .70, .84, .15) if f else _s(jaw, .78, .92, .15),
          "gonial": _s(gonial, 118, 134, 20) if f else _s(gonial, 112, 128, 20)}
    sc = {k: max(0.0, min(1.0, v)) for k, v in sc.items()}
    avg = lambda *k: sum(sc[x] for x in k) / len(k)
    to8 = lambda s: round(2 + 4 * s, 1)               # geometry alone tops out at 6: it can't see skin, eyes' depth, etc.
    yaw = min(d(1, 234), d(1, 454)) / max(d(1, 234), d(1, 454))
    roll = math.degrees(math.atan2(*(P[263] - P[33])[::-1]))
    return {"raw": {k: round(float(v), 3) for k, v in raw.items()}, "scores": {k: round(v, 2) for k, v in sc.items()},
            "harmony_g": to8(avg("symmetry", "thirds", "canthal", "eyes", "nose", "lips")),
            "dimorphism_g": to8(avg("fwhr", "jaw", "gonial")), "angularity_g": to8(avg("gonial", "jaw", "chin")),
            "pose": {"yaw_balance": round(yaw, 2), "roll_deg": round(roll, 1)}}


def measure(b64, sex="male"):
    global _mesh
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode the image."}
    h, w = img.shape[:2]
    with _lock:
        if _mesh is None:
            _mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.6)
        r = _mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks:
        return {"error": "No clearly visible face found. Use a well-lit, front-facing photo."}
    if len(r.multi_face_landmarks) > 1:
        return {"error": "More than one face detected. Use a photo of just you."}
    P = np.array([[p.x * w, p.y * h] for p in r.multi_face_landmarks[0].landmark])
    m = metrics(P, sex)
    if np.linalg.norm(P[234] - P[454]) / w < 0.22:
        return {"error": "Face is too small in the frame. Move closer."}
    warn = []
    if m["pose"]["yaw_balance"] < 0.8: warn.append("Head is turned; face the camera straight on (measurements are less reliable).")
    if abs(m["pose"]["roll_deg"]) > 6: warn.append("Head is tilted; keep your eyes level.")
    m["warnings"] = warn
    return m
