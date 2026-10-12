"""Model test for the PSL tracker's local vision model.

Measures how trustworthy the scoring is on YOUR machine:
  1. Format      - does it return valid JSON with four 0-8 category scores?
  2. Repeatable  - same photo N times: how much does the score wobble?
  3. Robust      - same face, different photo conditions (mirror, brightness,
                   contrast, crop, grayscale): does the score move when it shouldn't?
  4. Sanity      - blank / noise images should be rejected (valid:false).

Usage:  python test_model.py photo.jpg [--runs 5] [--sex male|female] [--model qwen2.5vl:7b]
Needs:  Ollama running + `pip install pillow`.  Use a photo of yourself (adult).
"""
import argparse, base64, io, os, random, statistics as st, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("photo")
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--model")
ap.add_argument("--sex", choices=["male", "female"], default="male")
a = ap.parse_args()
if a.model:
    os.environ["PSL_MODEL"] = a.model
from PIL import Image, ImageEnhance, ImageOps
import app  # reuses the exact Ollama call + JSON parsing the tracker uses

PROMPT = """You are the analysis engine of a PSL-scale progress tracker. Rate the face in the photo on the PSL scale (0-8, one decimal) using four categories: harmony (how well all features work together as a cohesive whole), dimorphism (how distinctly __DIM__ the features are for a __SEX__), angularity (sharpness and definition of jawline, cheekbones and chin; soft round features score lower), and misc (skin quality, eyebrows, nose, teeth/smile). Calibration: 4 = average adult; most adults fall between 3.5 and 5.0; 5-6 = clearly above average; 6+ = exceptionally rare. Do not inflate scores and do not default to the middle; judge each category separately. This is the baseline photo.
Rules: If there is no clearly visible adult face, several people, or the person looks under 18, return {"valid":false,"reason":"..."} and nothing else. Be honest but kind and constructive. Do not diagnose anything.
Return ONLY JSON: {"valid":true,"harmony":number,"dimorphism":number,"angularity":number,"misc":number,"confidence":"low|medium|high","photoQuality":["..."],"factors":["..."],"changeSummary":"...","photoConditionImpact":"","advice":"..."}"""
WEIGHTS = {"harmony": .40, "dimorphism": .25, "angularity": .20, "misc": .15}
PROMPT = PROMPT.replace("__SEX__", a.sex).replace("__DIM__", "feminine" if a.sex == "female" else "masculine")


def b64(im):
    im = im.convert("RGB"); im.thumbnail((512, 512))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def ask(im):
    """Returns (score or None, valid_json, said_valid)."""
    try:
        res = app.ollama("/api/chat", {
            "model": app.MODEL, "stream": False, "format": "json",
            "options": {"temperature": 0.2},
            "messages": [{"role": "user", "content": PROMPT, "images": [b64(im)]}]})
        out = app.parse_json(res["message"]["content"])
    except Exception as e:
        print("   error:", str(e)[:80]); return None, False, None
    if out.get("valid") is False:
        return None, True, False
    try:
        cats = [float(out[k]) for k in WEIGHTS]
        s = sum(float(out[k]) * w for k, w in WEIGHTS.items())
        good = all(0 <= c <= 8 for c in cats)
        return (round(s, 1) if good else None), good, True
    except Exception:
        return None, False, True


def verdict(ok, warn=None):
    return "PASS" if ok else ("WARN" if warn else "FAIL")


s = app.status()
if not s["ok"]:
    sys.exit(s["help"])
base = Image.open(a.photo)
print(f"Model: {app.MODEL}\nPhoto: {a.photo}\n")
t0 = time.time(); fmt_ok = fmt_n = 0

print(f"[2] Repeatability ({a.runs} identical runs)")
reps = []
for i in range(a.runs):
    sc, ok, _ = ask(base); fmt_n += 1; fmt_ok += ok
    if sc is not None: reps.append(sc)
    print(f"   run {i+1}: {sc}")
sd = st.pstdev(reps) if len(reps) > 1 else float("nan")

variants = {
    "mirrored": ImageOps.mirror(base.convert("RGB")),
    "brighter +25%": ImageEnhance.Brightness(base.convert("RGB")).enhance(1.25),
    "darker -25%": ImageEnhance.Brightness(base.convert("RGB")).enhance(0.75),
    "low contrast": ImageEnhance.Contrast(base.convert("RGB")).enhance(0.7),
    "grayscale": ImageOps.grayscale(base.convert("RGB")),
    "cropped 90%": base.crop((int(base.width*.05), int(base.height*.05), int(base.width*.95), int(base.height*.95))),
}
print("\n[3] Robustness (same face, different photo conditions)")
mean0 = st.mean(reps) if reps else None; devs = []
for name, im in variants.items():
    sc, ok, _ = ask(im); fmt_n += 1; fmt_ok += ok
    d = None if (sc is None or mean0 is None) else sc - mean0
    if d is not None: devs.append(abs(d))
    print(f"   {name:15} score={sc}  shift={'' if d is None else f'{d:+.1f}'}")
maxdev = max(devs) if devs else float("nan")

print("\n[4] Sanity (should be rejected)")
rnd = random.Random(1)
junk = {"solid gray": Image.new("RGB", (512, 512), (128, 128, 128)),
        "random noise": Image.frombytes("RGB", (256, 256), bytes(rnd.getrandbits(8) for _ in range(256*256*3)))}
rej = 0
for name, im in junk.items():
    sc, ok, said_valid = ask(im); fmt_n += 1; fmt_ok += ok
    rej += (said_valid is False)
    print(f"   {name:12} -> {'rejected' if said_valid is False else f'scored {sc} (bad)'}")

print(f"\n{'='*48}\nRESULTS  ({time.time()-t0:.0f}s)")
print(f"[1] Format valid     {fmt_ok}/{fmt_n}         {verdict(fmt_ok == fmt_n)}")
print(f"[2] Repeat stdev     {sd:.2f} (want <0.3)   {verdict(sd < .3, sd < .6)}")
print(f"[3] Max shift        {maxdev:.1f} (want <0.5)   {verdict(maxdev < .5, maxdev < 1.0)}")
print(f"[4] Junk rejected    {rej}/2         {verdict(rej == 2, rej == 1)}")
print("\nRule of thumb: if [2] or [3] is WARN/FAIL, ignore week-to-week changes smaller than the shift you measured, or try a bigger model (--model qwen2.5vl:7b).")
