# PSL Progress Tracker (local)

Runs entirely on your PC. A small Python server (packaged as an .exe) hosts the UI
and sends photos to a local vision model through Ollama. No cloud, no account.

## One-time setup
1. Install Ollama: https://ollama.com/download
2. Download the model (about 3 GB):  `ollama pull qwen2.5vl:3b`
   (weaker PCs: `ollama pull moondream`; stronger: `qwen2.5vl:7b`)
   To switch, set the env var `PSL_MODEL` (e.g. `set PSL_MODEL=qwen2.5vl:7b`).

## Build the .exe (on Windows)
Double-click `build.bat`  ->  `dist\PSLTracker.exe`

## Run
Make sure Ollama is running, then launch `PSLTracker.exe`. Your browser opens at
http://127.0.0.1:8765. Keep the console window open; close it to quit.
Your history is saved to `%APPDATA%\PSLTracker\store.json`.

No build needed? `pip install mediapipe==0.10.21` then `python app.py` (without mediapipe it still runs, AI-only).

## Get the .exe without installing Python (GitHub Actions)
1. Create a free GitHub repo and upload everything in this folder (keep `.github/workflows/build.yml`).
2. Open the repo's **Actions** tab, choose **Build Windows exe**, click **Run workflow**.
3. When it finishes (~2 min), open the run and download **PSLTracker-windows** (a zip containing `PSLTracker.exe`).

## Test how reliable your local model is
`pip install pillow`, make sure Ollama is running, then:
`python test_model.py my_photo.jpg --runs 5`
It checks JSON validity, repeatability (same photo N times), robustness (mirror, brightness,
contrast, grayscale, crop) and whether blank/noise images get rejected. Use it to decide how
big a week-to-week change you can trust, or to compare models (`--model qwen2.5vl:7b`).

## Scoring (v2)
The model rates four categories (Harmony, Dimorphism, Angularity, Miscellaneous). The app computes
the overall score itself: 40% / 25% / 20% / 15%, then maps it to a tier (True Adam/Eve, Chad/Stacy,
Chadlite/Stacylite, High/Mid/Low Tier Normie, Very Low Range). Pick Male/Female in the top bar so
dimorphism is judged correctly. `test_model.py --sex female` does the same for the test.

## What changed in v3 (more accurate, still 100% local)
- **Landmark measurements** (`geometry.py`, MediaPipe FaceMesh): asymmetry, facial thirds, canthal tilt, eye spacing,
  FWHR, jaw/cheek ratio, gonial angle, lip and nose ratios. Deterministic: same photo, same numbers.
- **Blended score**: Harmony/Dimorphism/Angularity = 60% measured + 40% AI; Miscellaneous (skin, brows, smile) = AI.
- **Less AI noise**: 3 passes per photo (median, `PSL_SAMPLES`), scores pulled toward the population mean of 4.
- **Better rejection**: no face / several faces / tiny face / turned head are caught by landmarks before the AI sees it.
- Bigger model recommended if your GPU allows: `ollama pull qwen2.5vl:7b` and `set PSL_MODEL=qwen2.5vl:7b`.
Tier cut-offs, the 0-8 scale and the four categories match what getpsl.app publicly describes; its exact formulas
are not public, so the weights here are my own estimate.
