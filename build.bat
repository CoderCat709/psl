@echo off
REM Builds dist\PSLTracker.exe (needs Python 3.9+ from python.org on PATH)
python -m pip install --upgrade pyinstaller mediapipe==0.10.21 || exit /b 1
python -m PyInstaller --onefile --name PSLTracker --collect-all mediapipe --add-data "index.html;." app.py || exit /b 1
echo.
echo Done: dist\PSLTracker.exe
