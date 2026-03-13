@echo off
echo Starting the AI-Based Traffic Computer Vision Demo...
echo.
echo Press 'q' at any time on the video windows to stop the demonstration.
call venv\Scripts\activate.bat
python yolo\yolo.py
pause
