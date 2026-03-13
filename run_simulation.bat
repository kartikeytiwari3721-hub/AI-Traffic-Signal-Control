@echo off
echo Starting the AI-Based Traffic Signal Control System Simulation...
echo.
echo Please ensure that SUMO-GUI is installed and "Play" is pressed when it opens!
call venv\Scripts\activate.bat
python test_generalised.py
pause
