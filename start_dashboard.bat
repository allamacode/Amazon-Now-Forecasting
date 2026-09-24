@echo off
echo ----------------------------------------------------
echo Amazon Now Enterprise AI Console - Boot Sequence
echo ----------------------------------------------------
echo.

echo [1/3] Checking for port collisions on 8501...
FOR /F "tokens=5" %%a in ('netstat -aon ^| find ":8501" ^| find "LISTENING"') do (
    echo [!] Found existing process blocking port 8501. Terminating...
    taskkill /F /PID %%a >nul 2>&1
)

echo [2/3] Starting the Backend Server...
:: Streamlit will automatically open your default web browser once it's ready!
streamlit run app.py
