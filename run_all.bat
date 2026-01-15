@echo off
setlocal

chcp 65001 >nul
cd /d "%~dp0"

echo ======================================
echo Starting backend (FastAPI)
echo ======================================
start "backend" cmd /k ^
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload

timeout /t 2 >nul

echo ======================================
echo Starting worker
echo ======================================
start "worker" cmd /k ^
python -m backend.worker

timeout /t 2 >nul

echo ======================================
echo Starting Streamlit UI
echo ======================================
start "ui" cmd /k ^
python -m streamlit run app\main.py --server.port 8501

echo ======================================
echo All services started
echo ======================================

endlocal
