@echo off
setlocal EnableExtensions

chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
set "API_HOST=127.0.0.1"
set "API_PORT=8000"
set "UI_PORT=8501"
if "%SMARTCAMPUS_WORKER_ROLE%"=="" set "SMARTCAMPUS_WORKER_ROLE=all"

echo ======================================
echo Starting backend (FastAPI)
echo ======================================
start "backend" cmd /k ""%PYTHON_EXE%" -m uvicorn backend.api:app --host %API_HOST% --port %API_PORT% --reload"

timeout /t 2 >nul

echo ======================================
echo Starting worker
echo ======================================
start "worker" cmd /k "set SMARTCAMPUS_WORKER_ROLE=%SMARTCAMPUS_WORKER_ROLE% && ""%PYTHON_EXE%"" -m backend.worker"

timeout /t 2 >nul

echo ======================================
echo Starting Streamlit UI
echo ======================================
start "ui" cmd /k ""%PYTHON_EXE%" -m streamlit run app\main.py --server.address %API_HOST% --server.port %UI_PORT%"

echo ======================================
echo All services started
echo ======================================
echo API: http://%API_HOST%:%API_PORT%
echo UI:  http://%API_HOST%:%UI_PORT%

endlocal
