@echo off
setlocal EnableExtensions

chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
set "API_HOST=127.0.0.1"
set "API_PORT=8000"
set "UI_PORT=8501"
if "%SMARTCAMPUS_START_GPU%"=="" set "SMARTCAMPUS_START_GPU=1"
if "%SMARTCAMPUS_START_CPU%"=="" set "SMARTCAMPUS_START_CPU=1"
if "%SMARTCAMPUS_START_MT%"=="" set "SMARTCAMPUS_START_MT=1"

echo ======================================
echo Starting backend (FastAPI)
echo ======================================
start "backend" cmd /k ""%PYTHON_EXE%" -m uvicorn backend.api:app --host %API_HOST% --port %API_PORT%"

timeout /t 2 >nul

if "%SMARTCAMPUS_START_GPU%"=="1" (
  echo ======================================
  echo Starting GPU worker
  echo ======================================
  start "worker_gpu" cmd /k "set SMARTCAMPUS_WORKER_ROLE=gpu && ""%PYTHON_EXE%"" -m backend.worker"
  timeout /t 2 >nul
)

if "%SMARTCAMPUS_START_CPU%"=="1" (
  echo ======================================
  echo Starting CPU worker
  echo ======================================
  start "worker_cpu" cmd /k "set SMARTCAMPUS_WORKER_ROLE=cpu && ""%PYTHON_EXE%"" -m backend.worker"
  timeout /t 2 >nul
)

if "%SMARTCAMPUS_START_MT%"=="1" (
  echo ======================================
  echo Starting MT worker
  echo ======================================
  start "worker_mt" cmd /k "set SMARTCAMPUS_WORKER_ROLE=mt && ""%PYTHON_EXE%"" -m backend.worker"
  timeout /t 2 >nul
)

echo ======================================
echo Starting Streamlit UI
echo ======================================
start "ui" cmd /k ""%PYTHON_EXE%" -m streamlit run app\main.py --server.address %API_HOST% --server.port %UI_PORT%"

echo ======================================
echo All services started
echo ======================================
echo API: http://%API_HOST%:%API_PORT%
echo UI:  http://%API_HOST%:%UI_PORT%
echo Workers: gpu=%SMARTCAMPUS_START_GPU% cpu=%SMARTCAMPUS_START_CPU% mt=%SMARTCAMPUS_START_MT%

endlocal
