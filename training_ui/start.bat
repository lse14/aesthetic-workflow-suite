@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

if not defined FUSION_JTP3_MODEL_ID set "FUSION_JTP3_MODEL_ID=RedRocket/JTP-3"
if not defined FUSION_JTP3_FALLBACK_MODEL_ID set "FUSION_JTP3_FALLBACK_MODEL_ID=none"
if not defined FUSION_MODEL_CACHE_ROOT set "FUSION_MODEL_CACHE_ROOT=%~dp0..\model\_models"
if not defined HF_HOME set "HF_HOME=%FUSION_MODEL_CACHE_ROOT%\hf_home"
if not defined HF_HUB_CACHE set "HF_HUB_CACHE=%HF_HOME%\hub"
if not exist "%HF_HOME%" mkdir "%HF_HOME%" >nul 2>nul
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%" >nul 2>nul
echo [training_ui] JTP3 model: %FUSION_JTP3_MODEL_ID%
echo [training_ui] model cache root: %FUSION_MODEL_CACHE_ROOT%
echo [training_ui] HF_HOME: %HF_HOME%
echo [training_ui] HF_HUB_CACHE: %HF_HUB_CACHE%

echo [training_ui 1/4] Checking Python...
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>nul
  if not %errorlevel%==0 (
    echo [ERROR] Python not found. Install Python 3.10+ first.
    goto :fail
  )
  set "PY_CMD=python"
)

echo [training_ui 2/4] Creating virtual environment...
if not exist ".venv\Scripts\python.exe" (
  %PY_CMD% -m venv .venv
  if not %errorlevel%==0 (
    echo [ERROR] Failed to create .venv
    goto :fail
  )
)

echo [training_ui 3/4] Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if not %errorlevel%==0 (
  echo [ERROR] Failed to upgrade pip.
  goto :fail
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if not %errorlevel%==0 (
  echo [ERROR] Failed to install dependencies.
  goto :fail
)

set "WEBUI_HOST="
set "WEBUI_PORT="
set "WEBUI_RESOLVE_TMP=%TEMP%\training_ui_webui_%RANDOM%%RANDOM%.txt"
".venv\Scripts\python.exe" "%APP_DIR%scripts\resolve_webui_port.py" --config "%APP_DIR%config.yaml" > "%WEBUI_RESOLVE_TMP%" 2>nul
if exist "%WEBUI_RESOLVE_TMP%" (
  for /f "usebackq tokens=1,2" %%a in ("%WEBUI_RESOLVE_TMP%") do (
    set "WEBUI_HOST=%%a"
    set "WEBUI_PORT=%%b"
  )
  del /q "%WEBUI_RESOLVE_TMP%" >nul 2>nul
)
if "%WEBUI_HOST%"=="" set "WEBUI_HOST=127.0.0.1"
if "%WEBUI_PORT%"=="" set "WEBUI_PORT=9300"

set "OPEN_HOST=%WEBUI_HOST%"
if /I "%OPEN_HOST%"=="0.0.0.0" set "OPEN_HOST=127.0.0.1"

echo [training_ui 4/4] Starting Training UI...
echo Training UI: http://%OPEN_HOST%:%WEBUI_PORT%/
start "" "http://%OPEN_HOST%:%WEBUI_PORT%/"

".venv\Scripts\python.exe" "%APP_DIR%run.py" --config "%APP_DIR%config.yaml" --host %WEBUI_HOST% --port %WEBUI_PORT%
if not %errorlevel%==0 (
  echo [ERROR] Training UI process exited unexpectedly.
  goto :fail
)
echo [INFO] Training UI exited normally.
goto :end

:fail
echo.
echo Press any key to close...
pause >nul
exit /b 1

:end
endlocal
