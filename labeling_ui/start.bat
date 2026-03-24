@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

echo [labeling_ui 1/4] Checking Python...
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

echo [labeling_ui 2/4] Creating virtual environment...
if not exist ".venv\Scripts\python.exe" (
  %PY_CMD% -m venv .venv
  if not %errorlevel%==0 (
    echo [ERROR] Failed to create .venv
    goto :fail
  )
)

echo [labeling_ui 3/4] Installing dependencies...
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
set "WEBUI_RESOLVE_TMP=%TEMP%\labeling_ui_webui_%RANDOM%%RANDOM%.txt"
".venv\Scripts\python.exe" "%APP_DIR%scripts\resolve_webui_port.py" --config "%APP_DIR%config.yaml" > "%WEBUI_RESOLVE_TMP%" 2>nul
if exist "%WEBUI_RESOLVE_TMP%" (
  for /f "usebackq tokens=1,2" %%a in ("%WEBUI_RESOLVE_TMP%") do (
    set "WEBUI_HOST=%%a"
    set "WEBUI_PORT=%%b"
  )
  del /q "%WEBUI_RESOLVE_TMP%" >nul 2>nul
)
if "%WEBUI_HOST%"=="" set "WEBUI_HOST=127.0.0.1"
if "%WEBUI_PORT%"=="" set "WEBUI_PORT=9100"

set "OPEN_HOST=%WEBUI_HOST%"
if /I "%OPEN_HOST%"=="0.0.0.0" set "OPEN_HOST=127.0.0.1"

echo [labeling_ui 4/4] Starting Labeling UI...
echo Labeling UI: http://%OPEN_HOST%:%WEBUI_PORT%/
start "" /min ".venv\Scripts\python.exe" "%APP_DIR%scripts\open_when_ready.py" --url "http://%OPEN_HOST%:%WEBUI_PORT%/" --health-url "http://%OPEN_HOST%:%WEBUI_PORT%/api/health" --timeout 120 --interval 0.8

".venv\Scripts\python.exe" "%APP_DIR%run.py" --config "%APP_DIR%config.yaml" --host %WEBUI_HOST% --port %WEBUI_PORT%
if not %errorlevel%==0 (
  echo [ERROR] Labeling UI process exited unexpectedly.
  goto :fail
)
echo [INFO] Labeling UI exited normally.
goto :end

:fail
echo.
echo Press any key to close...
pause >nul
exit /b 1

:end
endlocal
