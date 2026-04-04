@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"
set "RUN_PY="
set "PYTHONNOUSERSITE=1"
set "EMBED_HELPER=%APP_DIR%..\scripts\ensure_embedded_python.bat"
if exist "%EMBED_HELPER%" (
  call "%EMBED_HELPER%" "%APP_DIR%..\runtime\python"
  if not errorlevel 1 if defined EMBED_PYTHON_EXE set "RUN_PY=!EMBED_PYTHON_EXE!"
)
if "%RUN_PY%"=="" (
  set "RUN_PY=%APP_DIR%..\runtime\python\python.exe"
)
if not exist "%RUN_PY%" (
  echo [ERROR] Embedded Python unavailable: %RUN_PY%
  goto :fail
)
echo [labeling_ui 1/3] Using embedded runtime: %RUN_PY%

:install_deps
echo [labeling_ui 2/3] Installing dependencies...
%RUN_PY% -m pip install --upgrade pip
if not %errorlevel%==0 (
  echo [ERROR] Failed to upgrade pip.
  goto :fail
)

%RUN_PY% -m pip install -r requirements.txt
if not %errorlevel%==0 (
  echo [ERROR] Failed to install dependencies.
  goto :fail
)

set "WEBUI_HOST="
set "WEBUI_PORT="
set "WEBUI_RESOLVE_TMP=%TEMP%\labeling_ui_webui_%RANDOM%%RANDOM%.txt"
%RUN_PY% "%APP_DIR%scripts\resolve_webui_port.py" --config "%APP_DIR%config.yaml" > "%WEBUI_RESOLVE_TMP%" 2>nul
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

echo [labeling_ui 3/3] Starting Labeling UI...
echo Labeling UI: http://%OPEN_HOST%:%WEBUI_PORT%/
start "" /min %RUN_PY% "%APP_DIR%scripts\open_when_ready.py" --url "http://%OPEN_HOST%:%WEBUI_PORT%/" --health-url "http://%OPEN_HOST%:%WEBUI_PORT%/api/health" --timeout 120 --interval 0.8

%RUN_PY% "%APP_DIR%run.py" --config "%APP_DIR%config.yaml" --host %WEBUI_HOST% --port %WEBUI_PORT%
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

