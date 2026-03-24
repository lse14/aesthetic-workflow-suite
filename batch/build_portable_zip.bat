@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY_CMD="
if exist ".venv\Scripts\python.exe" (
  set "PY_CMD=.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "PY_CMD=py -3"
  ) else (
    where python >nul 2>nul
    if errorlevel 1 (
      echo [ERROR] Python not found.
      pause
      exit /b 1
    )
    set "PY_CMD=python"
  )
)

echo [build] Python: %PY_CMD%
%PY_CMD% -X utf8 build_portable_zip.py --force %*
if errorlevel 1 (
  echo [ERROR] build failed.
  pause
  exit /b 1
)
echo [build] done.
pause
exit /b 0
