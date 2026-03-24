@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "FAIL_REASON="
set "PY_CMD="

echo [prefetch_jtp3] infer_ui root: %cd%

where py >nul 2>nul
if errorlevel 1 goto :check_python
set "PY_CMD=py -3"
goto :ensure_venv

:check_python
where python >nul 2>nul
if errorlevel 1 (
  set "FAIL_REASON=python_not_found"
  goto :fail
)
set "PY_CMD=python"

:ensure_venv
if not exist ".venv\Scripts\python.exe" (
  echo [prefetch_jtp3] create .venv ...
  %PY_CMD% -m venv .venv
  if errorlevel 1 (
    set "FAIL_REASON=venv_create_failed"
    goto :fail
  )
)

echo [prefetch_jtp3] install deps (huggingface_hub + tqdm) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install huggingface_hub tqdm >nul
if errorlevel 1 (
  set "FAIL_REASON=pip_install_failed"
  goto :fail
)

if not defined HF_HUB_DISABLE_SYMLINKS_WARNING set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not defined FUSION_MODEL_CACHE_ROOT set "FUSION_MODEL_CACHE_ROOT=%~dp0_models"

echo [prefetch_jtp3] cache root: %FUSION_MODEL_CACHE_ROOT%
echo [prefetch_jtp3] target: %FUSION_MODEL_CACHE_ROOT%\repos\RedRocket__JTP-3

".venv\Scripts\python.exe" "scripts\prefetch_jtp3.py" --root "%FUSION_MODEL_CACHE_ROOT%" %*
if errorlevel 1 (
  set "FAIL_REASON=download_failed"
  goto :fail
)

echo.
echo [prefetch_jtp3] Success.
echo Press any key to close...
pause >nul
exit /b 0

:fail
echo.
echo [ERROR] prefetch_jtp3 failed: %FAIL_REASON%
if "%FAIL_REASON%"=="python_not_found" echo Fix: install Python 3.10+ and add to PATH.
if "%FAIL_REASON%"=="venv_create_failed" echo Fix: delete .venv and retry.
if "%FAIL_REASON%"=="pip_install_failed" echo Fix: run .venv\Scripts\python.exe -m pip install huggingface_hub tqdm
if "%FAIL_REASON%"=="download_failed" echo Fix: check network/HF token and retry.
echo.
echo Press any key to close...
pause >nul
exit /b 1

