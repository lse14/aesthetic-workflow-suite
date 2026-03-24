@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY_CMD="
set "FAIL_REASON="
if not defined FUSION_AUTO_FIX_CUDA_TORCH set "FUSION_AUTO_FIX_CUDA_TORCH=1"
set "TQDM_DISABLE=0"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not defined FUSION_JTP3_MODEL_ID set "FUSION_JTP3_MODEL_ID=RedRocket/JTP-3"
if not defined FUSION_STRICT_JTP3_ONLY set "FUSION_STRICT_JTP3_ONLY=1"
if not defined FUSION_JTP3_FALLBACK_MODEL_ID set "FUSION_JTP3_FALLBACK_MODEL_ID=google/siglip2-so400m-patch16-naflex"
if /I "%FUSION_STRICT_JTP3_ONLY%"=="1" set "FUSION_JTP3_FALLBACK_MODEL_ID=none"
if /I "%FUSION_STRICT_JTP3_ONLY%"=="true" set "FUSION_JTP3_FALLBACK_MODEL_ID=none"
if /I "%FUSION_STRICT_JTP3_ONLY%"=="yes" set "FUSION_JTP3_FALLBACK_MODEL_ID=none"
if /I "%FUSION_STRICT_JTP3_ONLY%"=="on" set "FUSION_JTP3_FALLBACK_MODEL_ID=none"
if not defined FUSION_MODEL_CACHE_ROOT set "FUSION_MODEL_CACHE_ROOT=%CD%\_models"
if not defined FUSION_WAIFU_V3_HEAD_PATH set "FUSION_WAIFU_V3_HEAD_PATH=%FUSION_MODEL_CACHE_ROOT%\waifu-scorer-v3\model.safetensors"
if not defined HF_HOME set "HF_HOME=%FUSION_MODEL_CACHE_ROOT%\hf_home"
if not defined HF_HUB_CACHE set "HF_HUB_CACHE=%HF_HOME%\hub"
if not exist "%HF_HOME%" mkdir "%HF_HOME%" >nul 2>nul
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%" >nul 2>nul

echo [infer_ui] model cache root: %FUSION_MODEL_CACHE_ROOT%
echo [infer_ui] HF_HOME: %HF_HOME%
echo [infer_ui] HF_HUB_CACHE: %HF_HUB_CACHE%
echo [infer_ui] JTP3 model id: %FUSION_JTP3_MODEL_ID%
echo [infer_ui] JTP3 fallback id: %FUSION_JTP3_FALLBACK_MODEL_ID%
echo [infer_ui] waifu-head path: %FUSION_WAIFU_V3_HEAD_PATH%

if exist ".venv\Scripts\python.exe" (
  set "PY_CMD=.venv\Scripts\python.exe"
  goto :install_deps
)

where py >nul 2>nul
if not errorlevel 1 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    set "FAIL_REASON=python_not_found"
    goto :fail
  )
  set "PY_CMD=python"
)

echo [infer_ui] creating .venv ...
%PY_CMD% -m venv .venv
if errorlevel 1 (
  set "FAIL_REASON=venv_create_failed"
  goto :fail
)
set "PY_CMD=.venv\Scripts\python.exe"

:install_deps
echo [infer_ui] install deps ...
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
if exist ".venv\Lib\site-packages" (
  for /d %%D in (".venv\Lib\site-packages\~ip*") do (
    echo [infer_ui] cleanup broken package metadata: %%~nxD
    rmdir /S /Q "%%~fD" >nul 2>nul
  )
  for %%F in (".venv\Lib\site-packages\~ip*") do (
    if exist "%%~fF" (
      echo [infer_ui] cleanup broken package metadata file: %%~nxF
      del /Q "%%~fF" >nul 2>nul
    )
  )
)
%PY_CMD% -m pip install --upgrade pip
if errorlevel 1 (
  set "FAIL_REASON=pip_upgrade_failed"
  goto :fail
)
%PY_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
  set "FAIL_REASON=pip_install_failed"
  goto :fail
)
call :maybe_fix_cuda_torch
if errorlevel 1 (
  set "FAIL_REASON=cuda_torch_fix_failed"
  goto :fail
)

if not "%~1"=="" (
  set "INFER_UI_CHECKPOINT=%~1"
  echo [infer_ui] checkpoint arg: %INFER_UI_CHECKPOINT%
) else (
  echo [infer_ui] No checkpoint arg. Set checkpoint in WebUI.
)

set "WEBUI_HOST=127.0.0.1"
set "WEBUI_PORT=9400"
set "WEBUI_RESOLVE_TMP=%TEMP%\infer_ui_webui_%RANDOM%%RANDOM%.txt"
%PY_CMD% "scripts\resolve_webui_port.py" --config "config.yaml" > "%WEBUI_RESOLVE_TMP%" 2>nul
if exist "%WEBUI_RESOLVE_TMP%" (
  for /f "usebackq tokens=1,2" %%a in ("%WEBUI_RESOLVE_TMP%") do (
    set "WEBUI_HOST=%%a"
    set "WEBUI_PORT=%%b"
  )
  del /q "%WEBUI_RESOLVE_TMP%" >nul 2>nul
)
if /I "%WEBUI_HOST%"=="0.0.0.0" set "OPEN_HOST=127.0.0.1"
if not defined OPEN_HOST set "OPEN_HOST=%WEBUI_HOST%"

echo.
echo [infer_ui] start web ui ...
echo Infer UI: http://%OPEN_HOST%:%WEBUI_PORT%/
start "" "http://%OPEN_HOST%:%WEBUI_PORT%/"
%PY_CMD% -X utf8 run_web.py --config "config.yaml" --host %WEBUI_HOST% --port %WEBUI_PORT%
if errorlevel 1 (
  set "FAIL_REASON=run_failed"
  goto :fail
)

echo.
echo [infer_ui] done.
pause
exit /b 0

:fail
echo.
echo [ERROR] infer ui failed: %FAIL_REASON%
if "%FAIL_REASON%"=="python_not_found" echo Fix: install Python 3.10+.
if "%FAIL_REASON%"=="venv_create_failed" echo Fix: remove .venv and retry.
if "%FAIL_REASON%"=="pip_upgrade_failed" echo Fix: run .venv\Scripts\python.exe -m pip install --upgrade pip
if "%FAIL_REASON%"=="pip_install_failed" echo Fix: ensure internet/proxy for pip, or preinstall wheels.
if "%FAIL_REASON%"=="cuda_torch_fix_failed" echo Fix: install CUDA PyTorch manually or set FUSION_AUTO_FIX_CUDA_TORCH=0 to skip auto-fix.
if "%FAIL_REASON%"=="run_failed" echo Fix: check config/model path and traceback in console.
pause
exit /b 1

:maybe_fix_cuda_torch
set "HAS_NVIDIA=0"
set "TORCH_HAS_CUDA=0"
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "HAS_NVIDIA=1"
if "%HAS_NVIDIA%"=="0" goto :cuda_done

> ".venv\\_tmp_cuda_check.py" echo import torch
>> ".venv\\_tmp_cuda_check.py" echo print("1" if torch.cuda.is_available() else "0")
for /f %%A in ('%PY_CMD% ".venv\\_tmp_cuda_check.py" 2^>nul') do set "TORCH_HAS_CUDA=%%A"
del /Q ".venv\\_tmp_cuda_check.py" >nul 2>nul

if "%TORCH_HAS_CUDA%"=="1" (
  echo [infer_ui] CUDA torch detected.
  goto :cuda_done
)

if /I not "%FUSION_AUTO_FIX_CUDA_TORCH%"=="1" (
  echo [infer_ui] NVIDIA GPU detected but torch CUDA unavailable. Auto-fix disabled.
  goto :cuda_done
)

echo [infer_ui] NVIDIA GPU detected but torch CUDA unavailable. Auto-fixing torch...
%PY_CMD% -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
%PY_CMD% -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
if errorlevel 1 exit /b 1

set "TORCH_HAS_CUDA=0"
> ".venv\\_tmp_cuda_check.py" echo import torch
>> ".venv\\_tmp_cuda_check.py" echo print("1" if torch.cuda.is_available() else "0")
for /f %%A in ('%PY_CMD% ".venv\\_tmp_cuda_check.py" 2^>nul') do set "TORCH_HAS_CUDA=%%A"
del /Q ".venv\\_tmp_cuda_check.py" >nul 2>nul
if not "%TORCH_HAS_CUDA%"=="1" exit /b 1
echo [infer_ui] CUDA torch ready.

:cuda_done
exit /b 0
