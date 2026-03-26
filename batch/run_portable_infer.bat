@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY_CMD="
set "RUN_PY="
set "USE_EMBEDDED=0"
set "FAIL_REASON="
if not defined FUSION_AUTO_FIX_CUDA_TORCH set "FUSION_AUTO_FIX_CUDA_TORCH=1"
if not defined FUSION_PREFETCH_JTP3 set "FUSION_PREFETCH_JTP3=1"
if not defined FUSION_PREFETCH_OPENCLIP set "FUSION_PREFETCH_OPENCLIP=1"
if not defined FUSION_PREFETCH_WAIFU_HEAD set "FUSION_PREFETCH_WAIFU_HEAD=1"
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

echo [portable] model cache root: %FUSION_MODEL_CACHE_ROOT%
echo [portable] HF_HOME: %HF_HOME%
echo [portable] HF_HUB_CACHE: %HF_HUB_CACHE%
echo [portable] JTP3 model id: %FUSION_JTP3_MODEL_ID%
echo [portable] JTP3 fallback id: %FUSION_JTP3_FALLBACK_MODEL_ID%
echo [portable] waifu-head path: %FUSION_WAIFU_V3_HEAD_PATH%
echo [portable] Prefetch open-clip: %FUSION_PREFETCH_OPENCLIP%
echo [portable] Prefetch waifu-head: %FUSION_PREFETCH_WAIFU_HEAD%

set "EMBED_PY=%CD%\runtime\python\python.exe"
if exist "%EMBED_PY%" (
  set "USE_EMBEDDED=1"
  set "PY_CMD=%EMBED_PY%"
  set "RUN_PY=%EMBED_PY%"
  echo [portable] using embedded runtime: %RUN_PY%
  goto :install_deps
)
set "EMBED_PY=%CD%\..\runtime\python\python.exe"
if exist "%EMBED_PY%" (
  set "USE_EMBEDDED=1"
  set "PY_CMD=%EMBED_PY%"
  set "RUN_PY=%EMBED_PY%"
  echo [portable] using embedded runtime: %RUN_PY%
  goto :install_deps
)
set "EMBED_HELPER=%CD%\..\scripts\ensure_embedded_python.bat"
if exist "%EMBED_HELPER%" (
  call "%EMBED_HELPER%" "%CD%\..\runtime\python"
  if not errorlevel 1 if defined EMBED_PYTHON_EXE (
    set "USE_EMBEDDED=1"
    set "PY_CMD=%EMBED_PYTHON_EXE%"
    set "RUN_PY=%EMBED_PYTHON_EXE%"
    echo [portable] using embedded runtime: %RUN_PY%
    goto :install_deps
  )
)

if exist ".venv\Scripts\python.exe" (
  set "PY_CMD=.venv\Scripts\python.exe"
  set "RUN_PY=.venv\Scripts\python.exe"
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

echo [portable] creating .venv ...
%PY_CMD% -m venv .venv
if errorlevel 1 (
  set "FAIL_REASON=venv_create_failed"
  goto :fail
)
set "PY_CMD=.venv\Scripts\python.exe"
set "RUN_PY=.venv\Scripts\python.exe"

:install_deps
echo [portable] install deps ...
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
if exist ".venv\Lib\site-packages" (
  for /d %%D in (".venv\Lib\site-packages\~ip*") do (
    echo [portable] cleanup broken package metadata: %%~nxD
    rmdir /S /Q "%%~fD" >nul 2>nul
  )
  for %%F in (".venv\Lib\site-packages\~ip*") do (
    if exist "%%~fF" (
      echo [portable] cleanup broken package metadata file: %%~nxF
      del /Q "%%~fF" >nul 2>nul
    )
  )
)
%RUN_PY% -m pip install --upgrade pip
if errorlevel 1 (
  set "FAIL_REASON=pip_upgrade_failed"
  goto :fail
)
%RUN_PY% -m pip install -r requirements.txt
if errorlevel 1 (
  set "FAIL_REASON=pip_install_failed"
  goto :fail
)
call :maybe_fix_cuda_torch
if errorlevel 1 (
  set "FAIL_REASON=cuda_torch_fix_failed"
  goto :fail
)
call :maybe_prefetch_jtp3
if errorlevel 1 (
  set "FAIL_REASON=prefetch_jtp3_failed"
  goto :fail
)

echo.
if "%~1"=="" (
  echo [portable] start GUI sorter ...
  %RUN_PY% -X utf8 sort_images_by_score.py --gui
) else (
  echo [portable] start sorter with args ...
  %RUN_PY% -X utf8 sort_images_by_score.py %*
)
if errorlevel 1 (
  set "FAIL_REASON=run_failed"
  goto :fail
)

echo.
echo [portable] done.
pause
exit /b 0

:fail
echo.
echo [ERROR] portable infer failed: %FAIL_REASON%
if "%FAIL_REASON%"=="python_not_found" echo Fix: install Python 3.10+.
if "%FAIL_REASON%"=="venv_create_failed" echo Fix: remove .venv and retry.
if "%FAIL_REASON%"=="pip_upgrade_failed" echo Fix: run .venv\Scripts\python.exe -m pip install --upgrade pip
if "%FAIL_REASON%"=="pip_install_failed" echo Fix: ensure internet/proxy for pip, or preinstall wheels.
if "%FAIL_REASON%"=="cuda_torch_fix_failed" echo Fix: install CUDA PyTorch manually or set FUSION_AUTO_FIX_CUDA_TORCH=0 to skip auto-fix.
if "%FAIL_REASON%"=="prefetch_jtp3_failed" echo Fix: check network/HF token, or set FUSION_PREFETCH_JTP3=0 to skip prefetch.
if "%FAIL_REASON%"=="run_failed" echo Fix: check path/model arguments in console.
pause
exit /b 1

:maybe_fix_cuda_torch
set "HAS_NVIDIA=0"
set "TORCH_HAS_CUDA=0"
set "CUDA_CHECK_PY=%TEMP%\batch_cuda_check_%RANDOM%%RANDOM%.py"
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "HAS_NVIDIA=1"
if "%HAS_NVIDIA%"=="0" goto :cuda_done

> "%CUDA_CHECK_PY%" echo import torch
>> "%CUDA_CHECK_PY%" echo print("1" if torch.cuda.is_available() else "0")
for /f %%A in ('%RUN_PY% "%CUDA_CHECK_PY%" 2^>nul') do set "TORCH_HAS_CUDA=%%A"
del /Q "%CUDA_CHECK_PY%" >nul 2>nul

if "%TORCH_HAS_CUDA%"=="1" (
  echo [portable] CUDA torch detected.
  goto :cuda_done
)

if /I not "%FUSION_AUTO_FIX_CUDA_TORCH%"=="1" (
  echo [portable] NVIDIA GPU detected but torch CUDA unavailable. Auto-fix disabled.
  goto :cuda_done
)

echo [portable] NVIDIA GPU detected but torch CUDA unavailable. Auto-fixing torch...
%RUN_PY% -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
%RUN_PY% -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
if errorlevel 1 exit /b 1

set "TORCH_HAS_CUDA=0"
> "%CUDA_CHECK_PY%" echo import torch
>> "%CUDA_CHECK_PY%" echo print("1" if torch.cuda.is_available() else "0")
for /f %%A in ('%RUN_PY% "%CUDA_CHECK_PY%" 2^>nul') do set "TORCH_HAS_CUDA=%%A"
del /Q "%CUDA_CHECK_PY%" >nul 2>nul
if not "%TORCH_HAS_CUDA%"=="1" exit /b 1
echo [portable] CUDA torch ready.

:cuda_done
exit /b 0

:maybe_prefetch_jtp3
if /I not "%FUSION_PREFETCH_JTP3%"=="1" (
  echo [portable] skip JTP3 prefetch ^(FUSION_PREFETCH_JTP3=%FUSION_PREFETCH_JTP3%^).
  exit /b 0
)
set "JTP3_DIR=%FUSION_MODEL_CACHE_ROOT%\repos\RedRocket__JTP-3"
set "WAIFU_HEAD_FILE=%FUSION_MODEL_CACHE_ROOT%\waifu-scorer-v3\model.safetensors"
set "NEED_PREFETCH=1"
if exist "%JTP3_DIR%\model.py" if exist "%JTP3_DIR%\models\jtp-3-hydra.safetensors" (
  set "NEED_PREFETCH=0"
)
if /I "%FUSION_PREFETCH_WAIFU_HEAD%"=="1" (
  if not exist "%WAIFU_HEAD_FILE%" set "NEED_PREFETCH=1"
)
if "%NEED_PREFETCH%"=="0" (
  echo [portable] JTP3/waifu cache exists, skip prefetch.
  exit /b 0
)
echo [portable] prefetch JTP3 base...
set "PREFETCH_OPENCLIP_ARG="
if /I not "%FUSION_PREFETCH_OPENCLIP%"=="1" set "PREFETCH_OPENCLIP_ARG=--no-prefetch-openclip"
set "PREFETCH_WAIFU_ARG="
if /I not "%FUSION_PREFETCH_WAIFU_HEAD%"=="1" set "PREFETCH_WAIFU_ARG=--no-prefetch-waifu-head"
%RUN_PY% -X utf8 runtime\prefetch_jtp3.py --root "%FUSION_MODEL_CACHE_ROOT%" --repo-id "%FUSION_JTP3_MODEL_ID%" %PREFETCH_OPENCLIP_ARG% %PREFETCH_WAIFU_ARG%
if errorlevel 1 exit /b 1
exit /b 0
