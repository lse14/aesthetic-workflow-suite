@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>nul

set "EMBED_HELPER=%ROOT%scripts\ensure_embedded_python.bat"
if not exist "%EMBED_HELPER%" (
  echo [ERROR] Missing helper: %EMBED_HELPER%
  goto :fail
)

echo [start_all] preparing embedded python runtime...
call "%EMBED_HELPER%" "%ROOT%runtime\python"
if errorlevel 1 (
  echo [ERROR] failed to prepare embedded python runtime.
  goto :fail
)
if defined EMBED_PYTHON_EXE echo [start_all] runtime ready: %EMBED_PYTHON_EXE%

set "TARGET=%~1"
if "%TARGET%"=="" goto :menu
call :launch_target "%TARGET%"
if errorlevel 1 goto :menu
goto :end

:menu
echo.
echo ===============================
echo Aesthetic Workflow Suite
echo Choose UI to launch:
echo   1. labeling_ui
echo   2. training_ui
echo   3. infer_ui
echo   4. batch sorter
echo   5. all UIs
echo   0. exit
echo ===============================
set "CHOICE="
set /p CHOICE=Enter choice [0-5]: 

if "%CHOICE%"=="1" call :launch_labeling & goto :end
if "%CHOICE%"=="2" call :launch_training & goto :end
if "%CHOICE%"=="3" call :launch_infer & goto :end
if "%CHOICE%"=="4" call :launch_batch & goto :end
if "%CHOICE%"=="5" call :launch_all & goto :end
if "%CHOICE%"=="0" goto :end

echo [WARN] invalid choice: %CHOICE%
goto :menu

:launch_target
set "T=%~1"
if /I "%T%"=="1" call :launch_labeling & exit /b 0
if /I "%T%"=="2" call :launch_training & exit /b 0
if /I "%T%"=="3" call :launch_infer & exit /b 0
if /I "%T%"=="4" call :launch_batch & exit /b 0
if /I "%T%"=="5" call :launch_all & exit /b 0
if /I "%T%"=="labeling" call :launch_labeling & exit /b 0
if /I "%T%"=="training" call :launch_training & exit /b 0
if /I "%T%"=="infer" call :launch_infer & exit /b 0
if /I "%T%"=="batch" call :launch_batch & exit /b 0
if /I "%T%"=="all" call :launch_all & exit /b 0
echo [WARN] unknown target: %T%
exit /b 1

:launch_labeling
echo [start_all] launching labeling_ui...
start "labeling_ui" cmd /k "cd /d \"%ROOT%labeling_ui\" && call start.bat"
exit /b 0

:launch_training
echo [start_all] launching training_ui...
start "training_ui" cmd /k "cd /d \"%ROOT%training_ui\" && call start.bat"
exit /b 0

:launch_infer
echo [start_all] launching infer_ui...
start "infer_ui" cmd /k "cd /d \"%ROOT%infer_ui\" && call start.bat"
exit /b 0

:launch_batch
echo [start_all] launching batch...
start "batch" cmd /k "cd /d \"%ROOT%batch\" && call run_portable_infer.bat"
exit /b 0

:launch_all
call :launch_labeling
call :launch_training
call :launch_infer
call :launch_batch
exit /b 0

:fail
echo.
echo Press any key to close...
pause >nul
popd >nul 2>nul
exit /b 1

:end
popd >nul 2>nul
endlocal
exit /b 0
