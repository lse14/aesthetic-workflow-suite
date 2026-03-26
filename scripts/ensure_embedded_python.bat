@echo off
setlocal EnableExtensions

set "TARGET_DIR=%~1"
if "%TARGET_DIR%"=="" set "TARGET_DIR=%~dp0..\runtime\python"
for %%I in ("%TARGET_DIR%") do set "TARGET_DIR=%%~fI"

if not defined EMBED_PY_VERSION set "EMBED_PY_VERSION=3.13.9"

set "ARCH_TAG=amd64"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH_TAG=arm64"
if /I "%PROCESSOR_ARCHITECTURE%"=="x86" set "ARCH_TAG=win32"

set "ZIP_NAME=python-%EMBED_PY_VERSION%-embed-%ARCH_TAG%.zip"
set "PY_URL=https://www.python.org/ftp/python/%EMBED_PY_VERSION%/%ZIP_NAME%"
set "PY_EXE=%TARGET_DIR%\python.exe"

if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%" >nul 2>nul

if exist "%PY_EXE%" goto :configure

echo [embedded-python] downloading %ZIP_NAME% ...
set "TMP_ZIP=%TEMP%\%ZIP_NAME%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%TMP_ZIP%';"
if errorlevel 1 (
  echo [embedded-python] download failed: %PY_URL%
  goto :fail
)

echo [embedded-python] extracting to %TARGET_DIR% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; Expand-Archive -Path '%TMP_ZIP%' -DestinationPath '%TARGET_DIR%' -Force;"
if errorlevel 1 (
  echo [embedded-python] extract failed.
  goto :fail
)

del /Q "%TMP_ZIP%" >nul 2>nul

if not exist "%PY_EXE%" (
  echo [embedded-python] python.exe missing after extract.
  goto :fail
)

:configure
for %%F in ("%TARGET_DIR%\python*._pth") do (
  set "PTH_FILE=%%~fF"
  goto :got_pth
)
set "PTH_FILE="

:got_pth
if not "%PTH_FILE%"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$p='%PTH_FILE%'; $t=Get-Content -Raw -Encoding UTF8 $p; $t=$t -replace '(?m)^\s*#\s*import\s+site\s*$','import site'; Set-Content -Encoding UTF8 -NoNewline -Path $p -Value $t;"
)

echo [embedded-python] ensuring pip ...
"%PY_EXE%" -m pip --version >nul 2>nul
if errorlevel 1 (
  set "GET_PIP=%TEMP%\get-pip.py"
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop'; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GET_PIP%';"
  if errorlevel 1 (
    echo [embedded-python] get-pip download failed.
    goto :fail
  )
  "%PY_EXE%" "%GET_PIP%" --no-warn-script-location
  if errorlevel 1 (
    echo [embedded-python] pip bootstrap failed.
    goto :fail
  )
  del /Q "%GET_PIP%" >nul 2>nul
)

endlocal & set "EMBED_PYTHON_EXE=%PY_EXE%" & exit /b 0

:fail
endlocal & exit /b 1
