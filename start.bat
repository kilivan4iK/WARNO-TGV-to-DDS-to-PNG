@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% -c "import zstandard" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python module "zstandard" not found.
  echo Run: %PY% -m pip install zstandard
  pause
  exit /b 1
)

%PY% -c "from PIL import Image" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python module "Pillow" not found.
  echo Run: %PY% -m pip install pillow
  pause
  exit /b 1
)

if not exist "png_out" mkdir "png_out"

set "MIRROR_FLAG="
set /p "MIRROR_INPUT=Mirror textures horizontally? [y/N]: "
if /I "!MIRROR_INPUT!"=="y" set "MIRROR_FLAG=--mirror"
if /I "!MIRROR_INPUT!"=="yes" set "MIRROR_FLAG=--mirror"

echo Converting all .tgv in: %CD%
echo.

%PY% tgv_to_png.py "." "png_out" --split auto --recursive !MIRROR_FLAG!

echo.
echo Done! PNG files are in: png_out
pause
popd
