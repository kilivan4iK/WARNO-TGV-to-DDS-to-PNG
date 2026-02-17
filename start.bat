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

echo Converting all .tgv in: %CD%
echo.

for %%F in (*.tgv) do (
  echo [%%F] -^> png_out\%%~nF.png
  %PY% tgv_to_png.py "%%F" "png_out\%%~nF.png" --split auto
)

echo.
echo Done! PNG files are in: png_out
pause
popd
