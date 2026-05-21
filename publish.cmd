@echo off
REM -- publish.cmd -- Build and upload olympus-browser-qt to PyPI --
REM
REM Usage:
REM   publish.cmd          Upload to PyPI (production)
REM   publish.cmd test     Upload to TestPyPI first
REM
REM Before first use:
REM   pip install build twine
REM   Create an API token at https://pypi.org/manage/account/token/
REM   (optional) store it:  keyring set https://upload.pypi.org/legacy/ __token__

setlocal

set "DECONVOLVE_PYTHON=C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve\python.exe"
if exist "%DECONVOLVE_PYTHON%" (
    set "PYTHON=%DECONVOLVE_PYTHON%"
) else (
    set "PYTHON=python"
)

REM 1. Clean previous builds
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM 2. Build sdist + wheel
echo === Building package ===
"%PYTHON%" -m build
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

REM 3. Validate
echo === Checking dist ===
"%PYTHON%" -m twine check dist\*
if errorlevel 1 (
    echo TWINE CHECK FAILED
    exit /b 1
)

REM 4. Upload
if /i "%~1"=="test" (
    echo === Uploading to TestPyPI ===
    "%PYTHON%" -m twine upload --verbose --repository testpypi dist\*
) else (
    echo === Uploading to PyPI ===
    "%PYTHON%" -m twine upload --verbose dist\*
)

echo === Done ===
endlocal
