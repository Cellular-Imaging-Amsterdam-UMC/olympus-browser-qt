@echo off
setlocal
rem Adapted from omero-browser-qt's run_viewer.cmd for olympus-browser-qt.

set "REPO_DIR=%~dp0"
set "PYTHONPATH=%REPO_DIR%src;%PYTHONPATH%"
set "DECONVOLVE_ENV=C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve"
set "DECONVOLVE_PYTHON=%DECONVOLVE_ENV%\python.exe"

if exist "%DECONVOLVE_PYTHON%" (
    "%DECONVOLVE_PYTHON%" -m olympus_browser_qt.olympus_viewer %*
) else if defined CONDA_EXE (
    "%CONDA_EXE%" run -n deconvolve python -m olympus_browser_qt.olympus_viewer %*
) else (
    conda run -n deconvolve python -m olympus_browser_qt.olympus_viewer %*
)
