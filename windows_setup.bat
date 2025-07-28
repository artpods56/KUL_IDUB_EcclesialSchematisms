@echo off
REM ============================================================
REM  AI_Osrodek – one-click Windows setup
REM ------------------------------------------------------------
REM  • Creates an isolated Python venv in .venv
REM  • Upgrades pip / build tools
REM  • Installs project dependencies from pyproject.toml
REM ============================================================

REM --- 1. Check that Python 3.12+ is available ----------------
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python not found on PATH. ^
          Please install Python 3.12 or newer from https://python.org and re-run this script.
    pause
    exit /b 1
)

REM --- 2. Create virtual environment -------------------------
if exist .venv (
    echo .venv already exists – skipping creation.
) else (
    echo Creating virtual environment in %%CD%%\.venv ...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM --- 3. Activate venv --------------------------------------
call ".venv\Scripts\activate.bat"

REM --- 4. Upgrade packaging tools ----------------------------
echo Upgrading pip, setuptools, wheel ...
python -m pip install --upgrade pip setuptools wheel

REM --- 5. Install project in editable mode -------------------
echo Installing project dependencies ...
python -m pip install --upgrade .
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

REM --- 6. Finished -------------------------------------------
echo.
echo ============================================================
echo  Setup completed successfully.
echo.
echo  To activate this environment in a new terminal:
echo     call %%CD%%\.venv\Scripts\activate.bat

echo  To run the inference script:
echo     python src\scripts\inference.py

echo ============================================================

pause 