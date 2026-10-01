@echo off
setlocal EnableExtensions
title Document Understanding Agent - Launcher
cd /d "%~dp0"

echo ==========================================================
echo   Document Understanding Agent for Handwritten Text
echo   Recognition - one-click launcher
echo ==========================================================
echo.
echo   This window only sets things up. The app itself runs in
echo   two other windows that stay open.
echo.

REM ---------- 1. Virtual environment -------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [OK]     Virtual environment already exists.
) else (
    echo [SETUP]  Creating virtual environment .venv ... one time only.
    python -m venv .venv 2>nul || py -3 -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo.
        echo [ERROR] Could not create .venv - Python was not found.
        echo         Install Python 3.10+ from https://www.python.org
        echo         and tick "Add python.exe to PATH" during setup.
        echo.
        pause
        exit /b 1
    )
    echo [OK]     Virtual environment created.
)

set "PY=.venv\Scripts\python.exe"

REM ---------- 2. Installed packages --------------------------------
echo [CHECK]  Verifying installed packages ...
%PY% -c "import importlib.metadata as m; [m.version(p) for p in ['fastapi','uvicorn','python-multipart','pydantic','pydantic-settings','python-dotenv','requests','PyMuPDF','python-docx','torch','transformers','opencv-python-headless','Pillow','numpy','chromadb','rank-bm25','sentence-transformers']]" >nul 2>&1
if errorlevel 1 (
    echo [SETUP]  Missing packages found. Installing requirements ...
    echo          Needs internet. One time only.
    %PY% -m pip install -r requirements.txt
    %PY% -c "import importlib.metadata as m; [m.version(p) for p in ['fastapi','uvicorn','python-multipart','pydantic','pydantic-settings','python-dotenv','requests','PyMuPDF','python-docx','torch','transformers','opencv-python-headless','Pillow','numpy','chromadb','rank-bm25','sentence-transformers']]" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [ERROR] Installation did not finish correctly.
        echo         Check the pip messages above, then run start.bat again.
        echo.
        pause
        exit /b 1
    )
    echo [OK]     All requirements installed.
) else (
    echo [OK]     All requirements already installed.
)

REM ---------- 3. Ollama ----------------------------------------------
curl -s -o nul -m 3 http://127.0.0.1:11434/api/tags
if errorlevel 1 (
    echo [WARN]   Ollama is not running - chat answers will not work.
    echo          Start it with:   ollama serve
    echo          Then pull once:  ollama pull qwen2.5:1.5b
) else (
    echo [OK]     Ollama is running.
)
echo.

REM ---------- 4. Backend window --------------------------------------
echo [START]  Backend  window - http://127.0.0.1:8000
start "DUA Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

REM ---------- 5. Frontend window -------------------------------------
echo [START]  Frontend window - http://localhost:5500
start "DUA Frontend" cmd /k ".venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory frontend"

REM ---------- 6. Wait for the backend, then open Edge ---------------
echo [WAIT]   Waiting for the backend to come up ...
set /a TRIES=0
:wait_backend
ping -n 3 127.0.0.1 >nul
curl -s -o nul -m 3 http://127.0.0.1:8000/health
if not errorlevel 1 goto backend_ready
set /a TRIES+=1
if %TRIES% LSS 40 goto wait_backend
echo [ERROR]  The backend did not answer within about two minutes.
echo          Look at the "DUA Backend" window for the real error.
goto open_browser

:backend_ready
echo [OK]     Backend is up.

:open_browser
set "EDGE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not exist "%EDGE%" set "EDGE=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if exist "%EDGE%" (
    echo [OPEN]   Opening Microsoft Edge ...
    start "" "%EDGE%" "http://localhost:5500/"
) else (
    echo [OPEN]   Microsoft Edge not found - opening the default browser ...
    start "" "http://localhost:5500/"
)

echo.
echo ==========================================================
echo   The app is running. Keep these two windows open:
echo     - DUA Backend   - the API. Closing it stops answering.
echo     - DUA Frontend  - the web page server.
echo   Both are local only; nothing leaves this computer.
echo ==========================================================
echo.
echo   This launcher window closes in 8 seconds.
ping -n 9 127.0.0.1 >nul
endlocal
exit /b 0
