@echo off
REM One-time setup: virtual environment, dependencies, API key, self-check
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found. Install it from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist .venv\Scripts\python.exe (
  if exist .venv rmdir /s /q .venv
  echo Creating virtual environment...
  python -m venv .venv || goto :fail
)
echo Installing packages (first time takes a few minutes)...
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.venv\Scripts\python.exe -m pip install --quiet -e ".[voiceprint]" || goto :fail
if not exist .env (
  set /p KEY=Paste your OpenAI API key and press Enter: 
  call echo OPENAI_API_KEY=%%KEY%%> .env
)
.venv\Scripts\minutesman.exe check || goto :fail
echo.
echo Setup done. Drag a recording onto transcribe.bat to transcribe it.
pause
exit /b 0
:fail
echo.
echo Setup failed, see the error above.
pause
exit /b 1
