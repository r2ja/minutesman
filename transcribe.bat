@echo off
REM Transcribe a recording: drag it onto this file, or run transcribe.bat "path\to\file" [options]
cd /d "%~dp0"
if not exist .venv\Scripts\minutesman.exe (echo Run setup.bat first. & pause & exit /b 1)
if "%~1"=="" (echo Drag a recording onto transcribe.bat, or run: transcribe.bat "C:\path\to\recording.m4a" & pause & exit /b 1)
.venv\Scripts\minutesman.exe estimate "%~1"
.venv\Scripts\minutesman.exe run %* || (echo. & echo Transcription failed, see the error above. Run it again to resume from where it stopped. & pause & exit /b 1)
start "" "output\%~n1\transcript.md"
pause
