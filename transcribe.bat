@echo off
REM Transcribe a recording: drag it onto this file, or run transcribe.bat "path\to\file" [options]
cd /d "%~dp0"
if not exist .venv\Scripts\minutesman.exe (echo Run setup.bat first. & pause & exit /b 1)
if "%~1"=="" (echo Drag a recording onto transcribe.bat, or run: transcribe.bat "C:\path\to\recording.m4a" & pause & exit /b 1)
set "OPTS="
set "SAVED=output\%~n1\run_options.txt"
REM Options typed after the file name skip the questions
if not "%~2"=="" goto run
if exist "%SAVED%" (
  set /p OPTS=<"%SAVED%"
  call echo Previous answers for this recording: %%OPTS%%
  set "REUSE=Y"
  set /p REUSE=Use them again? [Y/n] 
  call :reuse
  if defined OPTS goto run
)
echo.
echo Optional details, press Enter to skip. Do not use quotes or the ^& character.
set "CTX="
set /p CTX=Meeting context, e.g. FBL group heads session with CIO, IT and Digital Banking: 
set "KW="
set /p KW=Names and terms, comma separated, e.g. Tahir,Athar,FBL,core banking: 
set "ENDAT="
set /p ENDAT=Meeting ended at, e.g. 1:52:00, if recording was left running: 
set "STARTAT="
set /p STARTAT=Meeting started at, e.g. 2:30, to skip what came before: 
if defined CTX set OPTS=%OPTS% --context "%CTX%"
if defined KW set OPTS=%OPTS% --keywords "%KW%"
if defined ENDAT set OPTS=%OPTS% --end %ENDAT%
if defined STARTAT set OPTS=%OPTS% --start %STARTAT%
if not exist "output\%~n1" mkdir "output\%~n1"
if defined OPTS (>"%SAVED%" echo %OPTS%)
:run
.venv\Scripts\minutesman.exe estimate "%~1"
.venv\Scripts\minutesman.exe run %* %OPTS% || (echo. & echo Transcription failed, see the error above. Run it again to resume from where it stopped. & pause & exit /b 1)
start "" "output\%~n1\transcript.md"
pause
exit /b 0

:reuse
if /i "%REUSE%"=="n" set "OPTS="
exit /b 0
