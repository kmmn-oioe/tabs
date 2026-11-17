@echo off
setlocal
cd /d "%~dp0"
set "HANDOFF_DEST=%~dp0tabs"

if not exist "%~dp0tabs.git.bundle" (
    echo Missing tabs.git.bundle. Extract the complete private handoff ZIP first.
    pause
    exit /b 1
)
if not exist "%~dp0local-input\manifest.json" (
    echo Missing local-input\manifest.json. Extract the complete handoff ZIP first.
    pause
    exit /b 1
)
where git.exe >nul 2>nul
if errorlevel 1 (
    echo Install Git for Windows, then run this script again:
    echo https://git-scm.com/download/win
    pause
    exit /b 1
)
if exist "%HANDOFF_DEST%" (
    echo The tabs folder already exists. It will NOT be overwritten.
    echo Extract the handoff ZIP into a new empty location and try again.
    pause
    exit /b 1
)

git clone --branch improve-playable-tabs "%~dp0tabs.git.bundle" "%HANDOFF_DEST%"
if errorlevel 1 goto failed
git -C "%HANDOFF_DEST%" remote set-url origin https://github.com/kmmn-oioe/tabs.git
if errorlevel 1 goto failed
xcopy "%~dp0local-input" "%HANDOFF_DEST%\local-input" /E /I /Y >nul
if errorlevel 1 goto failed
git -C "%HANDOFF_DEST%" branch --show-current
git -C "%HANDOFF_DEST%" log -1 --oneline
git -C "%HANDOFF_DEST%" status --short
echo.
echo Source history and private test data are ready in:
echo "%HANDOFF_DEST%"
echo Read docs\CODEX_HANDOFF.md and docs\WINDOWS_HANDOFF.md in that folder.
echo Python, dependencies and Codex CLI have NOT been installed by this script.
pause
exit /b 0

:failed
echo.
echo Preparation stopped. Existing files have not been deleted.
echo Read the error above. Use a new empty extraction folder for another attempt.
pause
exit /b 1
