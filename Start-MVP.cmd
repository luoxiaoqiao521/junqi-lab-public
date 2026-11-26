@echo off
if "%~1"=="" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-mvp.ps1"
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-mvp.ps1" -ArchiveFile "%~1"
)
if errorlevel 1 pause
