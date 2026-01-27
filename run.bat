@echo off
setlocal

set ROOT_DIR=%~dp0

if not exist "%ROOT_DIR%.venv\\Scripts\\activate.bat" (
  echo Missing .venv. Please create it first.
  exit /b 1
)

call "%ROOT_DIR%.venv\\Scripts\\activate.bat"
python "%ROOT_DIR%main.py"
