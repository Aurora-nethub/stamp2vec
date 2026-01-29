@echo off
setlocal

set ROOT_DIR=%~dp0

if not exist "%ROOT_DIR%.venv\\Scripts\\activate.bat" (
  echo Missing .venv. Please create it first.
  exit /b 1
)

call "%ROOT_DIR%.venv\\Scripts\\activate.bat"

echo Initializing Milvus database...
python "%ROOT_DIR%scripts\\init_milvus.py"

if %ERRORLEVEL% neq 0 (
  echo Milvus initialization failed
  exit /b 1
)

echo Starting Seal Embedding API...
python "%ROOT_DIR%main.py"
