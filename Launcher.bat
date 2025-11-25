@echo off
cd /d "%~dp0"

echo Activating Virtual Environment...
call venv\Scripts\activate

echo Running NLP Pipeline...
python src\main.py

echo.
echo Execution Complete.
pause