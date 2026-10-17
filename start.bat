@echo off
cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt -q

if not exist .env (
    set /p KEY=Enter your GEMINI_API_KEY: 
    echo GEMINI_API_KEY=%KEY%> .env
)

echo Starting StyleSage...
streamlit run app.py
pause
