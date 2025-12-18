@echo off
setlocal
REM Adjust if your Anaconda path differs:
set "ACT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not exist "%ACT%" (
  echo Cannot find: %ACT%
  echo Edit Start_RAGDesk.bat and set ACT to your activate.bat path.
  pause
  exit /b 1
)

call "%ACT%" ragdesk
cd /d "%~dp0"
pip install -r requirements.txt
start "" http://localhost:8501
python -m streamlit run app.py
