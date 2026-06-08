@echo off
cd /d "%~dp0.."

call .venv\Scripts\activate
streamlit run app.py --server.port 8503 --server.headless=true --browser.serverAddress=127.0.0.1

