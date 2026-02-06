@echo off
cd /d %~dp0
echo ============================================
echo  RAG Document Intelligence System
echo  LLM: Groq Cloud (llama-3.3-70b-versatile)
echo ============================================
echo.
echo Open http://localhost:8000/docs in your browser
echo.
python main.py
pause
