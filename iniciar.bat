@echo off
chcp 65001 >NUL
title ollamaRAG
cd /d "%~dp0"

echo ============================================================
echo   Iniciando ollamaRAG
echo ============================================================

REM --- 1) Ollama: arrancarlo si no esta corriendo ---
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if errorlevel 1 (
    echo [1/4] Ollama no esta corriendo. Iniciando...
    start "" ollama serve
    timeout /t 4 >NUL
) else (
    echo [1/4] Ollama ya esta corriendo.
)

REM --- 2) Precargar el modelo en GPU (queda fijo por OLLAMA_KEEP_ALIVE=-1) ---
echo [2/4] Precargando gemma4:e2b en GPU...
curl -s http://localhost:11434/api/generate -d "{\"model\":\"gemma4:e2b\",\"keep_alive\":-1}" >NUL 2>&1

REM --- 3) Abrir el chat en el navegador cuando la API este lista (en paralelo) ---
echo [3/4] Programando apertura del navegador...
start "" cmd /c "timeout /t 6 >NUL & start http://localhost:8000"

REM --- 4) Activar el entorno y levantar la API (en ESTA ventana) ---
echo [4/4] Levantando API en http://localhost:8000
echo ============================================================
echo  (Para apagar todo: cerra esta ventana o presiona Ctrl+C)
echo ============================================================
call venv\Scripts\activate.bat
python api.py
