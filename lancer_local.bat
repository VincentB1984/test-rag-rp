@echo off
echo ============================================================
echo   RAG Recensement - Lancement local
echo ============================================================
echo.

REM Vérifier que Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR : Python n'est pas installé ou pas dans le PATH.
    pause
    exit /b 1
)

REM Installer les dépendances si nécessaire
echo Installation des dépendances...
pip install fastapi uvicorn langchain langchain-community langchain-text-splitters langchain-openai langchain-huggingface "huggingface-hub>=0.33.4,<1.0.0" sentence-transformers faiss-cpu pypdf python-pptx lxml ddgs rank-bm25 unstructured python-dotenv

echo.
echo Démarrage du serveur...
echo Ouvrez votre navigateur sur : http://localhost:8000
echo Appuyez sur Ctrl+C pour arrêter.
echo.

uvicorn app:app --host 0.0.0.0 --port 8000

pause
