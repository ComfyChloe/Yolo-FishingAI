@echo off
setlocal
echo ==========================
echo   Smart fella be fishing soon
echo ==========================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Pythonがインストールされていないか、PATHが通っていません。
    echo Pythonをインストールしてから再度実行してください。
    pause
    exit /b
)

python -c "import ultralytics" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing requirements...
    echo requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo requirements.txt?
        pause
        exit /b
    )
    echo [SUCCESS] Setup done.
) else (
    echo [INFO] OK
)

echo.
echo --------------------------------------------------
python main.py

echo.
echo --------------------------------------------------
echo Program is stop
pause