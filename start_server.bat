@echo off
echo 启动Whisper语音识别MCP服务器...

:: 激活虚拟环境（如果存在）
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
)

:: 运行MCP服务器
python whisper_server.py

:: 如果出错，暂停以查看错误信息
if %ERRORLEVEL% neq 0 (
    echo 服务器启动失败，错误代码: %ERRORLEVEL%
    pause
)
