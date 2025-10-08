@echo off
set "PYTHON=python"
set "SCRIPT=%~dp0smart_postprocess.py"
set "CONFIG=%~dp0smart_postprocess_config.json"
start "SmartWatcher" %PYTHON% "%SCRIPT%" --config "%CONFIG%"
echo Started watcher. Check smart_postprocess.log for details.
pause