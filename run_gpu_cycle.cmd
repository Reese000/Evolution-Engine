@echo off
cd /d "C:\Users\reese\Projects\Evolution Engine"
C:\Users\reese\miniconda3\python.exe run_multi_cycle.py --cycles 5 --gens 100 --clean-state 2>&1
echo EXIT_CODE=%ERRORLEVEL%
