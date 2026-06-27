@echo off
cd /d "C:\Users\reese\Projects\Evolution Engine"
python -B -u run_multi_cycle.py --cycles 20 --gens 100 > engine_run_full.log 2>&1
