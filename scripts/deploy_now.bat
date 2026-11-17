@echo off
cd /d "c:\Users\reese\Projects\Evolution Engine"
echo === pytest === > _deploy_out.txt
python -m pytest tests/unit/test_progress_controller.py tests/unit/test_collapse_reseed.py tests/unit/test_epi_momentum_governor.py tests/unit/test_clean_depth_escape.py -q >> _deploy_out.txt 2>&1
echo === restart === >> _deploy_out.txt
powershell -ExecutionPolicy Bypass -File "scripts\restart_evolution_daemon.ps1" >> _deploy_out.txt 2>&1
echo === health === >> _deploy_out.txt
python scripts\verify_loop_health.py --repair >> _deploy_out.txt 2>&1
echo === epi === >> _deploy_out.txt
python scripts\verify_epi_momentum_goal.py --json >> _deploy_out.txt 2>&1
type _deploy_out.txt
