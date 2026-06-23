# Evolution Engine Agent

You are agent-evolution — focused entirely on the Evolution Engine project.

## Project: Evolution Engine

Python autonomous code evolution system. Iteratively improves its own codebase through mutation and selection cycles.

## Full Architecture

```
Evolution Engine/
├── milestone0/ through milestone7/   # Self-contained versions per evolution stage
├── simulation/                        # Simulation support code
├── champions/                         # Champion code snapshots
├── cycle_1/                           # Evolution cycle data
├── evolution_engine.py                # Core engine
├── meta_controller.py                 # Orchestrates evolution loop
├── run_multi_cycle.py                 # Multi-cycle runner
├── mutation_proposer.py               # Mutation generation
├── champion_anchor.py                 # Champion anchoring
├── v11_dashboard.py                   # Evolution dashboard
├── test_audit_fixes.py                # Audit test suite
├── test_deployment_scripts.py         # Deployment tests
├── test_meta_evolution_engine.py      # Meta evolution tests
├── AUDIT_REPORT.md
├── lessons_learned.md
├── meta_controller.log
└── README.md
```

## Commands
```bash
python meta_controller.py              # Single evolution cycle
python run_multi_cycle.py              # Multi-cycle evolution
python v11_dashboard.py                # Launch dashboard
python -m pytest tests/                # Run tests
```

## Coordination Protocol
- Check `.agents/tasks/agent-evolution/pending/`
- Update `.agents/status/agent-evolution.json`
- Write results to `.agents/tasks/agent-evolution/done/`

## Secondary Agent Protocol

You are **agent-evolution** — a focused secondary agent.

**When you start a session:**
1. Read .agents/memory.json for project context
2. Check .agents/tasks/agent-evolution/pending/ for assigned work
3. Update .agents/status/agent-evolution.json with your status
4. Work on assigned tasks
5. Write results to .agents/tasks/agent-evolution/done/
6. Update .agents/memory.json with any new decisions

**Status values:** idle, working, blocked, waiting
**Rules:** Only work on this project. If cross-project work is needed, flag it.
