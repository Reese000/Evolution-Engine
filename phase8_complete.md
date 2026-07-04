# Phase 8 (Production Hardening) — Complete

**Date:** 2026-06-27
**Commit:** (pending)
**Tests:** 198/198 passing (63 new + 135 existing)

---

## What Was Done

### 8.1 Docker Setup

#### `Dockerfile` (NEW ~60 lines)
Multi-stage build with CPU/GPU variants:
- **Base:** `python:3.11-slim` (CPU) or `nvidia/cuda:12.1.1-runtime-ubuntu22.04` (GPU)
- **Build arg:** `CUDA=true/false` to select variant
- **Non-root user:** `evo` user for security
- **Layer caching:** `requirements.txt` copied first for fast rebuilds
- **Health check:** Built-in `HEALTHCHECK` directive reads `health.json`
- **Entrypoint:** `python autonomous_runner.py --resume`

```bash
# CPU build
docker build -t evolution-engine .

# GPU build
docker build --build-arg CUDA=true -t evolution-engine-gpu .
```

#### `docker-compose.yml` (NEW ~70 lines)
Full orchestration with:
- **Volume mounts** for: cycle data, checkpoints, metrics, logs, config, state files
- **Environment variables** for all config overrides (EE_MAX_CYCLES, EE_TARGET_FITNESS, etc.)
- **Webhook env vars:** EE_DISCORD_WEBHOOK, EE_SLACK_WEBHOOK
- **Resource limits:** 8GB memory limit, 2GB reservation
- **Docker health check** with 60s interval
- **Log rotation:** json-file driver, 10MB max, 5 files
- **GPU profile** (commented, ready to uncomment)

```bash
# Start
docker compose up -d

# Start with GPU
docker compose --profile gpu up -d --build

# Stop
docker compose down
```

---

### 8.2 Monitoring Alerts

#### `monitoring_alerts.py` (NEW ~270 lines)

**`AlertThresholds` dataclass:**
| Threshold | Default | Purpose |
|-----------|---------|---------|
| `fitness_regression_pct` | 20% | Trigger when fitness drops this much from avg |
| `oom_per_cycle_limit` | 3 | Max OOMs before critical alert |
| `crash_loop_consecutive` | 5 | Consecutive errors before crash loop alert |
| `crash_loop_window_minutes` | 30 | Time window for crash detection |
| `max_alerts_per_hour` | 10 | Rate limiting |
| `health_stale_seconds` | 300 | Alert if health.json not updated in 5 min |

**`AlertManager` class:**

| Method | Purpose |
|--------|---------|
| `send_alert(level, title, message)` | Send to all configured channels |
| `send_fitness_regression(current, prev_avg, cycle)` | Alert on fitness drop |
| `send_oom_alert(cycle, pop, consecutive)` | Alert on OOM with escalation |
| `send_crash_loop_alert(errors, cycle)` | Alert on repeated crashes |
| `send_health_stale_alert(last_update_s)` | Alert on stale health.json |
| `send_retrain_failure_alert(cycle, attempts, reason)` | Alert on retrain failure |
| `get_alert_history()` | Session alert log |
| `get_config_summary()` | Show configured channels |

**Delivery channels:**
- **Discord:** Color-coded embeds via webhook (green/yellow/red)
- **Slack:** Formatted messages with emoji severity indicators
- **Email:** SMTP with TLS (critical alerts only, to avoid spam)

**Rate limiting:** Max 10 alerts/hour to prevent webhook flooding.

---

### 8.3 Log Rotation Tuning

#### `log_rotation.py` (NEW ~210 lines)

**`CompressedRotatingFileHandler`:**
- Extends Python's `RotatingFileHandler`
- Automatically gzip-compresses rotated log files
- Default: 10MB max size, 10 backups

**`setup_production_logging()`** creates separate files per component:
| File | Content | Level |
|------|---------|-------|
| `logs/evolution.log` | All evolution engine messages | DEBUG+ |
| `logs/retrain.log` | Training/retrain messages | DEBUG+ |
| `logs/errors.log` | Errors only | ERROR+ |
| `logs/alerts.log` | Alert system messages | DEBUG+ |
| `logs/monitoring.log` | Health/metrics messages | DEBUG+ |
| `logs/evolution.log` (root) | Catch-all (backward compat) | DEBUG+ |

**`compress_old_logs(log_dir, max_age_days=30)`:**
- Scans for `.log` files older than threshold
- Gzip compresses them in place
- Returns list of compressed paths

**`get_log_stats(log_dir)`:**
- Returns file count, total size, compressed count, oldest file, per-component breakdown

---

### 8.4 Deployment Scripts

#### `deploy.sh` (NEW ~100 lines)
One-command deployment with options:
```bash
./deploy.sh              # Start with defaults (auto-resume)
./deploy.sh --gpu        # Start with GPU support
./deploy.sh --dry-run    # Validate config only
./deploy.sh --status     # Show health status
./deploy.sh --docker     # Start via Docker Compose
```

**Features:**
- Pre-flight checks (python3, config files, dependencies)
- Creates required directories (logs, metrics, cycle_data, archive)
- PID file management (`.runner.pid`)
- Duplicate process detection
- Auto-resume from saved state
- Waits for health.json before reporting success
- Non-daemonized via `nohup`

#### `stop.sh` (NEW ~80 lines)
Graceful shutdown:
```bash
./stop.sh              # Graceful (SIGTERM + 120s timeout)
./stop.sh --force      # Immediate kill (SIGKILL)
./stop.sh --docker     # Stop Docker containers
```

**Features:**
- Finds PID from `.runner.pid` or `pgrep`
- Creates `stop` file as fallback signal
- Waits up to 120s for graceful exit
- Force kills after timeout

#### `status.sh` (NEW ~120 lines)
Health check dashboard:
```bash
./status.sh              # Formatted health display
./status.sh --json       # Raw JSON output
./status.sh --logs       # Log file statistics
./status.sh --alerts     # Alert manager status
```

**Features:**
- Process status (RUNNING/STOPPED + PID)
- Health data display (status, cycles, fitness, level, errors, uptime)
- Color-coded output
- Log directory stats
- Alert configuration summary

---

## Tests — 63 new tests in `tests/test_phase8_production.py`

| Test Class | Tests | What's Tested |
|-----------|-------|---------------|
| `TestDockerfile` | 9 | File exists, Python 3.11, CUDA support, healthcheck, user, entrypoint |
| `TestDockerCompose` | 9 | Services, volumes, environment, healthcheck, logging, webhook env vars |
| `TestMonitoringAlerts` | 17 | AlertManager init, thresholds, config, rate limiting, all alert types, HTTP failure |
| `TestLogRotation` | 13 | Import, component names, setup, file creation, compression, stats |
| `TestDeploymentScripts` | 11 | All scripts exist, shebangs, key features (docker, force, json, PID) |
| `TestAlertIntegration` | 4 | Fitness regression detection, OOM escalation, crash loop levels, multi-alert tracking |

---

## Test Results

```
198 passed in 6.37s

TestVMFarmImport (5):               ✅
TestPaths (4):                      ✅
TestConfigFiles (4):                ✅
TestEvolutionEngineImport (4):      ✅
TestMutationProposerImport (4):     ✅
TestStrategyArchive (3):            ✅
TestFitnessLandscape (3):           ✅
TestTrainingSignalFilter (3):       ✅
TestAutonomousRunner (9):           ✅
TestMetaController (2):             ✅
TestChampionAnchor (1):             ✅
TestLoggingConfig (5):              ✅
TestMetricsCollector (10):          ✅
TestConfig (12):                    ✅
TestAutonomousRunnerConfigInteg (5): ✅
TestVMFactoryFallback (5):          ✅
TestGPUCPUBitExactMatch (4):        ✅
TestGPUOOMRecovery (3):             ✅
TestGPUFunctions (4):               ✅
TestEvolutionEngineGPUWiring (3):   ✅
TestCycleState (10):                ✅
TestCleanupStaleState (5):          ✅
TestOOMRecovery (5):                ✅
TestRetrainFailureRecovery (5):     ✅
TestDashboardHealthMetrics (11):    ✅
TestProgressReport (3):             ✅
TestDockerfile (9):                 ✅  ← NEW
TestDockerCompose (9):              ✅  ← NEW
TestMonitoringAlerts (17):          ✅  ← NEW
TestLogRotation (13):               ✅  ← NEW
TestDeploymentScripts (11):         ✅  ← NEW
TestAlertIntegration (4):           ✅  ← NEW
```

---

## Files Changed/Created

| File | Status | Lines |
|------|--------|-------|
| `Dockerfile` | **NEW** | ~60 |
| `docker-compose.yml` | **NEW** | ~70 |
| `monitoring_alerts.py` | **NEW** | ~270 |
| `log_rotation.py` | **NEW** | ~210 |
| `deploy.sh` | **NEW** | ~100 |
| `stop.sh` | **NEW** | ~80 |
| `status.sh` | **NEW** | ~120 |
| `tests/test_phase8_production.py` | **NEW** | ~500 |

---

## Architecture: Production Deployment Flow

```
┌─────────────────────────────────────────────────────────┐
│                  Production Deployment                   │
│                                                         │
│  ./deploy.sh ──→ Pre-flight checks                     │
│       │         ↓                                       │
│       ├─ Docker mode: docker compose up -d              │
│       │   └─ Container with healthcheck                 │
│       │                                                 │
│       └─ Direct mode: nohup python3 autonomous_runner   │
│           ├─ Writes PID to .runner.pid                  │
│           ├─ Auto-resumes from runner_state.json        │
│           └─ Writes health.json                         │
│                                                         │
│  ./status.sh ──→ Read health.json + process status     │
│                                                         │
│  ./stop.sh ────→ SIGTERM → 120s wait → SIGKILL         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Alert Pipeline                         │
│                                                         │
│  Event detected:                                        │
│    ├─ Fitness regression → AlertManager.send_fitness_.. │
│    ├─ OOM error         → AlertManager.send_oom_alert   │
│    ├─ Crash loop        → AlertManager.send_crash_loop..│
│    ├─ Retrain failure   → AlertManager.send_retrain_..  │
│    └─ Health stale      → AlertManager.send_health_..   │
│                                                         │
│  Rate limited (10/hour) → Delivery:                     │
│    ├─ Discord webhook (color embeds)                    │
│    ├─ Slack webhook (formatted text)                    │
│    └─ Email SMTP (critical only, with TLS)              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Log Rotation                            │
│                                                         │
│  Production logging setup:                              │
│    logs/evolution.log  ← All components (DEBUG+)        │
│    logs/retrain.log    ← Retrain messages (DEBUG+)      │
│    logs/errors.log     ← Errors only (ERROR+)           │
│    logs/alerts.log     ← Alert system (DEBUG+)          │
│    logs/monitoring.log ← Health/metrics (DEBUG+)        │
│                                                         │
│  Rotation: 10MB/file, 10 backups                        │
│  Compression: auto-gzip on rotation                     │
│  Retention: compress files >30 days old                 │
└─────────────────────────────────────────────────────────┘
```

---

## What's Left

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Fix Foundations | ✅ | |
| Phase 2: Autonomous Runner | ✅ | |
| Phase 3: Structured Logging + Metrics | ✅ | |
| Phase 4: GPU VM Farm Integration | ✅ | |
| Phase 5: Cycle Resilience | ✅ | |
| Phase 6: Unified Config | ✅ | |
| Phase 7: Monitoring Dashboard | ✅ | |
| **Phase 8: Production Hardening** | **✅** | **Docker, alerts, log rotation, deploy scripts** |
