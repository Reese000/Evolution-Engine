# Run Diagnosis: 50-Cycle GPU with L26-L30

## Date: 2026-07-18

## Key Findings

### 1. F=888.9 is an ARTIFACT (NOT a record)
- C11 champion (7 words) FAILS Level 30 verify(). Score=0.0
- Program only sets x3=11 (L0). All L28-L31 registers MISSING
- Fitness was computed during evolution with a different task instance
- The champion doesn't actually solve L30

### 2. Fitness is level-scaled (NOT comparable across levels)
- Formula: `level_bonus = champion_level * 25.0` (line 1897)
- F=888.9@L30 vs F=790.9@L25 is NOT a fair comparison
- Task-solving fitness (without level bonus): L30=138.9 vs L25=165.9
- L25 actually has HIGHER task-solving fitness than L30

### 3. Crash-truncated cycles are credited
- 2 cycles (C9, C10) have elapsed times of 3.2s and 5.9s
- A 3-second "cycle" with 10 generations + retrain is impossible
- Resume logic credits ANY cycle with results in the file
- C10 has F=-25.0 (CUDA error), but is still counted

### 4. Negative fitness mechanism
- C10: F=-25.0 caused by "CUDA error: device-side assert triggered"
- The cycle crashed mid-evolution (5.87s), wrote partial results with error reason
- The cycle is credited because results are in the file

### 5. Crashes persist (both R49 commits are falsified)
- Crashes are ACCESS_VIOLATION (0xC0000005) during in-process retrain
- Cannot reproduce in isolation — intermittent, CUDA-related
- R49 (cuda_cleanup skip) did not solve the crash
- R49 (in-process retrain cleanup) did not solve the crash
- Need: CUDA_LAUNCH_BLOCKING=1 + faulthandler to get fault stack

## Unreliable Data
- C9, C10: crash-truncated, should be excluded from analysis
- C11: F=888.9 is artifact, champion doesn't pass verify()
- C10: F=-25.0 is CUDA error artifact

## Valid Data
- C1-C8, C11-C12: completed cycles with valid data
- Peak valid fitness: 717.9 (C8, L24)
- Best valid val_loss: 0.0525 (C6)
