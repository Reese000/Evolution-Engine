## 2026-06-04 — v10 CROSSOVER Monoculture (99.4%) Worse Than v9 GROW Monoculture (89.8%)
- **Problem**: v10's dedicated CROSSOVER probability (10%) + archive preference for CROSSOVER (100% success) created a 99.4% CROSSOVER monoculture. Champion stuck at L2 with 3 words for all 34 cycles — worse than v9's L7-L11.
- **Cause**: CROSSOVER of 2-3 word programs produces 2-3 word children — zero innovation. The `model_guided_ratio_low=0.10` at L0-L5 gave the model too little influence (10%), so 90% of challengers came from mutation/CROSSOVER. GROW was suppressed to near-zero (3 uses vs 484 CROSSOVER).
- **Fix**: Three changes: (1) crossover_dedicated_prob: 0.10→0.05, (2) Added L4+ CROSSOVER guard (useless at L0-L3), (3) model_guided_ratio_low: 0.10→0.25.
- **Result**: v10.1 C1 reached L9 with 25 words in 17s (vs v10's L2 with 3 words). C5 achieved positive fitness (+0.770 at L11) — only the 2nd positive fitness cycle ever across all runs.
- **Prevention**: CROSSOVER of tiny programs (<5 words) is meaningless. Always add curriculum-level guards for crossover at low levels. The dedicated CROSSOVER probability combined with archive bias can create worse monocultures than the original problem.

## 2026-06-04 — Monoculture Cap Threshold Too Low (8) Caused 3,237 Spurious Cap Events
- **Problem**: The monoculture cap at `max_consecutive_same_strategy=8` fired 3,237 times across 25 v10.1 cycles. Each time it interrupted CROSSOVER (the most successful strategy) after only 8 consecutive recommendations, forcing GROW fallback.
- **Cause**: The archive always prefers CROSSOVER (100% success rate). After 8 consecutive recommendations, CROSSOVER gets capped → GROW used → GROW fails → archive reverts to CROSSOVER → cycle repeats. The stall-aware guard (generations_stalled >= 10) was active, but once stalled, the low 8-threshold caused constant oscillation.
- **Fix**: Raised `max_consecutive_same_strategy` from 8 to 30. The stall guard (circuit_breaker_stall_limit=10) provides primary protection; the consecutive cap is secondary. 30 gives successful strategies adequate runway.
- **Prevention**: When the stall guard already prevents cap firing during progress, the consecutive threshold should be high enough to avoid constant oscillation. 30 is a minimum; monitor and adjust.

## 2026-06-04 — v10.1 Achieved Positive Fitness (+0.770 at L11) in Only 5 Cycles
- **Problem**: Positive fitness had only been achieved once before (v9 C20: F=+0.737 at L12 in 20 cycles). v10.1 achieved F=+0.770 at L11 in only 5 cycles — the fastest positive fitness ever.
- **Cause**: The v10.1 fixes (GROW restored, CROSSOVER reduced, model_guided_ratio increased) created better strategy diversity (GROW 71%, CROSSOVER 20.6%, HYBRID 8.5%) vs v9 (GROW 89.8%) or v10 (CROSSOVER 99.4%).
- **Key Insight**: Strategy diversity is the primary driver of positive fitness. Neither extreme monoculture (GROW or CROSSOVER) works. The sweet spot appears to be GROW dominant (50-70%) with meaningful CROSSOVER (15-25%) and HYBRID (5-15%) contributions.
- **Prevention**: Track strategy distribution as a primary metric alongside fitness and val_loss. Target GROW ≤ 70%, CROSSOVER ≥ 15%, HYBRID ≥ 5%.