"""EPI momentum governor — drives sustained exponential progress.

Tracks rolling EPI momentum from progress_trajectory.jsonl and triggers
interventions when growth falls below the goal threshold (default avg > 0.5 / 5000 cycles).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

DEFAULT_WINDOW = int(os.environ.get("EE_EPI_MOMENTUM_WINDOW", "5000"))
DEFAULT_TARGET = float(os.environ.get("EE_EPI_MOMENTUM_TARGET", "0.5"))
INTERVENTION_WINDOW = int(os.environ.get("EE_EPI_INTERVENTION_WINDOW", "100"))
PLATEAU_STREAK_THRESHOLD = int(os.environ.get("EE_EPI_PLATEAU_STREAK", "5"))
PLATEAU_RESEED_STREAK = int(os.environ.get("EE_EPI_PLATEAU_RESEED_STREAK", "10"))
PLATEAU_INCOMING_WINDOW = int(os.environ.get("EE_EPI_PLATEAU_INCOMING_WINDOW", "20"))
TERMINAL_STAGNATION_CYCLES = int(os.environ.get("EE_TERMINAL_STAGNATION_CYCLES", "12"))
GROWTH_LOOKBACK = int(os.environ.get("EE_GROWTH_LOOKBACK", "200"))
GROWTH_MIN_FITNESS_GAIN = float(os.environ.get("EE_GROWTH_MIN_FITNESS_GAIN", "0.1"))
GROWTH_MIN_EPI_GAIN = float(os.environ.get("EE_GROWTH_MIN_EPI_GAIN", "0.5"))
GROWTH_MIN_SAMPLES = int(os.environ.get("EE_GROWTH_MIN_SAMPLES", "50"))
CAPABILITY_STALL_CYCLES = int(os.environ.get("EE_CAPABILITY_STALL_CYCLES", "20"))
CAPABILITY_STALL_EPSILON = float(os.environ.get("EE_CAPABILITY_STALL_EPSILON", "0.01"))


@dataclass
class GrowthGateReport:
    growth_met: bool
    lookback: int
    fitness_gain: float
    epi_gain: float
    sample_count: int
    reason: str = ""


@dataclass
class GrowthGateLookahead:
    """Early warning before the full-window growth gate fails."""

    at_risk: bool
    projected_fitness_gain: float
    projected_epi_gain: float
    recent_fitness_gain: float
    recent_epi_gain: float
    samples_in_window: int
    cycles_to_window_end: int
    reason: str = ""


@dataclass
class EpiMomentumReport:
    window: int
    target_avg: float
    sample_count: int
    avg_momentum: float
    recent_avg: float
    min_momentum: float
    max_momentum: float
    latest_momentum: float
    goal_met: bool
    needs_intervention: bool
    momentum_gate_met: bool = False
    growth_gate_met: bool = False


@dataclass
class PlateauBoostReport:
    needs_boost: bool
    negative_streak: int
    incoming_avg: float


def _read_tick(project_root: str) -> dict[str, Any]:
    path = _tick_state_path(project_root)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_tick(project_root: str, tick: dict[str, Any]) -> None:
    path = _tick_state_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tick, f, indent=2)


def _trajectory_path(project_root: str) -> str:
    return os.path.join(project_root, ".agents", "progress_trajectory.jsonl")


def _tick_state_path(project_root: str) -> str:
    return os.path.join(project_root, ".agents", "tick_state.json")


def load_recent_momentum(
    project_root: str,
    window: int = DEFAULT_WINDOW,
) -> list[float]:
    path = _trajectory_path(project_root)
    if not os.path.isfile(path):
        return []
    by_cycle: dict[int, float] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cycle = int(row.get("cycle", idx))
            if cycle < 0:
                continue
            by_cycle[cycle] = float(row.get("epi_momentum", 0.0))
    values = [by_cycle[c] for c in sorted(by_cycle)]
    if window <= 0:
        return values
    return values[-window:]


def load_recent_trajectory_metrics(
    project_root: str,
    window: int = DEFAULT_WINDOW,
) -> list[dict[str, Any]]:
    """Trajectory rows sorted by cycle (fitness, epi, momentum)."""
    path = _trajectory_path(project_root)
    if not os.path.isfile(path):
        return []
    by_cycle: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cycle = int(row.get("cycle", idx))
            if cycle < 0:
                continue
            by_cycle[cycle] = row
    rows = [by_cycle[c] for c in sorted(by_cycle)]
    if window <= 0:
        return rows
    return rows[-window:]


def assess_exponential_growth(
    project_root: str,
    window: int = DEFAULT_WINDOW,
) -> GrowthGateReport:
    """Require measurable fitness and EPI improvement — not metric accumulation alone."""
    rows = load_recent_trajectory_metrics(project_root, window)
    if len(rows) < GROWTH_MIN_SAMPLES:
        return GrowthGateReport(
            growth_met=False,
            lookback=0,
            fitness_gain=0.0,
            epi_gain=0.0,
            sample_count=len(rows),
            reason="insufficient_samples",
        )
    lookback = min(GROWTH_LOOKBACK, len(rows))
    recent = rows[-lookback:]
    fitnesses = [float(r.get("fitness", 0.0)) for r in recent if r.get("fitness") is not None]
    epis = [float(r.get("epi", 0.0)) for r in recent if r.get("epi") is not None]
    if len(fitnesses) < GROWTH_MIN_SAMPLES or len(epis) < GROWTH_MIN_SAMPLES:
        return GrowthGateReport(
            growth_met=False,
            lookback=lookback,
            fitness_gain=0.0,
            epi_gain=0.0,
            sample_count=len(rows),
            reason="missing_fitness_or_epi",
        )
    fitness_gain = fitnesses[-1] - fitnesses[0]
    epi_gain = epis[-1] - epis[0]
    if fitness_gain < GROWTH_MIN_FITNESS_GAIN:
        return GrowthGateReport(
            growth_met=False,
            lookback=lookback,
            fitness_gain=round(fitness_gain, 4),
            epi_gain=round(epi_gain, 4),
            sample_count=len(rows),
            reason="fitness_gain_below_threshold",
        )
    if epi_gain < GROWTH_MIN_EPI_GAIN:
        return GrowthGateReport(
            growth_met=False,
            lookback=lookback,
            fitness_gain=round(fitness_gain, 4),
            epi_gain=round(epi_gain, 4),
            sample_count=len(rows),
            reason="epi_gain_below_threshold",
        )
    if len(rows) >= window:
        full_fit = [float(r.get("fitness", 0.0)) for r in rows[-window:]]
        full_epi = [float(r.get("epi", 0.0)) for r in rows[-window:]]
        if full_fit and full_fit[-1] - full_fit[0] < GROWTH_MIN_FITNESS_GAIN:
            return GrowthGateReport(
                growth_met=False,
                lookback=lookback,
                fitness_gain=round(fitness_gain, 4),
                epi_gain=round(epi_gain, 4),
                sample_count=len(rows),
                reason="full_window_fitness_flat",
            )
        if full_epi and full_epi[-1] - full_epi[0] < GROWTH_MIN_EPI_GAIN:
            return GrowthGateReport(
                growth_met=False,
                lookback=lookback,
                fitness_gain=round(fitness_gain, 4),
                epi_gain=round(epi_gain, 4),
                sample_count=len(rows),
                reason="full_window_epi_flat",
            )
    return GrowthGateReport(
        growth_met=True,
        lookback=lookback,
        fitness_gain=round(fitness_gain, 4),
        epi_gain=round(epi_gain, 4),
        sample_count=len(rows),
        reason="ok",
    )


def assess_growth_gate_lookahead(
    project_root: str,
    window: int = DEFAULT_WINDOW,
) -> GrowthGateLookahead:
    """Project end-of-window fitness/EPI gain from recent slope — warn before gate fails."""
    rows = load_recent_trajectory_metrics(project_root, window)
    cycles_to_end = max(0, window - len(rows))
    if len(rows) < GROWTH_MIN_SAMPLES:
        return GrowthGateLookahead(
            at_risk=False,
            projected_fitness_gain=0.0,
            projected_epi_gain=0.0,
            recent_fitness_gain=0.0,
            recent_epi_gain=0.0,
            samples_in_window=len(rows),
            cycles_to_window_end=cycles_to_end,
            reason="insufficient_samples",
        )

    lookback = min(GROWTH_LOOKBACK, len(rows))
    recent = rows[-lookback:]
    fitnesses = [float(r.get("fitness", 0.0)) for r in recent]
    epis = [float(r.get("epi", 0.0)) for r in recent]
    recent_fit_gain = fitnesses[-1] - fitnesses[0]
    recent_epi_gain = epis[-1] - epis[0]
    span = max(1, len(recent) - 1)
    fit_slope = recent_fit_gain / span
    epi_slope = recent_epi_gain / span
    projected_fit_gain = recent_fit_gain + fit_slope * cycles_to_end
    projected_epi_gain = recent_epi_gain + epi_slope * cycles_to_end

    at_risk = False
    reason = "ok"
    if len(rows) >= window // 2:
        if projected_fit_gain < GROWTH_MIN_FITNESS_GAIN:
            at_risk = True
            reason = "projected_fitness_gain_below_threshold"
        elif projected_epi_gain < GROWTH_MIN_EPI_GAIN:
            at_risk = True
            reason = "projected_epi_gain_below_threshold"
    if not at_risk and len(rows) >= int(window * 0.8):
        full_fit = [float(r.get("fitness", 0.0)) for r in rows]
        full_epi = [float(r.get("epi", 0.0)) for r in rows]
        if full_fit and full_fit[-1] - full_fit[0] < GROWTH_MIN_FITNESS_GAIN * 0.5:
            at_risk = True
            reason = "full_window_fitness_stalling"
        elif full_epi and full_epi[-1] - full_epi[0] < GROWTH_MIN_EPI_GAIN * 0.5:
            at_risk = True
            reason = "full_window_epi_stalling"

    return GrowthGateLookahead(
        at_risk=at_risk,
        projected_fitness_gain=round(projected_fit_gain, 4),
        projected_epi_gain=round(projected_epi_gain, 4),
        recent_fitness_gain=round(recent_fit_gain, 4),
        recent_epi_gain=round(recent_epi_gain, 4),
        samples_in_window=len(rows),
        cycles_to_window_end=cycles_to_end,
        reason=reason,
    )


def _momentum_recovery_active(project_root: str, samples: list[float]) -> bool:
    """Stable depth with sustained positive incoming momentum — skip intervention."""
    if not samples or samples[-1] <= 0:
        return False
    tail = samples[-10:] if len(samples) >= 10 else samples
    if _terminal_depth_stagnation(project_root):
        return _positive_momentum_streak(tail) >= 2
    if not _deep_stable_champion(project_root):
        return False
    return _positive_momentum_streak(tail) >= 4


def assess_epi_momentum(
    project_root: str,
    window: int = DEFAULT_WINDOW,
    target_avg: float = DEFAULT_TARGET,
) -> EpiMomentumReport:
    samples = load_recent_momentum(project_root, window)
    growth = assess_exponential_growth(project_root, window)
    if not samples:
        return EpiMomentumReport(
            window=window,
            target_avg=target_avg,
            sample_count=0,
            avg_momentum=0.0,
            recent_avg=0.0,
            min_momentum=0.0,
            max_momentum=0.0,
            latest_momentum=0.0,
            goal_met=False,
            needs_intervention=True,
            momentum_gate_met=False,
            growth_gate_met=False,
        )
    avg = sum(samples) / len(samples)
    recent = samples[-INTERVENTION_WINDOW:] if len(samples) >= INTERVENTION_WINDOW else samples
    recent_avg = sum(recent) / len(recent) if recent else 0.0
    min_recent = min(20, INTERVENTION_WINDOW)
    recovering = _momentum_recovery_active(project_root, samples)
    momentum_gate_met = len(samples) >= window and avg > target_avg
    growth_gate_met = growth.growth_met
    return EpiMomentumReport(
        window=window,
        target_avg=target_avg,
        sample_count=len(samples),
        avg_momentum=round(avg, 4),
        recent_avg=round(recent_avg, 4),
        min_momentum=round(min(samples), 4),
        max_momentum=round(max(samples), 4),
        latest_momentum=round(samples[-1], 4),
        goal_met=momentum_gate_met and growth_gate_met,
        momentum_gate_met=momentum_gate_met,
        growth_gate_met=growth_gate_met,
        # Intervene only on recent weakness — do not reseed while the rolling
        # window is healthy even if older samples still drag the 1000-cycle avg down.
        needs_intervention=(len(recent) >= min_recent and recent_avg <= target_avg and not recovering),
    )


def apply_growth_interventions(project_root: str, report: EpiMomentumReport) -> None:
    """Record intervention flags for evolution loop (reseed + register repair boost)."""
    if not report.needs_intervention:
        return
    tick = _read_tick(project_root)
    depth_nudge = tick.get("depth_nudge_level")
    tick["epi_momentum_intervention"] = True
    tick["epi_momentum_avg"] = report.avg_momentum
    tick["epi_momentum_target"] = report.target_avg
    if _terminal_depth_stagnation(project_root):
        tick["terminal_diversity_boost"] = True
        tick.pop("epi_momentum_reseed", None)
        tick.pop("force_champion_reseed", None)
        tick.pop("force_reference_champion", None)
        tick.pop("force_register_repair", None)
        tick.pop("reseed_level", None)
    else:
        tick["epi_momentum_reseed"] = True
        tick["force_register_repair"] = True
        if should_suppress_register_repair(project_root, report):
            tick.pop("force_register_repair", None)
            tick.pop("epi_plateau_boost", None)
        if not should_suppress_champion_reseed(project_root, report) and not tick.get("collapse_reseed"):
            tick["force_champion_reseed"] = True
            tick["force_reference_champion"] = True
            level = _latest_champion_level(project_root)
            if level is not None:
                tick["reseed_level"] = level
        tick.pop("epi_plateau_boost", None)
    if depth_nudge is not None:
        tick["depth_nudge_level"] = depth_nudge
    _write_tick(project_root, tick)


def _plateau_recovery_sustained(project_root: str, report: EpiMomentumReport) -> bool:
    """Clear plateau boost only after incoming momentum stays above target."""
    tail = load_recent_momentum(project_root, 10)
    if not tail:
        return False
    tail_avg = sum(tail) / len(tail)
    return report.latest_momentum >= report.target_avg and tail_avg >= report.target_avg


def _recent_l30_capability_scores(project_root: str, limit: int) -> list[float]:
    """Recent promotion_raw scores for L30 rows (primary capability metric)."""
    from evolution.fitness_helpers import row_capability_score

    results = os.path.join(project_root, "multi_cycle_results.jsonl")
    if not os.path.isfile(results):
        return []
    try:
        with open(results, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
    except OSError:
        return []
    from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

    scores: list[float] = []
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("final_level", row.get("level", 0))) < 30:
            continue
        verify = float(row.get("champion_verify_score", row.get("verify_score", 0.0)))
        words = int(row.get("champion_words", row.get("words", 0)))
        stored = row.get("promotion_raw_fitness")
        if stored is not None:
            try:
                cap = float(stored)
            except (TypeError, ValueError):
                continue
            if cap <= 0.0 or cap > 330.0:
                continue
        elif verify >= 0.99 - 1e-6 and 12 <= words <= L30_COMPACT_MAX_WORDS:
            cap = float(row_capability_score(row))
            if cap <= 0.0 or cap > 330.0:
                continue
        else:
            continue
        scores.append(cap)
        if len(scores) >= limit:
            break
    return scores


def promotion_raw_capability_stalled(
    project_root: str,
    *,
    min_cycles: int | None = None,
    improvement_epsilon: float | None = None,
) -> bool:
    """True when primary promotion_raw has not improved over recent L30 cycles."""
    stall = min_cycles if min_cycles is not None else CAPABILITY_STALL_CYCLES
    eps = improvement_epsilon if improvement_epsilon is not None else CAPABILITY_STALL_EPSILON
    if stall <= 0:
        return False
    scores = _recent_l30_capability_scores(project_root, stall)
    if len(scores) < stall:
        return False
    return (max(scores) - min(scores)) < eps


def scan_floor_length_prom_stalled(
    project_root: str,
    floor_words: int,
    *,
    min_cycles: int | None = None,
    improvement_epsilon: float | None = None,
) -> bool:
    """True when promotion_raw is flat over recent cycles at the armed scan-floor length."""
    from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

    stall = min_cycles if min_cycles is not None else int(os.environ.get("EE_SCAN_FLOOR_LOCAL_STALL_CYCLES", "5"))
    eps = improvement_epsilon if improvement_epsilon is not None else CAPABILITY_STALL_EPSILON
    if stall <= 0 or floor_words <= 0:
        return False
    results = os.path.join(project_root, "multi_cycle_results.jsonl")
    if not os.path.isfile(results):
        return False
    try:
        with open(results, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
    except OSError:
        return False
    scores: list[float] = []
    floor_n = int(floor_words)
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("final_level", row.get("level", 0))) < 30:
            continue
        words = int(row.get("champion_words", row.get("words", 0)))
        if words != floor_n or not (12 <= words <= L30_COMPACT_MAX_WORDS):
            continue
        stored = row.get("promotion_raw_fitness")
        if stored is None:
            continue
        try:
            cap = float(stored)
        except (TypeError, ValueError):
            continue
        if cap <= 0.0 or cap > 330.0:
            continue
        scores.append(cap)
        if len(scores) >= stall:
            break
    if len(scores) < stall:
        return False
    return (max(scores) - min(scores)) < eps


def should_suppress_terminal_diversity_boost(project_root: str) -> bool:
    """Public gate: skip terminal diversity boost while goal flush is healthy."""
    # Primary metric wins: flat promotion_raw means capability plateau even if
    # EPI momentum looks healthy from inflated raw_fitness / trajectory noise.
    if promotion_raw_capability_stalled(project_root):
        return False
    report = assess_epi_momentum(project_root)
    return _sustained_goal_recovery(project_root, report)


def _sustained_goal_recovery(project_root: str, report: EpiMomentumReport) -> bool:
    """Suppress terminal plateau boosts while growth + recent momentum prove healthy flush."""
    if report.needs_intervention:
        return False
    if not report.growth_gate_met:
        return False
    if report.recent_avg <= report.target_avg:
        return False
    if report.latest_momentum < report.target_avg:
        return False
    tail = load_recent_momentum(project_root, 10)
    if not tail:
        return False
    tail_avg = sum(tail) / len(tail)
    return tail_avg >= report.target_avg


def clear_plateau_boost(project_root: str) -> None:
    tick = _read_tick(project_root)
    if not tick.get("epi_plateau_boost") and not tick.get("terminal_diversity_boost"):
        return
    tick.pop("epi_plateau_boost", None)
    tick.pop("terminal_diversity_boost", None)
    tick.pop("terminal_archive_reset", None)
    if tick.get("force_register_repair") and not tick.get("epi_momentum_intervention"):
        tick.pop("force_register_repair", None)
    tick["epi_negative_streak"] = 0
    _write_tick(project_root, tick)


def _negative_momentum_streak(samples: list[float]) -> int:
    streak = 0
    for momentum in reversed(samples):
        if momentum < 0:
            streak += 1
        else:
            break
    return streak


def _positive_momentum_streak(samples: list[float]) -> int:
    streak = 0
    for momentum in reversed(samples):
        if momentum > 0:
            streak += 1
        else:
            break
    return streak


def _certified_high_level_champion(project_root: str) -> bool:
    """True when progress state shows certify-passing champion at L20+."""
    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    if not os.path.isfile(state_path):
        return False
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        from evolution.progress_controller import ProgressController

        level = int(state.get("level", 0))
        verify = float(state.get("verify_score", 0.0))
        stall = max(2, int(state.get("cycles_at_level", 2)))
        return level >= 20 and ProgressController.passes_certify_verify(level, verify, stall=stall, stall_threshold=2)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, ImportError):
        return False


def _curriculum_max_level() -> int:
    return int(os.environ.get("EE_MAX_LEVEL", "30"))


def _terminal_depth_stagnation(project_root: str) -> bool:
    """At max curriculum depth with long in-cycle stall — allow register repair to break verify freeze."""
    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    if not os.path.isfile(state_path):
        return False
    try:
        from evolution.terminal_stagnation import terminal_depth

        depth = terminal_depth()
    except ImportError:
        depth = 30
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        level = int(state.get("level", 0))
        cycles = int(state.get("cycles_at_level", 0))
        return level >= depth and cycles >= TERMINAL_STAGNATION_CYCLES
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _deep_stable_champion(project_root: str) -> bool:
    """Stable nudge-quality champion at L22+ — depth hold must not be broken by EPI governor."""
    if _terminal_depth_stagnation(project_root):
        return False
    if not _stable_nudge_champion(project_root):
        return False
    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        level = int(state.get("level", 0))
        cycles = int(state.get("cycles_at_level", 0))
        return level >= 22 and cycles >= 3
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def should_suppress_register_repair(project_root: str, report: EpiMomentumReport) -> bool:
    """Skip aggressive register repair while incoming momentum is recovering."""
    if _terminal_depth_stagnation(project_root):
        return False
    if _deep_stable_champion(project_root):
        return True
    if _certified_high_level_champion(project_root):
        return True
    if _stable_nudge_champion(project_root):
        if report.recent_avg >= report.target_avg:
            return True
        if report.latest_momentum > 0:
            return True
    tail = load_recent_momentum(project_root, 10)
    pos_streak = _positive_momentum_streak(tail)
    if pos_streak >= 3:
        return True
    if report.latest_momentum > 0 and _negative_momentum_streak(tail) == 0:
        return True
    return False


def _stable_nudge_champion(project_root: str) -> bool:
    """True when progress state shows clean champion at nudge quality (not certify)."""
    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    if not os.path.isfile(state_path):
        return False
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        from evolution.curriculum_advance import nudge_fitness_passes, nudge_verify_passes

        level = int(state.get("level", 0))
        verify = float(state.get("verify_score", 0.0))
        fitness = float(state.get("champion_fitness", state.get("raw_fitness", 0.0)))
        cycles = max(1, int(state.get("cycles_at_level", 1)))
        if level < 20:
            return False
        return nudge_verify_passes(level, verify, cycles) and nudge_fitness_passes(level, fitness, verify)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, ImportError):
        return False


def _fresh_depth_hold(project_root: str) -> bool:
    """Newly reached L22+ with carry-quality verify — avoid disruptive reseed."""
    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    if not os.path.isfile(state_path):
        return False
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        from evolution.curriculum_advance import depth_carry_verify_floor, nudge_fitness_passes

        level = int(state.get("level", 0))
        cycles = int(state.get("cycles_at_level", 0))
        verify = float(state.get("verify_score", 0.0))
        fitness = float(state.get("champion_fitness", state.get("raw_fitness", 0.0)))
        if level < 22 or cycles > 2:
            return False
        return verify >= depth_carry_verify_floor(level) and nudge_fitness_passes(level, fitness, verify)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, ImportError):
        return False


def should_suppress_champion_reseed(project_root: str, report: EpiMomentumReport) -> bool:
    """Skip per-cycle champion reseed when clean certified or stable nudge champion holds."""
    if _terminal_depth_stagnation(project_root):
        return True
    if _deep_stable_champion(project_root):
        return True
    if _fresh_depth_hold(project_root):
        return True
    if _certified_high_level_champion(project_root):
        tail = load_recent_momentum(project_root, 10)
        if _positive_momentum_streak(tail) >= 2:
            return True
        return report.latest_momentum > 0
    if _stable_nudge_champion(project_root):
        tail = load_recent_momentum(project_root, 10)
        if _positive_momentum_streak(tail) >= 1 or report.latest_momentum > 0:
            return True
    return False


def assess_plateau_boost(project_root: str, report: EpiMomentumReport) -> PlateauBoostReport:
    """Detect EPI plateau decay — recent window healthy but incoming momentum falling."""
    incoming = load_recent_momentum(project_root, PLATEAU_INCOMING_WINDOW)
    incoming_avg = sum(incoming) / len(incoming) if incoming else 0.0
    tail = load_recent_momentum(project_root, 50)
    streak = _negative_momentum_streak(tail)
    tick = _read_tick(project_root)
    depth_nudge = tick.get("depth_nudge_level")
    tick["epi_negative_streak"] = streak
    if depth_nudge is not None:
        tick["depth_nudge_level"] = depth_nudge
    _write_tick(project_root, tick)
    if _sustained_goal_recovery(project_root, report):
        return PlateauBoostReport(
            needs_boost=False,
            negative_streak=streak,
            incoming_avg=round(incoming_avg, 4),
        )
    terminal_stall = _terminal_depth_stagnation(project_root)
    needs_boost = terminal_stall or (
        not report.needs_intervention
        and report.recent_avg > report.target_avg
        and (
            streak >= PLATEAU_STREAK_THRESHOLD
            or (report.avg_momentum < report.target_avg and incoming_avg < report.target_avg + 0.1)
        )
    )
    return PlateauBoostReport(
        needs_boost=needs_boost,
        negative_streak=streak,
        incoming_avg=round(incoming_avg, 4),
    )


def _latest_champion_level(project_root: str) -> int | None:
    path = _trajectory_path(project_root)
    if not os.path.isfile(path):
        return None
    level: int | None = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "level" in row:
                level = int(row["level"])
    return level


def apply_plateau_boost(
    project_root: str,
    *,
    escalate_reseed: bool = False,
    champion_level: int | None = None,
    report: EpiMomentumReport | None = None,
) -> None:
    """Soft diversity boost at plateau; escalate to champion reseed on long negative streak."""
    if report is not None and (
        _sustained_goal_recovery(project_root, report) or should_suppress_register_repair(project_root, report)
    ):
        clear_plateau_boost(project_root)
        os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
        return
    tick = _read_tick(project_root)
    tick["epi_plateau_boost"] = True
    if _terminal_depth_stagnation(project_root):
        tick["terminal_diversity_boost"] = True
        tick.pop("force_register_repair", None)
    else:
        tick["force_register_repair"] = True
        tick.pop("terminal_diversity_boost", None)
    if escalate_reseed and champion_level is not None:
        tick["force_champion_reseed"] = True
        tick["force_reference_champion"] = True
        tick["reseed_level"] = champion_level
    _write_tick(project_root, tick)


def run_cycle_governor(project_root: str) -> tuple[EpiMomentumReport, PlateauBoostReport | None]:
    """Assess momentum, apply intervention or plateau boost, return reports."""
    report = assess_epi_momentum(project_root)
    plateau_report: PlateauBoostReport | None = None
    if report.needs_intervention:
        apply_growth_interventions(project_root, report)
        if should_suppress_register_repair(project_root, report):
            os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
        else:
            os.environ["EE_REGISTER_REPAIR_BOOST"] = "1"
    else:
        clear_intervention_flags(project_root)
        if _plateau_recovery_sustained(project_root, report) or _sustained_goal_recovery(project_root, report):
            clear_plateau_boost(project_root)
        plateau_report = assess_plateau_boost(project_root, report)
        if plateau_report.needs_boost:
            escalate = plateau_report.negative_streak >= PLATEAU_RESEED_STREAK
            if _terminal_depth_stagnation(project_root):
                escalate = False
            level = _latest_champion_level(project_root) if escalate else None
            apply_plateau_boost(
                project_root,
                escalate_reseed=escalate,
                champion_level=level,
                report=report,
            )
            if _terminal_depth_stagnation(project_root):
                os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
            elif not should_suppress_register_repair(project_root, report):
                os.environ["EE_REGISTER_REPAIR_BOOST"] = "1"
            else:
                os.environ.pop("EE_REGISTER_REPAIR_BOOST", None)
    return report, plateau_report


def clear_intervention_flags(project_root: str) -> None:
    """Clear EPI-governor flags only — preserve timeout/clamp reseed flags."""
    path = _tick_state_path(project_root)
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            tick = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if not tick.get("epi_momentum_intervention"):
        return
    epi_owned_reseed = bool(tick.get("epi_momentum_reseed"))
    for key in (
        "epi_momentum_intervention",
        "epi_momentum_reseed",
        "force_register_repair",
        "epi_momentum_avg",
        "epi_momentum_target",
    ):
        tick.pop(key, None)
    if epi_owned_reseed and not tick.get("collapse_reseed"):
        tick.pop("force_champion_reseed", None)
        tick.pop("force_reference_champion", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tick, f, indent=2)
