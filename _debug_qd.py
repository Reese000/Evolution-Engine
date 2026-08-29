from evolution_engine import EvolutionConfig, EvolutionEngine

cfg = EvolutionConfig(seed=42, qd_bonus_weight=0.1)
eng = EvolutionEngine(config=cfg, vm_farm=None)
print('_HAS_QD:', eng._compute_qd_axes_fn is not None)
print('_compute_qd_axes_fn:', eng._compute_qd_axes_fn)
print('qd_archive:', eng.qd_archive)
print('qd_bonus_weight:', eng.config.qd_bonus_weight)
