"""Disk-backed VM score cache survives probe-pool cold starts."""


def test_heldout_disk_cache_hit(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        _heldout_l30_verify_mean_cached,
        _vm_score_disk_put,
        _vm_score_disk_reset,
        _vm_score_program_key,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    _vm_score_disk_reset()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    program = (0x00100093, 0x00200113, 0x00100073)
    disk_key = f"{_vm_score_program_key(program)}|8"
    _vm_score_disk_put("heldout_l30", disk_key, 0.88)
    _heldout_l30_verify_mean_cached.cache_clear()
    assert _heldout_l30_verify_mean_cached(program, 8) == 0.88
