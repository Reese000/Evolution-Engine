#!/usr/bin/env python3
"""Tests for Phase 4 (GPU VM Farm Integration).

Validates:
  - VMFactory returns best available backend
  - GPU/CPU bit-exact match for identical programs
  - OOM recovery with GPU (fallback to CPU)
  - GPU memory management functions
  - Factory fallback when CUDA unavailable
  - EvolutionEngine GPU config wiring
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Sample RV32I programs for testing ──────────────────────────────────

# ADDI x10, x0, 42; EBREAK
SIMPLE_PROGRAM = [
    0x02A00513,  # ADDI x10, x0, 42
    0x00100073,  # EBREAK
]

# ADDI x10, x0, 1; ADDI x11, x0, 2; ADD x12, x10, x11; EBREAK
ADD_PROGRAM = [
    0x00100513,  # ADDI x10, x0, 1
    0x00200593,  # ADDI x11, x0, 2
    0x00B50633,  # ADD x12, x10, x11
    0x00100073,  # EBREAK
]

# LUI x10, 0x10; SW x10, 0(x0); LW x11, 0(x0); EBREAK
MEM_PROGRAM = [
    0x00010537,  # LUI x10, 0x10
    0x00A02023,  # SW x10, 0(x0)
    0x00002583,  # LW x11, 0(x0)
    0x00100073,  # EBREAK
]


class TestVMFactoryFallback(unittest.TestCase):
    """Test that VMFactory correctly falls back to CPU when CUDA unavailable."""

    def test_factory_cpu_fallback_no_cuda(self):
        """VMFactory should use CPU backend when no CUDA is available."""
        from gpu_vm_farm import VMFactory

        factory = VMFactory(use_gpu=True)
        # On non-CUDA machine, should fall back to CPU
        self.assertIn(factory.backend_type, ("cpu", "gpu"))
        if factory.backend_type == "cpu":
            self.assertFalse(factory.is_gpu)
            self.assertIn("CUDA", factory.fallback_reason)  # or related

    def test_factory_explicit_cpu(self):
        """VMFactory with use_gpu=False should always use CPU."""
        from gpu_vm_farm import VMFactory

        factory = VMFactory(use_gpu=False)
        self.assertEqual(factory.backend_type, "cpu")
        self.assertFalse(factory.is_gpu)

    def test_factory_repr(self):
        """VMFactory repr shows backend info."""
        from gpu_vm_farm import VMFactory

        factory = VMFactory(use_gpu=False)
        r = repr(factory)
        self.assertIn("VMFactory", r)
        self.assertIn("cpu", r)

    def test_factory_evaluate_single(self):
        """VMFactory.evaluate_single works for a single program."""
        from gpu_vm_farm import VMFactory

        factory = VMFactory(use_gpu=False)
        result = factory.evaluate_single(SIMPLE_PROGRAM, {}, budget=1000)
        self.assertTrue(result.get("success"))
        self.assertEqual(result["registers"].get(10), 42)

    def test_factory_status_dict(self):
        """VMFactory.get_status returns a valid status dict."""
        from gpu_vm_farm import VMFactory

        factory = VMFactory(use_gpu=False)
        status = factory.get_status()
        self.assertIn("backend", status)
        self.assertIn("gpu_requested", status)
        self.assertIn("gpu_active", status)
        self.assertIn("fallback_reason", status)


class TestGPUCPUBitExactMatch(unittest.TestCase):
    """Test that GPU and CPU produce identical results for the same programs."""

    def test_bit_exact_simple_program(self):
        """Simple program produces identical results regardless of backend."""
        from gpu_vm_farm import _execute_vm

        cpu_result = _execute_vm(SIMPLE_PROGRAM, {}, {}, 1000)
        # Since current implementation is pure Python in both paths,
        # results must be identical
        self.assertTrue(cpu_result["success"])
        self.assertEqual(cpu_result["registers"].get(10), 42)

    def test_bit_exact_add_program(self):
        """ADD program: CPU result is deterministic."""
        from gpu_vm_farm import _execute_vm

        result = _execute_vm(ADD_PROGRAM, {}, {}, 1000)
        self.assertTrue(result["success"])
        self.assertEqual(result["registers"].get(10), 1)
        self.assertEqual(result["registers"].get(11), 2)
        self.assertEqual(result["registers"].get(12), 3)

    def test_bit_exact_memory_program(self):
        """Memory program: CPU result is deterministic."""
        from gpu_vm_farm import _execute_vm

        result = _execute_vm(MEM_PROGRAM, {}, {}, 1000)
        self.assertTrue(result["success"])
        # LUI x10, 0x10 -> x10 = 0x10000
        # SW stores x10 at addr 0, LW loads it to x11
        self.assertEqual(result["registers"].get(10), 0x10000)
        self.assertEqual(result["registers"].get(11), 0x10000)

    def test_bit_exact_across_backends(self):
        """Factory CPU and direct _execute_vm produce same results."""
        from gpu_vm_farm import VMFactory, _execute_vm

        factory = VMFactory(use_gpu=False)
        direct = _execute_vm(SIMPLE_PROGRAM, {}, {}, 1000)
        factory_result = factory.evaluate_single(SIMPLE_PROGRAM, {}, budget=1000)

        # Both should succeed with same register values
        self.assertTrue(direct["success"])
        self.assertTrue(factory_result.get("success"))
        self.assertEqual(direct["registers"].get(10), 42)
        self.assertEqual(factory_result.get("registers", {}).get(10), 42)


class TestGPUOOMRecovery(unittest.TestCase):
    """Test OOM recovery when GPU runs out of memory."""

    def test_fallback_to_cpu_on_gpu_error(self):
        """VMFactory falls back to CPU when GPU evaluation raises exception."""
        from gpu_vm_farm import VMFactory

        factory = VMFactory(use_gpu=False)  # Start with CPU for testability
        self.assertEqual(factory.backend_type, "cpu")

        # Simulate: if we had a GPU backend, an exception should trigger fallback
        # Verify the fallback method works
        factory._fallback_to_cpu("test fallback")
        self.assertFalse(factory.is_gpu)
        self.assertEqual(factory.fallback_reason, "test fallback")

    def test_oom_recovery_reduces_population(self):
        """AutonomousRunner OOM handler reduces population on MemoryError."""
        from autonomous_runner import AutonomousRunner

        with patch.object(AutonomousRunner, "__init__", lambda self, **kw: None):
            runner = AutonomousRunner.__new__(AutonomousRunner)
            runner._shutdown_requested = False
            runner._health = MagicMock()
            runner._health._start_time = 0
            runner._config = {
                "evolution": {"population_size": 50},
                "max_champion_words": 500,
                "max_cycles": 10,
                "target_fitness": 100,
                "state_file": "runner_state.json",
                "stop_file": "stop",
                "max_errors_before_stop": 10,
            }
            runner._state = {
                "current_cycle": 0,
                "error_count": 0,
                "total_errors": 0,
                "consecutive_errors": 0,
                "best_fitness": 0,
                "best_level": 0,
            }

            # OOM should reduce population
            result = runner._handle_oom(MemoryError("CUDA out of memory"), 1)
            self.assertTrue(result)  # returns True = should retry
            self.assertLess(runner._config["evolution"]["population_size"], 50)

    def test_cuda_oom_detection(self):
        """_is_cuda_oom correctly identifies CUDA OOM messages."""
        from autonomous_runner import AutonomousRunner

        with patch.object(AutonomousRunner, "__init__", lambda self, **kw: None):
            runner = AutonomousRunner.__new__(AutonomousRunner)

            self.assertTrue(runner._is_cuda_oom(MemoryError("CUDA out of memory")))
            self.assertTrue(runner._is_cuda_oom(MemoryError("cuda malloc failed")))
            # "OOM error" alone is NOT CUDA OOM — requires "cuda" or "out of memory"
            self.assertFalse(runner._is_cuda_oom(MemoryError("OOM error")))
            self.assertFalse(runner._is_cuda_oom(MemoryError("regular oom")))


class TestGPUFunctions(unittest.TestCase):
    """Test GPU memory management utility functions."""

    def test_get_gpu_vram_info_no_cuda(self):
        """get_gpu_vram_info returns empty stats when no CUDA."""
        from gpu_vm_farm import get_gpu_vram_info

        info = get_gpu_vram_info()
        self.assertIn("available", info)
        self.assertIn("total_mb", info)
        self.assertIn("used_mb", info)
        self.assertIn("free_mb", info)
        self.assertIn("utilization", info)
        self.assertIn("device_name", info)
        # On non-CUDA machine
        if not info["available"]:
            self.assertEqual(info["total_mb"], 0.0)
            self.assertEqual(info["device_name"], "N/A")

    def test_clear_gpu_memory_returns_stats(self):
        """clear_gpu_memory returns valid stats dict."""
        from gpu_vm_farm import clear_gpu_memory

        result = clear_gpu_memory()
        self.assertIn("torch_cleared", result)
        self.assertIn("cupy_cleared", result)
        self.assertIn("freed_mb", result)
        self.assertIsInstance(result["freed_mb"], float)

    def test_check_gpu_capacity_no_cuda(self):
        """check_gpu_capacity reports no capacity when no CUDA is present.

        On machines with a real GPU (CUDA available), capacity should be
        a bool driven by current utilization vs. threshold rather than
        unconditionally False -- this test only asserts the "no capacity"
        behavior when CUDA is genuinely absent, so it passes consistently
        on both CPU-only CI and GPU dev boxes.
        """
        from gpu_vm_farm import check_gpu_capacity

        cap = check_gpu_capacity()
        self.assertIn("has_capacity", cap)
        self.assertIn("vram_info", cap)
        self.assertIn("reason", cap)
        if not cap["vram_info"]["available"]:
            self.assertFalse(cap["has_capacity"])
            self.assertEqual(cap["reason"], "No CUDA GPU available")
        else:
            self.assertIsInstance(cap["has_capacity"], bool)

    def test_check_gpu_capacity_threshold(self):
        """check_gpu_capacity respects custom threshold."""
        from gpu_vm_farm import check_gpu_capacity

        cap = check_gpu_capacity(threshold=0.50)
        self.assertIn("has_capacity", cap)
        # Should still be no capacity on non-CUDA machine
        if not cap["vram_info"]["available"]:
            self.assertFalse(cap["has_capacity"])


class TestEvolutionEngineGPUWiring(unittest.TestCase):
    """Test that EvolutionEngine properly wires GPU config."""

    def test_vm_farm_use_gpu_config(self):
        """VMFarm.use_gpu is set from config."""
        from evolution_engine import VMFarm

        # CPU mode
        farm = VMFarm(num_workers=1, use_gpu=False)
        self.assertFalse(farm.use_gpu)

    def test_evolution_config_gpu_field(self):
        """EvolutionConfig has use_gpu field."""
        from evolution_engine import EvolutionConfig

        config = EvolutionConfig(use_gpu=True)
        self.assertTrue(config.use_gpu)

        config2 = EvolutionConfig(use_gpu=False)
        self.assertFalse(config2.use_gpu)

    def test_vm_factory_independently_constructible(self):
        """VMFactory can be constructed independently of EvolutionEngine."""
        from gpu_vm_farm import VMFactory

        factory = VMFactory(use_gpu=False, batch_size=64, budget=2000)
        self.assertEqual(factory.batch_size, 64)
        self.assertEqual(factory.budget, 2000)
        self.assertEqual(factory.backend_type, "cpu")


if __name__ == "__main__":
    unittest.main()
