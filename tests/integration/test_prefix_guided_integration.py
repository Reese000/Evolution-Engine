#!/usr/bin/env python3
"""Integration tests for prefix-guided generation (MODEL_GUIDED strategy).

Validates the full path from PrefixGenerator.complete() through
EvolutionEngine._replace_with_model_guided(), ensuring the model-guided
mutation strategy is properly wired into the evolution loop.

Uses a fresh (randomly initialized) model — no checkpoint needed.
Tests are CPU-only and lightweight.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPrefixGeneratorFreshModel(unittest.TestCase):
    """Test PrefixGenerator with a fresh (no-checkpoint) model."""

    def test_fallback_creates_fresh_model(self):
        """PrefixGenerator with non-existent checkpoint creates fresh model."""
        from prefix_generator import PrefixGenerator

        pg = PrefixGenerator(
            checkpoint_path="/nonexistent/path.pt",
            temperature=0.9,
            max_new_tokens=64,
            device="cpu",
            verbose=False,
        )
        self.assertIsNotNone(pg.model)
        self.assertEqual(pg.device.type, "cpu")

    def test_complete_returns_list_of_lists(self):
        """complete() returns List[List[str]] of hex instruction strings."""
        from prefix_generator import PrefixGenerator

        pg = PrefixGenerator(
            checkpoint_path="/nonexistent/path.pt",
            temperature=0.9,
            max_new_tokens=64,
            device="cpu",
            verbose=False,
        )
        prefix = [
            "0x02a00093",  # ADDI x1, x0, 42
            "0x03a00113",  # ADDI x2, x0, 58
            "0x002080b3",  # ADD x1, x1, x2
            "0x00000013",  # NOP
        ]
        result = pg.complete(
            prefix_source=prefix,
            prefix_count=4,
            num_samples=2,
            temperature=0.9,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        self.assertIsInstance(result, list)
        # Each completion should be a list of hex strings
        for comp in result:
            self.assertIsInstance(comp, list)
            self.assertGreaterEqual(len(comp), 2, "Each completion needs >= 2 instructions")
            for hex_str in comp:
                self.assertIsInstance(hex_str, str)
                self.assertTrue(
                    hex_str.startswith("0x"),
                    f"Instruction should be hex: {hex_str}",
                )

    def test_complete_respects_num_samples(self):
        """complete() generates up to num_samples completions."""
        from prefix_generator import PrefixGenerator

        pg = PrefixGenerator(
            checkpoint_path="/nonexistent/path.pt",
            temperature=0.9,
            max_new_tokens=64,
            device="cpu",
            verbose=False,
        )
        prefix = ["0x02a00093", "0x03a00113"]
        result = pg.complete(
            prefix_source=prefix,
            prefix_count=2,
            num_samples=3,
            temperature=0.9,
        )
        # Should return <= num_samples (some may be filtered for too few instructions)
        self.assertLessEqual(len(result), 3)

    def test_complete_empty_prefix_returns_empty(self):
        """complete() with empty prefix returns empty list."""
        from prefix_generator import PrefixGenerator

        pg = PrefixGenerator(
            checkpoint_path="/nonexistent/path.pt",
            device="cpu",
            verbose=False,
        )
        result = pg.complete(prefix_source=[], prefix_count=4, num_samples=2)
        self.assertEqual(result, [])

    def test_complete_ends_with_ebreak(self):
        """Generated completions should end with EBREAK (0x00100073)."""
        from prefix_generator import PrefixGenerator

        pg = PrefixGenerator(
            checkpoint_path="/nonexistent/path.pt",
            temperature=0.9,
            max_new_tokens=64,
            device="cpu",
            verbose=False,
        )
        prefix = [
            "0x02a00093",
            "0x03a00113",
            "0x002080b3",
        ]
        result = pg.complete(
            prefix_source=prefix,
            prefix_count=3,
            num_samples=2,
            temperature=0.9,
        )
        for comp in result:
            if comp:
                self.assertEqual(
                    comp[-1].lower(),
                    "0x00100073",
                    f"Completion should end with EBREAK: {comp[-1]}",
                )


class TestPrefixGuidedEngineWiring(unittest.TestCase):
    """Test that _replace_with_model_guided is wired into the evolution loop."""

    def _make_engine(self, tmpdir):
        """Create an EvolutionEngine with minimal config for testing."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig(
            population_size=10,
            max_generations=3,
            mutation_rate=0.3,
            elite_ratio=0.1,
            model_guided_ratio=0.2,
        )
        engine = EvolutionEngine(config=config)
        # Set a dummy champion so MODEL_GUIDED has something to work with
        engine.champion = [0x02A00093, 0x03A00113, 0x002080B3, 0x00000013, 0x00100073]
        engine.champion_fitness = 1.0
        engine.champion_level = 5
        return engine

    def test_replace_with_model_guided_creates_prefix_generator(self):
        """_replace_with_model_guided lazily initializes PrefixGenerator."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            # No prefix generator yet
            self.assertIsNone(engine._prefix_generator)

            # Create a list of challengers
            challengers = [[0x02A00093, 0x03A00113, 0x00100073] for _ in range(10)]
            engine._last_challengers = challengers

            # Mock the PrefixGenerator to avoid model loading
            mock_pg = MagicMock()
            mock_pg.complete.return_value = [
                ["0x02a00093", "0x03a00113", "0x00100073"],
            ]

            with (
                patch("prefix_generator.PrefixGenerator", return_value=mock_pg),
                patch("os.path.isfile", return_value=True),
            ):
                engine._replace_with_model_guided(challengers, model_ratio=0.2)

            # PrefixGenerator should now be set
            self.assertIsNotNone(engine._prefix_generator)

    def test_model_guided_replaces_challengers(self):
        """_replace_with_model_guided replaces some challengers with model output."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            challengers = [[0x02A00093, 0x03A00113, 0x00100073] for _ in range(10)]
            engine._last_challengers = list(challengers)  # copy
            [c[:] for c in challengers]  # deep copy

            mock_pg = MagicMock()
            # Return 2 completions (each is list of hex strings)
            mock_pg.complete.return_value = [
                ["0x02a00093", "0x03a00113", "0x00100073"],
                ["0x02a00093", "0x002080b3", "0x00100073"],
            ]

            with (
                patch("prefix_generator.PrefixGenerator", return_value=mock_pg),
                patch("os.path.isfile", return_value=True),
            ):
                replaced = engine._replace_with_model_guided(challengers, model_ratio=0.2)

            # At least one challenger should have been replaced
            self.assertGreater(len(replaced), 0)
            # The replaced indices should be valid
            for idx in replaced:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(challengers))

    def test_model_guided_noop_when_no_champion(self):
        """_replace_with_model_guided returns empty when no champion exists."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            engine.champion = None
            challengers = [[0x02A00093, 0x03A00113, 0x00100073] for _ in range(10)]
            result = engine._replace_with_model_guided(challengers, model_ratio=0.2)
            self.assertEqual(result, [])

    def test_model_guided_noop_when_ratio_zero(self):
        """_replace_with_model_guided returns empty when model_guided_ratio is 0."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            engine.config.model_guided_ratio = 0
            challengers = [[0x02A00093, 0x03A00113, 0x00100073] for _ in range(10)]
            result = engine._replace_with_model_guided(challengers, model_ratio=0.2)
            self.assertEqual(result, [])

    def test_apply_model_guided_slots_in_generation(self):
        """_apply_model_guided_slots is called during challenger generation."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            engine.config.model_guided_ratio = 0.2
            challengers = [[0x02A00093, 0x03A00113, 0x00100073] for _ in range(10)]
            # Mock _replace_with_model_guided to track calls
            with patch.object(engine, "_replace_with_model_guided", return_value=[0, 1]):
                engine._apply_model_guided_slots(challengers)
                engine._replace_with_model_guided.assert_called_once()

    def test_model_guided_uses_temperature_annealing(self):
        """_compute_model_temperature returns different values over generations."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine(tmpdir)
            engine.current_generation = 0
            temp0 = engine._compute_model_temperature()
            engine.current_generation = 50
            temp50 = engine._compute_model_temperature()
            # Temperature should decrease over generations
            self.assertLess(temp50, temp0)


class TestPrefixGeneratorBatch(unittest.TestCase):
    """Test batch_generate for multiple prefixes."""

    def test_batch_generate_multiple_prefixes(self):
        """batch_generate handles multiple prefixes in one call."""
        from prefix_generator import PrefixGenerator

        pg = PrefixGenerator(
            checkpoint_path="/nonexistent/path.pt",
            temperature=0.9,
            max_new_tokens=64,
            device="cpu",
            verbose=False,
        )
        prefixes = [
            ["0x02a00093", "0x03a00113"],
            ["0x002080b3", "0x00000013"],
        ]
        result = pg.batch_generate(
            prefixes=prefixes,
            n=2,
            prefix_count=2,
            temperature=0.9,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # one per prefix
        for prefix_result in result:
            self.assertIsInstance(prefix_result, list)


if __name__ == "__main__":
    unittest.main()
