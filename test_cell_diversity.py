"""Test per-cell program diversity tracking in QDArchive."""
import sys
from paths import ensure_project_on_path
ensure_project_on_path()

from qd_archive import MAPElitesArchive

# Descriptor that reliably maps to cell 0 with bins=10
_DESC = [-0.99, -0.99]
_BINS = 10


def _cell(desc=None):
    """Helper: return the cell index for a descriptor."""
    a = MAPElitesArchive(bins=_BINS)
    return a._cell_index(desc or _DESC)


def test_cell_diversity_empty_cell():
    """Diversity of an empty cell should be 0.0."""
    archive = MAPElitesArchive(bins=_BINS)
    cell = _cell()
    result = archive.get_cell_program_diversity(cell)
    assert result["diversity"] == 0.0, f"Expected 0.0, got {result['diversity']}"
    assert result["unique_programs"] == 0
    assert result["total_programs"] == 0
    print("PASS: empty cell diversity")


def test_cell_diversity_single_program():
    """A cell with one program should have diversity 0.0."""
    archive = MAPElitesArchive(bins=_BINS)
    cell = _cell()
    prog = [0x00310093, 0x00510113]
    archive.add_solution(_DESC, 1.0, prog)
    result = archive.get_cell_program_diversity(cell)
    assert result["unique_programs"] == 1
    assert result["total_programs"] == 1
    assert result["diversity"] == 0.0, f"Expected 0.0, got {result['diversity']}"
    print("PASS: single program diversity")


def test_cell_diversity_identical_programs():
    """Replacing with identical programs should not increase diversity."""
    archive = MAPElitesArchive(bins=_BINS)
    cell = _cell()
    prog = [0x00310093, 0x00510113]
    archive.add_solution(_DESC, 1.0, prog)
    archive.add_solution(_DESC, 2.0, list(prog))
    result = archive.get_cell_program_diversity(cell)
    assert result["unique_programs"] == 1, f"Expected 1 unique, got {result['unique_programs']}"
    assert result["total_programs"] == 2
    assert result["diversity"] == 0.0, f"Expected 0.0, got {result['diversity']}"
    print("PASS: identical programs diversity")


def test_cell_diversity_different_programs():
    """Different programs in the same cell should increase diversity."""
    archive = MAPElitesArchive(bins=_BINS)
    cell = _cell()
    prog1 = [0x00310093, 0x00510113]
    prog2 = [0x00310093, 0x00510113, 0x00A10213]
    prog3 = [0x00310093]

    archive.add_solution(_DESC, 1.0, prog1)
    archive.add_solution(_DESC, 2.0, prog2)
    archive.add_solution(_DESC, 3.0, prog3)

    result = archive.get_cell_program_diversity(cell)
    assert result["unique_programs"] == 3, f"Expected 3 unique, got {result['unique_programs']}"
    assert result["total_programs"] == 3
    assert result["diversity"] > 0.0, f"Expected diversity > 0, got {result['diversity']}"
    print(f"PASS: different programs diversity = {result['diversity']:.4f}")


def test_cell_diversity_mixed():
    """Mix of identical and different programs."""
    archive = MAPElitesArchive(bins=_BINS)
    cell = _cell()
    prog1 = [0x00310093, 0x00510113]
    prog2 = [0x00310093]

    archive.add_solution(_DESC, 1.0, prog1)
    archive.add_solution(_DESC, 2.0, list(prog1))
    archive.add_solution(_DESC, 3.0, prog2)
    archive.add_solution(_DESC, 4.0, list(prog2))

    result = archive.get_cell_program_diversity(cell)
    assert result["unique_programs"] == 2, f"Expected 2, got {result['unique_programs']}"
    assert result["total_programs"] == 4
    assert 0.0 < result["diversity"] <= 1.0
    print(f"PASS: mixed diversity = {result['diversity']:.4f} (unique_ratio={result['unique_ratio']:.4f})")


def test_cell_diversity_invalid_cell():
    """Invalid cell index should return 0.0."""
    archive = MAPElitesArchive(bins=_BINS)
    result = archive.get_cell_program_diversity(-1)
    assert result["diversity"] == 0.0
    result = archive.get_cell_program_diversity(999)
    assert result["diversity"] == 0.0
    print("PASS: invalid cell index")


def test_cell_diversity_clear():
    """After clear, all cell histories should be empty."""
    archive = MAPElitesArchive(bins=_BINS)
    cell1 = _cell([-0.99, -0.99])
    cell2 = _cell([-0.9, -0.9])
    archive.add_solution([-0.99, -0.99], 1.0, [0x00310093])
    archive.add_solution([-0.9, -0.9], 2.0, [0x00510113])
    archive.clear()
    result1 = archive.get_cell_program_diversity(cell1)
    result2 = archive.get_cell_program_diversity(cell2)
    assert result1["unique_programs"] == 0
    assert result2["unique_programs"] == 0
    print("PASS: clear resets cell histories")


def test_cell_diversity_serialize_deserialize():
    """Diversity data should survive serialization round-trip."""
    archive = MAPElitesArchive(bins=_BINS)
    cell = _cell()
    archive.add_solution(_DESC, 1.0, [0x00310093, 0x00510113])
    archive.add_solution(_DESC, 2.0, [0x00310093])

    data = archive.serialize()
    restored = MAPElitesArchive.deserialize(data)

    orig = archive.get_cell_program_diversity(cell)
    rest = restored.get_cell_program_diversity(cell)
    assert orig["unique_programs"] == rest["unique_programs"]
    assert orig["total_programs"] == rest["total_programs"]
    assert abs(orig["diversity"] - rest["diversity"]) < 1e-10
    print(f"PASS: serialize/deserialize round-trip, diversity={rest['diversity']:.4f}")


def test_cell_diversity_entropy():
    """Entropy-based diversity should be higher with more unique programs."""
    archive = MAPElitesArchive(bins=_BINS)
    cell = _cell()
    # All same programs -> low diversity
    prog = [0x00310093]
    for i in range(5):
        archive.add_solution(_DESC, float(i + 1), list(prog))
    low = archive.get_cell_program_diversity(cell)

    archive.clear()
    # All different programs -> high diversity
    for i in range(5):
        archive.add_solution(_DESC, float(i + 1), [0x00310093 + i])
    high = archive.get_cell_program_diversity(cell)

    assert low["diversity"] == 0.0, f"Expected 0.0 for same programs, got {low['diversity']}"
    assert high["diversity"] > 0.0, f"Expected > 0 for different programs, got {high['diversity']}"
    print(f"PASS: entropy diversity low={low['diversity']:.4f} high={high['diversity']:.4f}")


if __name__ == "__main__":
    test_cell_diversity_empty_cell()
    test_cell_diversity_single_program()
    test_cell_diversity_identical_programs()
    test_cell_diversity_different_programs()
    test_cell_diversity_mixed()
    test_cell_diversity_invalid_cell()
    test_cell_diversity_clear()
    test_cell_diversity_serialize_deserialize()
    test_cell_diversity_entropy()
    print("\nAll tests passed!")
