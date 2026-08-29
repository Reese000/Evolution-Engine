try:
    from qd_archive import MAPElitesArchive, compute_qd_axes
    print("qd_archive import OK")
    print("compute_qd_axes:", compute_qd_axes)
except ImportError as e:
    print(f"ImportError: {e}")

import evolution_engine

print("_HAS_QD:", evolution_engine._HAS_QD)
