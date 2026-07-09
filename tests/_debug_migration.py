import os
import sys

proj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, proj_root)
sys.path.insert(0, os.path.join(proj_root, "milestone1"))
import torch

torch.manual_seed(42)
from rv32i_model import RV32ITransformer

from milestone1.training.scaling.model_migrator import ModelMigrator

OLD = dict(d_model=256, n_layers=8, n_heads=8, d_ff=1024)
NEW = dict(d_model=512, n_layers=14, n_heads=8, d_ff=1365)

torch.manual_seed(42)
old = RV32ITransformer(vocab_size=2048, d_model=256, n_layers=8, n_heads=8, d_ff=1024, max_seq_len=256, dropout=0.0)
torch.manual_seed(999)
new = RV32ITransformer(vocab_size=2048, d_model=512, n_layers=14, n_heads=8, d_ff=1365, max_seq_len=256, dropout=0.0)

migrator = ModelMigrator()
result = migrator.migrate(old, new, OLD, NEW)
print("Migration:", result.message)

old.eval()
new.eval()

# Test with various inputs
torch.manual_seed(0)
x1 = torch.randint(1, 2048, (1, 64))
with torch.no_grad():
    o1 = old(x1)
    o2 = new(x1)
d = (o1 - o2).abs().max().item()
print(f"randint(1,64) max_diff: {d:.2e}")

x2 = torch.tensor([[42] * 16])
with torch.no_grad():
    o1 = old(x2)
    o2 = new(x2)
d = (o1 - o2).abs().max().item()
print(f"fixed input max_diff: {d:.2e}")

torch.manual_seed(123)
x3 = torch.randint(1, 2048, (1, 32))
with torch.no_grad():
    o1 = old(x3)
    o2 = new(x3)
d = (o1 - o2).abs().max().item()
print(f"manual_seed(123) max_diff: {d:.2e}")
