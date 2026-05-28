import sys

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch is not installed")
    sys.exit(2)

if not torch.cuda.is_available():
    print("CUDA not available (CPU only)")
    sys.exit(1)

count = torch.cuda.device_count()
for i in range(count):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
    print(f"  GPU {i}: {name} ({mem} GB)")

try:
    x = torch.ones(4, device="cuda")
    y = (x * 2).sum().item()
except Exception as exc:
    print(f"CUDA device allocation failed: {exc}")
    sys.exit(1)

print(f"  smoke test: ones(4) * 2 -> sum = {y}")
print(f"PyTorch {torch.__version__} - CUDA available ({count} device(s))")
