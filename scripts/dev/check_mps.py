import sys

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch is not installed")
    sys.exit(2)

if not hasattr(torch.backends, "mps"):
    print("MPS backend not present in this PyTorch build")
    sys.exit(1)

if not torch.backends.mps.is_built():
    print("PyTorch built without MPS support")
    sys.exit(1)

if not torch.backends.mps.is_available():
    print("MPS not available (requires macOS 12.3+ on Apple Silicon)")
    sys.exit(1)

try:
    x = torch.ones(4, device="mps")
    y = (x * 2).sum().item()
except Exception as exc:
    print(f"MPS device allocation failed: {exc}")
    sys.exit(1)

print(f"  smoke test: ones(4) * 2 -> sum = {y}")
print(f"PyTorch {torch.__version__} - MPS available")
