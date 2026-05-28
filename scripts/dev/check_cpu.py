"""Report CPU features that affect faster-whisper / CTranslate2 throughput.

Prints physical and logical core counts and which SIMD ISA extensions the
runtime CPU supports. Use this to pick TRANSCRIPT_CPU_THREADS and to decide
whether int8 / bfloat16 / float16 compute_types will run natively or fall
back.

Best-effort and read-only: cpuid is queried via py-cpuinfo when available,
falling back to /proc/cpuinfo (Linux), `sysctl` (macOS), or the Windows
registry. No third-party dependency is required — outputs degrade gracefully.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys


def _physical_logical_cores() -> tuple[int | None, int]:
    logical = os.cpu_count() or 0
    physical: int | None = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    if physical is None and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cores = {
                    line.split(":", 1)[1].strip()
                    for line in f
                    if line.startswith("core id")
                }
            if cores:
                physical = len(cores)
        except OSError:
            pass
    if physical is None and platform.system() == "Darwin":
        try:
            physical = int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True,
            ).strip())
        except (subprocess.CalledProcessError, OSError, ValueError):
            pass
    return physical, logical


def _cpu_flags() -> tuple[str, set[str]]:
    """Return (brand string, lowercase flag set). Empty set if undetected."""
    try:
        import cpuinfo  # py-cpuinfo
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw", "unknown")
        flags = {f.lower() for f in (info.get("flags") or [])}
        if not flags and platform.machine().lower() in {"arm64", "aarch64"}:
            flags = {"neon", "asimd"}
        if flags:
            return brand, flags
    except ImportError:
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                text = f.read()
            brand = "unknown"
            flags: set[str] = set()
            for line in text.splitlines():
                if line.startswith("model name") and brand == "unknown":
                    brand = line.split(":", 1)[1].strip()
                elif line.startswith("flags"):
                    flags = {f.lower() for f in line.split(":", 1)[1].split()}
                    break
            return brand, flags
        except OSError:
            return "unknown", set()

    if platform.system() == "Darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
            ).strip()
        except (subprocess.CalledProcessError, OSError):
            brand = "unknown"
        try:
            features = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                text=True,
            ).split()
            flags = {f.lower() for f in features}
        except (subprocess.CalledProcessError, OSError):
            flags = set()
        if not flags and platform.machine().lower() in {"arm64", "aarch64"}:
            flags = {"neon", "asimd"}
        return brand, flags

    if platform.system() == "Windows":
        brand = platform.processor() or "unknown"
        return brand, set()

    return "unknown", set()


def _check(flags: set[str], *names: str) -> bool:
    return any(n.lower() in flags for n in names)


def main() -> int:
    physical, logical = _physical_logical_cores()
    brand, flags = _cpu_flags()

    print(f"CPU:        {brand}")
    print(f"Platform:   {platform.system()} {platform.machine()}")
    if physical is None:
        print(f"Cores:      {logical} logical (physical unknown - install py-cpuinfo or psutil)")
    else:
        smt = "SMT/HT" if logical > physical else "no SMT"
        print(f"Cores:      {physical} physical / {logical} logical ({smt})")

    print()

    suggested = max(1, (physical or logical // 2 or 1))
    print(f"TRANSCRIPT_CPU_THREADS suggestion: {suggested}")
    print("  (physical core count - SMT siblings share execution units and usually hurt int8 GEMM)")

    print()
    print("SIMD ISA support relevant to CTranslate2 / faster-whisper:")

    if not flags:
        print("  (flag detection unavailable on this platform - install py-cpuinfo for details)")
        return 0

    checks = [
        ("AVX2",          "Required for fast int8/fp32 GEMM on x86. Almost universal post-2013.",
         _check(flags, "avx2")),
        ("AVX-512F",      "Wider vectors, helps fp32 throughput on Skylake-X+ / Zen 4+.",
         _check(flags, "avx512f")),
        ("AVX-512_VNNI",  "Native int8 dot-product. Big int8 speedup on Cascade Lake+ / Ice Lake+ / Zen 4+.",
         _check(flags, "avx512_vnni", "avx512vnni")),
        ("AVX-VNNI",      "int8 VNNI on Alder Lake / Sapphire Rapids E-cores (no AVX-512).",
         _check(flags, "avx_vnni", "avxvnni")),
        ("AVX-512_BF16",  "Native bfloat16 GEMM. Required for compute_type=bfloat16 to be fast (Sapphire Rapids, Zen 4+).",
         _check(flags, "avx512_bf16", "avx512bf16")),
        ("AVX-512_FP16",  "Native float16 GEMM. Without this, compute_type=float16 emulates and is SLOWER than fp32.",
         _check(flags, "avx512_fp16", "avx512fp16")),
        ("AMX-INT8",      "Tile-matmul int8 (Sapphire Rapids+ Xeon). Largest int8 boost when CT2 uses it.",
         _check(flags, "amx_int8", "amx-int8", "amxint8")),
        ("AMX-BF16",      "Tile-matmul bf16 (Sapphire Rapids+ Xeon).",
         _check(flags, "amx_bf16", "amx-bf16", "amxbf16")),
        ("NEON",          "ARM equivalent of SSE/AVX (Apple Silicon, Graviton, RPi 4+).",
         _check(flags, "neon", "asimd")),
    ]

    for name, note, ok in checks:
        mark = "yes" if ok else "no "
        print(f"  [{mark}] {name:14s} {note}")

    print()
    print("Recommended compute_type:")
    has_vnni = _check(flags, "avx512_vnni", "avx512vnni", "avx_vnni", "avxvnni", "amx_int8")
    has_bf16 = _check(flags, "avx512_bf16", "avx512bf16", "amx_bf16")
    has_fp16 = _check(flags, "avx512_fp16", "avx512fp16")
    is_arm = _check(flags, "neon", "asimd")

    if has_vnni:
        print("  int8       (native VNNI - best perf/quality tradeoff)")
    elif _check(flags, "avx2") or is_arm:
        print("  int8       (no VNNI but AVX2/NEON int8 path is still 3-4x faster than fp32)")
    else:
        print("  float32    (no AVX2/NEON detected - int8 may not give the expected speedup)")
    if has_bf16:
        print("  bfloat16   (also available - try if you want a quality-leaning fast option)")
    if has_fp16:
        print("  float16    (also available)")
    else:
        print("  float16    NOT recommended - emulated on this CPU and slower than fp32")

    return 0


if __name__ == "__main__":
    sys.exit(main())
