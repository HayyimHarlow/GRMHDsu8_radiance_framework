"""
startup.py
Unified startup environment configuration for GRMHDsu8_radiance_framework.
Sets working directory, adds module paths, verifies repo structure,
and prints pipeline readiness.
"""

import os
import sys

REPO = "/content/GRMHDsu8_radiance_framework"

DIRS = {
    "su8_core": f"{REPO}/su8_core",
    "grmhd_bridge": f"{REPO}/grmhd_bridge",
    "grrt_synthetic": f"{REPO}/grrt_synthetic",
    "inference_pipeline": f"{REPO}/inference_pipeline",
    "examples": f"{REPO}/examples",
    "tests": f"{REPO}/tests",
}

FILES = {
    "basis_ordering.txt": f"{REPO}/examples/basis_ordering.txt",
}

def initialize_environment():
    print("=== SU(8 Radiance Framework Startup ===\n")

    # 1. Working directory
    if os.path.exists(REPO):
        os.chdir(REPO)
    else:
        raise FileNotFoundError(f"Repository not found at {REPO}")

    print(f"Working directory set to: {os.getcwd()}")

    # 2. Add module paths
    if REPO not in sys.path:
        sys.path.append(REPO)
    if "/content" not in sys.path:
        sys.path.append("/content")

    for name, path in DIRS.items():
        if path not in sys.path:
            sys.path.append(path)

    print("Module paths added.\n")

    # 3. Check directories
    missing_dirs = [name for name, path in DIRS.items() if not os.path.isdir(path)]
    missing_files = [name for name, path in FILES.items() if not os.path.isfile(path)]

    print("Directory check:")
    if missing_dirs:
        print("Missing directories:")
        for d in missing_dirs:
            print(f"  - {d}")
    else:
        print("All required directories present.")

    print("\nFile check:")
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("All required files present.")

    if not missing_dirs and not missing_files:
        print("\nPipeline ready ✔")
    else:
        print("\nPipeline NOT ready — fix missing items above.")

    print("\nStartup complete ✔")

