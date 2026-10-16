"""
executor.py
Unified pipeline executor for GRMHDsu8_radiance_framework.
Runs module pipeline + SU(8) inference + archiving.
"""

import importlib
import inspect
import os
import numpy as np

from GRMHDsu8_radiance_framework.logging_utils import PipelineLogger
from GRMHDsu8_radiance_framework.inference.inference_hmc_nuts import run_su8_hmc
from GRMHDsu8_radiance_framework.data.load_eht import load_eht_subset

logger = PipelineLogger()
execution_context = {}
execution_log = []

# ------------------------------------------------------------
# Module loader
# ------------------------------------------------------------
def load_module(module_name):
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return module

# ------------------------------------------------------------
# Constructor finder
# ------------------------------------------------------------
def find_constructor(module):
    prefixes = ["generate_", "build_", "create_"]
    for prefix in prefixes:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith(prefix):
                return getattr(module, name)
    return None

# ------------------------------------------------------------
# Execute module
# ------------------------------------------------------------
def execute(module_name, step_label):
    logger.log(f"Executing {step_label}: {module_name}")

    module = load_module(module_name)
    constructor = find_constructor(module)

    if constructor is None:
        execution_context[module_name] = module
        execution_log.append({
            "step": step_label,
            "module": module_name,
            "constructor": None,
            "status": "imported",
        })
        return module

    signature = inspect.signature(constructor)
    args = {}

    for param in signature.parameters.values():
        pname = param.name
        if pname in execution_context:
            args[pname] = execution_context[pname]
        elif param.default is not inspect.Parameter.empty:
            continue
        else:
            raise RuntimeError(f"Missing dependency '{pname}' for {module_name}")

    result = constructor(**args)

    if result is not None:
        execution_context[module_name] = result

    execution_log.append({
        "step": step_label,
        "module": module_name,
        "constructor": constructor.__name__,
        "status": "complete",
    })

    return result

# ------------------------------------------------------------
# Pipeline definition
# ------------------------------------------------------------
PIPELINE = [
    ("1.1", "GRMHDsu8_radiance_framework.su8_core.geometry"),
    ("1.2", "GRMHDsu8_radiance_framework.su8_core.acoustic"),
    ("1.3", "GRMHDsu8_radiance_framework.su8_core.stability"),

    ("2.1", "GRMHDsu8_radiance_framework.grmhd_bridge.source_terms"),

    ("3.1", "GRMHDsu8_radiance_framework.grrt_synthetic.radiative_transfer"),
    ("3.2", "GRMHDsu8_radiance_framework.grrt_synthetic.polarization"),

    ("4.1", "GRMHDsu8_radiance_framework.inference_pipeline.feature_space"),
    ("4.2", "GRMHDsu8_radiance_framework.inference_pipeline.likelihood"),
]

# ------------------------------------------------------------
# Unified entry point
# ------------------------------------------------------------
def run_all():
    logger.log("=== SU(8 Radiance Framework Pipeline Execution ===")

    # 1) Run module pipeline
    for step, module_name in PIPELINE:
        execute(module_name, step)

    logger.log("Module pipeline complete ✔")

    # 2) Load EHT data (absolute path to repo-local eht_data)
    eht_data = load_eht_subset("/content/GRMHDsu8_radiance_framework/eht_data")

    # 3) Run SU(8) inference
    samples = run_su8_hmc(eht_data, num_warmup=500, num_samples=1000, num_chains=1)

    # 4) Archive results (canonical archive directory)
    ARCHIVE_DIR = "/content/GRMHDsu8_radiance_framework/archive_v2"
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    posterior_path = os.path.join(ARCHIVE_DIR, "posterior_samples_su8.npz")
    np.savez(posterior_path, posterior_samples=samples)

    logger.log("Posterior samples archived ✔")
    logger.summary()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    run_all()

