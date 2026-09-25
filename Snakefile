############################################
# Snakefile for PVfit
############################################

from pathlib import Path
import glob

import yaml

# ------------------------------------------------------------
# Global config
# (override: --config reconstruction_config=... analysis_config=...)
# ------------------------------------------------------------
UNKNOWN_KEYS = set(config) - {"reconstruction_config", "analysis_config"}
if UNKNOWN_KEYS:
    raise ValueError(
        f"Unknown Snakemake config keys: {sorted(UNKNOWN_KEYS)}. "
        "Use --config reconstruction_config=<yaml> analysis_config=<yaml> "
        "(do not pass a stage YAML via --configfile)."
    )

RECONSTRUCTION_CONFIG = config.get("reconstruction_config", "configs/reconstruction.yaml")
ANALYSIS_CONFIG = config.get("analysis_config", "configs/analysis.yaml")

with open(RECONSTRUCTION_CONFIG) as f:
    PATHS = yaml.safe_load(f)["paths"]

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
INPUT_PATTERN = PATHS["input_root"]      # e.g. input/version6/run_0003*.root

# ------------------------------------------------------------
# Outputs (output_dir == experiment ID)
# ------------------------------------------------------------
OUTPUT_DIR   = Path(PATHS["output_dir"])
HDF5_FILE    = OUTPUT_DIR / PATHS["hdf5_file"]
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

# ------------------------------------------------------------
# Logs (per experiment)
# ------------------------------------------------------------
LOG_DIR       = OUTPUT_DIR / "logs"
FIT_LOG       = LOG_DIR / "fit.log"
ANALYSIS_LOG  = LOG_DIR / "analysis.log"


# ------------------------------------------------------------
# Final target (equivalent to: uv run pvfit run)
# ------------------------------------------------------------
rule all:
    input:
        ANALYSIS_DIR / "analysis.done"


# ------------------------------------------------------------
# FIT: ROOT -> HDF5
# ------------------------------------------------------------
rule fit:
    input:
        tracks=glob.glob(INPUT_PATTERN),
        cfg=RECONSTRUCTION_CONFIG
    output:
        hdf5=HDF5_FILE
    log:
        FIT_LOG
    message:
        "Running PVfit fitting → {output.hdf5}"
    shell:
        """
        mkdir -p $(dirname {log})
        uv run pvfit fit \
          --config {input.cfg:q} \
          2>&1 | tee {log}
        """


# ------------------------------------------------------------
# ANALYSIS: HDF5 -> plots / tables
# ------------------------------------------------------------
rule analysis:
    input:
        hdf5=HDF5_FILE,
        cfg=ANALYSIS_CONFIG
    output:
        done=ANALYSIS_DIR / "analysis.done"
    log:
        ANALYSIS_LOG
    message:
        "Running PVfit analysis → {input.hdf5}"
    shell:
        """
        mkdir -p $(dirname {log})
        mkdir -p {ANALYSIS_DIR}
        uv run pvfit analyze \
          --input {input.hdf5:q} \
          --config {input.cfg:q} \
          --output {ANALYSIS_DIR:q} \
          2>&1 | tee {log}

        """
