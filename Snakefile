############################################
# Snakefile for PVfit
############################################

from pathlib import Path
import glob

# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------
configfile: "config.yaml"

PATHS = config["paths"]
WORKFLOW = config.get("workflow", {})

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
# Final target (equivalent to: uv run python main.py --mode all)
# ------------------------------------------------------------
rule all:
    input:
        ANALYSIS_DIR / "analysis.done"


# ------------------------------------------------------------
# FIT: ROOT -> HDF5
# ------------------------------------------------------------
rule fit:
    input:
        glob.glob(INPUT_PATTERN)
    output:
        hdf5=HDF5_FILE
    log:
        FIT_LOG
    message:
        "Running PVfit fitting → {output.hdf5}"
    params:
        cfg=workflow.configfiles[0]
    shell:
        """
        mkdir -p $(dirname {log})
        uv run python main.py \
          --mode fit \
          --config-file {params.cfg} \
          2>&1 | tee {log}
        """


# ------------------------------------------------------------
# ANALYSIS: HDF5 -> plots / tables
# ------------------------------------------------------------
rule analysis:
    input:
        hdf5=HDF5_FILE
    output:
        done=ANALYSIS_DIR / "analysis.done"
    log:
        ANALYSIS_LOG
    params:
        cfg=workflow.configfiles[0]
    message:
        "Running PVfit analysis → {input.hdf5}"
    shell:
        """
        mkdir -p $(dirname {log})
        mkdir -p {ANALYSIS_DIR}
        uv run python main.py \
          --mode analysis \
          --config-file {params.cfg} \
          2>&1 | tee {log}

        touch {output.done}
        """
