"""
config_loader.py - Algorithm configuration only
"""

import yaml


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def get_parameters(config):
    algo = config["algorithm"]

    params = {
        "m_minTracks": algo["minTracks"],
        "m_r": algo["m_r"],
        "nsigma": algo["nsigma"],
        "smear_window_size": algo["smear_window_size"],
        "z_scale": algo.get("z_scale", 1.5),
        "t_scale": algo.get("t_scale", 0.06),
        "zmin": algo["z_range"]["min"],
        "zmax": algo["z_range"]["max"],
        "tmin": algo["t_range"]["min"],
        "tmax": algo["t_range"]["max"],
        "Nbins_z": algo["bins"]["z"],
        "Nbins_t": algo["bins"]["t"],
        "max_chi2": algo.get("max_chi2", 10),
        "is_adaptive_tukey": algo.get("is_adaptive_tukey", True),
        "drop_low_track_seeds": algo.get("drop_low_track_seeds", True),
        "merge_vertices": algo.get("merge_vertices", False),
        "nsigma_merge": algo.get("nsigma_merge", 5),
        "recall_threshold": algo.get("recall_threshold", 0.7),
        "purity_threshold": algo.get("purity_threshold", 0.7),
        "plot_event_id": algo.get("plot_event_id", 0),
        "draw_fixed_ellipses": algo.get("draw_fixed_ellipses", False),
        "draw_fitted_ellipses": algo.get("draw_fitted_ellipses", False),
        "beamline_chi2_cut": algo.get("beamline_chi2_cut", False),
        "max_track_bl_chi2": algo.get("max_track_bl_chi2", 10.0),
        "z0_err_cut": algo.get("z0_err_cut", False),
        "pp_max_track_z0_err": algo.get("pp_max_track_z0_err", 1.5),
    }

    params["bin_size_z"] = (params["zmax"] - params["zmin"]) / params["Nbins_z"]
    params["bin_size_t"] = (params["tmax"] - params["tmin"]) / params["Nbins_t"]

    # Scale minTracks by bin area relative to reference bin size,
    # so the threshold remains physically consistent across different binnings.
    ref_bin_size_z = 0.5  # reference: (200 - (-200)) / 800
    ref_bin_size_t = 0.025  # reference: (0.8 - (-0.6)) / 56
    params["m_minTracks"] *= (params["bin_size_z"] * params["bin_size_t"]) / (
        ref_bin_size_z * ref_bin_size_t
    )

    return params


def print_parameters(params):
    print("Algorithm parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
