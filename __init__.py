"""
opus_convert
============

A small toolkit for processing Bruker OPUS spectra alongside EC-Lab
electrochemistry data.

Public API
----------
    from opus_convert import OpusConvert, ECSynchronizer, OpusPlotter

Typical workflow
----------------
    opus = OpusConvert("Sample")
    opus.load_files(["spec.0", "spec.1", ...])

    sync = ECSynchronizer(opus)
    sync.load_eclab_binaries(["01_OCV.mpr", "02_LSV.mpr"])
    sync.sync_and_label(tolerance_seconds=5.0)

    opus.process()                       # builds df_ab / df_metadata / df_ftir
    opus.export("my_run", "./out")

    plotter = OpusPlotter(opus, sync)
    plotter.load_references("refs.xlsx")
    keys, labels = sync.select_spectra_by_potential(2.5, 1.5, 6, step_filter="LSV")
    plotter.calculate_differentials(keys, labels, sync.get_step_baseline("OCV"))
    plotter.plot_3_panel_experiment(
        diff_keys=keys,
        ax1_peaks=[], ax2_peaks=[], ax3_peaks=[],
        background_key=sync.get_step_baseline("OCV"),
    )
"""

from .parser import OpusConvert
from .synchronizer import ECSynchronizer
from .plotter import OpusPlotter

__all__ = ["OpusConvert", "ECSynchronizer", "OpusPlotter"]
