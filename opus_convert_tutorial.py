"""
TUTORIAL BLUEPRINT: Processing FTIR + EC-Lab data with OpusConvert
===================================================================

This template walks through a complete experiment-processing workflow:
    1. Load OPUS spectra and EC-Lab .mpr files
    2. Time-sync them so each spectrum has a potential and cycle label
    3. Pick the spectra you want to plot
    4. Compute differentials and render multi-panel figures

To use it for a new experiment, copy this file, rename it to
    YYMMDD-<electrolyte>-<replicate>.py
and edit the CONFIGURATION block. The downstream sections rarely need
changes once your conventions stabilize.

Common tuning parameters are flagged with `# TUNE:` comments — those are
the places you'll most often revisit when a plot doesn't look right.
"""

import glob
import matplotlib.pyplot as plt
import numpy as np

from parser import OpusConvert
from synchronizer import ECSynchronizer
from plotter import OpusPlotter


# =====================================================================
# CONFIGURATION
# =====================================================================
# Everything that varies per-experiment lives here. Treat the rest of
# the script as a fixed pipeline driven by these values.

# --- Identifiers ---
yymmdd = "yymmdd"
electrolyte = "electrolyte name"
replicate = 1 
prism = "optical element from experiment"
scans = 32 # See the 'Optic' section of your spectrometer
scan_rate = float # See the 'Optic' section of your spectrometer

# --- Input paths ---
# Glob pattern *.[0-9]* matches OPUS extensions like .0, .1, .2, ... .999
# without picking up companion files (.txt, .log, etc).
ftir_dir = f"path to your FTIR experiment directory"
filepaths = glob.glob(f"{ftir_dir}/*.[0-9]*")

eclab_dir = f"path to your EC-lab experiment directory"
eclab_files = glob.glob(f"{eclab_dir}/*.mpr")

ref_xlsx = "path to your references"

# --- Output path stem (figures get suffixes appended below) ---
out = (
    f"root path where you want your figures to go"
    f"{yymmdd}-{electrolyte}-{replicate}/"
    f"{yymmdd}-{electrolyte}-{replicate}"
)

# --- Sync parameters ---
# TUNE: time_shift_hours corrects for clock drift between the FTIR and
# EC-Lab computers. Run the script once with shift=0; the synchronizer
# prints the first timestamp from each source — eyeball the difference,
# round to the nearest hour, set it here, and rerun.
sync_tolerance_seconds = 30
sync_time_shift_hours = 3

# --- Spectrum selection ---
# TUNE: pick the voltage window you care about. The selector returns
# spectra evenly spaced across this window in chronological order.
v_start = 1.8
v_end = -0.5
num_samples = 10
selection_step = "04_LSV"   # Substring of the .mpr filename to scope the search

# --- Plot text and limits ---
title_successive = f"{electrolyte} - {replicate}\n(successive differentials)"
title_baseline = f"{electrolyte} - {replicate}\n(vs. OCV)"
ftir_conditions = (
    f"{yymmdd[2:4]}/{yymmdd[4:6]}/{yymmdd[0:2]} on {prism}\n"
    f"{scans} scans @ {scan_rate}kHz"
)
eclab_conditions = (
    "OCV/PEIS/OCV (step 1-3, 90 min)\n"
    " CV; (step 4, 0.25mV/s, 2.4V$_{Na^{0}/Na^{+}}$ <-> -0.5V$_{Na^{0}/Na^{+}}$)"
)
target_ec_step = "LSV"
ax2_ylim = (-30, 10)

# TUNE: plot_spacing controls vertical gap between differential traces.
# ref_spacing controls gap between reference overlays at the top.
# ref_pad is the gap right above the topmost differential.
# Increase if traces overlap, decrease if they look too far apart.
plot_spacing = 0.0000625
ref_spacing = -0.0005
ref_pad = 0.0001

# Peaks of interest — drawn as vertical guides on each panel.
# Format: (wavenumber, (label_x_offset, label_y_offset))
ax1_peaks = [
    (1760, (0, 0)),
    (1660, (0, -6)),
    (1260, (0, 0)),
    (1437, (0, 0)),
    (867, (0, 0)),
]
ax2_peaks = [
    (1700, (0, 0)),
]
ax3_peaks = [
    (1760, (0, -10)),
    (1660, (0, -16)),
    (1260, (0, -10)),
    (1437, (0, -10)),
    (867, (0, -10)),
]


# =====================================================================
# STEP 1: PARSE FTIR SPECTRA
# =====================================================================
# OpusConvert reads each Bruker OPUS file, sorts by extension number
# (the chronological order in which they were measured), trims the
# common artifact at the end of each spectrum, and interpolates onto a
# common wavenumber grid so all spectra are comparable.
oc = OpusConvert()
oc.load_files(filepaths)
oc.apply_timestamps()
oc.build_dataframes()


# =====================================================================
# STEP 2: SYNCHRONIZE WITH ELECTROCHEMISTRY
# =====================================================================
# ECSynchronizer reads .mpr files, concatenates them in chronological
# order, and tags each OPUS spectrum with the closest EC-Lab row's
# Potential, Current, Cycle, and Source_File (within tolerance).
sync = ECSynchronizer(oc)
sync.load_eclab_binaries(eclab_files)

# sync_and_label prints the first OPUS and first EC-Lab timestamps
# BEFORE applying time_shift_hours. Use this output to calibrate the
# shift value (see TUNE comment in CONFIGURATION).
sync.sync_and_label(
    tolerance_seconds=sync_tolerance_seconds,
    time_shift_hours=sync_time_shift_hours,
)


# =====================================================================
# STEP 3: SELECT TARGET SPECTRA
# =====================================================================
# Two selection methods are available — pick whichever fits the data.
#
# (A) BY POTENTIAL — picks spectra evenly spaced across a voltage
#     window, useful for any technique with a monotonic sweep (LSV,
#     half of a CV, etc).
target_keys, target_labels = sync.select_spectra_by_potential(
    v_start=v_start,
    v_end=v_end,
    num_samples=num_samples,
    step_filter=selection_step,
)

# (B) BY HALF-CYCLE — picks spectra chronologically across one half
#     of one CV cycle, useful when you want to trace chemistry along
#     the sweep direction. Uncomment the block below and comment out
#     (A) if your experiment is a CV.
#
# target_keys, target_labels = sync.select_spectra_by_half_cycle(
#     cycle=4,
#     direction="anodic",       # or "cathodic"
#     num_samples=num_samples,
#     step_filter="04_CV",
#     v_start=v_start,          # optional voltage window within the half
#     v_end=v_end,
# )

# The "background" is the spectrum used as the OCV reference for
# baseline-mode differentials. By default it's the first spectrum of
# the requested step.
bkgd = sync.get_step_baseline(step_filter=selection_step)


# =====================================================================
# STEP 4: SET UP PLOTTER AND LOAD REFERENCES
# =====================================================================
plotter = OpusPlotter(oc, sync)

# Reference spectra are loaded from a wide-format Excel file. Each
# `Wavenumbers` column header starts a new "Reference Set" — useful
# when reference data was collected on different days or instruments.
# Use `set_indices=[0,1]` to combine sets, or `cols_to_drop` to hide
# specific columns from the plot.
plotter.load_references(
    ref_xlsx,
    cols_to_drop=[
        "Write names of references exactly as they appear in your reference sheet",
    ],
)


# =====================================================================
# STEP 5: COMPUTE DIFFERENTIALS — SUCCESSIVE MODE
# =====================================================================
# "successive": each spectrum minus the previous one (highlights what
# changes between adjacent voltages — useful for catching transient
# intermediates).
plotter.calculate_differentials(target_keys, target_labels, bkgd, mode="successive")


# =====================================================================
# STEP 6: PLOT — SUCCESSIVE 3-PANEL
# =====================================================================
# The 3-panel layout: differentials (left), echem trace (middle), raw
# spectra (right), with sample markers shared across panels.
plotter.plot_3_panel_experiment(
    target_keys, ax1_peaks, ax2_peaks, ax3_peaks, bkgd,
    title=title_successive,
    ax2_ylim=ax2_ylim,
    export_path=f"{out}-successive_3panel",
    target_ec_step=target_ec_step,
    ax1_plot_spacing=plot_spacing,
    ax1_ref_spacing=ref_spacing,
    ax1_ref_pad=ref_pad,
    ax1_conditions=ftir_conditions,
    ax2_conditions=eclab_conditions,
    # TUNE: ax1_diff_scale and ax1_ref_scale let you independently
    # amplify the differentials or the references/OCV when amplitudes
    # are mismatched.
    # ax1_diff_scale=10.0,
    # ax1_ref_scale=0.1,
)

# Standalone variant: just the differentials panel, no echem or raw.
fig, ax = plt.subplots(figsize=(8.68 / 3, 6.93), dpi=300)
n_diffs = len(plotter.differential_spectra)
diff_colors = plt.cm.plasma(np.linspace(0.7, 0, n_diffs + 2)).tolist()

plotter._plot_differentials_panel(
    ax=ax,
    background_key=bkgd,
    colors=diff_colors,
    x_bounds=(2000, 600),
    title=title_successive,
    peaks=ax1_peaks,
    line_width=0.5,
    font_size=6,
    conditions=ftir_conditions,
    plot_spacing=plot_spacing,
    ref_spacing=ref_spacing,
    pad=ref_pad,
)
fig.savefig(f"{out}-successive_diff_only.png", bbox_inches="tight")


# =====================================================================
# STEP 7: REPEAT FOR BASELINE MODE
# =====================================================================
# "baseline": each spectrum minus the OCV (highlights total change
# from the start of the experiment — useful for tracking accumulated
# SEI or other persistent products).
plotter.calculate_differentials(target_keys, target_labels, bkgd, mode="baseline")

plotter.plot_3_panel_experiment(
    target_keys, ax1_peaks, ax2_peaks, ax3_peaks, bkgd,
    title=title_baseline,
    ax2_ylim=ax2_ylim,
    export_path=f"{out}-baseline_3panel",
    target_ec_step=target_ec_step,
    ax1_plot_spacing=plot_spacing,
    ax1_ref_spacing=ref_spacing,
    ax1_ref_pad=ref_pad,
    ax1_conditions=ftir_conditions,
    ax2_conditions=eclab_conditions,
)

# Standalone differentials panel — baseline mode
fig, ax = plt.subplots(figsize=(8.68 / 3, 6.93), dpi=300)
plotter._plot_differentials_panel(
    ax=ax,
    background_key=bkgd,
    colors=diff_colors,
    x_bounds=(2000, 600),
    title=title_baseline,
    peaks=ax1_peaks,
    line_width=0.5,
    font_size=6,
    conditions=ftir_conditions,
    plot_spacing=plot_spacing,
    ref_spacing=ref_spacing,
    pad=ref_pad,
)
fig.savefig(f"{out}-baseline_diff_only.png", bbox_inches="tight")


# =====================================================================
# STEP 8: STANDALONE ECHEM PANEL
# =====================================================================
# Useful for slides where the FTIR and the cyclic voltammogram need to
# be separate figures. Pass `cycle=` and `direction=` to filter the
# trace to a single half-cycle of a CV; omit them for the full step.
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
plotter._plot_electrochemistry_panel(
    ax=ax,
    diff_keys=target_keys,
    marker_colors=diff_colors,
    target_ec_step=target_ec_step,
    peaks=ax2_peaks,
    xlim=None,
    ylim=ax2_ylim,
    line_width=0.5,
    font_size=6,
    conditions=eclab_conditions,
    # cycle=1,
    # direction="cathodic",
)
ax.set_title(f"{electrolyte} - {replicate}")
fig.savefig(f"{out}-echem.png", bbox_inches="tight")


# =====================================================================
# TROUBLESHOOTING NOTES
# =====================================================================
# - Spectra missing from the cycle filter? Check the synchronizer's
#   summary output — the most common cause is a different .mpr file
#   reusing the cycle counter (e.g., PEIS has its own cycle 1). Use
#   step_filter to scope to a single technique.
#
# - Differentials look flat? Could be real (LSV before SEI formation)
#   or visual (amplitude mismatch with refs). Try ax1_diff_scale=10.0.
#
# - OCV or references inflate the panel ylim? The panel already clips
#   them to x_bounds, but if you've added new traces via custom code,
#   make sure they're windowed before plotting.
#
# - Time shift unknown? Run once with sync_time_shift_hours=0 — the
#   synchronizer prints both first-timestamps so you can eyeball the
#   offset.