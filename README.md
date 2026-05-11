# OpusConvert

A small Python toolkit for processing Bruker OPUS FTIR spectra alongside BioLogic EC-Lab electrochemistry data. Time-synchronizes the two streams, lets you select spectra by voltage or cycle position, computes differentials, and renders publication-ready multi-panel figures.

Built for in-situ FTIR experiments where you want to correlate spectral changes with applied potential without the manual overhead of matching timestamps, labeling spectra by voltage, and hand-tuning matplotlib.

---

## What it does

- **Parses** Bruker OPUS files (`.0`, `.1`, …) — handles extension-based ordering, common artifacts, and interpolation onto a shared wavenumber grid.
- **Synchronizes** OPUS spectra with BioLogic `.mpr` files via timestamp matching, with a manual clock-drift correction for cases where the two instruments have different system times.
- **Selects** target spectra by voltage window, by cycle, or by half-cycle direction (anodic/cathodic).
- **Plots** three-panel publication figures (differentials / electrochemistry trace / raw spectra) with stackable reference overlays from an Excel library.

---

## Installation

```bash
git clone https://github.com/robert-colley/opus_convert.git
cd opus_convert

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `openpyxl`, `brukeropusreader`, `galvani`.

Tested on Python 3.10+.

---

## Quick start

```python
import glob
from parser import OpusConvert
from synchronizer import ECSynchronizer
from plotter import OpusPlotter

# Parse FTIR
oc = OpusConvert()
oc.load_files(glob.glob("path/to/ftir/*.[0-9]*"))
oc.apply_timestamps()
oc.build_dataframes()

# Sync with EC-Lab
sync = ECSynchronizer(oc)
sync.load_eclab_binaries(glob.glob("path/to/eclab/*.mpr"))
sync.sync_and_label(tolerance_seconds=30, time_shift_hours=0)

# Pick spectra spanning a voltage window
target_keys, target_labels = sync.select_spectra_by_potential(
    v_start=1.8, v_end=-0.5, num_samples=10, step_filter="04_CV"
)

# Plot
plotter = OpusPlotter(oc, sync)
plotter.load_references("references.xlsx")
plotter.calculate_differentials(
    target_keys, target_labels,
    sync.get_step_baseline(step_filter="04_CV"),
    mode="successive"
)
plotter.plot_3_panel_experiment(
    target_keys, ax1_peaks=[], ax2_peaks=[], ax3_peaks=[],
    background_key=sync.get_step_baseline(step_filter="04_CV"),
    title="My Experiment",
    export_path="my_experiment_3panel"
)
```

See `tutorial_blueprint.py` for a complete worked example covering every option.

---

## Project structure

```
opus_convert/
├── parser.py                 # OpusConvert: load and clean OPUS spectra
├── synchronizer.py           # ECSynchronizer: time-align with EC-Lab data
├── plotter.py                # OpusPlotter: multi-panel figures with overlays
├── __init__.py               # Exports the three classes
├── tutorial_blueprint.py     # Worked example — copy and adapt per experiment
├── FEED_TO_AI.md             # Drop into Claude/ChatGPT to get help with usage
├── requirements.txt
├── LICENSE
└── README.md
```

The three classes are designed to chain: `OpusConvert → ECSynchronizer → OpusPlotter`. Each takes the previous one as input.

---

## Documentation

- `tutorial_blueprint.py` — heavily commented end-to-end example. The fastest way to learn the pipeline.
- `FEED_TO_AI.md` — context file written for AI assistants. Paste it as your first message when asking ChatGPT/Claude for help with OpusConvert, and you'll get answers that actually match the codebase instead of hallucinated API.

---

## Status

Active development. The API is stable for the core pipeline (parse → sync → select → plot) but visualization knobs are still being added as new experiments surface new needs. Pin a commit hash if you need reproducibility.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If this tool contributes to a publication, a citation or acknowledgment is appreciated but not required.

```
Colley, R. (2026). OpusConvert: Synchronized processing of FTIR and electrochemistry data.
https://github.com/robert-colley/opus_convert
```
