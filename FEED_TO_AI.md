# FEED_TO_AI.md

## Purpose of this file

You are an AI assistant helping someone use **OpusConvert**, a small Python package for processing Bruker OPUS FTIR spectra synchronized with BioLogic EC-Lab electrochemistry data. This document gives you the context to answer their questions accurately. Read it end-to-end before responding.

The user is most likely an electrochemist or spectroscopist running cyclic voltammetry, linear sweep voltammetry, or similar experiments while collecting FTIR spectra. They want to correlate spectral changes with applied potential.

When the user asks how to do something, prefer concrete code over conceptual explanation, and refer them to `tutorial_blueprint.py` for a complete worked example.

---

## What OpusConvert does

In one sentence: it loads a folder of OPUS spectra and a folder of EC-Lab `.mpr` files, time-aligns them, lets you pick spectra by voltage or by cycle position, computes differentials against an OCV baseline or against the previous spectrum, and produces multi-panel publication-style figures with reference overlays.

---

## Installation

OpusConvert is distributed as a flat directory of Python files. There's no `pip install`. Setup:

```bash
# 1. Create a virtual environment in the project root
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# or: source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `matplotlib`, `openpyxl`, `brukeropusreader`, `galvani`. The first three are common; the last two are domain-specific (OPUS and BioLogic readers respectively).

The user's processing scripts go in the same directory as the modules so the imports `from parser import OpusConvert` etc. resolve as flat modules.

---

## The user's data

What they bring:

| Input | Format | Source |
|---|---|---|
| FTIR spectra | Bruker OPUS files (extensions `.0`, `.1`, `.2`, ... `.999`) | OPUS spectrometer software |
| Electrochemistry | BioLogic `.mpr` binaries, one per technique step | EC-Lab software |
| Reference spectra (optional) | Wide-format Excel file | Library of known compounds |

Typical experiment: open-circuit voltage (OCV) → impedance (PEIS) → another OCV → linear sweep or CV. The FTIR runs continuously through all of this, taking one spectrum every ~10–120 seconds. After the experiment, the user has hundreds of OPUS files and 4–8 `.mpr` files representing one continuous timeline.

The two computers (FTIR PC and EC-Lab PC) often have **clock drift** of minutes to hours. OpusConvert applies a manual `time_shift_hours` to align them.

---

## Module organization

```
parser.py        →  Loads and cleans OPUS spectra
synchronizer.py  →  Time-aligns FTIR with EC-Lab, lets you pick spectra
plotter.py       →  Renders multi-panel figures with reference overlays
__init__.py      →  Exposes the three classes
tutorial_blueprint.py  →  Worked example showing the full pipeline
```

The three classes are designed to be **chained**: each takes the previous one as input.

```
filepaths --> OpusConvert --> ECSynchronizer --> OpusPlotter --> figures
                  (oc)            (sync)            (plotter)
```

`OpusConvert` knows nothing about EC-Lab. `ECSynchronizer` takes an `oc` instance and adds EC data. `OpusPlotter` takes both and produces figures. The user can use `OpusConvert` standalone if they only care about the FTIR data.

---

## Standard pipeline

Every processing script follows this shape:

```python
from parser import OpusConvert
from synchronizer import ECSynchronizer
from plotter import OpusPlotter
import glob

# 1. Parse FTIR
oc = OpusConvert()
oc.load_files(glob.glob("path/to/ftir/*.[0-9]*"))
oc.apply_timestamps()
oc.build_dataframes()

# 2. Sync with EC-Lab
sync = ECSynchronizer(oc)
sync.load_eclab_binaries(glob.glob("path/to/eclab/*.mpr"))
sync.sync_and_label(tolerance_seconds=30, time_shift_hours=3)

# 3. Pick spectra to plot
target_keys, target_labels = sync.select_spectra_by_potential(
    v_start=1.8, v_end=-0.5, num_samples=10, step_filter="04_LSV"
)
bkgd = sync.get_step_baseline(step_filter="04_LSV")

# 4. Plot
plotter = OpusPlotter(oc, sync)
plotter.load_references("references.xlsx")
plotter.calculate_differentials(target_keys, target_labels, bkgd, mode="successive")
plotter.plot_3_panel_experiment(
    target_keys, ax1_peaks, ax2_peaks, ax3_peaks, bkgd,
    title="My Experiment", export_path="output_path"
)
```

`tutorial_blueprint.py` contains a complete, runnable version with all options shown.

---

## Key methods reference

### OpusConvert
| Method | Purpose |
|---|---|
| `load_files(filepaths)` | Reads OPUS files, sorts by extension number, interpolates onto common wavenumber grid |
| `apply_timestamps()` | Pulls the absolute measurement time from each OPUS file's header |
| `build_dataframes()` | Constructs `df_ab`, `df_metadata`, `df_ftir` for later use |

After running these, `oc.spectra` is a dict of `{sample_name: {"AB": absorbance, "Absolute_Time": datetime, ...}}`.

### ECSynchronizer
| Method | Purpose |
|---|---|
| `load_eclab_binaries(eclab_files)` | Reads `.mpr` files and concatenates into one timeline |
| `sync_and_label(tolerance_seconds, time_shift_hours)` | Tags each OPUS spectrum with closest EC-Lab row's potential, current, cycle, source file |
| `select_spectra_by_potential(v_start, v_end, num_samples, step_filter)` | Picks N spectra evenly spaced across a voltage window |
| `select_spectra_by_half_cycle(cycle, direction, num_samples, step_filter, v_start, v_end)` | Picks N spectra chronologically across one half-sweep of one CV cycle |
| `get_step_baseline(step_filter)` | Returns the first spectrum of a given EC-Lab step (for OCV reference) |
| `summary()` | Diagnostic: prints counts of spectra per cycle and per source file |

`step_filter` is a substring of the `.mpr` filename. **Always use it** when the EC-Lab session contains multiple techniques (PEIS, GCPL, CV) — these techniques have their own per-technique cycle counters that share column names with CV cycles, and without `step_filter` they'll be conflated.

### OpusPlotter
| Method | Purpose |
|---|---|
| `load_references(filepath, cols_to_drop, set_indices)` | Loads reference spectra from Excel; `set_indices` selects which "Reference Set" to use (int or list of ints) |
| `calculate_differentials(target_keys, target_labels, bkgd, mode)` | Computes either `"baseline"` (vs OCV) or `"successive"` (each minus previous) differentials |
| `plot_3_panel_experiment(...)` | Main entry point — renders left/middle/right panels: differentials / echem trace / raw spectra |
| `_plot_differentials_panel(...)` | Standalone differentials panel (use when you want a single-panel figure) |
| `_plot_electrochemistry_panel(...)` | Standalone echem panel |

The 3-panel orchestrator accepts `ax1_diff_scale` and `ax1_ref_scale` to independently amplify the differentials or the references/OCV — useful when the differential amplitudes are mismatched with the references.

---

## Common pitfalls

These come up regularly. If the user describes any of them, you'll save them debugging time.

### "Wrong cycle number / spectra from PEIS leaking in"
BioLogic's PEIS technique has an internal cycle counter starting from 1 that gets concatenated with the CV's cycle counter under the same column name. **Fix:** pass `step_filter="04_CV"` (or whatever your CV file is named) to `select_spectra_by_half_cycle` and `select_spectra_by_potential`.

### "Spectra from before the experiment showing up"
If the FTIR ran before EC-Lab started (or the user restarted EC-Lab mid-experiment), some OPUS spectra have no matching EC-Lab data. They'll show up with `Source_File = None`. The synchronizer correctly excludes them when filtering by step.

### "Time shift unknown"
Run `sync.sync_and_label(time_shift_hours=0)` first. The synchronizer prints `First OPUS spectrum: ...` and `First EC-Lab record: ...`. Eyeball the difference, round to the nearest hour, set `time_shift_hours` accordingly, rerun.

### "Differentials look flat"
Three possible causes:
1. They genuinely are flat (chemistry hasn't started yet — common for early voltages in an LSV)
2. References are scaled to the middle differential, but middle has small amplitude → references look prominent and differentials look squished
3. `plot_spacing` is too large for the actual differential amplitudes
**Fix:** try `ax1_diff_scale=10.0, ax1_ref_scale=0.1` to amplify just the differentials.

### "Panel y-axis is way too tall, plots look squished"
Off-window peaks (e.g., OCV's broad water/OH absorbance ~3300 cm⁻¹) inflate matplotlib's autoscaled ylim even though they're not visible. The current panel code clips OCV and references to the visible window before plotting; if the user has added custom plotting, they need to do the same.

### "Half-cycle selection includes a sample with weird voltage"
Usually means the OPUS spectra span a longer real-world time than the EC-Lab session, and a stray spectrum's shifted timestamp landed inside the half-cycle's time range by coincidence. Use `step_filter` and the built-in time-range sanity check usually catches this; if it doesn't, look at `Source_File` for the suspicious sample.

---

## How to help the user effectively

- **For "how do I X" questions:** give them code, refer to the tutorial blueprint for context.
- **For "why is X broken" questions:** ask them to paste the relevant terminal output AND describe what they expected. The synchronizer prints diagnostic info that often reveals the issue immediately.
- **For "my plot looks weird" questions:** ask them to upload the PNG. Layout, scaling, and overlap issues are obvious in the image.
- **For modifications to the modules:** the user owns these files and can edit freely. Suggest changes to `plotter.py` if the issue is purely visual; suggest changes to `synchronizer.py` if the issue is data-selection.
- **Avoid suggesting:** package restructuring, type annotations beyond what's already there, or additional dependencies. The user values keeping the codebase small and inspectable.
