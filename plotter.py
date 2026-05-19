"""
opus_convert/plotter.py

Three-panel plot generator combining differential spectra, electrochemistry,
and raw spectra into a single figure.

Original Author: Robert Colley (2026)
License: MIT
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parser import OpusConvert
from synchronizer import ECSynchronizer


class OpusPlotter:
    """
    Build the canonical three-panel figure for an OPUS + EC-Lab experiment.

    Layout
    ------
        +-----------+-----------------+
        |   ax3     |                 |
        | (raw)     |     ax1         |
        +-----------+ (differentials  |
        |   ax2     |  + references)  |
        | (E-chem)  |                 |
        +-----------+-----------------+
    """

    def __init__(
        self,
        opus_converter: OpusConvert,
        ec_synchronizer: ECSynchronizer | None = None,
    ):
        self.parser = opus_converter
        self.synchronizer = ec_synchronizer

        # Reference spectra (loaded by `load_references`)
        self.df_references: pd.DataFrame | None = None
        self.df_reference_set: pd.DataFrame | None = None
        self.reference_sets: dict[str, list[str]] = {}
        self._reference_x_columns: dict[str, str] = {}

        # Chronoamperometry data (loaded by `load_chronoamperometry`)
        self.df_ca: pd.DataFrame | None = None
        self.ca: dict[str, np.ndarray] = {}

        # Differential spectra (built by `calculate_differentials`)
        # Maps label -> shifted differential array
        self.differential_spectra: dict[str, np.ndarray] = {}
        # 'mode' of plotting. Smart plot labeling. baseline vs. successive
        self.differential_mode: str = "baseline"

    # ==================================================================
    # Data loading
    # ==================================================================
    def load_chronoamperometry(self, filepath: str) -> dict[str, np.ndarray]:
        """Load chronoamperometry data: column 0 = time, column 1 = voltage."""
        self.df_ca = pd.read_excel(filepath)
        self.ca["t"] = self.df_ca[self.df_ca.columns[0]].to_numpy()
        self.ca["V"] = self.df_ca[self.df_ca.columns[1]].to_numpy()
        return self.ca

    def load_references(
        self,
        filepath: str,
        cols_to_drop: Sequence[str] | None = None,
        set_indices: int | Sequence[int] = 0,
    ) -> None:
        """
        Load wide-format reference spectra from an Excel file. Each
        ``Wavenumbers`` column starts a new "Reference Set", and the columns
        between two ``Wavenumbers`` headers belong to that set.

        Parameters
        ----------
        filepath
            Path to the Excel file.
        cols_to_drop
            Specific reference columns to exclude from the displayed set.
            Missing names are silently ignored.
        set_indices
            Which reference set(s) to display, by integer index. Pass an
            ``int`` for a single set or a ``list[int]`` to combine columns
            from multiple sets. Default is ``0`` (the first set).
            Each spectrum is plotted against the wavenumbers from its own
            set, so combining sets with different x-axes works correctly.
        """
        self.df_references = pd.read_excel(filepath)

        # Group columns into reference sets, demarcated by "Wavenumbers" headers
        current_set = -1
        self.reference_sets = {}
        for col in self.df_references.columns:
            if "Wavenumbers" in str(col):
                current_set += 1
                self.reference_sets[f"Reference Set {current_set}"] = []
            if current_set > -1:
                self.reference_sets[f"Reference Set {current_set}"].append(col)

        # Normalize set_indices to a list and validate
        if isinstance(set_indices, int):
            set_indices = [set_indices]
        n_sets = len(self.reference_sets)
        for i in set_indices:
            if not (0 <= i < n_sets):
                raise IndexError(
                    f"set_indices contains {i}, but only {n_sets} reference "
                    f"set(s) found in the file (valid indices: 0..{n_sets-1})."
                )

        # Build the displayed subset: spectrum columns from each requested
        # set (skipping each set's own Wavenumbers column), plus a map from
        # spectrum-column-name -> wavenumber-column-name for plotting.
        spectrum_cols: list[str] = []
        self._reference_x_columns: dict[str, str] = {}
        for i in set_indices:
            set_name = f"Reference Set {i}"
            cols = self.reference_sets[set_name]
            wavenumber_col = cols[0]
            for col in cols[1::]:
                spectrum_cols.append(col)
                self._reference_x_columns[col] = wavenumber_col

        self.df_reference_set = self.df_references[spectrum_cols].copy()
        self.df_reference_set.dropna(axis=0, how="all", inplace=True)

        if cols_to_drop:
            drop = [c for c in cols_to_drop if c in self.df_reference_set.columns]
            self.df_reference_set.drop(columns=drop, inplace=True)
            for c in drop:
                self._reference_x_columns.pop(c, None)

    # ==================================================================
    # Differential calculation
    # ==================================================================
    def calculate_differentials(
        self,
        diff_keys: Sequence[str],
        diff_labels: Sequence[str],
        background_key: str | None = None,
        mode: str = "baseline",
    ) -> dict[str, np.ndarray]:
        """
        Compute differential spectra and store them on ``self.differential_spectra``
        as ``{label: array}``. Each result is zero-shifted to its first-index value.

        Parameters
        ----------
        diff_keys
            Sample keys (in ``self.parser.spectra``) to compute differentials for,
            in the order they should appear in the output.
        diff_labels
            Display labels, one per ``diff_keys`` entry. In ``"successive"`` mode
            these label the *minuend* (the later spectrum); the method automatically
            formats the final label as ``"<this> − <previous>"``.
        background_key
            The single reference spectrum to subtract from every sample.
            Required when ``mode="baseline"``; ignored in ``"successive"`` mode.
        mode
            - ``"baseline"`` (default): each result is ``spectrum[i] - background``.
              Output has ``len(diff_keys)`` entries.
            - ``"successive"``: each result is ``spectrum[i] - spectrum[i-1]``.
              Output has ``len(diff_keys) - 1`` entries (the first spectrum has
              no predecessor).
        """
        if len(diff_keys) != len(diff_labels):
            raise ValueError(
                f"diff_keys (n={len(diff_keys)}) and diff_labels "
                f"(n={len(diff_labels)}) must be the same length."
            )

        if mode == "baseline":
            if background_key is None:
                raise ValueError("mode='baseline' requires a background_key.")
            background = self.parser.spectra[background_key]["AB"]
            self.differential_spectra = {
                label: self.shift_to_origin(self.parser.spectra[key]["AB"] - background)
                for key, label in zip(diff_keys, diff_labels)
            }

        elif mode == "successive":
            if len(diff_keys) < 2:
                raise ValueError(
                    f"mode='successive' requires at least 2 spectra; got {len(diff_keys)}."
                )
            result: dict[str, np.ndarray] = {}
            for i in range(1, len(diff_keys)):
                current = self.parser.spectra[diff_keys[i]]["AB"]
                previous = self.parser.spectra[diff_keys[i - 1]]["AB"]
                label = f"{diff_labels[i]} − {diff_labels[i - 1]}"
                result[label] = self.shift_to_origin(current - previous)
            self.differential_spectra = result

        else:
            raise ValueError(
                f"mode must be 'baseline' or 'successive'; got {mode!r}."
            )
        self.differential_mode = mode
        return self.differential_spectra

    # ==================================================================
    # Integral calculation
    # ==================================================================    

    def integrate_peaks(
        self,
        peak_windows: list[tuple[float, float]],
        peak_labels: list[str] | None = None,
        baseline: str = "trapezoid",
    ) -> pd.DataFrame:
        """
        Integrate each differential spectrum across each peak window.

        Parameters
        ----------
        peak_windows
            List of (x_left, x_right) wavenumber windows defining each peak.
        peak_labels
            Display labels for each window. Defaults to "X-Y cm-1" auto-format.
        baseline
            "trapezoid" (default): straight line between the window endpoints,
            equivalent to baseline-subtracted trapezoid rule integration.
            "fitted": uses self.parser.spectra[key]["Baseline"] from prior
            baseline_correct() call. Falls back to trapezoid if baseline absent.

        Returns
        -------
        DataFrame indexed by spectrum key (matching self.differential_spectra),
        with one column per peak. Also stored as self.integrated_areas.
        """
        if not self.differential_spectra:
            raise RuntimeError(
                "No differentials available. Run calculate_differentials() first."
            )

        if peak_labels is None:
            peak_labels = [f"{int(min(w))}-{int(max(w))} cm⁻¹" for w in peak_windows]
        if len(peak_labels) != len(peak_windows):
            raise ValueError("peak_labels and peak_windows must have same length.")

        wn = self.parser.wavenumbers
        rows = {}

        for diff_label, y in self.differential_spectra.items():
            areas_for_this_diff = {}
            for window, peak_label in zip(peak_windows, peak_labels):
                i_lo, i_hi = self._x_window_indices(wn, window[0], window[1])
                x_seg = wn[i_lo:i_hi + 1]
                y_seg = y[i_lo:i_hi + 1]

                # Baseline construction
                if baseline == "trapezoid":
                    # Straight line from (x_seg[0], y_seg[0]) to (x_seg[-1], y_seg[-1])
                    bl = np.linspace(y_seg[0], y_seg[-1], len(y_seg))
                elif baseline == "fitted":
                    # Would need a spectrum-key handle to look up the fitted baseline.
                    # Skip for now or raise informatively.
                    raise NotImplementedError(
                        "baseline='fitted' requires per-spectrum baseline lookup; "
                        "not yet supported for differential traces."
                    )
                else:
                    raise ValueError(f"Unknown baseline mode: {baseline!r}")

                # np.trapezoid wants ascending x; OPUS wavenumbers are descending,
                # so flip both before integrating (sign comes out correct).
                area = np.trapezoid(y_seg - bl, x_seg)
                # Take absolute value: the sign just reflects integration direction
                # since OPUS wavenumbers go high-to-low.
                areas_for_this_diff[peak_label] = -area

            rows[diff_label] = areas_for_this_diff

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "differential"
        self.integrated_areas = df
        self._integration_windows = peak_windows  # save for plotting
        self._integration_labels = peak_labels
        return df


    # ==================================================================
    # Top-level plotting entry point
    # ==================================================================
    def plot_3_panel_experiment(
        self,
        diff_keys: Sequence[str],
        ax1_peaks: Sequence[tuple[float, tuple[float, float]]],
        ax2_peaks: Sequence[tuple[float, tuple[float, float]]],
        ax3_peaks: Sequence[tuple[float, tuple[float, float]]],
        background_key: str,
        title: str = "Default Title",
        x_bounds: tuple[float, float] = (2000, 600),
        export_path: str | None = None,
        target_ec_step: str = "LSV",
        target_ec_cycle: int | float | None = None,
        target_ec_direction: str | None = None,
        ax1_plot_spacing: float = 0.00125,
        ax1_ref_spacing: float = 0.05,
        ax1_ref_pad: float = 0.05,
        ax1_diff_scale: float = 1.0,
        ax1_ref_scale: float = 1.0,
        ax1_conditions: str|None = None,
        ax2_conditions: str|None = None,
        ax2_xlim: tuple[float, float] | None = None,
        ax2_ylim: tuple[float, float] | None = None,
        ax3_ylim: tuple[float, float] | None = None,
    ) -> plt.Figure:
        """
        Generate the canonical 3-panel layout. Auto-scales the EC-Lab and
        raw-spectra panels when explicit limits are not provided.

        ``ax{1,2,3}_peaks`` are lists of ``(peak_x, (offset_x, offset_y))``
        used to draw labeled vertical lines on each panel.

        For multi-cycle CVs, ``target_ec_cycle`` and ``target_ec_direction``
        restrict the echem panel (ax2) to a single cycle or half-cycle.
        Pair this with ``ECSynchronizer.select_spectra_by_half_cycle()`` so
        the spectra and the echem trace cover the same window.
        """
        # Shared style constants
        line_width = 0.5
        font_size = 6

        # ---- Figure layout ----
        fig = plt.figure(figsize=(8.68, 6.93), dpi=300)
        ax1 = fig.add_axes([0.5, 0.10, 0.30, 0.80])  # right column: differentials
        ax2 = fig.add_axes([0.1, 0.10, 0.35, 0.50])  # bottom-left: electrochemistry
        ax3 = fig.add_axes([0.1, 0.65, 0.35, 0.25])  # top-left: raw spectra

        # Color palettes
        n_diffs = len(self.differential_spectra)
        diff_colors = plt.cm.plasma(np.linspace(0.7, 0, n_diffs + 2)).tolist()
        marker_colors = plt.cm.plasma(np.linspace(0.7, 0, len(diff_keys))).tolist()

        # ---- Build each panel ----
        self._plot_differentials_panel(
            ax1, background_key, diff_colors, x_bounds,
            title, ax1_peaks, line_width, font_size,
            plot_spacing = ax1_plot_spacing,
            ref_spacing = ax1_ref_spacing,
            pad = ax1_ref_pad,
            diff_scale = ax1_diff_scale,
            ref_scale = ax1_ref_scale,
            conditions = ax1_conditions
        )
        self._plot_electrochemistry_panel(
            ax2, diff_keys, marker_colors, target_ec_step,
            ax2_peaks, ax2_xlim, ax2_ylim, line_width, font_size, ax2_conditions,
            cycle = target_ec_cycle,
            direction = target_ec_direction,
        )
        self._plot_raw_spectra_panel(
            ax3, diff_keys, marker_colors, x_bounds,
            ax3_peaks, ax3_ylim, line_width, font_size,
        )

        if export_path:
            export_dir = os.path.dirname(export_path)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
            fig.savefig(export_path)

        # plt.show()
        return fig


    # ==================================================================
    # Panel: differential spectra + reference overlays
    # ==================================================================
    def _plot_differentials_panel(
        self,
        ax: plt.Axes,
        background_key: str,
        colors: list,
        x_bounds: tuple[float, float],
        title: str,
        peaks: Sequence[tuple[float, tuple[float, float]]],
        line_width: float,
        font_size: int,
        conditions: str|None = None,
        plot_spacing: float = 0.00125,
        ref_spacing: float = 0.005,
        pad: float = 0.005,
        diff_scale: float = 1.0,
        ref_scale: float = 1.0,
    ) -> None:
        x_left, x_right = x_bounds
        wn = self.parser.wavenumbers
        # Window indices for the main wavenumber axis — used so the
        # stacking offset reflects only what's visible in the plot.
        i_lo, i_hi = self._x_window_indices(wn, x_left, x_right)
 
        wn_w = wn[i_lo:i_hi + 1]
 
        # --- Stack the differential traces, ground-floor up ---
        floor = 0.0
        diffs = list(self.differential_spectra.items())
        for i, (label, y) in enumerate(diffs):
            y = y * diff_scale
            y = y[i_lo:i_hi + 1]
            if y.min() < 0:
                y = y - y.min()
            stacked = y + floor
            ax.plot(wn_w, stacked, color=colors[i])
            ax.annotate(
                label,
                xy=(x_right, self._y_at_x(wn_w, stacked, 800)), #Do not change on future updates, this keeps label positions from moving wildly
                xytext=(2.5, -1),
                textcoords="offset points",
                fontsize=font_size,
            )
            floor = stacked.max() + plot_spacing
 
        # Snapshot the height just above the topmost differential — used
        # later to anchor the peak-marker vlines so they bracket the
        # differentials only, not the references or OCV stacked above.
        diffs_top = floor
 
        # --- Reference overlays (if loaded) ---
        # Use the middle (scaled) differential as the amplitude reference,
        # falling back gracefully if there are fewer than three differentials.
        if diffs:
            amplitude_ref = diffs[len(diffs) // 2][1] * diff_scale
        else:
            amplitude_ref = None
 
        if self.df_reference_set is not None and amplitude_ref is not None:
            x_lo, x_hi = sorted(x_bounds)
            for col in self.df_reference_set.columns:
                # Look up this reference's own wavenumber column (each
                # reference set may have a different x-axis).
                wn_col = self._reference_x_columns.get(col)
                if wn_col is None:
                    continue
                ref_x_full = self.df_references[wn_col].to_numpy()
                ref_y_full = self.df_references[col].to_numpy()
                # Mask NaN rows (reference files often have trailing empty rows).
                valid = ~np.isnan(ref_y_full) & ~np.isnan(ref_x_full)
                if not valid.any():
                    continue
                ref_x = ref_x_full[valid]
                ref_y = ref_y_full[valid]
                # Clip to visible x-window before normalizing, for the same
                # reasons as the OCV block below.
                in_window = (ref_x >= x_lo) & (ref_x <= x_hi)
                if not in_window.any():
                    continue
                ref_x = ref_x[in_window]
                ref_y = ref_y[in_window]
                ref_y = self._scale_to(amplitude_ref, self._normalize(ref_y)) * ref_scale
                stacked = ref_y + floor + pad
                ax.plot(ref_x, stacked, color="black", lw=0.5)
                ax.annotate(
                    str(col),
                    xy=(x_right, self._y_at_x(ref_x, stacked, x_right)),
                    xytext=(2.5, -1),
                    textcoords="offset points",
                    fontsize=font_size,
                )
                # ref_x is already windowed, so stacked.max() is the
                # windowed max — no further slicing needed.
                floor = stacked.max() + ref_spacing
 
        # --- Background spectrum overlay ---
        if amplitude_ref is not None:
            background_raw = self.parser.spectra[background_key]["AB"]
            # Clip to visible x-window before normalizing — otherwise an
            # off-window peak (e.g. broad water/OH band ~3300 cm⁻¹) becomes
            # the normalization max, distorting the visible portion AND
            # inflating the panel ylim via matplotlib's autoscale.
            x_lo, x_hi = sorted(x_bounds)
            in_window = (wn >= x_lo) & (wn <= x_hi)
            bg_x = wn[in_window]
            bg_y = background_raw[in_window]
            background_y = self._scale_to(
                amplitude_ref, self._normalize(bg_y)
            ) * ref_scale
            stacked = background_y + floor + pad
            ax.plot(bg_x, stacked, color="black")
            background_label = self.parser.spectra[background_key].get(
                "Label", background_key
            )
            ax.annotate(
                f"OCV {background_label}",
                xy=(x_right, self._y_at_x(bg_x, stacked, x_right)),
                xytext=(2.5, -1),
                textcoords="offset points",
                fontsize=font_size,
            )
 
        # --- Axis chrome + peak markers ---
        self.preset_format(
            ax,
            axis_label_fontsize=font_size,
            xlabel="Wavenumbers (cm$^{-1}$)",
            ylabel=r"$\Delta$A.U.",
            title=title,
            xlim=(x_left, x_right),
            linewidth=line_width,
        )
        # Tag the differential block at the lower-left
        if diffs:
            mode_tag = {
                "baseline": "differential (vs. baseline)",
                "successive": "differential (successive)",
            }.get(self.differential_mode, "differential")
            ax.annotate(
                mode_tag,
                xy=(x_left, diffs[0][1][0]),
                xytext=(2.5, -7.5),
                textcoords="offset points",
                fontsize=font_size,
            )
        if conditions:
            ax.annotate(
                f"{conditions}",
                xy=(x_right, diffs[0][1][0]),
                xytext=(2.5, -15),
                textcoords="offset points",
                fontsize=font_size,
            )
 
        anchor = ax.get_ylim()
        for peak, offset in peaks:
            self.vline(ax, peak, 0, wn, fs=font_size, offset=offset,
                       anchor_y=anchor, lw=0.4, unit="")
    # ==================================================================
    # Plot Integrals panel
    # ==================================================================            
    def plot_peak_areas(
        self,
        target_ec_step: str | None = None,
        show_lsv: bool = True,
        eclab_ylim: tuple[float, float] | None = None,
        shade_differentials: bool = True,
        differentials_xlim: tuple[float, float] | None = None,
        plot_spacing: float = 1e-4,
        title: str | None = None,
        export_path: str | None = None,
        figsize: tuple[float, float] = (10, 5),
    ) -> plt.Figure:
        """
        Two-panel figure: differentials with shaded integration regions (left),
        peak area vs. potential with optional LSV underlay (right).
        """
        if not hasattr(self, "integrated_areas") or self.integrated_areas is None:
            raise RuntimeError("Run integrate_peaks() first.")

        fig = plt.figure(figsize=figsize, dpi=300)
        ax1 = fig.add_axes([0.1,0.1,0.4,0.8])
        ax2 = fig.add_axes([0.65,0.1,0.2,0.8])

        # --- Left panel: differentials with shading ---
        wn = self.parser.wavenumbers
        diffs = list(self.differential_spectra.items())
        colors = plt.cm.plasma(np.linspace(0.7, 0, len(diffs) + 2)).tolist()
        x_lo, x_hi = self._x_window_indices(wn, differentials_xlim[0], differentials_xlim[1])
        floor = 0.0
        # plot_spacing = 1e-4  # could be exposed as a parameter
        for i, (label, y) in enumerate(diffs):
            y_shifted = y - y[x_lo:x_hi + 1].min()
            stacked = y_shifted + floor
            ax1.plot(wn, stacked, color=colors[i], lw=0.6)
            ax1.annotate(
                label, 
                xy = (wn[x_lo], stacked[0]),
                xytext = (6,0),
                textcoords = "offset points",
                fontsize = 6
                )

            if shade_differentials:
                for window in self._integration_windows:
                    i_lo, i_hi = self._x_window_indices(wn, window[0], window[1])
                    x_seg = wn[i_lo:i_hi + 1]
                    y_seg = stacked[i_lo:i_hi + 1]
                    bl_seg = np.linspace(y_seg[0], y_seg[-1], len(y_seg))
                    
                    ax1.fill_between(
                        x_seg, bl_seg, y_seg,
                        # where=[(y_seg > bl_seg)],
                        color=colors[i], alpha=0.25,
                    )




                    # floor = y_seg.max()
            floor = stacked[i_lo:i_hi + 1].max() + plot_spacing

        for window in self._integration_windows:
            self.vline(ax1, window[0], 0, wn, fs=6, offset=(0,0),
               anchor_y=(0,floor), lw=0.4, unit="")
            self.vline(ax1, window[1], 0, wn, fs=6, offset=(0,0),
               anchor_y=(0,floor), lw=0.4, unit="")

        if differentials_xlim:
            ax1.set_xlim(differentials_xlim)
        # ax1.invert_xaxis()  # standard FTIR convention
        ax1.set_xlabel("Wavenumber (cm⁻¹)")
        ax1.set_ylabel("Δ Absorbance (stacked)")
        ax1.set_yticklabels([])
        if title:
            ax1.set_title(title)

        # --- Right panel: area vs. potential ---
        # Pull potential for each differential. The differential label encodes
        # the "current" spectrum; look up its potential from synchronizer.
        potentials = []
        for diff_label in self.integrated_areas.index:
            # Differential labels look like "0.5 V − 0.3 V" or "0.5 V - OCV".
            # Extract the first voltage (the "current" spectrum's potential).
            try:
                v_str = diff_label.split("V")[0].strip()
                potentials.append(float(v_str))
            except ValueError:
                potentials.append(np.nan)
        potentials = np.array(potentials)

        # LSV underlay
        if show_lsv and target_ec_step:
            df = self.synchronizer.eclab_df
            df = df[df["Source_File"].str.contains(target_ec_step, case=False, na=False)]
            if not df.empty:
                current_col = self.synchronizer.find_current_column(df)
                ax2_lsv = ax2.twinx()
                ax2_lsv.plot(df["Potential_V"], df[current_col]*1000,
                             color="gray", alpha=0.5, lw=0.8, zorder=1)
                ax2_lsv.set_ylim(eclab_ylim)
                ax2_lsv.set_ylabel(r"I ($\mu$A)", color="gray")
                ax2_lsv.tick_params(axis="y", labelcolor="gray")

        # Area scatter, one series per peak
        # peak_colors = plt.cm.viridis(np.linspace(0, 0.85, len(self._integration_labels)))
        for j, peak_label in enumerate(self._integration_labels):
            ax2.plot(
                potentials, self.integrated_areas[peak_label], marker = "o", linestyle = '-',
                 label=peak_label, zorder=2,
            )

        ax2.set_xlabel("Potential (V)")
        ax2.set_ylabel("Integrated peak area (a.u.)")
        ax2.legend(loc="best", fontsize=8)
        # ax2.invert_xaxis()  # match LSV convention (high V on left)

        # fig.tight_layout()

        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            fig.savefig(f"{export_path}.png", bbox_inches="tight")

        return fig

    # ==================================================================
    # Panel: electrochemistry trace + sample markers
    # ==================================================================
    def _plot_electrochemistry_panel(
        self,
        ax: plt.Axes,
        diff_keys: Sequence[str],
        marker_colors: list,
        target_ec_step: str,
        peaks: Sequence[tuple[float, tuple[float, float]]],
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
        line_width: float,
        font_size: int,
        conditions: str|None = None,
        cycle: int | float | None = None,
        direction: str | None = None,
    ) -> None:
        plot_df = self._extract_ec_trace(target_ec_step, cycle=cycle, direction=direction)

        # --- Draw the underlying CV / LSV trace ---
        if plot_df is not None and not plot_df.empty:
            ax.plot(plot_df["E"], plot_df["I"] * 1000, label="_nolegend_")

            # Auto-scale axes if the user didn't pass explicit limits
            if ylim is None:
                ylim = self._padded_range(plot_df["I"], pad_frac=0.10)
            if xlim is None:
                xlim = self._padded_range(plot_df["E"], pad_frac=0.05)

            for peak, offset in peaks:
                self.vline(
                    ax, peak, 2, plot_df["E"].to_numpy(),
                    fs=font_size, offset=offset, anchor_y=ylim,
                )

        # --- Overlay one dot per sampled spectrum ---
        # In baseline mode there are len(diff_keys) differential labels (1:1).
        # In successive mode there are len(diff_keys)-1 labels, each describing
        # spec[i] - spec[i-1]; that label belongs to the *later* (minuend)
        # marker, so we shift the alignment by one. The first marker has no
        # preceding differential and is annotated with its own spectrum label
        # to mark the starting point of the sequence.
        diff_labels = list(self.differential_spectra.keys())
        successive_mode = len(diff_labels) == len(diff_keys) - 1

        for i, key in enumerate(diff_keys):
            if key not in self.parser.spectra:
                continue
            data = self.parser.spectra[key]
            voltage = data.get("Potential_V", 0)
            current_ma = data.get("Current", 0) * 1000

            ax.scatter(voltage, current_ma, color=marker_colors[i],
                       s=10, zorder=5, edgecolor="none")

            if successive_mode:
                sample_label = data.get("Label", f"{round(voltage, 2)} V")
            else:
                sample_label = (
                    diff_labels[i] if i < len(diff_labels)
                    else f"{round(voltage, 2)} V"
                )

            ax.annotate(
                sample_label,
                xy=(voltage, current_ma),
                xytext=(-10, 5),
                textcoords="offset points",
                fontsize=font_size,
            )

        self.preset_format(
            ax,
            axis_label_fontsize=font_size,
            xlabel=r"E vs. V$_{Na^{0}/Na^{+}}$",
            ylabel=r"I ($\mu$A)",
            xlim=xlim,
            ylim=ylim,
            ytick_labels_off=False,
            legend=False,
            linewidth=line_width,
        )
        # Step-type label in the high-potential / low-current corner (top-right
        # for a cathodic LSV plotted left=low-E, right=high-E).
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        ax.annotate(
            target_ec_step,
            xy=(x_hi, y_hi),
            xytext=(-3, -8),
            textcoords="offset points",
            fontsize=font_size,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

        if conditions:
            ax.annotate(
                conditions,
                xy=(x_hi, y_lo),
                xytext=(-3, 10),
                textcoords="offset points",
                fontsize=font_size,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    def _extract_ec_trace(
        self,
        target_ec_step: str | None,
        cycle: int | float | None = None,
        direction: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Pull a clean (E, I) DataFrame from the synchronizer's eclab_df.

        Parameters
        ----------
        target_ec_step
            Substring of ``Source_File`` to filter rows by (e.g. ``"04_CV"``).
        cycle
            If given, restrict to this cycle number (uses the cycle column
            found by ``ECSynchronizer.find_cycle_column``). Required if
            ``direction`` is also given.
        direction
            ``"cathodic"`` (decreasing voltage) or ``"anodic"`` (increasing
            voltage). Filters to the corresponding half of ``cycle``.
            Ignored with a warning if ``cycle`` is not provided.
        """
        if self.synchronizer is None or self.synchronizer.eclab_df is None:
            return None
        df = self.synchronizer.eclab_df.copy()

        if target_ec_step:
            df = df[
                df["Source_File"].str.contains(target_ec_step, case=False, na=False)
            ]

        # Unify potential
        if "Potential_V" not in df.columns:
            ECSynchronizer.unify_potential_column(df)
        if "Potential_V" not in df.columns:
            return None

        # Cycle filter
        if cycle is not None:
            cycle_col = ECSynchronizer.find_cycle_column(df)
            if cycle_col is None:
                print("Warning: cycle filter requested but no cycle column found.")
            else:
                df = df[df[cycle_col] == cycle]

        # Direction filter — needs cycle to know which reversal to use
        if direction is not None:
            if direction not in ("cathodic", "anodic"):
                print(f"Warning: invalid direction {direction!r}; ignoring.")
            elif cycle is None:
                print("Warning: direction filter requires a cycle; ignoring.")
            else:
                halves = self.synchronizer._detect_cycle_halves(step_filter=target_ec_step)
                match = next((k for k in halves if k == cycle), None)
                if match is None:
                    print(
                        f"Warning: cycle {cycle} has no detectable reversal; "
                        f"direction filter ignored."
                    )
                else:
                    rev_time = pd.Timestamp(halves[match]["reversal_time"])
                    first_half_dir = halves[match]["first_half_direction"]
                    want_first = (direction == first_half_dir)
                    times = pd.to_datetime(df["Absolute_Time"])
                    df = df[(times < rev_time) == want_first]

        e_series = df["Potential_V"]

        # Combine all current columns into one (some experiments split across
        # I/mA and <I>/mA depending on technique step).
        cur_cols = [c for c in df.columns if c.startswith("I/mA") or c.startswith("<I>/mA")]
        if not cur_cols:
            return None
        i_series = df[cur_cols[0]]
        for col in cur_cols[1:]:
            i_series = i_series.combine_first(df[col])

        return pd.DataFrame({"E": e_series, "I": i_series}).dropna()

    # ==================================================================
    # Panel: raw absorbance spectra
    # ==================================================================
    def _plot_raw_spectra_panel(
        self,
        ax: plt.Axes,
        diff_keys: Sequence[str],
        colors: list,
        x_bounds: tuple[float, float],
        peaks: Sequence[tuple[float, tuple[float, float]]],
        ylim: tuple[float, float] | None,
        line_width: float,
        font_size: int,
    ) -> None:
        x_left, x_right = x_bounds
        wn = self.parser.wavenumbers

        # Plot each requested raw spectrum, shifted to share an origin
        traces = []
        for i, key in enumerate(diff_keys):
            shifted = self.shift_to_origin(self.parser.spectra[key]["AB"], 1000)
            ax.plot(wn, shifted, color=colors[i])
            traces.append(shifted)

        # Auto-compute ylim from the visible window if not provided
        if ylim is None and traces:
            i_left, i_right = self._x_window_indices(wn, x_left, x_right)
            visible = np.concatenate([t[i_left:i_right] for t in traces])
            ylim = self._padded_range(visible, pad_frac=0.10)

        self.preset_format(
            ax,
            axis_label_fontsize=font_size,
            xlabel="Wavenumbers (cm$^{-1}$)",
            ylabel="Absorbance Units (A.U.)",
            xlim=(x_left, x_right),
            ylim=ylim,
            linewidth=line_width,
        )
        ax.annotate(
            "raw",
            xy=(x_left, ax.get_ylim()[1]),
            xytext=(2.5, -6),
            textcoords="offset points",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

        anchor = ax.get_ylim()
        for peak, offset in peaks:
            self.vline(ax, peak, 0, wn, fs=font_size, offset=offset,
                       anchor_y=anchor, lw=0.4, unit="")

    # ==================================================================
    # Static helpers (numerical)
    # ==================================================================
    @staticmethod
    def normalize(y: np.ndarray) -> np.ndarray:
        """Min-max normalize ``y`` to the [0, 1] range."""
        return (y - y.min()) / (y.max() - y.min())

    # Backwards-compatible alias
    _normalize = normalize
    normalize_values = normalize

    @staticmethod
    def scale_to(reference: np.ndarray, normalized_y: np.ndarray) -> np.ndarray:
        """Scale a 0–1 normalized array up to the amplitude of ``reference``."""
        return normalized_y * (reference.max() - reference.min())

    # Backwards-compatible alias
    _scale_to = scale_to
    ballpark_scale = scale_to

    @staticmethod
    def shift_to_origin(arr: np.ndarray, shift_index: int = 0) -> np.ndarray:
        """Subtract ``arr[shift_index]`` so the chosen point sits at zero."""
        return arr - arr[shift_index]

    @staticmethod
    def _y_at_x(x_array: np.ndarray, y_array: np.ndarray, x_target: float) -> float:
        """Value of ``y_array`` at the index whose ``x`` is closest to ``x_target``."""
        idx = int(np.argmin(np.abs(np.asarray(x_array) - x_target)))
        return float(y_array[idx])

    @staticmethod
    def _padded_range(values, pad_frac: float = 0.1) -> tuple[float, float]:
        """``(min, max)`` of ``values`` expanded outward by ``pad_frac`` of the span."""
        v_min, v_max = float(np.min(values)), float(np.max(values))
        span = v_max - v_min if v_max != v_min else 1.0
        return (v_min - pad_frac * span, v_max + pad_frac * span)

    @staticmethod
    def _x_window_indices(
        x_array: np.ndarray, x_left: float, x_right: float
    ) -> tuple[int, int]:
        """First/last indices of ``x_array`` falling inside [x_right, x_left]."""
        # OPUS wavenumbers are typically sorted descending (high → low).
        i_left = int(np.argmin(np.abs(x_array - x_left)))
        i_right = int(np.argmin(np.abs(x_array - x_right)))
        if i_left > i_right:
            i_left, i_right = i_right, i_left
        return i_left, i_right

    # ==================================================================
    # Static helpers (matplotlib styling)
    # ==================================================================
    @staticmethod
    def preset_format(
        ax: plt.Axes,
        *,
        axis_label_fontsize: int = 8,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim=None,
        ylim=None,
        linewidth: float | None = None,
        ytick_labels_off: bool = True,
        grid: bool = False,
        minor_grid: bool = False,
        legend: bool = False,
    ) -> plt.Axes:
        """Apply consistent aesthetic formatting to a Matplotlib axis."""
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=0, fontsize=axis_label_fontsize)
            ax.yaxis.set_label_position("left")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if linewidth is not None:
            for line in ax.lines:
                line.set_linewidth(linewidth)

        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(direction="in", length=2, width=0.5)
        ax.tick_params(which="minor", length=2)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")
        ax.minorticks_off()

        if ytick_labels_off:
            ax.set_yticklabels([])
        if grid:
            ax.grid(True, which="major", alpha=0.25)
        if minor_grid:
            ax.grid(True, which="minor", alpha=0.10)
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(frameon=False, fontsize=6)
        return ax

    @staticmethod
    def vline(
        ax: plt.Axes,
        x: float,
        precision: int,
        x_ref: np.ndarray,
        c: str = "black",
        a: float = 0.2,
        fs: int = 8,
        unit: str = "V",
        offset: tuple[float, float] = (0, 0),
        lw: float = 0.2,
        anchor_y: tuple[float, float] | None = None,
    ) -> plt.Axes:
        """
        Draw a vertical line at ``x`` (snapped to the closest entry in
        ``x_ref`` within ``10**(-precision)``) and annotate it.

        ``anchor_y`` pins the line to a specific y-range; if ``None``,
        the current axis ylim is used.
        """
        tolerance = 10 ** (-precision)
        for value in x_ref:
            if abs(x - round(value, precision)) > tolerance:
                continue

            if anchor_y is not None:
                ax.plot([x, x], [anchor_y[0], anchor_y[1]],
                        color=c, alpha=a, linewidth=lw)
                label_x = x if precision == 0 else value
                label_val = x if precision == 0 else round(x, precision)
                ax.annotate(f"{label_val}{unit}", xy=(label_x, anchor_y[1]),
                            xytext=offset, textcoords="offset points", fontsize=fs)
            else:
                ax.plot([value, value], ax.get_ylim(),
                        color=c, alpha=a, linewidth=lw)
                ax.annotate(f"{round(value, precision)}{unit}",
                            xy=(value, ax.get_ylim()[1]),
                            xytext=offset, textcoords="offset points", fontsize=fs)
            break
        return ax