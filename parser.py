"""
opus_convert/parser.py

Load, align, and export Bruker OPUS spectral data.

Original Author: Robert Colley (2026)
License: MIT
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from brukeropusreader import read_file


class OpusConvert:
    """
    Loads, aligns, and exports Bruker OPUS (.0, .1, .2, …) spectra.

    Each loaded spectrum is stored in ``self.spectra`` keyed by sample name.
    The shared interpolated wavenumber axis lives in ``self.wavenumbers``.

    After ``process()`` runs, three DataFrames are populated:

    - ``self.df_ab``       : wide format, one column per sample (rows = wavenumbers)
    - ``self.df_metadata`` : per-sample metadata (timestamps, EC-Lab values, labels)
    - ``self.df_ftir``     : transposed view, one row per sample (columns = wavenumbers)

    Typical usage
    -------------
        opus = OpusConvert(key="Sample")
        opus.load_files(["spec.0", "spec.1", "spec.2"])
        opus.process()
        opus.export("my_run", output_dir="./out")
    """

    # Strips trailing "(GMT-5)" / "(GMT+2)" timezone hints from OPUS time strings
    _GMT_SUFFIX_RE = re.compile(r"\s*\(GMT[+-]?\d+\)\s*$")

    def __init__(self, key: str | Sequence[str] = "Sample"):
        """
        Parameters
        ----------
        key
            If a string, samples are auto-named "<key> 0", "<key> 1", ….
            If a sequence, those names are used directly (must be at least
            as long as the file list passed to ``load_files``).
        """
        self.key = key
        self.spectra: dict[str, dict] = {}
        self.wavenumbers: np.ndarray | None = None

        # Populated after `process()`
        self.df_ab: pd.DataFrame | None = None
        self.df_metadata: pd.DataFrame | None = None
        self.df_ftir: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Loading & cleaning
    # ------------------------------------------------------------------
    def load_files(self, filepaths: Iterable[str]) -> None:
        """
        Read OPUS files from disk, sort them by extension number, and align
        every spectrum onto a common wavenumber axis.

        Sorting by extension number means ``spec.2`` correctly precedes
        ``spec.10`` (lexical sort would not).
        """
        sorted_paths = sorted(filepaths, key=self._sequence_number)

        #Is the given key a list/tuple? Must match number of filepaths provided.
        if isinstance(self.key, (list, tuple)):
            if len(self.key) < len(sorted_paths):
                raise ValueError(
                    f"Got {len(sorted_paths)} files but only {len(self.key)} "
                    f"names in `key`."
                )
            names = list(self.key)[: len(sorted_paths)]
        #If not a Sequence, then assign standard names for samples (ie. Sample 0, Sample 1, Sample 2 ...)
        else:
            names = [f"{self.key} {i}" for i in range(len(sorted_paths))]

        #Populate self.spectra dictionary using brukeropusreader read_file method. 
        for name, path in zip(names, sorted_paths):
            if not os.path.exists(path):
                print(f"Warning: file not found, skipping: {path}")
                continue
            self.spectra[name] = read_file(path)

        #Cleaning and making each spectrum the same length
        self._trim_artifacts()
        self._align_wavenumber_axis()

    @staticmethod
    def _sequence_number(filepath: str) -> int:
        """Return the integer extension of an OPUS file (e.g. ``spec.7`` -> 7)."""
        try:
            return int(str(filepath).rsplit(".", 1)[-1])
        except ValueError:
            return 10**9  # files without numeric extensions go last

    def _trim_artifacts(self) -> None:
        """Remove the trailing artifact sample at index NPT, when present."""
        for data in self.spectra.values():
            try:
                npt = data["AB Data Parameter"]["NPT"]
                data["AB"] = np.delete(data["AB"], npt)
            except IndexError:
                # NPT was already at or past the array end – nothing to drop
                pass

    def _align_wavenumber_axis(self) -> None:
        """
        Interpolate every spectrum onto the *shortest* wavenumber grid found
        across loaded samples, so they all share ``self.wavenumbers``.
        """
        names = list(self.spectra)
        if not names:
            return

        def grid_of(data: dict) -> np.ndarray:
            params = data["AB Data Parameter"]
            return np.linspace(params["FXV"], params["LXV"], params["NPT"])

        # As we walk through the samples, whenever we find a shorter grid,
        # re-interpolate everything we've seen up to this point onto it.
        self.wavenumbers = grid_of(self.spectra[names[0]])
        anchor = 0
        for i, name in enumerate(names):
            comparator = grid_of(self.spectra[name])

            if len(self.wavenumbers) > len(comparator):
                # New shortest grid – downsample everything before this point
                for prev_name in names[anchor:i]:
                    prev = self.spectra[prev_name]
                    prev["AB"] = np.interp(
                        comparator, self.wavenumbers[::-1], prev["AB"][::-1]
                    )
                self.wavenumbers = comparator
                anchor = i
            elif len(self.wavenumbers) < len(comparator):
                # This sample is longer – downsample it onto our axis
                self.spectra[name]["AB"] = np.interp(
                    self.wavenumbers, comparator[::-1], self.spectra[name]["AB"][::-1]
                )

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------
    @classmethod
    def _parse_opus_datetime(cls, ab_params: dict) -> datetime:
        """Parse the absolute datetime stored in an OPUS AB Data Parameter dict."""
        time_str = ab_params.get("TIM") or ab_params.get("TIME")
        if not time_str:
            raise KeyError("Neither 'TIM' nor 'TIME' present in AB Data Parameter.")
        time_str = cls._GMT_SUFFIX_RE.sub("", time_str.strip())
        return datetime.strptime(
            f"{ab_params['DAT']} {time_str}", "%d/%m/%Y %H:%M:%S.%f"
        )

    def apply_timestamps(self) -> None:
        """Annotate every spectrum with ``Absolute_Time`` and ``Elapsed_Time_s``."""
        names = list(self.spectra)
        if not names:
            return
        origin = self._parse_opus_datetime(self.spectra[names[0]]["AB Data Parameter"])
        for name in names:
            ts = self._parse_opus_datetime(self.spectra[name]["AB Data Parameter"])
            self.spectra[name]["Absolute_Time"] = ts
            self.spectra[name]["Elapsed_Time_s"] = (ts - origin).total_seconds()

    # ------------------------------------------------------------------
    # DataFrame assembly
    # ------------------------------------------------------------------
    def build_dataframes(self) -> None:
        """
        Build the three primary DataFrames. Should be called after
        ``apply_timestamps()`` (and after any synchronizer has finished
        attaching EC-Lab labels to the spectra).
        """
        # df_ab: rows = wavenumbers, columns = sample names
        self.df_ab = pd.DataFrame(
            {name: data["AB"] for name, data in self.spectra.items()}
        )
        self.df_ab.insert(0, "Wavenumbers/cm⁻¹", self.wavenumbers)

        # df_metadata: per-sample info, with optional EC-Lab columns
        records = []
        for name, data in self.spectra.items():
            record = {
                "Sample": name,
                "Absolute_Time": data.get("Absolute_Time"),
                "Elapsed_Time_s": data.get("Elapsed_Time_s"),
            }
            if "Potential_V" in data:
                record["Potential_V"] = data["Potential_V"]
            if "Current" in data:
                unit = data.get("Current_Unit", "unknown")
                record[f"Current_{unit}"] = data["Current"]
            if "Label" in data:
                record["Label"] = data["Label"]
            records.append(record)
        self.df_metadata = pd.DataFrame(records).set_index("Sample")

        # df_ftir: transposed FTIR view. Use EC-Lab labels as row labels if
        # available, otherwise fall back to elapsed seconds.
        if "Label" in self.df_metadata.columns:
            row_labels = self.df_metadata["Label"].to_numpy()
        else:
            row_labels = self.df_metadata["Elapsed_Time_s"].to_numpy()
        df_a = pd.DataFrame(
            {name: data["AB"] for name, data in self.spectra.items()}
        ).set_axis(row_labels, axis=1)
        self.df_ftir = df_a.T.set_axis(self.wavenumbers, axis=1)

    def process(self) -> None:
        """Apply timestamps and build all DataFrames in one step."""
        self.apply_timestamps()
        self.build_dataframes()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, filename_prefix: str, output_dir: str = ".") -> None:
        """Write ``df_ab``, ``df_metadata``, and ``df_ftir`` to CSV."""
        if self.df_ab is None or self.df_metadata is None or self.df_ftir is None:
            raise RuntimeError("Call `process()` before exporting.")
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, filename_prefix)
        self.df_ab.to_csv(f"{base}_all.csv")
        self.df_metadata.to_csv(f"{base}_metadata.csv")
        self.df_ftir.to_csv(f"{base}_tir.csv")

    # ------------------------------------------------------------------
    # Spectrum manipulation
    # ------------------------------------------------------------------
    def shift_correction(self, reference_index: int = 0) -> None:
        """Subtract ``AB[reference_index]`` from every spectrum (zero-baseline)."""
        for data in self.spectra.values():
            data["AB"] = data["AB"] - data["AB"][reference_index]

    def rebase(self, background: str = "Sample 0") -> None:
        """
        Recompute absorbance against a different background spectrum:

            A_i = log(ScSm_background / ScSm_i)
        """
        if background not in self.spectra:
            raise KeyError(f"Background sample '{background}' not loaded.")
        ref_scsm = self.spectra[background]["ScSm"]
        for data in self.spectra.values():
            data["AB"] = np.log(ref_scsm / data["ScSm"])
        self.build_dataframes()

    def match_closest(
        self,
        elapsed_seconds: Sequence[float],
        export_filename: str | None = None,
    ) -> pd.DataFrame:
        """
        For each value in ``elapsed_seconds``, find the sample whose
        ``Elapsed_Time_s`` is closest. Returns a DataFrame containing only
        those spectra (with a wavenumber column inserted at position 0).
        """
        if self.df_metadata is None:
            raise RuntimeError("Call `process()` before matching.")

        elapsed = self.df_metadata["Elapsed_Time_s"].to_numpy()
        names = self.df_metadata.index.to_numpy()

        matches: list[str] = []
        for target in elapsed_seconds:
            idx = int(np.argmin(np.abs(elapsed - target)))
            print(
                f"{names[idx]}: {elapsed[idx]:.4f} s "
                f"(d {elapsed[idx] - target:+.4f} s)"
            )
            matches.append(names[idx])

        df = self.df_ab[matches].copy()
        df.insert(0, "Wavenumbers/cm⁻¹", self.wavenumbers)
        if export_filename:
            print(f"Selected data exported to {export_filename}_selected.csv")
            df.to_csv(f"{export_filename}_selected.csv")
        return df

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def voltage_step_count(
        v_initial: float, v_final: float, step_size: float = 0.1
    ) -> int:
        """Inclusive count of equally spaced points from ``v_initial`` to ``v_final``."""
        return int(round(abs(v_initial - v_final) / step_size)) + 1

    @staticmethod
    def voltage_step_labels(
        v_initial: float, v_final: float, step_size: float = 0.1
    ) -> list[str]:
        """E.g. ``voltage_step_labels(2.0, 1.0)`` -> ``['2.0V', '1.9V', …, '1.0V']``."""
        n = OpusConvert.voltage_step_count(v_initial, v_final, step_size)
        return [f"{round(v, 2)}V" for v in np.linspace(v_initial, v_final, n)]
