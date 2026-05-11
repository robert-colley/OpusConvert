"""
opus_convert/synchronizer.py

Match each OPUS spectrum to its nearest EC-Lab data point in time.

Original Author: Robert Colley (2026)
License: MIT
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from galvani import MPRfile

from parser import OpusConvert


class ECSynchronizer:
    """
    Synchronize OPUS spectra (parsed by ``OpusConvert``) with EC-Lab ``.mpr``
    binary recordings using absolute timestamps.

    Typical workflow
    ----------------
        opus = OpusConvert("Sample")
        opus.load_files([...])

        sync = ECSynchronizer(opus)
        sync.load_eclab_binaries(["01_OCV.mpr", "02_LSV.mpr"])
        sync.sync_and_label(tolerance_seconds=5.0, time_shift_hours=0)

        opus.process()   # now picks up Potential_V / Label / Current / etc.
    """

    def __init__(self, opus_converter: OpusConvert):
        self.parser = opus_converter
        self.eclab_df: pd.DataFrame | None = None
        self.synced_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # EC-Lab loading
    # ------------------------------------------------------------------
    def load_eclab_binaries(
        self, filepaths: str | Iterable[str]
    ) -> pd.DataFrame:
        """
        Read one or more ``.mpr`` files into a single, chronologically sorted
        DataFrame stored on ``self.eclab_df``.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        filepaths = list(filepaths)

        frames = []
        for path in filepaths:
            print(f"Parsing EC-Lab binary: {path}")
            mpr = MPRfile(path)
            df = pd.DataFrame(mpr.data)
            df["Absolute_Time"] = mpr.timestamp + pd.to_timedelta(
                df["time/s"], unit="s"
            )
            df["Source_File"] = os.path.basename(path)
            frames.append(df)

        # `merge_asof` requires a monotonically increasing key – sort now.
        self.eclab_df = (
            pd.concat(frames, ignore_index=True)
            .sort_values("Absolute_Time")
            .reset_index(drop=True)
        )
        print(
            f"Loaded {len(filepaths)} EC-Lab file(s); "
            f"{len(self.eclab_df)} total rows."
        )
        return self.eclab_df

    # ------------------------------------------------------------------
    # OPUS timestamp extraction
    # ------------------------------------------------------------------
    def _opus_timestamps(self) -> pd.DataFrame:
        """Pull absolute datetimes out of the parser's spectra into a DataFrame."""
        records = [
            {
                "Sample_Name": name,
                "Absolute_Time": OpusConvert._parse_opus_datetime(
                    data["AB Data Parameter"]
                ),
            }
            for name, data in self.parser.spectra.items()
        ]
        return pd.DataFrame(records).sort_values("Absolute_Time")

    # ------------------------------------------------------------------
    # EC-Lab column unification
    # ------------------------------------------------------------------
    @staticmethod
    def unify_potential_column(df: pd.DataFrame) -> str | None:
        """
        Combine ``Ewe/V`` and ``<Ewe>/V`` into a single ``Potential_V`` column
        on ``df`` (in place).

        Returns
        -------
        str or None
            ``"Potential_V"`` if a source column was found, else ``None``.
        """
        has_inst = "Ewe/V" in df.columns
        has_avg = "<Ewe>/V" in df.columns
        if has_inst and has_avg:
            df["Potential_V"] = df["Ewe/V"].fillna(df["<Ewe>/V"])
        elif has_inst:
            df["Potential_V"] = df["Ewe/V"]
        elif has_avg:
            df["Potential_V"] = df["<Ewe>/V"]
        else:
            return None
        return "Potential_V"

    @staticmethod
    def find_current_column(df: pd.DataFrame) -> str | None:
        """First current column in ``df`` (e.g. ``I/mA`` or ``<I>/uA``), if any."""
        return next(
            (c for c in df.columns if c.startswith("I/") or c.startswith("<I>/")),
            None,
        )

    @staticmethod
    def find_cycle_column(df: pd.DataFrame) -> str | None:
        return next((c for c in df.columns if "cycle" in c.lower()), None)

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------
    def sync_and_label(
        self,
        tolerance_seconds: float = 5.0,
        time_shift_hours: float = 0.0,
    ) -> pd.DataFrame:
        """
        Use ``merge_asof`` to attach the nearest EC-Lab row to each OPUS
        spectrum. For every successful match, the parser's spectrum dict
        gets these extra keys:

        - ``Potential_V``  : voltage at the matched time
        - ``Label``        : ``"<voltage> V"``
        - ``Current``      : current at the matched time (if available)
        - ``Current_Unit`` : ``'mA'``, ``'uA'``, …
        - ``Source_File``  : the ``.mpr`` file the match came from
        - ``Cycle``        : cycle number, if a cycle column exists

        Parameters
        ----------
        tolerance_seconds
            Maximum allowable gap between an OPUS spectrum's timestamp and
            its nearest EC-Lab row. Spectra with no row inside this window
            are left unlabeled.
        time_shift_hours
            Offset added to OPUS timestamps before matching, to correct for
            timezone or clock drift between instruments.
        """
        if self.eclab_df is None:
            raise RuntimeError("Call `load_eclab_binaries()` before syncing.")

        pot_col = self.unify_potential_column(self.eclab_df)
        if pot_col is None:
            print(
                "Warning: neither 'Ewe/V' nor '<Ewe>/V' found in EC-Lab data; "
                "skipping sync."
            )
            return pd.DataFrame()

        cur_col = self.find_current_column(self.eclab_df)
        cycle_col = self.find_cycle_column(self.eclab_df)

        opus_times = self._opus_timestamps()
        # Align datetime precision so merge_asof doesn't raise MergeError.
        # Python datetime -> datetime64[us]; EC-Lab timestamps are datetime64[ns].
        ec_res = self.eclab_df["Absolute_Time"].dtype
        opus_times["Absolute_Time"] = opus_times["Absolute_Time"].astype(ec_res)

        # Clock comparison — print the earliest unshifted timestamp from each
        # source so the user can spot timezone or clock-drift mismatches and
        # set time_shift_hours accordingly on a rerun.
        first_opus = opus_times["Absolute_Time"].iloc[0]
        first_eclab = self.eclab_df["Absolute_Time"].iloc[0]
        print(f"First OPUS spectrum: {first_opus}")
        print(f"First EC-Lab record: {first_eclab}")

        if time_shift_hours:
            opus_times["Absolute_Time"] += pd.Timedelta(hours=time_shift_hours)
            print(f"Applied {time_shift_hours:+g} h shift to OPUS timestamps.")

        print("Synchronizing OPUS spectra with EC-Lab data...")
        self.synced_df = pd.merge_asof(
            opus_times,
            self.eclab_df,
            on="Absolute_Time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tolerance_seconds),
        )

        # Push results back into the parser's spectra dict
        successes = 0
        for _, row in self.synced_df.iterrows():
            potential = row[pot_col]
            if pd.isna(potential):
                continue

            target = self.parser.spectra[row["Sample_Name"]]
            target["Potential_V"] = potential
            target["Label"] = f"{round(potential, 1)} V"

            if not pd.isna(row.get("Source_File")):
                target["Source_File"] = row["Source_File"]
            if cycle_col and not pd.isna(row[cycle_col]):
                target["Cycle"] = row[cycle_col]
            if cur_col and not pd.isna(row[cur_col]):
                target["Current"] = row[cur_col]
                target["Current_Unit"] = (
                    cur_col.split("/", 1)[1] if "/" in cur_col else "unknown"
                )
            successes += 1

        print(f"Labeled {successes}/{len(self.synced_df)} spectra.")
        if cur_col:
            print(f"Current source column: {cur_col!r}")
        return self.synced_df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """
        Print a breakdown of what got attached to spectra during
        ``sync_and_label()``. Useful when ``select_spectra_by_potential()``
        returns fewer matches than expected — the breakdown usually shows
        why (spectra outside tolerance, synced to a non-cyclic step, etc.).
        """
        from collections import Counter

        spectra = self.parser.spectra
        total = len(spectra)
        with_cycle = sum(1 for d in spectra.values() if "Cycle" in d)
        with_source = sum(1 for d in spectra.values() if d.get("Source_File"))

        print(f"\n--- Sync summary ---")
        print(f"Total spectra:            {total}")
        print(f"Synced (within tolerance): {with_source}/{total}")
        print(f"With cycle data:          {with_cycle}/{total}")

        cycle_counts = Counter(d.get("Cycle") for d in spectra.values())
        print(f"\nCycle distribution:")
        for cyc in sorted(cycle_counts, key=lambda x: (x is None, x)):
            label = f"{cyc!r}" if cyc is not None else "None (no cycle data)"
            print(f"  {label:25s} {cycle_counts[cyc]} spectra")

        source_counts = Counter(d.get("Source_File") for d in spectra.values())
        print(f"\nSource file distribution:")
        for src, count in source_counts.most_common():
            label = src if src is not None else "None (failed sync)"
            print(f"  {count:5d}  {label}")
        print()

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def select_spectra_by_potential(
        self,
        v_start: float,
        v_end: float,
        num_samples: int,
        step_filter: str | None = None,
        cycle_filter: int | None = None,
    ) -> tuple[list[str], list[str]]:
        """
        Pick ``num_samples`` evenly spaced spectra inside the [v_start, v_end]
        voltage window. Spectra are returned ordered by potential, descending
        if ``v_start > v_end`` (cathodic) and ascending otherwise (anodic).

        Parameters
        ----------
        step_filter
            Only consider spectra whose ``Source_File`` contains this
            substring (case-insensitive). Useful for picking out, e.g.,
            "02_LSV" from a multi-step sequence.
        cycle_filter
            Only consider spectra with this cycle number.
        """
        candidates: list[tuple[str, float]] = []
        # Track why spectra were dropped so we can report intelligibly
        dropped = {"no_potential": 0, "step_filter": 0,
                   "cycle_no_data": 0, "cycle_mismatch": 0}

        for name, data in self.parser.spectra.items():
            potential = data.get("Potential_V")
            if potential is None or pd.isna(potential):
                dropped["no_potential"] += 1
                continue
            if step_filter and step_filter.lower() not in str(
                data.get("Source_File", "")
            ).lower():
                dropped["step_filter"] += 1
                continue
            if cycle_filter is not None:
                spectrum_cycle = data.get("Cycle")
                if spectrum_cycle is None or pd.isna(spectrum_cycle):
                    dropped["cycle_no_data"] += 1
                    continue
                if spectrum_cycle != cycle_filter:
                    dropped["cycle_mismatch"] += 1
                    continue
            candidates.append((name, potential))

        # Emit a warning if cycle_filter dropped many spectra for lack of
        # cycle data — usually means it should be combined with step_filter
        if cycle_filter is not None and dropped["cycle_no_data"] > 0:
            print(
                f"Note: cycle_filter={cycle_filter} excluded "
                f"{dropped['cycle_no_data']} spectra that have no cycle data "
                f"(typically OCV/PEIS steps or spectra outside sync tolerance). "
                f"If you only want CV cycles, also pass step_filter='<CV step>'."
            )

        v_min, v_max = sorted((v_start, v_end))
        in_window = [c for c in candidates if v_min <= c[1] <= v_max]
        if not in_window:
            print(
                f"Warning: no spectra found between {v_start} V and {v_end} V "
                f"after filtering. Drop counts: {dropped}. "
                f"Run synchronizer.summary() for more detail."
            )
            return [], []

        in_window.sort(key=lambda c: c[1], reverse=v_start > v_end)

        # Choose evenly spaced indices across the filtered list
        if num_samples >= len(in_window):
            print(
                f"Note: only {len(in_window)} spectra in window; returning all."
            )
            picks = in_window
        elif num_samples == 1:
            picks = [in_window[0]]
        else:
            step = (len(in_window) - 1) / (num_samples - 1)
            picks = [in_window[round(i * step)] for i in range(num_samples)]

        keys = [name for name, _ in picks]
        labels = [self.parser.spectra[k]["Label"] for k in keys]
        print(
            f"Selected {len(keys)} spectra from "
            f"{picks[0][1]:.3f} V to {picks[-1][1]:.3f} V."
        )
        return keys, labels

    # ------------------------------------------------------------------
    # Half-cycle detection
    # ------------------------------------------------------------------
    def _detect_cycle_halves(self, step_filter: str | None = None) -> dict:
        """
        For each cycle in ``self.eclab_df``, find the reversal time and
        the direction of the first half-sweep.

        Parameters
        ----------
        step_filter
            If provided, only consider rows whose ``Source_File`` contains
            this substring (case-insensitive) before grouping by cycle.
            **Important** when the EC-Lab session contains multiple
            techniques: BioLogic's PEIS, GCPL, and other techniques have
            their own per-technique cycle counters that share column names
            with CV cycles. Without ``step_filter``, those counters get
            mixed and "cycle 1" can refer to the first PEIS sweep as well
            as the first CV cycle.

        Returns
        -------
        dict
            ``{cycle_value: {'reversal_time': Timestamp,
                              'first_half_direction': 'cathodic'|'anodic'}}``

        Within each cycle a reversal is defined as the index of the most
        extreme potential — the trajectory reaches one of the scan limits
        and turns around. This is robust to noise without smoothing,
        and handles cycles that start either anodic or cathodic.
        Cycles too short for a meaningful reversal are skipped.
        """
        if self.eclab_df is None:
            raise RuntimeError("Call `load_eclab_binaries()` and `sync_and_label()` first.")

        df = self.eclab_df
        if step_filter:
            df = df[df["Source_File"].str.contains(step_filter, case=False, na=False)]
            if df.empty:
                raise RuntimeError(
                    f"step_filter={step_filter!r} matched no EC-Lab rows."
                )

        cycle_col = self.find_cycle_column(df)
        if cycle_col is None:
            raise RuntimeError("No cycle column found in EC-Lab data.")
        if "Potential_V" not in df.columns:
            raise RuntimeError("Run `sync_and_label()` first to unify potential column.")

        halves: dict = {}
        for cyc, group in df.groupby(cycle_col):
            if pd.isna(cyc) or len(group) < 5:
                continue

            v = group["Potential_V"].to_numpy(dtype=float)
            t = group["Absolute_Time"].to_numpy()
            v_start = v[0]

            # The reversal is wherever the trajectory is most extreme.
            # Whether that's the min or max tells us the start direction.
            i_min, i_max = int(np.argmin(v)), int(np.argmax(v))
            if abs(v[i_min] - v_start) > abs(v[i_max] - v_start):
                rev_idx, first_dir = i_min, "cathodic"  # went down to min
            else:
                rev_idx, first_dir = i_max, "anodic"    # went up to max

            # Skip degenerate cycles where reversal is at the very edge
            if rev_idx in (0, len(v) - 1):
                continue

            halves[cyc] = {
                "reversal_time": t[rev_idx],
                "first_half_direction": first_dir,
            }
        return halves

    def select_spectra_by_half_cycle(
        self,
        cycle: int | float,
        direction: str | None = None,
        num_samples: int = 10,
        step_filter: str | None = None,
        v_start: float | None = None,
        v_end: float | None = None,
    ) -> tuple[list[str], list[str]]:
        """
        Pick ``num_samples`` evenly-spaced spectra from one cycle (or one
        half-cycle), ordered **chronologically** by absolute time.

        Use this when tracing chemistry along a sweep — unlike
        ``select_spectra_by_potential``, this preserves the time-ordering
        within a sweep so each spectrum represents one step forward in
        the experiment.

        Parameters
        ----------
        cycle
            Cycle number (matches the values stored on each spectrum's
            ``Cycle`` attribute by ``sync_and_label``).
        direction
            ``"cathodic"`` (decreasing voltage) or ``"anodic"`` (increasing
            voltage). If ``None`` (default), returns chronologically-
            ordered spectra from the full cycle (both halves).
        num_samples
            How many spectra to return.
        step_filter
            Source-file substring (e.g. ``"04_CV"``) to scope the search to
            one EC-Lab technique. **Strongly recommended** for sessions
            containing PEIS / GCPL / multi-technique runs, because those
            techniques have their own cycle counters that share column
            names with CV cycles and would otherwise be mixed in.
        v_start, v_end
            Optional voltage window. If both are provided, only spectra
            with ``min(v_start, v_end) <= Potential_V <= max(v_start, v_end)``
            are considered before the evenly-spaced pick. Order doesn't
            matter — it's a window, not a direction. Useful for zooming
            into a specific potential region within a half-cycle.

        Returns
        -------
        (keys, labels)
            Lists of sample keys and their voltage labels, in chronological
            order. Empty lists if no matches were found.

        Convention
        ----------
        Cathodic = decreasing voltage, anodic = increasing voltage. For a
        Na half-cell ``cathodic`` corresponds to sodiation. If your
        convention is reversed for a particular system, you can detect
        which half a cycle starts in by inspecting the dict returned by
        ``_detect_cycle_halves()``.

        Notes
        -----
        Filtering uses the synchronizer's merged DataFrame
        (``self.synced_df``), where every row's ``Absolute_Time`` is in
        the same frame as the EC-Lab data (i.e. ``time_shift_hours`` has
        been applied). This sidesteps any spectra-vs-EC-Lab time-frame
        mismatch when ``time_shift_hours`` is non-zero.
        """
        if direction is not None and direction not in ("cathodic", "anodic"):
            raise ValueError(
                f"direction must be 'cathodic', 'anodic', or None; got {direction!r}"
            )
        if self.synced_df is None:
            raise RuntimeError("Run `sync_and_label()` before selecting spectra.")

        halves = self._detect_cycle_halves(step_filter=step_filter)
        if not halves:
            print("Warning: no cycles with detectable reversals found.")
            return [], []

        match = next((k for k in halves if k == cycle), None)
        if match is None:
            print(
                f"Warning: cycle {cycle} not found. "
                f"Available cycles: {sorted(halves)}"
            )
            return [], []

        rev_time = pd.Timestamp(halves[match]["reversal_time"])
        first_half_dir = halves[match]["first_half_direction"]

        # Find the actual time bounds of this cycle in EC-Lab data — used
        # below to reject spectra that synced_df claims belong to this cycle
        # but whose timestamp is way outside the cycle's real time range
        # (a merge_asof+tolerance anomaly that can happen at session edges).
        ec_df = self.eclab_df
        if step_filter:
            ec_df = ec_df[ec_df["Source_File"].str.contains(step_filter, case=False, na=False)]
        ec_cycle_col = self.find_cycle_column(ec_df)
        ec_cycle_rows = ec_df[ec_df[ec_cycle_col] == cycle]
        cycle_t_start = pd.Timestamp(ec_cycle_rows["Absolute_Time"].min())
        cycle_t_end = pd.Timestamp(ec_cycle_rows["Absolute_Time"].max())

        # Filter synced_df: by source (if requested), same cycle, in time range
        sub = self.synced_df.copy()
        if step_filter:
            sub = sub[sub["Source_File"].str.contains(step_filter, case=False, na=False)]
        cycle_col = self.find_cycle_column(self.synced_df)
        sub = sub[sub[cycle_col] == cycle]
        sub = sub.dropna(subset=["Absolute_Time"])

        # Sanity check: drop spectra whose time falls outside the cycle's
        # actual time range, regardless of what synced_df reported.
        times = pd.to_datetime(sub["Absolute_Time"])
        in_range = (times >= cycle_t_start) & (times <= cycle_t_end)
        n_dropped = int((~in_range).sum())
        sub = sub[in_range]
        if n_dropped > 0:
            print(
                f"Note: dropped {n_dropped} spectra whose timestamp fell "
                f"outside cycle {cycle}'s actual time range "
                f"({cycle_t_start} to {cycle_t_end}) despite being tagged "
                f"with this cycle by merge_asof. Likely a tolerance-window "
                f"anomaly at a session edge."
            )

        if direction is not None:
            want_first_half = (direction == first_half_dir)
            times = pd.to_datetime(sub["Absolute_Time"])
            sub = sub[(times < rev_time) == want_first_half]

        # Optional voltage window — both bounds must be provided together
        if (v_start is None) != (v_end is None):
            raise ValueError(
                "v_start and v_end must both be provided or both omitted."
            )
        if v_start is not None:
            v_min, v_max = sorted((v_start, v_end))
            sub = sub[(sub["Potential_V"] >= v_min) & (sub["Potential_V"] <= v_max)]

        if sub.empty:
            scope = f"cycle {cycle}" + (f" {direction} half" if direction else "")
            if v_start is not None:
                scope += f" within [{min(v_start, v_end)}, {max(v_start, v_end)}] V"
            print(f"Warning: no spectra found for {scope}.")
            return [], []

        # Sort chronologically, then pick evenly spaced indices
        sub = sub.sort_values("Absolute_Time").reset_index(drop=True)
        n = len(sub)
        if num_samples >= n:
            print(f"Note: only {n} spectra available; returning all.")
            idx = list(range(n))
        elif num_samples == 1:
            idx = [0]
        else:
            step = (n - 1) / (num_samples - 1)
            idx = [round(i * step) for i in range(num_samples)]

        keys = sub["Sample_Name"].iloc[idx].tolist()
        labels = [self.parser.spectra[k]["Label"] for k in keys]
        v_first = self.parser.spectra[keys[0]]["Potential_V"]
        v_last = self.parser.spectra[keys[-1]]["Potential_V"]
        scope = f"cycle {cycle}" + (f" ({direction})" if direction else "")
        print(
            f"Selected {len(keys)} spectra from {scope}, "
            f"chronologically from {v_first:.3f} V to {v_last:.3f} V."
        )
        return keys, labels

    def get_step_baseline(self, step_filter: str) -> str | None:
        """
        First (chronologically) OPUS spectrum matched to the given EC-Lab step.
        Useful as a baseline for differential calculations.
        """
        if self.synced_df is None:
            raise RuntimeError("Call `sync_and_label()` first.")

        mask = self.synced_df["Source_File"].str.contains(
            step_filter, case=False, na=False
        )
        step_df = self.synced_df[mask]
        if step_df.empty:
            print(f"Warning: no spectra synced to step '{step_filter}'.")
            return None

        baseline = step_df.iloc[0]["Sample_Name"]
        voltage = self.parser.spectra[baseline].get("Potential_V", 0.0)
        print(f"Baseline for '{step_filter}': {baseline} ({voltage:.3f} V)")
        return baseline