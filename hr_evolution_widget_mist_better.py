from __future__ import annotations

"""
Interactive H-R diagram backed by the raw MIST *isochrone* dataframe from
`isochrones.mist.MISTIsochroneGrid`.

Why this file exists
--------------------
Earlier versions used either
1) a hand-built toy stellar-evolution model, or
2) the raw MIST evolution-track grid.

The toy model was only qualitative. The track-grid version was more physical,
but it is not the natural object for a *coeval cluster* snapshot. A coeval
cluster at one age is described by an *isochrone*: the locus of stars of
various initial masses at one fixed age and metallicity.

This file therefore keeps the *working widget / plotting layer* but replaces the
physics backend with the raw MIST isochrone grid. Importantly, it still avoids
fragile high-level calls such as `tracks.generate(...)`; instead it accesses the
underlying dataframe directly and performs interpolation with NumPy/Pandas.

Raw MIST isochrone dataframe structure
--------------------------------------
In the user's environment, `MISTIsochroneGrid().df` has

    index names: ['log10_isochrone_age_yr', 'feh', 'EEP']

and columns including

    ['eep', 'age', 'feh', 'mass', 'initial_mass', 'radius', 'density',
     'logTeff', 'Teff', 'logg', 'logL', 'Mbol', 'delta_nu', 'nu_max',
     'phase', 'dm_deep']

The raw dataframe already stores the physically relevant quantities we need:
- Teff or logTeff          -> horizontal axis / marker color
- logL                     -> luminosity axis via L/Lsun = 10**logL
- radius                   -> marker size
- phase                    -> diagnostics and optional pre-main-sequence masking
- initial_mass             -> the coordinate along an isochrone used for our
                              interpolation onto a synthetic stellar population

Physics choices in this implementation
--------------------------------------
1. Sample a synthetic coeval population once:
       - initial masses from a simple IMF-like power law
       - [Fe/H] values from a user-controlled Gaussian scatter

2. For each redraw at a cluster age t:
       - convert age [Myr] -> log10(age/yr)
       - for each star metallicity, use the nearest available MIST metallicity
         grid value (feh_idx)
       - for the chosen age, interpolate *between neighboring isochrone ages*
         in log10(age/yr)
       - within each isochrone slice, interpolate in initial mass to obtain
         Teff, logL, radius for each star

3. Omit pre-main-sequence stars:
       - within each isochrone slice we keep only rows with phase >= 0 when
         possible. This respects the user's request to "start with all stars on
         the main sequence".
       - stars outside the visible mass range of that main-sequence-or-later
         slice are hidden by default, or optionally shown faded at the boundary.

4. Preserve the plotting style from earlier versions:
       - x-axis: Teff decreasing to the right
       - y-axis: L/Lsun on a log scale
       - color: Teff
       - size: radius with compressed dynamic range

This file is intentionally heavily commented so the physics/implementation logic
is easy to inspect and modify in a notebook or script.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import ipywidgets as widgets
from IPython.display import display

# -----------------------------------------------------------------------------
# Global caches
# -----------------------------------------------------------------------------

_ISO_DF_CACHE: pd.DataFrame | None = None
_AGE_GRID_CACHE: np.ndarray | None = None
_FEH_GRID_CACHE: np.ndarray | None = None
_ISO_SLICE_CACHE: Dict[tuple[float, float], "PreparedIsochroneSlice | None"] = {}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class IsoKey:
    """Identity of one raw MIST isochrone slice.

    Parameters
    ----------
    log10_age_idx
        Grid age in log10(years), taken directly from the dataframe index.
    feh_idx
        Grid metallicity [Fe/H], taken directly from the dataframe index.
    """

    log10_age_idx: float
    feh_idx: float


@dataclass
class PreparedIsochroneSlice:
    """One isochrone slice prepared for interpolation in initial mass.

    Attributes
    ----------
    key
        Age/metallicity identity of the isochrone slice.
    initial_mass
        Monotone initial-mass coordinate used for interpolation.
    teff, logl, radius
        Physical quantities along the isochrone.
    phase
        MIST phase label retained for diagnostics / optional styling.
    mass_min, mass_max
        Visible mass range after filtering (for hiding or faded-boundary logic).
    """

    key: IsoKey
    initial_mass: np.ndarray
    teff: np.ndarray
    logl: np.ndarray
    radius: np.ndarray
    phase: np.ndarray
    mass_min: float
    mass_max: float


@dataclass
class ClusterPopulation:
    """Synthetic coeval stellar population.

    We sample masses and [Fe/H] once and reuse them for all ages.
    """

    sampled_mass: np.ndarray
    sampled_feh: np.ndarray
    assigned_feh: np.ndarray
    mean_feh: float
    feh_scatter: float


# -----------------------------------------------------------------------------
# Raw MIST isochrone grid loading
# -----------------------------------------------------------------------------


def load_mist_isochrone_dataframe() -> pd.DataFrame:
    """Load the raw MIST isochrone dataframe with explicit index columns.

    The raw dataframe has index names
        ['log10_isochrone_age_yr', 'feh', 'EEP']
    and also has ordinary columns named 'age', 'feh', 'eep'.  To avoid column
    name collisions when resetting the index, we first rename the *index axes*
    to distinct names.

    Returns
    -------
    pandas.DataFrame
        Dataframe with explicit columns:
        - log10_age_idx
        - feh_idx
        - eep_idx
        plus the original physical columns.
    """
    global _ISO_DF_CACHE, _AGE_GRID_CACHE, _FEH_GRID_CACHE

    if _ISO_DF_CACHE is not None:
        return _ISO_DF_CACHE

    from isochrones.mist import MISTIsochroneGrid

    igrid = MISTIsochroneGrid()
    df = igrid.df.rename_axis(
        index=["log10_age_idx", "feh_idx", "eep_idx"]
    ).reset_index()

    # Keep only finite, physically sensible values used by the H-R diagram.
    good = (
        np.isfinite(df["log10_age_idx"]) &
        np.isfinite(df["feh_idx"]) &
        np.isfinite(df["eep_idx"]) &
        np.isfinite(df["initial_mass"]) &
        np.isfinite(df["Teff"]) &
        np.isfinite(df["logL"]) &
        np.isfinite(df["radius"])
    )
    df = df.loc[good].copy()

    # Broad sanity cuts.  The raw dataframe may contain infinities/nonphysical
    # values in the ordinary 'feh' column, but our actual key is feh_idx.
    df = df[
        (df["feh_idx"] >= -4.5) & (df["feh_idx"] <= 0.5) &
        (df["initial_mass"] > 0.0) &
        (df["Teff"] > 0.0) &
        (df["radius"] > 0.0)
    ].copy()

    _ISO_DF_CACHE = df
    _AGE_GRID_CACHE = np.sort(df["log10_age_idx"].unique())
    _FEH_GRID_CACHE = np.sort(df["feh_idx"].unique())
    return df



def available_age_values() -> np.ndarray:
    load_mist_isochrone_dataframe()
    assert _AGE_GRID_CACHE is not None
    return _AGE_GRID_CACHE



def available_feh_values() -> np.ndarray:
    load_mist_isochrone_dataframe()
    assert _FEH_GRID_CACHE is not None
    return _FEH_GRID_CACHE



def nearest_grid_values(values: np.ndarray, grid_values: np.ndarray) -> np.ndarray:
    """Snap each input value to the nearest available grid value."""
    vals = np.asarray(values, dtype=float)
    grid = np.asarray(grid_values, dtype=float)
    idx = np.abs(vals[:, None] - grid[None, :]).argmin(axis=1)
    return grid[idx]


# -----------------------------------------------------------------------------
# Isochrone preparation and interpolation
# -----------------------------------------------------------------------------


def prepare_isochrone_slice(key: IsoKey) -> PreparedIsochroneSlice | None:
    """Prepare one raw MIST isochrone slice for interpolation in initial mass.

    Important choices:
    - group by the *index keys* (log10_age_idx, feh_idx), not by the ordinary
      columns 'age' or 'feh'
    - sort by initial_mass so we can interpolate stellar properties for a
      synthetic population sampled in initial mass
    - omit pre-main-sequence rows using phase >= 0 whenever that leaves at least
      two rows, matching the user's request to start on the main sequence
    """
    global _ISO_SLICE_CACHE

    cache_key = (float(key.log10_age_idx), float(key.feh_idx))
    if cache_key in _ISO_SLICE_CACHE:
        return _ISO_SLICE_CACHE[cache_key]

    df = load_mist_isochrone_dataframe()
    grp = df[(df["log10_age_idx"] == key.log10_age_idx) & (df["feh_idx"] == key.feh_idx)].copy()
    if len(grp) == 0:
        _ISO_SLICE_CACHE[cache_key] = None
        return None

    # Omit PMS if possible.
    ms = grp[grp["phase"] >= 0].copy()
    if len(ms) >= 2:
        grp = ms

    # Sort by initial mass and remove duplicate masses to keep np.interp stable.
    grp = grp.sort_values("initial_mass").drop_duplicates(subset=["initial_mass"], keep="first").copy()

    m = grp["initial_mass"].to_numpy(dtype=float)
    teff = grp["Teff"].to_numpy(dtype=float)
    logl = grp["logL"].to_numpy(dtype=float)
    radius = grp["radius"].to_numpy(dtype=float)
    phase = grp["phase"].to_numpy(dtype=float)

    keep = (
        np.isfinite(m) & np.isfinite(teff) & np.isfinite(logl) & np.isfinite(radius) &
        (m > 0) & (teff > 0) & (radius > 0)
    )
    m, teff, logl, radius, phase = m[keep], teff[keep], logl[keep], radius[keep], phase[keep]

    if len(m) < 2:
        _ISO_SLICE_CACHE[cache_key] = None
        return None

    out = PreparedIsochroneSlice(
        key=key,
        initial_mass=m,
        teff=teff,
        logl=logl,
        radius=radius,
        phase=phase,
        mass_min=float(m[0]),
        mass_max=float(m[-1]),
    )
    _ISO_SLICE_CACHE[cache_key] = out
    return out



def _interp_on_mass(slice_obj: PreparedIsochroneSlice, masses: np.ndarray, *, show_boundary: bool) -> pd.DataFrame:
    """Interpolate one prepared isochrone slice onto an array of initial masses.

    If a sampled star lies outside the visible mass range of the slice:
    - hide it by returning NaN, or
    - if `show_boundary=True`, pin it to the nearest boundary and flag it faded
      so the plotting layer can reduce its alpha.
    """
    m = np.asarray(masses, dtype=float)
    x = slice_obj.initial_mass

    low = m < slice_obj.mass_min
    high = m > slice_obj.mass_max
    out_of_range = low | high

    if show_boundary:
        m_eval = np.clip(m, slice_obj.mass_min, slice_obj.mass_max)
    else:
        m_eval = m.copy()

    teff = np.interp(m_eval, x, slice_obj.teff)
    logl = np.interp(m_eval, x, slice_obj.logl)
    radius = np.interp(m_eval, x, slice_obj.radius)
    phase = np.interp(m_eval, x, slice_obj.phase)

    if not show_boundary:
        teff[out_of_range] = np.nan
        logl[out_of_range] = np.nan
        radius[out_of_range] = np.nan
        phase[out_of_range] = np.nan

    return pd.DataFrame(
        {
            "initial_mass": m,
            "Teff": teff,
            "logL": logl,
            "radius": radius,
            "phase": phase,
            "at_boundary": out_of_range,
        }
    )



def evaluate_population(pop: ClusterPopulation, age_myr: float, *, show_terminal_faded_stars: bool) -> pd.DataFrame:
    """Evaluate the synthetic cluster on the raw MIST isochrone grid.

    Age handling
    ------------
    The raw MIST isochrone grid is indexed by log10(age / yr).  For a widget age
    in Myr, we convert via

        log10(age/yr) = log10(age_Myr) + 6.

    To make slider-driven animation smoother than nearest-neighbor stepping, we
    interpolate between the two neighboring MIST isochrone ages in log10(age/yr).

    Metallicity handling
    --------------------
    Each star is assigned a fixed metallicity at population construction time.
    For now we snap to the nearest available MIST metallicity grid value and do
    not interpolate between metallicities. That keeps the implementation robust.
    """
    age_myr = float(age_myr)
    if age_myr <= 0:
        raise ValueError("age_myr must be > 0 for log10(age/yr) conversion")

    log_age = np.log10(age_myr) + 6.0
    age_grid = available_age_values()

    if log_age <= age_grid[0]:
        age_lo = age_hi = age_grid[0]
        w_hi = 0.0
    elif log_age >= age_grid[-1]:
        age_lo = age_hi = age_grid[-1]
        w_hi = 0.0
    else:
        hi_idx = np.searchsorted(age_grid, log_age)
        age_lo = age_grid[hi_idx - 1]
        age_hi = age_grid[hi_idx]
        w_hi = (log_age - age_lo) / (age_hi - age_lo)
    w_lo = 1.0 - w_hi

    parts: List[pd.DataFrame] = []

    # Work group-by-group in metallicity; all stars in one group share the same
    # raw MIST isochrone slices and differ only by initial mass.
    for feh_val in np.unique(pop.assigned_feh):
        idx = np.where(pop.assigned_feh == feh_val)[0]
        masses = pop.sampled_mass[idx]

        key_lo = IsoKey(log10_age_idx=float(age_lo), feh_idx=float(feh_val))
        key_hi = IsoKey(log10_age_idx=float(age_hi), feh_idx=float(feh_val))
        iso_lo = prepare_isochrone_slice(key_lo)
        iso_hi = prepare_isochrone_slice(key_hi)

        if iso_lo is None and iso_hi is None:
            part = pd.DataFrame(
                {
                    "initial_mass": masses,
                    "Teff": np.full(len(masses), np.nan),
                    "logL": np.full(len(masses), np.nan),
                    "radius": np.full(len(masses), np.nan),
                    "phase": np.full(len(masses), np.nan),
                    "at_boundary": np.full(len(masses), False),
                }
            )
        elif iso_hi is None or w_hi == 0.0:
            part = _interp_on_mass(iso_lo, masses, show_boundary=show_terminal_faded_stars)
        elif iso_lo is None:
            part = _interp_on_mass(iso_hi, masses, show_boundary=show_terminal_faded_stars)
        else:
            lo = _interp_on_mass(iso_lo, masses, show_boundary=show_terminal_faded_stars)
            hi = _interp_on_mass(iso_hi, masses, show_boundary=show_terminal_faded_stars)
            part = lo.copy()
            part["Teff"] = w_lo * lo["Teff"].to_numpy() + w_hi * hi["Teff"].to_numpy()
            part["logL"] = w_lo * lo["logL"].to_numpy() + w_hi * hi["logL"].to_numpy()
            part["radius"] = w_lo * lo["radius"].to_numpy() + w_hi * hi["radius"].to_numpy()
            part["phase"] = w_lo * lo["phase"].to_numpy() + w_hi * hi["phase"].to_numpy()
            part["at_boundary"] = lo["at_boundary"].to_numpy() | hi["at_boundary"].to_numpy()

        part["sampled_feh"] = pop.sampled_feh[idx]
        part["assigned_feh"] = feh_val
        parts.append(part)

    df = pd.concat(parts, axis=0, ignore_index=True)
    df["L"] = 10.0 ** df["logL"]
    return df


# -----------------------------------------------------------------------------
# Population sampling
# -----------------------------------------------------------------------------


def sample_power_law_masses(
    n: int,
    m_min: float,
    m_max: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample initial masses from dN/dM ∝ M^{-alpha}."""
    if np.isclose(alpha, 1.0):
        u = rng.random(n)
        return m_min * (m_max / m_min) ** u

    expo = 1.0 - alpha
    u = rng.random(n)
    return (u * (m_max**expo - m_min**expo) + m_min**expo) ** (1.0 / expo)



def build_population(
    n_stars: int = 200,
    m_min: float = 0.8,
    m_max: float = 8.0,
    imf_alpha: float = 2.35,
    mean_feh: float = 0.0,
    feh_scatter: float = 0.10,
    random_seed: int | None = 0,
) -> ClusterPopulation:
    """Construct a coeval synthetic cluster population.

    Parameters
    ----------
    n_stars
        Number of stars in the synthetic cluster.
    m_min, m_max
        Initial mass range in solar units.
    imf_alpha
        Power-law slope in dN/dM ∝ M^{-alpha}. A Salpeter-like value is 2.35.
    mean_feh, feh_scatter
        Mean and Gaussian scatter of [Fe/H] assigned to stars.  These are then
        snapped to the nearest available MIST isochrone metallicity grid values.
    random_seed
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(random_seed)
    sampled_mass = sample_power_law_masses(n_stars, m_min, m_max, imf_alpha, rng)
    sampled_feh = rng.normal(loc=float(mean_feh), scale=float(feh_scatter), size=n_stars)

    feh_grid = available_feh_values()
    assigned_feh = nearest_grid_values(sampled_feh, feh_grid)

    return ClusterPopulation(
        sampled_mass=sampled_mass,
        sampled_feh=sampled_feh,
        assigned_feh=assigned_feh,
        mean_feh=float(mean_feh),
        feh_scatter=float(feh_scatter),
    )


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def compute_marker_sizes(radius: np.ndarray, marker_scale: float = 1.0) -> np.ndarray:
    """Map stellar radius to scatter-marker size.

    We deliberately compress the dynamic range because giant radii can become
    enormous. A simple sub-linear scaling works well pedagogically:

        size ∝ (radius)^{0.8}

    with an overall user-controlled multiplier.
    """
    r = np.asarray(radius, dtype=float)
    r = np.clip(r, 1e-3, None)
    return marker_scale * (18.0 * r**0.8)



def _draw_single_frame(
    pop: ClusterPopulation,
    age_myr: float,
    age_max_myr: float,
    playback_duration_s: float,
    show_guides: bool,
    show_terminal_faded_stars: bool,
    black_marker_edges: bool,
    marker_scale: float,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw one H-R snapshot for a given cluster age."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)

    plot_df = evaluate_population(pop, age_myr, show_terminal_faded_stars=show_terminal_faded_stars)

    # Stars with NaN are outside the visible slice (e.g. below the PMS cutoff
    # when PMS is omitted, or beyond the available mass range if hiding is used).
    plot_df = plot_df[np.isfinite(plot_df["Teff"]) & np.isfinite(plot_df["L"]) & np.isfinite(plot_df["radius"])].copy()

    colors = plot_df["Teff"].to_numpy()
    sizes = compute_marker_sizes(plot_df["radius"].to_numpy(), marker_scale=marker_scale)
    edgecolor = "k" if black_marker_edges else "none"

    if len(plot_df) == 0:
        ax.text(
            0.5, 0.5,
            "No stars visible for this age / selection",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="0.35",
        )
        sc = None
    else:
        alpha = np.where(plot_df["at_boundary"].to_numpy(), 0.25, 0.85)
        sc = ax.scatter(
            plot_df["Teff"],
            plot_df["L"],
            c=colors,
            s=sizes,
            cmap="plasma_r",
            norm=Normalize(vmin=3000, vmax=25000),
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=0.3,
        )

    ax.set_yscale("log")
    ax.set_xlim(10000, 1500)
    ax.set_ylim(1e-4, 3e5)
    ax.set_xlabel(r"Effective temperature $T_{\rm eff}$ [K]", fontsize=13)
    ax.set_ylabel(r"Luminosity $L/L_\odot$", fontsize=13)
    ax.grid(True, which="both", alpha=0.18)

    speed_myr_per_s = age_max_myr / playback_duration_s
    speedup = speed_myr_per_s * 1e6 * 365.25 * 24 * 3600

    shown = len(plot_df)
    ax.set_title(
        f"MIST raw isochrone H-R diagram | cluster age {age_myr:,.0f} Myr | "
        f"speed {speed_myr_per_s:.1f} Myr/s | x{speedup:.2e} real time",
        fontsize=14,
        pad=14,
    )

    box_text = (
        f"n={shown} shown / {len(pop.sampled_mass)} total | "
        f"mean [Fe/H]={pop.mean_feh:+.2f} | sigma[Fe/H]={pop.feh_scatter:.2f}"
    )
    ax.text(
        0.015,
        0.985,
        box_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.75", alpha=0.9),
    )

    if show_guides:
 #       ax.text(22000, 1.6e4, "Hot\nmain sequence", color="0.35", fontsize=18, alpha=0.85)
        ax.text(5200, 220, "Giants", color="0.35", fontsize=18, alpha=0.85)
 #       ax.text(15000, 2.5e-3, "White dwarf\nregion", color="0.35", fontsize=18, alpha=0.85)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r"$T_{\rm eff}$ [K]", fontsize=12)

    return fig, ax



def draw_snapshot(
    pop: ClusterPopulation,
    age_myr: float,
    age_max_myr: float = 3000.0,
    playback_duration_s: float = 30.0,
    show_guides: bool = True,
    show_terminal_faded_stars: bool = False,
    black_marker_edges: bool = True,
    marker_scale: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Convenience wrapper to draw one static H-R snapshot."""
    return _draw_single_frame(
        pop=pop,
        age_myr=age_myr,
        age_max_myr=age_max_myr,
        playback_duration_s=playback_duration_s,
        show_guides=show_guides,
        show_terminal_faded_stars=show_terminal_faded_stars,
        black_marker_edges=black_marker_edges,
        marker_scale=marker_scale,
    )



def snapshot_sequence(
    pop: ClusterPopulation,
    ages_myr: Sequence[float],
    ncols: int = 3,
    age_max_myr: float = 3000.0,
    playback_duration_s: float = 30.0,
    show_guides: bool = True,
    show_terminal_faded_stars: bool = False,
    black_marker_edges: bool = True,
    marker_scale: float = 0.8,
) -> tuple[plt.Figure, np.ndarray]:
    """Draw several static H-R snapshots in a grid of subplots."""
    ages_myr = list(ages_myr)
    n = len(ages_myr)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for ax, age in zip(axes.flat, ages_myr):
        _draw_single_frame(
            pop=pop,
            age_myr=float(age),
            age_max_myr=age_max_myr,
            playback_duration_s=playback_duration_s,
            show_guides=show_guides,
            show_terminal_faded_stars=show_terminal_faded_stars,
            black_marker_edges=black_marker_edges,
            marker_scale=marker_scale,
            fig=fig,
            ax=ax,
        )

    for ax in axes.flat[n:]:
        ax.axis("off")

    return fig, axes


# -----------------------------------------------------------------------------
# Widget-driven interactive figure
# -----------------------------------------------------------------------------


def build_hr_widget_demo(
    n_stars: int = 500,
    m_min: float = 0.8,
    m_max: float = 8.0,
    mean_feh: float = 0.0,
    feh_scatter: float = 0.10,
    age_max_myr: float = 300.0,
    playback_duration_s: float = 30.0,
    random_seed: int | None = 0,
):
    """Build the notebook widget UI for the raw-MIST-isochrone H-R animation.

    Notes
    -----
    This function redraws a fresh matplotlib figure in an ipywidgets Output area
    whenever a control changes. This strategy has proven robust in the user's
    notebook setup when `%matplotlib widget` and `ipympl` are enabled.
    """
    pop = build_population(
        n_stars=n_stars,
        m_min=m_min,
        m_max=m_max,
        mean_feh=mean_feh,
        feh_scatter=feh_scatter,
        random_seed=random_seed,
    )

    speed_myr_per_s = age_max_myr / playback_duration_s

    title_html = widgets.HTML(
        value=(
            f"<b>Population:</b> {n_stars} coeval stars, masses {m_min:.1f}–{m_max:.1f} M☉"
            f"<br><b>Physics:</b> real MIST <i>isochrones</i> from raw grid "
            f"(index keys log10_age_idx, feh_idx, eep_idx)"
            f"<br><b>Playback speed:</b> {speed_myr_per_s:.1f} Myr/s"
        )
    )

    age_slider = widgets.FloatSlider(
        value=min(140.0, age_max_myr),
        min=1.0,   # age must be >0 for log10(age/yr)
        max=age_max_myr,
        step=max(1.0, age_max_myr / 300.0),
        description="Age [Myr]",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="620px"),
    )
    guides_box = widgets.Checkbox(value=True, description="Show guide labels")
    terminal_box = widgets.Checkbox(value=False, description="Show terminal faded stars")
    edge_box = widgets.Checkbox(value=True, description="Black marker edges")
    marker_slider = widgets.FloatSlider(
        value=1.0,
        min=0.3,
        max=2.5,
        step=0.1,
        description="Marker scale",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="320px"),
    )

    out = widgets.Output()

    def redraw(*_args):
        with out:
            out.clear_output(wait=True)
            plt.close("all")
            fig, ax = _draw_single_frame(
                pop=pop,
                age_myr=age_slider.value,
                age_max_myr=age_max_myr,
                playback_duration_s=playback_duration_s,
                show_guides=guides_box.value,
                show_terminal_faded_stars=terminal_box.value,
                black_marker_edges=edge_box.value,
                marker_scale=marker_slider.value,
            )
            display(fig)
            plt.close(fig)

    for w in [age_slider, guides_box, terminal_box, edge_box, marker_slider]:
        w.observe(redraw, names="value")

    controls = widgets.VBox(
        [
            title_html,
            age_slider,
            widgets.HBox([guides_box, terminal_box, edge_box, marker_slider]),
        ]
    )

    redraw()
    return widgets.VBox([controls, out])
