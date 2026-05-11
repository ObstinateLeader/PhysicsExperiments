"""
Zeeman Splitting Analysis
=========================
Loads two spectra (field OFF and field ON), calibrates both using the
sodium D-lines (or any reference lines defined in the calibration script),
overlaps them, and extracts the Zeeman splitting / line broadening for
each spectral line.

This file imports from your existing calibration script — do NOT modify
spectrometer_calibration.py.

Usage
-----
Edit the file paths and manual_peaks in Section 1, then run:

    python zeeman_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,            # Controls default text sizes
    'axes.titlesize': 22,       # Title of the individual subplot
    'axes.labelsize': 18,       # x and y labels
    'xtick.labelsize': 14,      # x-axis tick labels
    'ytick.labelsize': 14,      # y-axis tick labels
    'legend.fontsize': 14,      # Legend font size
    'figure.titlesize': 24,     # Title of the entire figure
    'lines.linewidth': 2.5,     # Thicker lines for visibility
    'axes.titlepad': 20         # Space between title and plot
})

# ---------------------------------------------------------------------------
# Import from your existing calibration module (unchanged file)
# ---------------------------------------------------------------------------
import os
import sys
# Ensure the folder containing this script is on sys.path so the local
# calibration module can be imported reliably.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
# from callibration_newest import (
#     REFERENCE_LINES,
#     gaussian,
#     fit_peak,
#     run_calibration,
#     pixel_to_wavelength,
# )
REFERENCE_LINES = {
    "Ne 585.2": 585.2488,
    "Ne 603.0": 603.0,      # ~602.7 nm
    "Ne 615.6": 615.6283,
    "Ne 618.7": 618.6716,
    "Ne 626.8": 626.8232,
    "Ne 630.0": 629.7388,
    "Ne 640.2": 640.2248,
    "Ne 650.6": 650.6528,
}

def pixel_to_wavelength(pixel, coeffs):
    """Convert a pixel position to wavelength using the calibration polynomial."""
    return np.polyval(coeffs, pixel)

# ===========================================================================
# SECTION 1 — USER SETTINGS  (edit these)
# ===========================================================================

# File paths for the two spectra
# FILE_OFF = "NatriumBalk1.txt"          # B = 0  (no magnetic field)
# FILE_ON  = "NatriumBalk1.txt"          # B ≠ 0  (with magnetic field)
                                        # ← replace with your actual ON file
FILE_ON = 'measurement sessie 6 magnet  close(2).txt'
FILE_OFF = 'measurement sessie 6 magnet far(2).txt'

DELIMITER  = ","
SKIP_ROWS  = 1

# Polynomial order for wavelength calibration
POLY_ORDER = 2

# Manual peak guesses (pixel positions) — used for BOTH spectra.
# Adjust if the field-on spectrum has shifted peaks.
# MANUAL_PEAKS_OFF = {
#     "Na 589.0": 2158.0,
#     "Na 589.6": 2190.0,
# }

# MANUAL_PEAKS_ON = {
#     "Na 589.0": 2158.0,   # ← update for your ON spectrum if needed
#     "Na 589.6": 2190.0,
# }
MANUAL_PEAKS_OFF = None
MANUAL_PEAKS_ON = None

# Lines to analyse for Zeeman splitting (must be keys in REFERENCE_LINES)
LINES_TO_ANALYSE = list(REFERENCE_LINES.keys())

# Fit window (±pixels) around each peak for the Gaussian fit
PEAK_WINDOW = 15

# Optional: wavelength range (nm) to zoom the overlap plot.
# Set to None to use the full calibrated range.
# ZOOM_WL = (588.5, 590.2)   # nm, or None
ZOOM_WL = None


# ===========================================================================
# SECTION 2 — MULTI-PEAK GAUSSIAN MODELS
# ===========================================================================

def multi_gaussian(x, *params):
    """
    Sum of N Gaussians.
    params = [amp1, cen1, sig1,  amp2, cen2, sig2, …,  baseline]
    """
    n_peaks = (len(params) - 1) // 3
    baseline = params[-1]
    result = np.full_like(x, baseline, dtype=float)
    for i in range(n_peaks):
        amp, cen, sig = params[3*i], params[3*i+1], params[3*i+2]
        result += amp * np.exp(-0.5 * ((x - cen) / sig) ** 2)
    return result


def fit_zeeman_components(wavelengths_nm, intensities, center_nm,
                          n_components=2, window_nm=0.5):
    """
    Fit N Gaussian components around a central wavelength to resolve
    Zeeman sub-components (or measure broadening as a single wide peak).

    Parameters
    ----------
    wavelengths_nm : calibrated wavelength axis (nm)
    intensities    : spectral intensities
    center_nm      : approximate line centre (nm)
    n_components   : number of Gaussian sub-peaks to fit (1 or 2)
    window_nm      : ±nm around centre_nm to include in the fit

    Returns
    -------
    popt     : fitted parameters [amp, cen, sig, … , baseline]
    pcov     : covariance matrix
    x_fit    : wavelength array used for the fit
    y_fit    : evaluated fit curve
    success  : bool
    """
    mask = np.abs(wavelengths_nm - center_nm) <= window_nm
    x = wavelengths_nm[mask]
    y = intensities[mask]

    if len(x) < 5 * n_components:
        return None, None, x, y, False

    amp0   = (y.max() - y.min())
    base0  = y.min()
    sig0   = window_nm / 4

    if n_components == 1:
        p0     = [amp0, center_nm, sig0, base0]
        bounds_lo = [0,  center_nm - window_nm, 1e-4,  0]
        bounds_hi = [np.inf, center_nm + window_nm, window_nm, np.inf]
    else:
        # Two components: start them symmetrically offset by sig0
        offset = sig0 * 0.5
        p0     = [amp0*0.7, center_nm - offset, sig0,
                  amp0*0.7, center_nm + offset, sig0,
                  base0]
        bounds_lo = [0, center_nm - window_nm, 1e-4,
                     0, center_nm - window_nm, 1e-4,  0]
        bounds_hi = [np.inf, center_nm + window_nm, window_nm,
                     np.inf, center_nm + window_nm, window_nm, np.inf]

    try:
        popt, pcov = curve_fit(
            multi_gaussian, x, y, p0=p0,
            bounds=(bounds_lo, bounds_hi),
            maxfev=10000,
        )
        x_dense = np.linspace(x.min(), x.max(), 500)
        y_fit   = multi_gaussian(x_dense, *popt)
        return popt, pcov, x_dense, y_fit, True
    except RuntimeError:
        return None, None, x, y, False


# ===========================================================================
# SECTION 3 — LOADING & CALIBRATING
# ===========================================================================

def load_and_calibrate(filepath, manual_peaks, poly_order=POLY_ORDER, prominence_factor=0.2, fname="calibration_data.npz"):
    # """Load a spectrum file and return a calibrated wavelength axis."""
    # pixels, spectrum = np.loadtxt(
    #     filepath, delimiter=DELIMITER, skiprows=SKIP_ROWS, unpack=True
    # )
    # coeffs, residuals, peak_info = run_calibration(
    #     pixels, spectrum, REFERENCE_LINES,
    #     poly_order=poly_order,
    #     peak_window=PEAK_WINDOW,
    #     manual_peaks=manual_peaks,
    #     prominence_factor=prominence_factor
    # )
    # wavelengths = pixel_to_wavelength(pixels, coeffs)
    container = np.load(fname, allow_pickle=True)
    pixels = container['pixels']
    spectrum = container['spectrum']
    # wavelengths = container['wavelengths']
    coeffs = container['coeffs']
    residuals = container['residuals']
    peak_info = container['peak_info']
    wavelengths = pixel_to_wavelength(pixels, coeffs)
    return pixels, spectrum, wavelengths, coeffs, residuals, peak_info


# ===========================================================================
# SECTION 4 — ZEEMAN ANALYSIS
# ===========================================================================

def analyse_zeeman(wl_off, sp_off, wl_on, sp_on,
                   lines=LINES_TO_ANALYSE, window_nm=0.4,
                   peak_info_off=None, coeffs_off=None):
    """
    For each spectral line, fit the field-off (single peak) and
    field-on (potentially split / broadened) profiles and compute
    splitting metrics.

    peak_info_off : the peak_info dict loaded from the OFF calibration .npz
                    Used to seed the window centre from the *measured* centroid
                    pixel rather than the theoretical NIST wavelength, so the
                    mask always lands on the actual spectral line.
    coeffs_off    : calibration polynomial coefficients for the OFF spectrum,
                    used to convert the centroid pixel to wavelength.

    Returns
    -------
    results : dict  {line_label: {...}}
    """
    # Unwrap peak_info if it was stored as a 0-d object array (numpy .npz quirk)
    if peak_info_off is not None and isinstance(peak_info_off, np.ndarray):
        peak_info_off = peak_info_off.item()

    results = {}

    for label in lines:
        # Use the measured centroid (converted to nm) when available;
        # fall back to the NIST reference only if the calibration data is missing.
        nist_nm = REFERENCE_LINES[label]
        centre_nm = nist_nm   # default
        if (peak_info_off is not None and coeffs_off is not None
                and label in peak_info_off):
            info = peak_info_off[label]
            px = info.get("centroid_pixel") if isinstance(info, dict) else None
            if px is not None:
                centre_nm = float(pixel_to_wavelength(px, coeffs_off))
                print(f"  {label}: using measured centre {centre_nm:.4f} nm "
                      f"(NIST: {nist_nm:.4f} nm, offset: "
                      f"{(centre_nm - nist_nm)*1000:+.1f} pm)")

        # ── field OFF: fit as single Gaussian ──────────────────────────────
        popt_off, pcov_off, x_off, y_off, ok_off = fit_zeeman_components(
            wl_off, sp_off, centre_nm, n_components=1, window_nm=window_nm
        )

        # ── field ON: try 2-component first, fall back to 1 ────────────────
        popt_on2, pcov_on2, x_on2, y_on2, ok_on2 = fit_zeeman_components(
            wl_on, sp_on, centre_nm, n_components=2, window_nm=window_nm
        )
        popt_on1, pcov_on1, x_on1, y_on1, ok_on1 = fit_zeeman_components(
            wl_on, sp_on, centre_nm, n_components=1, window_nm=window_nm
        )

        # Choose better ON fit: prefer 2-component if both work
        use_2comp = ok_on2

        entry = {
            "centre_nm"     : centre_nm,
            # OFF
            "off_ok"        : ok_off,
            "off_popt"      : popt_off,
            "off_x"         : x_off,
            "off_y"         : y_off,
            # ON (1-comp)
            "on1_ok"        : ok_on1,
            "on1_popt"      : popt_on1,
            "on1_x"         : x_on1,
            "on1_y"         : y_on1,
            # ON (2-comp)
            "on2_ok"        : ok_on2,
            "on2_popt"      : popt_on2,
            "on2_x"         : x_on2,
            "on2_y"         : y_on2,
            "use_2comp"     : use_2comp,
        }

        # ── Derived quantities ──────────────────────────────────────────────
        if ok_off:
            sigma_off_nm = popt_off[2]   # Gaussian σ in nm
            fwhm_off_nm  = 2.355 * sigma_off_nm
            entry["sigma_off_nm"] = sigma_off_nm
            entry["fwhm_off_nm"]  = fwhm_off_nm
        else:
            entry["sigma_off_nm"] = entry["fwhm_off_nm"] = None

        if use_2comp and ok_on2:
            cen1, cen2 = popt_on2[1], popt_on2[4]
            splitting_nm  = abs(cen2 - cen1)
            splitting_pm  = splitting_nm * 1000
            mid_nm        = (cen1 + cen2) / 2
            sigma_on_nm   = (popt_on2[2] + popt_on2[5]) / 2
            fwhm_on_nm    = 2.355 * sigma_on_nm
            entry.update({
                "splitting_nm"  : splitting_nm,
                "splitting_pm"  : splitting_pm,
                "mid_nm"        : mid_nm,
                "sigma_on_nm"   : sigma_on_nm,
                "fwhm_on_nm"    : fwhm_on_nm,
                "cen1_nm"       : cen1,
                "cen2_nm"       : cen2,
            })
        elif ok_on1:
            # Report broadening instead of splitting
            sigma_on_nm = popt_on1[2]
            fwhm_on_nm  = 2.355 * sigma_on_nm
            broadening  = (fwhm_on_nm - (entry["fwhm_off_nm"] or 0)) * 1000
            entry.update({
                "splitting_nm"  : None,
                "splitting_pm"  : None,
                "sigma_on_nm"   : sigma_on_nm,
                "fwhm_on_nm"    : fwhm_on_nm,
                "broadening_pm" : broadening,
            })
        else:
            entry.update({
                "splitting_nm": None,
                "splitting_pm": None,
            })

        results[label] = entry

    return results


# ===========================================================================
# SECTION 5 — PLOTTING
# ===========================================================================

def plot_overlap_and_zeeman(wl_off, sp_off, wl_on, sp_on,
                             zeeman_results, zoom=ZOOM_WL):
    """
    Figure 1 : full-spectrum overlap (field OFF vs ON)
    Figure 2 : per-line zoom panels with component fits
    """

    # ── Figure 1: full overlap ───────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wl_off, sp_off, color="#3a86ff", lw=0.9, label="Field OFF", alpha=0.85)
    ax.plot(wl_on,  sp_on,  color="#ff006e", lw=0.9, label="Field ON",  alpha=0.85)

    if zoom is not None:
        ax.set_xlim(zoom)
        # auto-scale y within zoom window
        mask_off = (wl_off >= zoom[0]) & (wl_off <= zoom[1])
        mask_on  = (wl_on  >= zoom[0]) & (wl_on  <= zoom[1])
        ymax = max(
            sp_off[mask_off].max() if mask_off.any() else 0,
            sp_on [mask_on ].max() if mask_on .any() else 0,
        ) * 1.15
        ax.set_ylim(0, ymax)

    for label, res in zeeman_results.items():
        ax.axvline(res["centre_nm"], color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.text(res["centre_nm"] + 0.01, ax.get_ylim()[1] * 0.95,
                label, fontsize=7, color="gray", rotation=90, va="top")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("Overlaid Spectra — Field OFF vs Field ON")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()

    # ── Figure 2: per-line Zeeman zoom panels ────────────────────────────────
    n_lines = len(zeeman_results)
    fig2, axes = plt.subplots(1, n_lines, figsize=(6 * n_lines, 5),
                               sharey=False)
    if n_lines == 1:
        axes = [axes]

    fig2.suptitle("Zeeman Component Fits per Spectral Line",
                  fontsize=13, fontweight="bold")

    for ax, (label, res) in zip(axes, zeeman_results.items()):
        centre = res["centre_nm"]
        win    = 0.4   # nm

        # Raw data windows
        mask_off = np.abs(wl_off - centre) <= win
        mask_on  = np.abs(wl_on  - centre) <= win

        ax.plot(wl_off[mask_off], sp_off[mask_off],
                color="#3a86ff", lw=1.2, label="Field OFF", alpha=0.8)
        ax.plot(wl_on [mask_on ], sp_on [mask_on ],
                color="#ff006e", lw=1.2, label="Field ON",  alpha=0.8)

        # OFF single-Gaussian fit
        if res["off_ok"]:
            ax.plot(res["off_x"], res["off_y"],
                    color="#3a86ff", lw=2, ls="--", alpha=0.9,
                    label=f"OFF fit  FWHM={res['fwhm_off_nm']*1000:.0f} pm")

        # ON fit (2-component preferred)
        if res["use_2comp"] and res["on2_ok"]:
            ax.plot(res["on2_x"], res["on2_y"],
                    color="#ff006e", lw=2, ls="--", alpha=0.9,
                    label=(f"ON fit (2-comp)\n"
                           f"Δλ = {res['splitting_pm']:.1f} pm"))
            # Mark individual component centres
            for cen_key in ("cen1_nm", "cen2_nm"):
                if cen_key in res:
                    ax.axvline(res[cen_key], color="#fb5607",
                               lw=1, ls=":", alpha=0.7)
        elif res["on1_ok"]:
            broad_str = (f"  Δbroadening={res.get('broadening_pm', 0):.1f} pm"
                         if "broadening_pm" in res else "")
            ax.plot(res["on1_x"], res["on1_y"],
                    color="#ff006e", lw=2, ls="--", alpha=0.9,
                    label=f"ON fit (1-comp){broad_str}")

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (counts)")
        ax.set_title(f"{label}  ({centre:.3f} nm)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    return fig1, fig2


def print_zeeman_summary(zeeman_results):
    """Print a formatted summary table of Zeeman / broadening results."""
    print("\n" + "=" * 68)
    print("  Zeeman Splitting / Line Broadening Summary")
    print("=" * 68)
    hdr = (f"  {'Line':<14} {'λ₀ (nm)':>10} {'FWHM_off (pm)':>14} "
           f"{'FWHM_on (pm)':>13} {'Δλ (pm)':>10}  Notes")
    print(hdr)
    print("  " + "-" * 64)

    for label, res in zeeman_results.items():
        fwhm_off_pm = res["fwhm_off_nm"] * 1000 if res.get("fwhm_off_nm") else float("nan")
        fwhm_on_pm  = res.get("fwhm_on_nm", float("nan"))
        if fwhm_on_pm is not None:
            fwhm_on_pm *= 1000
        else:
            fwhm_on_pm = float("nan")

        if res.get("splitting_pm") is not None:
            delta_str = f"{res['splitting_pm']:>10.2f}"
            note = "resolved split"
        elif "broadening_pm" in res:
            delta_str = f"{res['broadening_pm']:>10.2f}"
            note = "broadening only"
        else:
            delta_str = f"{'—':>10}"
            note = "fit failed"

        print(f"  {label:<14} {res['centre_nm']:>10.3f} {fwhm_off_pm:>14.1f} "
              f"{fwhm_on_pm:>13.1f} {delta_str}  {note}")

    print("=" * 68)

    # Extra detail for resolved splits
    for label, res in zeeman_results.items():
        if res.get("splitting_pm") is not None:
            c = res["centre_nm"]
            dl = res["splitting_nm"]
            # Δν = c/λ² · Δλ  (c in nm/s → use 3e17 nm/s)
            delta_nu_GHz = (3e17 / c**2) * dl / 1e9
            print(f"\n  {label}: component centres "
                  f"{res['cen1_nm']:.4f} nm  &  {res['cen2_nm']:.4f} nm")
            print(f"    Δλ = {res['splitting_pm']:.2f} pm  →  "
                  f"Δν ≈ {delta_nu_GHz:.2f} GHz")



# --- 1. Define Model Functions ---
def gaussian(x, amp, cen, sig, offset):
    return amp * np.exp(-0.5 * ((x - cen) / sig) ** 2) + offset

def lorentzian(x, amp, cen, gamma, offset):
    return amp * (gamma**2 / ((x - cen)**2 + gamma**2)) + offset

# --- 2. Processing & Plotting Function ---
def plot_comparative_fits(wl_off, sp_off, wl_on, sp_on, center_nm, window=0.3):
    """
    Plots normalized spectra with both Gaussian and Lorentzian fits.
    """
    # Create masks for the specific line window
    mask_off = (wl_off > center_nm - window) & (wl_off < center_nm + window)
    mask_on = (wl_on > center_nm - window) & (wl_on < center_nm + window)
    
    x_off, y_off = wl_off[mask_off], sp_off[mask_off]
    x_on, y_on = wl_on[mask_on], sp_on[mask_on]

    # --- Scaling/Normalization ---
    # We scale both to a peak height of 1.0 (subtracting baseline first)
    y_off_norm = (y_off - y_off.min()) / (y_off.max() - y_off.min())
    y_on_norm = (y_on - y_on.min()) / (y_on.max() - y_on.min())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # --- FIT 1: GAUSSIAN ---
    cen_guess_off = x_off[np.argmax(y_off_norm)]
    cen_guess_on  = x_on[np.argmax(y_on_norm)]
    sig_guess     = window / 4
    popt_g_off, _ = curve_fit(gaussian, x_off, y_off_norm, p0=[1, cen_guess_off, sig_guess, 0])
    popt_g_on, _ = curve_fit(gaussian, x_on, y_on_norm, p0=[1, cen_guess_on, sig_guess, 0])
    
    # Calculate Gaussian FWHM: 2.355 * sigma
    fwhm_g_off = 2.355 * popt_g_off[2] * 1000 # in pm
    fwhm_g_on = 2.355 * popt_g_on[2] * 1000
    
    ax1.plot(x_off, y_off_norm, 'bo', alpha=0.3, label='Data OFF')
    ax1.plot(x_on, y_on_norm, 'ro', alpha=0.3, label='Data ON')
    ax1.plot(x_off, gaussian(x_off, *popt_g_off), 'b-', lw=2, label=f'Gauss OFF ({fwhm_g_off:.1f} pm)')
    ax1.plot(x_on, gaussian(x_on, *popt_g_on), 'r-', lw=2, label=f'Gauss ON ({fwhm_g_on:.1f} pm)')
    ax1.set_title(f"Gaussian Fit\nWidth Diff: {fwhm_g_on - fwhm_g_off:.1f} pm")
    ax1.legend()

    # --- FIT 2: LORENTZIAN ---
    popt_l_off, _ = curve_fit(lorentzian, x_off, y_off_norm, p0=[1, cen_guess_off, sig_guess, 0])
    popt_l_on, _ = curve_fit(lorentzian, x_on, y_on_norm, p0=[1, cen_guess_on, sig_guess, 0])
    
    # Calculate Lorentzian FWHM: 2 * gamma
    fwhm_l_off = 2 * popt_l_off[2] * 1000 # in pm
    fwhm_l_on = 2 * popt_l_on[2] * 1000
    
    ax2.plot(x_off, y_off_norm, 'bo', alpha=0.3, label='Data OFF')
    ax2.plot(x_on, y_on_norm, 'ro', alpha=0.3, label='Data ON')
    ax2.plot(x_off, lorentzian(x_off, *popt_l_off), 'b--', lw=2, label=f'Lorentz OFF ({fwhm_l_off:.1f} pm)')
    ax2.plot(x_on, lorentzian(x_on, *popt_l_on), 'r--', lw=2, label=f'Lorentz ON ({fwhm_l_on:.1f} pm)')
    ax2.set_title(f"Lorentzian Fit\nWidth Diff: {fwhm_l_on - fwhm_l_off:.1f} pm")
    ax2.legend()

    for ax in [ax1, ax2]:
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized Intensity")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_comparative_fits_grid(wl_off, sp_off, wl_on, sp_on, center_nm, axes, label, window=1.0):
    """
    Plots into a specific pair of axes provided by the caller.
    axes: a list/array of two matplotlib axes [ax_gauss, ax_lorentz]
    """
    ax1, ax2 = axes

    # Masking
    mask_off = (wl_off > center_nm - window) & (wl_off < center_nm + window)
    mask_on = (wl_on > center_nm - window) & (wl_on < center_nm + window)
    
    x_off, y_off = wl_off[mask_off], sp_off[mask_off]
    x_on, y_on = wl_on[mask_on], sp_on[mask_on]

    # Normalization (Simpler version to avoid cutting off the bottom)
    y_off_norm = y_off / y_off.max() if len(y_off) > 0 else y_off
    y_on_norm = y_on / y_on.max() if len(y_on) > 0 else y_on

    # --- GAUSSIAN FIT ---
    try:
        cen_guess_off = x_off[np.argmax(y_off_norm)] if len(x_off) > 0 else center_nm
        cen_guess_on  = x_on[np.argmax(y_on_norm)]   if len(x_on)  > 0 else center_nm
        sig_guess     = window / 4
        popt_g_off, _ = curve_fit(gaussian, x_off, y_off_norm, p0=[1, cen_guess_off, sig_guess, 0])
        popt_g_on, _ = curve_fit(gaussian, x_on, y_on_norm, p0=[1, cen_guess_on, sig_guess, 0])
        fwhm_g_off, fwhm_g_on = 2.355 * popt_g_off[2] * 1000, 2.355 * popt_g_on[2] * 1000
        
        ax1.plot(x_off, y_off_norm, 'bo', alpha=0.2)
        ax1.plot(x_on, y_on_norm, 'ro', alpha=0.2)
        ax1.plot(x_off, gaussian(x_off, *popt_g_off), 'b-', label=f'OFF: {fwhm_g_off:.1f}pm')
        ax1.plot(x_on, gaussian(x_on, *popt_g_on), 'r-', label=f'ON: {fwhm_g_on:.1f}pm')
        ax1.set_title(f"{label} (Gaussian)\n$\Delta$FWHM: {fwhm_g_on - fwhm_g_off:.1f} pm")
    except:
        ax1.set_title(f"{label} Gaussian Fit Failed")

    # --- LORENTZIAN FIT ---
    try:
        popt_l_off, _ = curve_fit(lorentzian, x_off, y_off_norm, p0=[1, cen_guess_off, sig_guess, 0])
        popt_l_on, _ = curve_fit(lorentzian, x_on, y_on_norm, p0=[1, cen_guess_on, sig_guess, 0])
        fwhm_l_off, fwhm_l_on = 2 * popt_l_off[2] * 1000, 2 * popt_l_on[2] * 1000
        
        ax2.plot(x_off, y_off_norm, 'bo', alpha=0.2)
        ax2.plot(x_on, y_on_norm, 'ro', alpha=0.2)
        ax2.plot(x_off, lorentzian(x_off, *popt_l_off), 'b--', label=f'OFF: {fwhm_l_off:.1f}pm')
        ax2.plot(x_on, lorentzian(x_on, *popt_l_on), 'r--', label=f'ON: {fwhm_l_on:.1f}pm')
        ax2.set_title(f"{label} (Lorentzian)\n$\Delta$FWHM: {fwhm_l_on - fwhm_l_off:.1f} pm")
    except:
        ax2.set_title(f"{label} Lorentzian Fit Failed")

    for ax in [ax1, ax2]:
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.2)



def plot_all_in_one(wl_off, sp_off, wl_on, sp_on, center_nm, label, window=0.8):
    """
    Plots OFF/ON data and BOTH Gaussian/Lorentzian fits on a single set of axes.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Prepare Data
    mask_off = (wl_off > center_nm - window) & (wl_off < center_nm + window)
    mask_on = (wl_on > center_nm - window) & (wl_on < center_nm + window)
    
    x_off, y_off = wl_off[mask_off], sp_off[mask_off]
    x_on, y_on = wl_on[mask_on], sp_on[mask_on]

    # Normalize to peak = 1.0
    y_off_norm = y_off / y_off.max()
    y_on_norm = y_on / y_on.max()

    # 2. Plot Raw Data (as faint dots)
    plt.scatter(x_off, y_off_norm, color='blue', s=10, alpha=0.15, label='Data OFF')
    plt.scatter(x_on, y_on_norm, color='red', s=10, alpha=0.15, label='Data ON')

    # 3. Fit and Plot Gaussian
    cen_guess_off = x_off[np.argmax(y_off_norm)]
    cen_guess_on  = x_on[np.argmax(y_on_norm)]
    sig_guess     = window / 4
    try:
        popt_g_off, _ = curve_fit(gaussian, x_off, y_off_norm, p0=[1, cen_guess_off, sig_guess, 0])
        popt_g_on, _ = curve_fit(gaussian, x_on, y_on_norm, p0=[1, cen_guess_on, sig_guess, 0])
        
        plt.plot(x_off, gaussian(x_off, *popt_g_off), color='blue', linestyle='-', lw=1.5, 
                 label=f'Gauss OFF ({2.355*popt_g_off[2]*1000:.1f} pm)')
        plt.plot(x_on, gaussian(x_on, *popt_g_on), color='red', linestyle='-', lw=1.5, 
                 label=f'Gauss ON ({2.355*popt_g_on[2]*1000:.1f} pm)')
    except: print(f"Gaussian fit failed for {label}")

    # 4. Fit and Plot Lorentzian
    try:
        popt_l_off, _ = curve_fit(lorentzian, x_off, y_off_norm, p0=[1, cen_guess_off, sig_guess, 0])
        popt_l_on, _ = curve_fit(lorentzian, x_on, y_on_norm, p0=[1, cen_guess_on, sig_guess, 0])
        
        plt.plot(x_off, lorentzian(x_off, *popt_l_off), color='blue', linestyle='--', lw=1.5, alpha=0.7,
                 label=f'Lorentz OFF ({2*popt_l_off[2]*1000:.1f} pm)')
        plt.plot(x_on, lorentzian(x_on, *popt_l_on), color='red', linestyle='--', lw=1.5, alpha=0.7,
                 label=f'Lorentz ON ({2*popt_l_on[2]*1000:.1f} pm)')
    except: print(f"Lorentzian fit failed for {label}")

    # 5. Formatting
    plt.axvline(center_nm, color='black', lw=1, ls=':', label=f'Ref: {center_nm}nm')
    plt.title(f"Detailed Line Profile: {label}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.2)
    plt.show()


def plot_complete_zeeman_spectrum(wl_off, sp_off, wl_on, sp_on, zeeman_results, window_nm=0.8):
    """
    Plots the entire spectrum with Gaussian and Lorentzian fits for all identified lines
    in a single coordinate system.
    """
    plt.figure(figsize=(15, 7))

    # 1. Normalize the full spectra for visual comparison
    # We use a global normalization so relative intensities between lines are preserved
    norm_factor = max(sp_off.max(), sp_on.max())
    y_off_norm = sp_off / norm_factor
    y_on_norm = sp_on / norm_factor

    # 2. Plot the raw data for the entire range
    plt.plot(wl_off, y_off_norm, color='blue', alpha=0.3, lw=0.5, label='Full Data OFF')
    plt.plot(wl_on, y_on_norm, color='red', alpha=0.3, lw=0.5, label='Full Data ON')

    # 3. Iterate through each analyzed line to add fits
    for label, res in zeeman_results.items():
        center_nm = res["centre_nm"]
        
        # Create a local mask for fitting/plotting this specific line
        mask = (wl_off > center_nm - window_nm) & (wl_off < center_nm + window_nm)
        x_fit = wl_off[mask]
        
        if len(x_fit) < 5: continue # Skip if no data in window

        # Local normalization for the fit to match the global plot
        local_max = sp_off[mask].max() / norm_factor

        # --- Gaussian Fit ---
        try:
            popt_g, _ = curve_fit(gaussian, wl_off[mask], y_off_norm[mask], 
                                  p0=[local_max, center_nm, 0.05, 0])
            plt.plot(x_fit, gaussian(x_fit, *popt_g), 'b-', lw=2, 
                     label=f'Gauss {label}' if label == list(zeeman_results.keys())[0] else "")
        except: pass

        # --- Lorentzian Fit ---
        try:
            popt_l, _ = curve_fit(lorentzian, wl_off[mask], y_off_norm[mask], 
                                  p0=[local_max, center_nm, 0.05, 0])
            plt.plot(x_fit, lorentzian(x_fit, *popt_l), 'k--', lw=1.5, alpha=0.7, zorder=3, # zorder=3 to be on top of Gaussian
                     label=f'Lorentz {label}' if label == list(zeeman_results.keys())[0] else "")
        except: pass
        
        # Mark the theoretical center
        plt.axvline(center_nm, color='green', linestyle=':', alpha=0.5)
        plt.text(center_nm, 0.95, label, rotation=90, verticalalignment='top', fontsize=9)

    # 4. Formatting
    plt.title("Complete Calibrated Zeeman Spectrum with Global Fits")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    
    # Zoom into the region where lines actually exist
    all_centers = [r["centre_nm"] for r in zeeman_results.values()]
    plt.xlim(min(all_centers) - 2, max(all_centers) + 2)
    plt.ylim(0, 1.1)
    
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

# ===========================================================================
# SECTION 6 — ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Zeeman Splitting Analysis")
    print("=" * 60)

    # ── Load & calibrate both spectra ────────────────────────────────────────
    print(f"\nLoading field-OFF spectrum : {FILE_OFF}")
    (pix_off, sp_off_raw,
     wl_off, coeffs_off,
     res_off, pinfo_off) = load_and_calibrate(FILE_OFF, MANUAL_PEAKS_OFF, prominence_factor=0.2, fname="calibration_off.npz")

    print(f"\nLoading field-ON  spectrum : {FILE_ON}")
    (pix_on, sp_on_raw,
     wl_on, coeffs_on,
     res_on, pinfo_on) = load_and_calibrate(FILE_ON, MANUAL_PEAKS_ON, prominence_factor=0.2, fname="calibration_on.npz")
    
    # --- SECTION 3 Update ---
    # # Load and calibrate OFF spectrum
    # (pix_off, sp_off_raw, wl_off, coeffs_off, res_off, pinfo_off) = load_and_calibrate(
    #     FILE_OFF, MANUAL_PEAKS_OFF, poly_order=2 # Increased to 2
    # )

    # # Load ON spectrum, but use the OFF coefficients
    # pixels_on, sp_on_raw = np.loadtxt(FILE_ON, delimiter=DELIMITER, skiprows=SKIP_ROWS, unpack=True)
    # wl_on = pixel_to_wavelength(pixels_on, coeffs_off) # USE COEFFS_OFF HERE

    # # 1. Calibrate OFF spectrum (B=0) to get the "Master" coefficients
    # print(f"\nLoading and Calibrating using OFF spectrum: {FILE_OFF}")
    # (pix_off, sp_off_raw, wl_off, coeffs_master, res_off, pinfo_off) = load_and_calibrate(
    #     FILE_OFF, MANUAL_PEAKS_OFF, prominence_factor=0.2
    # )

    # # 2. Load ON spectrum but DO NOT call load_and_calibrate
    # print(f"\nLoading field-ON spectrum: {FILE_ON}")
    # pix_on, sp_on_raw = np.loadtxt(FILE_ON, delimiter=DELIMITER, skiprows=SKIP_ROWS, unpack=True)
    
    # # 3. Apply the MASTER coefficients to the ON pixels
    # wl_on = pixel_to_wavelength(pix_on, coeffs_master)
    
    # # 4. Proceed with analysis using wl_on (calibrated by the OFF coefficients)
    # zeeman_results = analyse_zeeman(
    #     wl_off, sp_off_raw,
    #     wl_on,  sp_on_raw,
    #     lines=LINES_TO_ANALYSE,
    #     window_nm=0.4,
    # )

    print("\n── Calibration polynomials ──")
    for tag, c in [("OFF", coeffs_off), ("ON ", coeffs_on)]:
        poly_str = "  +  ".join(
            f"{coef:.6g}·x^{len(c)-1-i}" for i, coef in enumerate(c)
        )
        print(f"  {tag}: λ(x) = {poly_str}")

    # ── Zeeman analysis ──────────────────────────────────────────────────────
    print("\nRunning Zeeman / broadening fits …")
    zeeman_results = analyse_zeeman(
        wl_off, sp_off_raw,
        wl_on,  sp_on_raw,
        lines=LINES_TO_ANALYSE,
        window_nm=1.0,
        peak_info_off=pinfo_off,   # use measured centroids, not NIST values
        coeffs_off=coeffs_off,
    )

    print_zeeman_summary(zeeman_results)

    #######################

    print("\nGenerating Complete Spectrum Plot...")
    plot_complete_zeeman_spectrum(
        wl_off, sp_off_raw, 
        wl_on, sp_on_raw, 
        zeeman_results,
        window_nm=1.0 # Shows 1nm around each fit
    )

    #######################

    for label, result in zeeman_results.items():
        # Call the unified plot for each line
        plot_all_in_one(
            wl_off, sp_off_raw, 
            wl_on, sp_on_raw, 
            center_nm=result["centre_nm"], 
            label=label,
            window=1.0 # Large window to ensure peak is centered
        )

    # 3. NEW: Comparative Normalized Plots (Gaussian vs Lorentzian)
    # We loop through each line found in the analysis results
    # print("\nGenerating Normalized Gaussian vs Lorentzian comparison plots...")
    # for label, result in zeeman_results.items():
    #     center_wl = result["centre_nm"]
    #     print(f" -> Plotting fits for {label} at {center_wl:.3f} nm")
        
    #     # Call the new function we created
    #     plot_comparative_fits(
    #         wl_off, sp_off_raw, 
    #         wl_on, sp_on_raw, 
    #         center_nm=center_wl, 
    #         # window=0.4  # You can adjust this window if needed
    #         window = 1
    #     )

    # Determine how many lines we are plotting
    num_lines = len(zeeman_results)
    
    # Create ONE figure with 2 columns (Gauss, Lorentz) and N rows (one per line)
    fig_master, all_axes = plt.subplots(num_lines, 2, figsize=(12, 4 * num_lines))
    
    # Ensure all_axes is 2D even if only 1 line is analyzed
    if num_lines == 1:
        all_axes = np.expand_dims(all_axes, axis=0)

    print("\nGenerating Master Comparison Plot...")
    for i, (label, result) in enumerate(zeeman_results.items()):
        center_wl = result["centre_nm"]
        
        # Pass the specific row of axes to the function
        plot_comparative_fits_grid(
            wl_off, sp_off_raw, 
            wl_on, sp_on_raw, 
            center_nm=center_wl, 
            axes=all_axes[i], 
            label=label,
            window=1.0  # Increased to 1.0 to help center the peaks
        )

    fig_master.tight_layout()
    plt.show()

    # ── Plots ────────────────────────────────────────────────────────────────
    fig1, fig2 = plot_overlap_and_zeeman(
        wl_off, sp_off_raw,
        wl_on,  sp_on_raw,
        zeeman_results,
        zoom=ZOOM_WL,
    )
    plt.show()