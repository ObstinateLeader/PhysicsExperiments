"""
Spectrometer Calibration Script
================================
Calibrates a spectrometer by fitting a polynomial to known reference
emission lines, then uses the calibration to measure an unknown laser
wavelength from a CCD pixel position.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# 1. Reference emission lines (wavelength in nm → known atomic physics)
# ---------------------------------------------------------------------------

# REFERENCE_LINES = {
#     "Hg 405": 404.66,
#     "Hg 436": 435.83,
#     "Hg 546": 546.07,
#     "Hg 577": 576.96,
#     "Hg 579": 579.07,
# }

# REFERENCE_LINES = {
#     "Na 589.0": 588.995,
#     "Na 589.6": 589.592,
# }

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


# ---------------------------------------------------------------------------
# 2. Gaussian peak model
# ---------------------------------------------------------------------------

def gaussian(x, amplitude, center, sigma, baseline):
    """Single Gaussian with a constant baseline."""
    return baseline + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def fit_peak(pixels, intensities, guess_center, window=30):
    """
    Fit a Gaussian to a spectral peak.

    Parameters
    ----------
    pixels      : 1-D array of pixel indices
    intensities : 1-D array of CCD counts
    guess_center: approximate pixel position of the peak
    window      : ±pixels around the guess to include in the fit

    Returns
    -------
    center  : fitted centroid (sub-pixel precision)
    sigma   : Gaussian width in pixels
    success : bool
    """
    mask = np.abs(pixels - guess_center) <= window
    x, y = pixels[mask], intensities[mask]

    if len(x) < 5:
        return guess_center, np.nan, False

    amp0   = y.max() - y.min()
    base0  = y.min()
    sigma0 = window / 3

    try:
        popt, _ = curve_fit(
            gaussian, x, y,
            p0=[amp0, guess_center, sigma0, base0],
            bounds=([0, guess_center - window, 0.5, 0],
                    [np.inf, guess_center + window, window, np.inf]),
            maxfev=5000,
        )
        return popt[1], popt[2], True
    except RuntimeError:
        return guess_center, np.nan, False


# ---------------------------------------------------------------------------
# 3. Calibration
# ---------------------------------------------------------------------------

def calibrate(pixel_positions, wavelengths, poly_order=2):
    """
    Fit a polynomial λ(x) = a0 + a1·x + a2·x² + …

    Parameters
    ----------
    pixel_positions : list/array of measured centroid pixels
    wavelengths     : list/array of corresponding known wavelengths (nm)
    poly_order      : degree of the polynomial (2 recommended)

    Returns
    -------
    coeffs    : polynomial coefficients (highest degree first, numpy convention)
    residuals : fit residuals in nm for each reference line
    """
    coeffs = np.polyfit(pixel_positions, wavelengths, poly_order)
    fitted = np.polyval(coeffs, pixel_positions)
    residuals = np.array(wavelengths) - fitted
    return coeffs, residuals


def pixel_to_wavelength(pixel, coeffs):
    """Convert a pixel position to wavelength using the calibration polynomial."""
    return np.polyval(coeffs, pixel)


# ---------------------------------------------------------------------------
# 4. Synthetic demo data (replace with real CCD arrays in production)
# ---------------------------------------------------------------------------

def make_demo_spectrum(n_pixels=2048, noise_level=5, seed=42):
    """
    Generate a realistic-looking calibration spectrum with known line positions
    plus Gaussian noise, so the script can be run and tested stand-alone.
    """
    rng = np.random.default_rng(seed)
    pixels = np.arange(n_pixels, dtype=float)

    # True (ground-truth) linear mapping used to generate the fake data.
    # In a real experiment you do NOT know this — it is what you are solving for.
    true_coeffs = np.array([0.0, 0.28, 130.0])   # λ = 0·x² + 0.28·x + 130 nm

    spectrum = np.full(n_pixels, 50.0)            # baseline counts

    # Place synthetic emission lines
    demo_pixel_positions = {}
    for name, wl in REFERENCE_LINES.items():
        # invert: pixel = (λ - c) / b  (valid for the linear true mapping)
        px = (wl - true_coeffs[2]) / true_coeffs[1]
        demo_pixel_positions[name] = px
        sigma_px = 4.0
        amp = rng.uniform(3000, 8000)
        spectrum += amp * np.exp(-0.5 * ((pixels - px) / sigma_px) ** 2)

    spectrum += rng.normal(0, noise_level, n_pixels)
    spectrum = np.clip(spectrum, 0, None)
    return pixels, spectrum, demo_pixel_positions, true_coeffs


# ---------------------------------------------------------------------------
# 5. Main calibration workflow
# ---------------------------------------------------------------------------

def run_calibration(pixels, spectrum, reference_lines,
                    poly_order=2, peak_window=30,
                    manual_peaks=None, prominence_factor=0.2):
    """
    Full calibration pipeline.

    Parameters
    ----------
    pixels          : 1-D pixel array
    spectrum        : 1-D intensity array (CCD counts)
    reference_lines : dict  {label: wavelength_nm}
    poly_order      : polynomial degree for λ(x)
    peak_window     : window (±pixels) for Gaussian fitting
    manual_peaks    : dict {label: approx_pixel}  — supply if auto-detection
                      is unreliable.  If None, uses scipy peak-finding.

    Returns
    -------
    coeffs    : calibration polynomial coefficients
    residuals : residuals in nm
    peak_info : dict with per-line centroid and sigma
    """
    wl_values = np.array(list(reference_lines.values()))
    labels    = list(reference_lines.keys())

    # --- auto-detect prominent peaks if no manual hints supplied -------------
    if manual_peaks is None:
        prominence = (spectrum.max() - spectrum.min()) * prominence_factor
        peak_idx, _ = find_peaks(spectrum, prominence=prominence, distance=20)
        print(f"DEBUG: Prominence threshold: {prominence:.1f}")
        print(f"DEBUG: Found {len(peak_idx)} peaks at pixels: {pixels[peak_idx]}")
        print(f"DEBUG: Peak intensities: {spectrum[peak_idx]}")
        n_found = len(peak_idx)
        n_needed = len(reference_lines)
        if n_found < n_needed:
            print(f"  WARNING: Auto-detection found only {n_found} peaks but "
                  f"{n_needed} reference lines were supplied. "
                  f"Calibration will proceed using the {n_found} matched lines.")
        # Sort detected peaks by position and pair with lines sorted by λ
        peak_idx_sorted = peak_idx[np.argsort(pixels[peak_idx])]
        sort_order      = np.argsort(wl_values)
        # Only pair as many peaks as were detected
        guess_pixels    = {labels[sort_order[j]]: pixels[peak_idx_sorted[j]]
                           for j in range(min(n_found, n_needed))}
    else:
        guess_pixels = manual_peaks

    # --- fit each peak (skip lines with no pixel guess) ----------------------
    centroids  = []
    peak_info  = {}
    for label, wl in reference_lines.items():
        if label not in guess_pixels:
            print(f"  SKIPPING {label} ({wl} nm) — no peak position available.")
            peak_info[label] = {"wavelength_nm": wl,
                                "centroid_pixel": None,
                                "sigma_pixels": None,
                                "fit_ok": False,
                                "skipped": True}
            continue

        center, sigma, ok = fit_peak(
            pixels, spectrum, guess_pixels[label], window=peak_window
        )
        if not ok:
            print(f"  WARNING: Gaussian fit failed for {label} ({wl} nm). "
                  "Using initial guess.")
        centroids.append(center)
        peak_info[label] = {"wavelength_nm": wl,
                            "centroid_pixel": center,
                            "sigma_pixels": sigma,
                            "fit_ok": ok,
                            "skipped": False}

    # --- check we have enough points for the requested polynomial order ------
    n_pts = len(centroids)
    if n_pts < 2:
        raise ValueError(
            f"Only {n_pts} peak(s) could be identified. "
            "Need at least 2 points for any calibration fit."
        )
    if n_pts <= poly_order:
        new_order = n_pts - 1
        print(f"  WARNING: Only {n_pts} points available — reducing polynomial "
              f"order from {poly_order} to {new_order}.")
        poly_order = new_order

    # Only use wavelengths for lines that were actually fitted
    fitted_labels = [l for l, info in peak_info.items()
                     if not info.get("skipped")]
    fitted_wls    = np.array([peak_info[l]["wavelength_nm"]
                               for l in fitted_labels])

    # --- polynomial fit ------------------------------------------------------
    coeffs, residuals = calibrate(centroids, fitted_wls, poly_order)

    for i, label in enumerate(fitted_labels):
        peak_info[label]["residual_nm"] = residuals[i]

    return coeffs, residuals, peak_info


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def plot_calibration(pixels, spectrum, peak_info, coeffs, residuals,
                     laser_pixel=None, laser_wavelength=None):
    """Generate a 3-panel calibration figure."""

    fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})
    fig.suptitle("Spectrometer Calibration", fontsize=14, fontweight="bold")

    # ── Panel 1: raw spectrum ────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(pixels, spectrum, color="#3a86ff", lw=0.8, label="CCD spectrum")
    for label, info in peak_info.items():
        if info.get("skipped"):
            continue
        ax1.axvline(info["centroid_pixel"], color="#ff006e", lw=1.2,
                    ls="--", alpha=0.8)
        ax1.text(info["centroid_pixel"] + 10, spectrum.max() * 0.85,
                 f"{info['wavelength_nm']:.1f} nm",
                 fontsize=7, color="#ff006e", rotation=90, va="top")
    if laser_pixel is not None:
        ax1.axvline(laser_pixel, color="#fb5607", lw=2, ls="-",
                    label=f"Laser ({laser_wavelength:.2f} nm)")
        ax1.legend(fontsize=9)
    ax1.set_xlabel("Pixel")
    ax1.set_ylabel("Intensity (counts)")
    ax1.set_title("Raw CCD Spectrum with Identified Reference Lines")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: calibration curve ───────────────────────────────────────────
    ax2 = axes[1]
    centroids = [info["centroid_pixel"] for info in peak_info.values()
                 if not info.get("skipped")]
    wavelengths = [info["wavelength_nm"] for info in peak_info.values()
                   if not info.get("skipped")]

    px_range = np.linspace(pixels.min(), pixels.max(), 500)
    ax2.plot(px_range, np.polyval(coeffs, px_range),
             color="#8338ec", lw=2, label=f"Poly fit (order {len(coeffs)-1})")
    ax2.scatter(centroids, wavelengths, color="#ff006e", zorder=5,
                label="Reference lines")
    if laser_pixel is not None:
        ax2.scatter([laser_pixel], [laser_wavelength],
                    color="#fb5607", marker="*", s=200, zorder=6,
                    label=f"Laser: {laser_wavelength:.2f} nm")
    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("Wavelength (nm)")
    ax2.set_title("Calibration Curve  λ(x)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: residuals ───────────────────────────────────────────────────
    ax3 = axes[2]
    residuals_plot = [info["residual_nm"] for info in peak_info.values()
                      if not info.get("skipped")]
    ax3.stem(centroids, np.array(residuals_plot) * 1000,
             linefmt="#ff006e", markerfmt="o", basefmt="k-")
    ax3.axhline(0, color="k", lw=1)
    ax3.set_xlabel("Pixel")
    ax3.set_ylabel("Residual (pm)")
    ax3.set_title("Calibration Residuals  (1 pm = 0.001 nm)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_top_panel(pixels, spectrum, peak_info, coeffs, residuals,
                     laser_pixel=None, laser_wavelength=None):
    """Generate a 3-panel calibration figure."""

    fig, axes = plt.subplots(1, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [3]})
    fig.suptitle("Spectrometer Calibration", fontsize=14, fontweight="bold")

    # ── Panel 1: raw spectrum ────────────────────────────────────────────────
    ax1 = axes
    ax1.plot(pixels, spectrum, color="#3a86ff", lw=0.8, label="CCD spectrum")
    for label, info in peak_info.items():
        if info.get("skipped"):
            continue
        ax1.axvline(info["centroid_pixel"], color="#ff006e", lw=1.2,
                    ls="--", alpha=0.8)
        ax1.text(info["centroid_pixel"] + 10, spectrum.max() * 0.85,
                 f"{info['wavelength_nm']:.1f} nm",
                 fontsize=7, color="#ff006e", rotation=90, va="top")
    if laser_pixel is not None:
        ax1.axvline(laser_pixel, color="#fb5607", lw=2, ls="-",
                    label=f"Laser ({laser_wavelength:.2f} nm)")
        ax1.legend(fontsize=9)
    ax1.set_xlabel("Pixel")
    ax1.set_ylabel("Intensity (counts)")
    ax1.set_title("Raw CCD Spectrum with Identified Reference Lines")
    ax1.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# 7. Entry point — demo run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("  Spectrometer Calibration — Demo Run")
    print("=" * 60)

    # ── Generate synthetic data ──────────────────────────────────────────────
    # pixels, spectrum, demo_peaks, true_coeffs = make_demo_spectrum()
    # pixels, spectrum = np.loadtxt('Natriummaybef.txt', delimiter=',', skiprows=1, unpack=True)
    pixels, spectrum = np.loadtxt('NeonMeting1540.txt', delimiter=',', skiprows=1, unpack=True)
    pixels, spectrum = np.loadtxt('NeonMeting1540(1).txt', delimiter=',', skiprows=1, unpack=True)
    pixels, spectrum = np.loadtxt('NatriumBalk1.txt', delimiter=',', skiprows=1, unpack=True)
    pixels, spectrum = np.loadtxt('NeonMeasurementsession5.txt', delimiter=',', skiprows=1, unpack=True)
    # pixels, spectrum = np.loadtxt('NeonMeasurementsession5(1).txt', delimiter=',', skiprows=1, unpack=True)
    pixels, spectrum = np.loadtxt('measurement sessie 6 magnet  close(2).txt', delimiter=',', skiprows=1, unpack=True)
    # pixels, spectrum = np.loadtxt('measurement sessie 6 magnet far(2).txt', delimiter=',', skiprows=1, unpack=True)

    manual_peaks = None

    # manual_peaks = {
    # "Na 589.0": 2159,  # ← replace with pixel you read from the plot
    # "Na 589.6": 2185,  # ← replace with pixel you read from the plot
    # }

#     manual_peaks = {
#     "Na 589.0": 2163.0,
#     "Na 589.6": 2168.0,
# }
    
#     manual_peaks = {
#     "Na 589.0": 2158.0,
#     "Na 589.6": 2190.0,
# }


    plt.title("Raw CCD Spectrum")
    plt.plot(pixels, spectrum, color="#3a86ff", lw=0.8)
    plt.xlabel("Pixel")
    plt.ylabel("Intensity (counts)")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\nCCD size       : {len(pixels)} pixels")
    # print(f"True mapping   : λ = {true_coeffs[1]:.4f}·x + {true_coeffs[2]:.2f} nm")

    # ── Run calibration ──────────────────────────────────────────────────────
    coeffs, residuals, peak_info = run_calibration(
        pixels, spectrum, REFERENCE_LINES,
        poly_order=1,
        peak_window=15,
        manual_peaks=manual_peaks,  # use known positions in demo
    )

    print("\n── Fitted calibration polynomial ──")
    poly_str = " + ".join(
        f"{c:.6g}·x^{len(coeffs)-1-i}" for i, c in enumerate(coeffs)
    )
    print(f"  λ(x) = {poly_str}")

    print("\n── Per-line results ──")
    print(f"  {'Line':<12} {'True λ (nm)':>12} {'Centroid px':>12} "
          f"{'Residual (pm)':>15}")
    print("  " + "-" * 55)
    for label, info in peak_info.items():
        if info.get("skipped"):
            print(f"  {label:<12} {info['wavelength_nm']:>12.2f} "
                  f"{'skipped':>12} {'—':>15}")
        else:
            print(f"  {label:<12} {info['wavelength_nm']:>12.2f} "
                  f"{info['centroid_pixel']:>12.1f} "
                  f"{info['residual_nm']*1000:>15.2f}")

    rms = np.sqrt(np.mean(residuals**2)) * 1000
    print(f"\n  RMS residual : {rms:.2f} pm")

    # ── Measure a synthetic laser ────────────────────────────────────────────
    # Suppose the laser sits at pixel 1450 on our fake CCD
    laser_pixel = 1450.0
    laser_wl = pixel_to_wavelength(laser_pixel, coeffs)
    print(f"\n── Laser measurement ──")
    print(f"  Laser centroid pixel : {laser_pixel}")
    print(f"  Measured wavelength  : {laser_wl:.3f} nm")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plot_calibration(
        pixels, spectrum, peak_info, coeffs, residuals,
        # laser_pixel=laser_pixel, laser_wavelength=laser_wl,
    )
    # out_path = "/mnt/user-data/outputs/calibration_plot.png"
    # fig.savefig(out_path, dpi=150, bbox_inches="tight")
    # print(f"\nCalibration figure saved → {out_path}")
    plt.show()

    fig = plot_top_panel(
         pixels, spectrum, peak_info, coeffs, residuals,
        # laser_pixel=laser_pixel, laser_wavelength=laser_wl,
    )
    plt.show()

    # np.savetxt("calibration far.txt", np.array([coeffs, residuals, peak_info]), delimiter=",")
    # np.save("data.npy", np.array([pixels, spectrum,coeffs, residuals, peak_info]))
# Load with: arr = np.load("data.npy")

np.savez("calibration_on.npz", 
         pixels=pixels, 
         spectrum=spectrum, 
         coeffs=coeffs, 
         residuals=residuals, 
         peak_info=peak_info)

# To load it back:
# data = np.load("calibration_data.npz", allow_pickle=True)