"""Broadband PIAACMC coronagraph simulation for MagAO-X (z' and 875 filters).

Refactored from ``PIAACMC_BW.ipynb`` into an importable module so notebooks can
reuse the system without copy-pasting.

It also generalises the off-axis offset to 2D ``(dx, dy)`` [lambda/D] so that a
central star and an off-axis companion can be propagated through the same
coronagraph and combined. The two sources are *incoherent*: their intensities
(powers) add, never their electric fields.
"""

import os

import numpy as np
import hcipy as hp
from hcipy import *
import matplotlib.pyplot as plt

from utils.magaox import make_magaox_bump_mask
from utils.apodized_mask_generator import make_magaox_piaa_lyot_stop
from utils.PIAACMC import (
    load_fpm_csv, center_fpm_map, make_fpm_apodizer, make_measured_fpm_apodizer,
    DESIGN_RADII, DESIGN_HEIGHTS, F_NUMBER, SPATIAL_RESOLUTION,
)


FILTERS = {
    "875": {"name": "875 nm", "wavelength": 875e-9, "bandwidth": 0.03},
    "z": {"name": "z band", "wavelength": 908e-9, "bandwidth": 0.13 / 0.908},
}


def build_system(filter_spec):
    """Build every wavelength-independent part of the coronagraph.

    Returns a dict holding the grids, apodizers and propagators that the
    per-wavelength propagation reuses. Nothing here depends on wavelength
    except the refractive-index *functions* (``n(lambda)``), which HCIPy's
    ``SurfaceApodizer`` evaluates at each wavelength for us.
    """
    wavelength = filter_spec["wavelength"]

    # --- telescope pupil (full aperture) ------------------------------------
    telescope_diameter = 6.5          # [m], Magellan Clay
    pupil_diameter = 8.604e-3         # [m], bump mask
    pupil_diameter_full = 8.9e-3      # [m], full entrance pupil MagAO-X
    grid_diameter = 1.5 * telescope_diameter  # [m], oversize
    F_number = 69
    focal_length = float(F_number * pupil_diameter_full)
    pupil_grid_non_scaled = hp.make_pupil_grid(500, grid_diameter)
    pupil_grid = pupil_grid_non_scaled.scaled(pupil_diameter / telescope_diameter)
    aperture_non_scaled = hp.evaluate_supersampled(make_magaox_bump_mask(), pupil_grid_non_scaled, 8)
    aperture = hp.Field(aperture_non_scaled, pupil_grid)
    piaa_pupil_grid = pupil_grid

    # --- measured PIAA lens profiles and FPM height map ---------------------
    piaa0_csv = np.loadtxt("utils/piaa0_CaF2.csv", delimiter=",", skiprows=1)
    piaa1_csv = np.loadtxt("utils/piaa1_CaF2.csv", delimiter=",", skiprows=1)
    radial_d0, profile0 = piaa0_csv[:, 0], -piaa0_csv[:, 1]
    radial_d1, profile1 = piaa1_csv[:, 0], -piaa1_csv[:, 1]
    profile0 -= np.min(profile0)
    profile1 -= np.min(profile1)

    fpm_height, fpm_pixel_size = load_fpm_csv("utils/FPM.csv")
    fpm_height = center_fpm_map(fpm_height, fpm_pixel_size)
    print("FPM map:", fpm_height.shape, "  pixel size [m]:", fpm_pixel_size)

    # --- PIAA lens surfaces and Fresnel propagators -------------------------
    d_pupil_piaa0 = 37.2e-3   # m, pupil plane -> PIAA0
    d_piaa0_piaa1 = 60.2e-3   # m, PIAA0 -> PIAA1
    ref_ind_lenses = hp.get_refractive_index("CAF2")   # callable n(lambda)

    prop_pupil_piaa0 = hp.FresnelPropagator(aperture.grid, d_pupil_piaa0)
    prop_piaa0_piaa1 = hp.FresnelPropagator(aperture.grid, d_piaa0_piaa1)
    prop_piaa1_pupil = hp.FresnelPropagator(aperture.grid, d_piaa0_piaa1 + d_pupil_piaa0)

    prop_LS_piaa1 = hp.FresnelPropagator(aperture.grid, d_pupil_piaa0)
    prop_piaa1_piaa0 = hp.FresnelPropagator(aperture.grid, d_piaa0_piaa1)
    prop_piaa0_LS = hp.FresnelPropagator(aperture.grid, d_piaa0_piaa1 + d_pupil_piaa0)

    sag0 = hp.Field(np.interp(aperture.grid.as_("polar").r, radial_d0, profile0), aperture.grid)
    sag1 = hp.Field(np.interp(aperture.grid.as_("polar").r, radial_d1, profile1), aperture.grid)
    piaa0 = hp.SurfaceApodizer(sag0, ref_ind_lenses)
    piaa1 = hp.SurfaceApodizer(sag1, ref_ind_lenses)

    # --- focal plane, FPM, Lyot stop and detector ---------------------------
    spatial_resolution = SPATIAL_RESOLUTION   # 1 lambda/D at the central wavelength
    fpm_radius = 0.75 * spatial_resolution
    print(f"Focal-plane scale (1 lambda/D) = {spatial_resolution * 1e6:.2f} um,  "
          f"FPM radius = {fpm_radius * 1e6:.2f} um")

    focal_grid = hp.make_focal_grid(q=6, num_airy=16, spatial_resolution=spatial_resolution)
    # FPM focal-plane grid: must capture the full PSF (~16 lambda/D) *and* resolve
    # the ~4.7 um FPM rings, hence 1500 px over 1000 um (~0.67 um/px).
    cor_grid = hp.make_pupil_grid(1500, 1000e-6)

    n_mask = hp.get_refractive_index("IP_DIP")

    prop_piaa1_focal = hp.FraunhoferPropagator(piaa_pupil_grid, cor_grid, focal_length=focal_length)
    prop_piaa1_focal_big = hp.FraunhoferPropagator(piaa_pupil_grid, focal_grid, focal_length=focal_length)
    prop_fpm_lyot = hp.FraunhoferPropagator(cor_grid, piaa_pupil_grid, focal_length=focal_length)

    # Plain-telescope focal plane (before any coronagraph optics), used only
    # for the broadband telescope-PSF diagnostic.
    prop_telescope = hp.FraunhoferPropagator(aperture.grid, hp.make_focal_grid(
        q=8, num_airy=16, pupil_diameter=telescope_diameter,
        focal_length=focal_length, reference_wavelength=wavelength), focal_length=focal_length)

    fpm_apodizer = make_measured_fpm_apodizer(fpm_height, fpm_pixel_size, cor_grid, n_mask)
    fpm_apodizer_design = make_fpm_apodizer(DESIGN_RADII, DESIGN_HEIGHTS, cor_grid, n_mask)

    lyot_stop = hp.evaluate_supersampled(
        hp.make_rotated_aperture(make_magaox_piaa_lyot_stop(), np.deg2rad(245 + 90)), piaa_pupil_grid, 8)

    lyot_apodizer = hp.Apodizer(lyot_stop)

    detector_grid = hp.make_focal_grid(
        q=4,
        num_airy=20,
        pupil_diameter=pupil_diameter,
        focal_length=focal_length,
        reference_wavelength=wavelength,
    )

    prop_detector = hp.FraunhoferPropagator(aperture.grid, detector_grid, focal_length=focal_length)

    return {
        "filter": filter_spec,
        "pupil_diameter": pupil_diameter,
        "aperture": aperture,
        "focal_length": focal_length,
        "piaa0": piaa0,
        "piaa1": piaa1,
        "lyot_stop": lyot_stop,
        "lyot_apodizer": lyot_apodizer,
        "fpm_apodizer": fpm_apodizer,
        "spatial_resolution": spatial_resolution,
        "prop_pupil_piaa0": prop_pupil_piaa0,
        "prop_piaa0_piaa1": prop_piaa0_piaa1,
        "prop_piaa1_pupil": prop_piaa1_pupil,
        "prop_LS_piaa1": prop_LS_piaa1,
        "prop_piaa1_piaa0": prop_piaa1_piaa0,
        "prop_piaa0_LS": prop_piaa0_LS,
        "prop_piaa1_focal": prop_piaa1_focal,
        "prop_piaa1_focal_big": prop_piaa1_focal_big,
        "prop_fpm_lyot": prop_fpm_lyot,
        "prop_detector": prop_detector,
        "detector_grid": detector_grid,
        "prop_telescope": prop_telescope,
        "radial_d0": radial_d0,
        "profile0": profile0,
        "radial_d1": radial_d1,
        "profile1": profile1,
        "fpm_height": fpm_height,
        "fpm_pixel_size": fpm_pixel_size,
        "fpm_apodizer_design": fpm_apodizer_design,
    }


def _offaxis_tilt(aperture, offset, pupil_diameter):
    """Phase ramp that shifts a source to ``offset`` = (dx, dy) [lambda/D].

    A tilt ``exp(2j*pi*(dx*x + dy*y)/D)`` displaces the PSF by
    ``(dx, dy) * (lambda/D)`` in the detector plane.
    """
    dx, dy = offset
    return np.exp(2j * np.pi * (dx * aperture.grid.x + dy * aperture.grid.y) / pupil_diameter)


def propagate_monochromatic_fields(wavelength, sys, offset=(0.0, 0.0)):
    """Propagate one wavelength of a point source through the full coronagraph.

    ``offset`` is the source position ``(dx, dy)`` in units of lambda/D
    (``(0, 0)`` = on-axis). Returns a dict with every intermediate wavefront
    and the final PSF power.
    """
    aperture = sys["aperture"]
    pupil_diameter = sys["pupil_diameter"]

    wf_pupil = hp.Wavefront(aperture * _offaxis_tilt(aperture, offset, pupil_diameter),
                            wavelength=wavelength)
    wf_pupil.total_power = 1.0

    # Forward PIAA (apodize).
    wf_piaa0 = sys["piaa0"].forward(sys["prop_pupil_piaa0"].forward(wf_pupil))
    wf_piaa1 = sys["piaa1"].forward(sys["prop_piaa0_piaa1"].forward(wf_piaa0))
    wf_apodized = sys["prop_piaa1_pupil"].backward(wf_piaa1)

    # FPM and Lyot stop.
    wf_fpm = sys["fpm_apodizer"].forward(sys["prop_piaa1_focal"].forward(wf_apodized))
    wf_lyot = sys["prop_fpm_lyot"].forward(wf_fpm)
    wf_lyot_stopped = sys["lyot_apodizer"](wf_lyot)

    # Inverse PIAA (de-apodize back to a round pupil).
    wf_piaa1_out = sys["piaa1"].forward(sys["prop_LS_piaa1"].forward(wf_lyot_stopped))
    wf_piaa0_out = sys["piaa0"].forward(sys["prop_piaa1_piaa0"].forward(wf_piaa1_out))
    wf_pupil_out = sys["prop_piaa0_LS"].backward(wf_piaa0_out)

    psf_power = sys["prop_detector"].forward(wf_pupil_out).power

    return {
        "wf_pupil": wf_pupil,
        "wf_apodized": wf_apodized,
        "wf_fpm": wf_fpm,
        "wf_lyot": wf_lyot,
        "wf_lyot_stopped": wf_lyot_stopped,
        "wf_pupil_out": wf_pupil_out,
        "psf_power": psf_power,
    }


def propagate_monochromatic(wavelength, sys, offset=(0.0, 0.0)):
    """Monochromatic coronagraphic PSF power for a source at ``offset`` [lambda/D]."""
    return propagate_monochromatic_fields(wavelength, sys, offset)["psf_power"]


def propagate_unocculted(wavelength, sys, offset=(0.0, 0.0)):
    """Reference PSF with the focal-plane mask removed (the 'unocculted' PSF).

    Only the FPM is taken out of the beam; the PIAA lenses and Lyot stop stay,
    matching the FPM papers. Used to normalise the raw contrast.
    """
    aperture = sys["aperture"]
    pupil_diameter = sys["pupil_diameter"]

    wf_pupil = hp.Wavefront(aperture * _offaxis_tilt(aperture, offset, pupil_diameter),
                            wavelength=wavelength)
    wf_pupil.total_power = 1.0

    wf_piaa0 = sys["piaa0"].forward(sys["prop_pupil_piaa0"].forward(wf_pupil))
    wf_piaa1 = sys["piaa1"].forward(sys["prop_piaa0_piaa1"].forward(wf_piaa0))
    wf_apodized = sys["prop_piaa1_pupil"].backward(wf_piaa1)

    wf_lyot_stopped = sys["lyot_apodizer"](wf_apodized)

    wf_piaa1 = sys["piaa1"].forward(sys["prop_LS_piaa1"].forward(wf_lyot_stopped))
    wf_piaa0 = sys["piaa0"].forward(sys["prop_piaa1_piaa0"].forward(wf_piaa1))
    wf_pupil_out = sys["prop_piaa0_LS"].backward(wf_piaa0)

    return sys["prop_detector"].forward(wf_pupil_out).power


def _sum_over_band(sys, wavelengths, mono_func, spectrum=None):
    """Incoherently sum a per-wavelength PSF ``mono_func(wl, sys)`` over the band."""
    total = None
    for wl in wavelengths:
        weight = 1.0 if spectrum is None else spectrum(wl)
        psf_power = mono_func(wl, sys)
        total = weight * psf_power if total is None else total + weight * psf_power
    return total


# --------------------------------------------------------------------------- #
# Single-source broadband run (kept for parity with PIAACMC_BW.ipynb).
# --------------------------------------------------------------------------- #

def _radial_profile(values, grid, spatial_resolution, n_bins=100):
    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    r = np.hypot(x, y) / spatial_resolution
    vals = np.asarray(values)

    edges = np.linspace(0, r.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        if np.any(sel):
            profile[i] = np.mean(vals[sel])
    return centers, profile


def _mean_contrast_1_5(field, sys, n_bins=100):
    """Mean raw contrast averaged over the 1-5 lambda/D annulus (paper metric)."""
    centers, prof = _radial_profile(field, sys["detector_grid"], sys["spatial_resolution"], n_bins)
    sel = (centers >= 1) & (centers <= 5)
    return np.nanmean(prof[sel])


def run_bandwidth(filter_name="z", n_wavelengths=11, save=False, plot=True, out_dir="outputs"):
    """Run the coronagraph over a filter band and return the raw-contrast field."""
    filter_spec = FILTERS[filter_name]
    wavelengths = np.linspace(
        filter_spec["wavelength"] * (1 - filter_spec["bandwidth"] / 2),
        filter_spec["wavelength"] * (1 + filter_spec["bandwidth"] / 2),
        n_wavelengths,
    )

    sys = build_system(filter_spec)

    broadband = _sum_over_band(sys, wavelengths, propagate_monochromatic)
    unocculted = _sum_over_band(sys, wavelengths, propagate_unocculted)

    unocculted_peak = unocculted.max()
    contrast = broadband / unocculted_peak

    print(f"Unocculted PSF peak = {unocculted_peak:.6g}")
    print(f"On-axis raw contrast = {contrast.max():.3e}")
    print(f"1-5 lambda/D raw contrast = {_mean_contrast_1_5(contrast, sys):.3e}")

    return contrast


# --------------------------------------------------------------------------- #
# Two-source observation: central star + off-axis companion.
# --------------------------------------------------------------------------- #

def observe_star_and_companion(filter_name="z", separation_ld=1.5, position_angle_deg=0.0,
                                flux_ratio=5e-2, n_wavelengths=7, out_dir="outputs",
                                plot=True, save=False, vmin=-6, vmax=0):
    """Simulate a coronagraphic observation of a central star + off-axis companion.

    Both are incoherent point sources; each is propagated through the full
    coronagraph over the filter band and their *powers* are summed. The star is
    on-axis, the companion is at ``separation_ld`` lambda/D and position angle
    ``position_angle_deg`` (deg, from the +x axis), with a flux ratio
    ``flux_ratio`` relative to the star.

    ``vmin``/``vmax`` set the shared log colour scale of every PSF panel
    (``0`` == an un-suppressed star's peak, since all fields are normalised to
    the unocculted on-axis star peak). Keeping them shared is what lets the
    panels be compared in real intensity.

    Returns a dict of broadband fields, all normalised to the unocculted
    on-axis stellar PSF peak:

    * ``star``       - suppressed on-axis stellar residual (raw contrast)
    * ``companion``  - off-axis companion PSF at *unit* flux (multiply by
                       ``flux_ratio`` for its real intensity)
    * ``science``    - ``star + flux_ratio * companion`` (the actual image)
    * ``unocculted`` - on-axis reference PSF (no FPM), normalised to 1
    """
    filter_spec = FILTERS[filter_name]
    wl = filter_spec["wavelength"]
    bw = filter_spec["bandwidth"]
    wavelengths = np.linspace(wl * (1 - bw / 2), wl * (1 + bw / 2), n_wavelengths)

    sys = build_system(filter_spec)
    sr = sys["spatial_resolution"]

    theta = np.deg2rad(position_angle_deg)
    offset = (separation_ld * np.cos(theta), separation_ld * np.sin(theta))

    # Three independent broadband sums over the SAME built system. The star and
    # companion are independent -> trivially parallelisable if memory allows.
    star = _sum_over_band(sys, wavelengths, lambda w, s: propagate_monochromatic(w, s, (0.0, 0.0)))
    companion = _sum_over_band(sys, wavelengths, lambda w, s: propagate_monochromatic(w, s, offset))
    unocculted = _sum_over_band(sys, wavelengths, lambda w, s: propagate_unocculted(w, s, (0.0, 0.0)))

    unocc_peak = unocculted.max()
    star_raw = star / unocc_peak
    companion_raw = companion / unocc_peak
    science = star_raw + flux_ratio * companion_raw

    print(f"\nSeparation = {separation_ld} lambda/D,  PA = {position_angle_deg} deg,  "
          f"flux ratio = {flux_ratio:.2e}")
    print(f"Unocculted star peak          = {unocc_peak:.6g}")
    print(f"Star residual (1-5 lambda/D)   = {_mean_contrast_1_5(star_raw, sys):.3e}")
    print(f"Companion peak (raw)           = {companion_raw.max():.3e}")

    if plot or save:
        os.makedirs(out_dir, exist_ok=True)
    if plot:
        _plot_observation(star_raw, companion_raw, unocculted / unocc_peak, science, sys,
                          separation_ld, position_angle_deg, flux_ratio, out_dir, vmin, vmax)
    if save:
        np.savez(
            os.path.join(out_dir, f"{filter_name}_star_companion.npz"),
            star=star_raw, companion=companion_raw, science=science,
            unocculted=unocculted / unocc_peak,
            spatial_resolution=sr,
            separation_ld=separation_ld,
            position_angle_deg=position_angle_deg,
            flux_ratio=flux_ratio,
        )

    return {
        "detector_grid": sys["detector_grid"],
        "spatial_resolution": sr,
        "star": star_raw,
        "companion": companion_raw,
        "science": science,
        "unocculted": unocculted / unocc_peak,
        "offset": offset,
        "separation_ld": separation_ld,
        "flux_ratio": flux_ratio,
    }


def _cut_through(field, grid, spatial_resolution, max_ld=None):
    """Horizontal (y=0) cross-section through ``field``.

    Assumes the companion lies on the +x axis (position angle = 0 deg) so the
    companion peak appears on this cut. Returns ``(x_ld, values)``.
    """
    vals = np.asarray(field).reshape(grid.shape)
    y1d = grid.y.reshape(grid.shape)[:, 0]
    x1d = grid.x.reshape(grid.shape)[0, :]
    row = int(np.argmin(np.abs(y1d)))
    x_ld = x1d / spatial_resolution
    if max_ld is not None:
        sel = np.abs(x_ld) <= max_ld
        return x_ld[sel], vals[row, sel]
    return x_ld, vals[row, :]


def _plot_observation(star, companion, unocculted, science, sys, separation_ld, pa_deg, flux_ratio,
                      out_dir, vmin=-6, vmax=0):
    sr = sys["spatial_resolution"]
    name = sys["filter"]["name"]

    fig, ax = plt.subplots(2, 3, figsize=(19, 12))

    def show(field, ax, title):
        # All fields are already normalised to the unocculted on-axis star peak,
        # so one shared (vmin, vmax) makes the panels directly comparable in
        # "raw intensity" units (1.0 == an un-suppressed star's peak).
        floor = 10.0 ** vmin
        img = hp.Field(np.log10(np.maximum(np.asarray(field), floor)), field.grid)
        hp.imshow_field(img, cmap="inferno", grid_units=sr, vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(r"Detector plane [$\lambda/D$]")
        ax.set_ylabel(r"Detector plane [$\lambda/D$]")
        fig.colorbar(ax.images[0], ax=ax, label="log10(raw power)")

    show(star, ax[0, 0], "On-axis star (suppressed)")
    show(companion * flux_ratio, ax[0, 1],
         f"Off-axis companion ($\\times$ flux ratio {flux_ratio:.1e})")
    show(unocculted, ax[0, 2], "Unocculted reference (no FPM)")
    show(science, ax[1, 0], "Combined science image")

    # 1-D cut spanning the bottom-right two cells (valid for PA = 0).
    gs = ax[1, 1].get_gridspec()
    ax[1, 1].remove()
    ax[1, 2].remove()
    axcut = fig.add_subplot(gs[1, 1:])

    x_ld, star_cut = _cut_through(star, sys["detector_grid"], sr, max_ld=8)
    _, comp_cut = _cut_through(companion, sys["detector_grid"], sr, max_ld=8)
    _, sci_cut = _cut_through(science, sys["detector_grid"], sr, max_ld=8)
    axcut.semilogy(x_ld, star_cut, color="C0", lw=1.2, label="Star only")
    axcut.semilogy(x_ld, comp_cut * flux_ratio, color="C2", lw=1.2, ls="--",
                   label="Companion (scaled)")
    axcut.semilogy(x_ld, sci_cut, color="k", lw=1.5, label="Combined")
    axcut.axvline(separation_ld, color="C2", ls=":", alpha=0.6)
    axcut.set_title("Cross-section along +x (companion axis)")
    axcut.set_xlabel(r"x [$\lambda/D$]")
    axcut.set_ylabel("Raw power (1 = unocculted star peak)")
    axcut.set_ylim(1e-8, 1.0)
    axcut.grid(True, which="both", ls=":", alpha=0.4)
    axcut.legend()

    fig.suptitle(f"{name} — star + companion observation (separation {separation_ld} $\\lambda/D$, "
                 f"PA {pa_deg} deg)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, f"star_companion_observation_{separation_ld}_{pa_deg}_{flux_ratio}.png"), dpi=150)
    plt.close(fig)
    