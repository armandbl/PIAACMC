import csv
import numpy as np
import scipy.ndimage as ndi
from scipy.signal import fftconvolve
import hcipy as hp

N_RINGS = 10
WAVELENGTH = 0.908e-6            # m, z' filter central wavelength
F_NUMBER = 69
MAX_RADIUS_NORM = 0.75           # lambda/D, mask radius

SPATIAL_RESOLUTION = WAVELENGTH * F_NUMBER    # 1 lambda/D in the focal plane [m]

DESIGN_RADII = (
    np.linspace(MAX_RADIUS_NORM / N_RINGS, MAX_RADIUS_NORM, N_RINGS)
    * SPATIAL_RESOLUTION
)
DESIGN_HEIGHTS = np.array([
    1.26196831, 0.94074998, 0.77836148, 0.6366583, 0.7739299,
    3.19782967, 0.85539274, 0.79005693, 0.88346143, 0.87813155,
])   # um


def load_fpm_csv(path):
    """Load a Keyence VK-X1000 height-map CSV.

    Returns ``(height_um, pixel_size_m)``: the 2D surface height in microns
    (``shape = (rows, columns)``, row 0 = +y) and the XY calibration in m/px.
    """
    with open(path) as f:
        rows = list(csv.reader(f))

    header_i = next(i for i, row in enumerate(rows) if row and row[0] == "Height")
    meta = {row[0]: row[1] for row in rows[:header_i] if len(row) >= 2}

    pixel_size = float(meta["XY calibration"]) * 1e-6
    height_um = np.array([[float(v) for v in row] for row in rows[header_i + 1:] if row])
    height_um[height_um < 0.5] = 0.0            # zero the noise floor
    return height_um, pixel_size


def make_fpm_ring_field(radii, thicknesses, grid):
    """Surface sag [m] of a concentric-ring FPM evaluated on ``grid``.

    Ring ``i`` spans ``(radii[i-1], radii[i]]`` (ring 0 is the central disc
    ``[0, radii[0]]``) with constant height ``thicknesses[i]`` [um]; the region
    beyond ``radii[-1]`` is the unetched substrate (height 0).
    """
    radii = np.asarray(radii, dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    if radii.size != thicknesses.size:
        raise ValueError("`radii` and `thicknesses` must have the same length")

    r = np.hypot(grid.x, grid.y)
    sag = np.zeros(grid.size)
    inner = np.concatenate(([0.0], radii[:-1]))
    for r_in, r_out, t in zip(inner, radii, thicknesses):
        sag[(r > r_in) & (r <= r_out)] = t

    return hp.Field(sag * 1e-6, grid)


def make_fpm_apodizer(radii, thicknesses, grid, refractive_index):
    """Build the design FPM as an HCIPy :class:`SurfaceApodizer`."""
    sag = make_fpm_ring_field(radii, thicknesses, grid)
    return hp.SurfaceApodizer(sag, refractive_index)


def make_synthetic_fpm_map(radii, thicknesses, dims, diameter):
    """Design FPM height map [um] on a pupil grid with given dimensions.

    ``dims``/``diameter`` are passed to ``hp.make_pupil_grid``, so the returned
    array has the same pixel count and scale as that grid and is centred on the
    optical axis.  It follows the image convention (row 0 = +y), like a measured
    map, and reproduces the surface of :func:`make_fpm_apodizer`.
    """
    grid = hp.make_pupil_grid(dims, diameter)
    sag = make_fpm_ring_field(radii, thicknesses, grid)
    return sag.reshape(grid.shape)[::-1] * 1e6


def prep_map(image):
    """Subtract the median and replace non-finite values with zero."""
    out = np.array(image, dtype=float)
    median = np.nanmedian(out)
    if np.isfinite(median):
        out = out - median
    out[~np.isfinite(out)] = 0.0
    return out


def measure_shift_integer(reference_map, target_map):
    """Cross-correlate two maps via FFT convolution.

    Returns ``(lag_x, lag_y, peak_value, corr)`` where ``target`` is
    ``reference`` shifted by ``(-lag_x, -lag_y)`` pixels, so the shift that
    realigns ``target`` with ``reference`` is ``np.roll(target, (lag_y, lag_x))``.
    """
    ref = prep_map(reference_map)
    tgt = prep_map(target_map)
    corr = fftconvolve(ref, tgt[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    center_y, center_x = (s // 2 for s in corr.shape)
    return int(peak_x - center_x), int(peak_y - center_y), float(corr[peak_y, peak_x]), corr


def center_fpm_map(height_um, pixel_size, radii=DESIGN_RADII, thicknesses=DESIGN_HEIGHTS):
    """Centre a measured FPM height map on the optical axis.

    The measured map is cross-correlated against a synthetic design map built
    on a pupil grid with the same dimensions, then rolled so its ring pattern
    sits at the image centre (= the optical axis, (0, 0)).
    """
    rows, columns = height_um.shape
    dims = [columns, rows]
    diameter = [columns * pixel_size, rows * pixel_size]

    synthetic = make_synthetic_fpm_map(radii, thicknesses, dims, diameter)
    lag_x, lag_y, _, _ = measure_shift_integer(synthetic, height_um)
    return np.roll(height_um, shift=(lag_y, lag_x), axis=(0, 1))


def make_measured_fpm_apodizer(height_um, pixel_size, grid, refractive_index):
    """Sample a centred measured FPM map onto ``grid`` as a SurfaceApodizer.
    
    ``height_um`` must be centred (see :func:`center_fpm_map`), so its ring
    pattern sits at the image centre, which maps to the optical axis on ``grid``.
    """
    rows, columns = height_um.shape
    grid_x = grid.x.reshape(grid.shape)      # [m], row 0 = -y
    grid_y = grid.y.reshape(grid.shape)
    column = (columns - 1) / 2 + grid_x / pixel_size
    row = (rows - 1) / 2 - grid_y / pixel_size
    sag = ndi.map_coordinates(height_um, [row, column], order=1, mode="constant", cval=0.0) * 1e-6
    return hp.SurfaceApodizer(hp.Field(sag.ravel(), grid), refractive_index)
