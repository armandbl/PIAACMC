import numpy as np
import hcipy as hp
import hcipy as hcipy
from hcipy import *
import matplotlib.pyplot as plt
from utils.magaox import make_magaox_bump_mask
from utils.apodized_mask_generator import make_magaox_piaa_lyot_stop
import csv
import scipy.ndimage as ndi

datadir = '/media/armand/EXTRA/Sync/Master-Leiden/Y1/First Project/PIAACMC/magaox_data'


def prep_map(image: np.ndarray) -> np.ndarray:
    out = np.array(image, dtype=float)
    median = np.nanmedian(out)
    if np.isfinite(median):
        out = out - median
    out[~np.isfinite(out)] = 0.0
    return out

def measure_shift_integer(
    reference_map: np.ndarray, target_map: np.ndarray,
) -> tuple[int, int, float, np.ndarray]:
    """Cross-correlate two moment-0 maps via FFT convolution.

    Returns ``(lag_x, lag_y, peak_value, corr)`` where *lag_?* are the
    integer pixel lags of the correlation peak relative to centre and
    *peak_value* is the maximum correlation coefficient.
    """

    ref = prep_map(reference_map)
    tgt = prep_map(target_map)
    corr = fftconvolve(ref, tgt[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    center_y, center_x = (s // 2 for s in corr.shape)
    lag_y = int(peak_y - center_y)
    lag_x = int(peak_x - center_x)
    return lag_x, lag_y, float(corr[peak_y, peak_x]), corr