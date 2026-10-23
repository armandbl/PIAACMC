from hcipy import make_obstructed_circular_aperture, make_circular_aperture, make_spider_infinite, Field, make_regular_polygon_aperture, make_elliptical_aperture
import numpy as np

__all__ = [
    'make_magaox_bump_mask',
    'make_magaox_large_lyot_stop',
	'make_gmt_lyot_aperture'
]

def make_magaox_bump_mask(normalized=False, with_spiders=True, aperture_scale="telescope"):
    '''Make the Magellan bump mask.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be normalized by the telescope diameter such that the size of the pupil_grid is 1.
    with_spiders: boolean
        If this is False, the spiders will be left out.
    aperture_scale : string
        telescope : Scale the aperture dimensions to match the physical size of the telescope
		pupil: Scale the aperture dimensions to match the physical size of the pupil 
        Scale factor for the aperture dimensions.
        normalized: Same as `normalized=True`. The outer diameter will be scaled to 1.

    Returns
    -------
    Field generator
        The Magellan aperture.
    '''

    if normalized or aperture_scale == "normalized":
        pupil_diameter = 1.0
    elif aperture_scale == "telescope":
        pupil_diameter = 6.5
    elif aperture_scale == "pupil":
        pupil_diameter = 9e-3
    else:
        raise ValueError('Invalid aperture_scale. Must be "telescope", "pupil", or "normalized".')
    
    magnification_factor = pupil_diameter / 9e-3

    mask_inner = 2.79e-3 * magnification_factor # meter
    mask_outer = 8.604e-3 * magnification_factor # meter

    bump_mask_diameter = 0.5742e-3 * magnification_factor

    bump_mask_pos = np.array([2.853e-3, -0.6705e-3]) * magnification_factor

    radius = np.hypot(bump_mask_pos[0], bump_mask_pos[1])
    theta = np.arctan2(bump_mask_pos[1], bump_mask_pos[0]) - np.deg2rad(25.2) #+ np.pi/2 # Adjusted bump angle to better center it on spider
    bump_mask_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)])

    spider_width1 = 0.1917e-3 * magnification_factor  # meter 
    spider_width2 = 0.1917e-3  * magnification_factor  # meter
    central_obscuration_ratio = mask_inner / mask_outer 
    spider_offset = np.array([0.0, 0.34 / 6.5]) * pupil_diameter
    spider1_offset_correction = np.array([0.0, -0.0185 / 6.5]) * pupil_diameter
    spider2_offset_correction = np.array([0.0, 0.0200 / 6.5]) * pupil_diameter
    spider3_offset_correction = np.array([0.0, -0.0220 / 6.5]) * pupil_diameter
    spider4_offset_correction = np.array([0.0, 0.0230 / 6.5]) *  pupil_diameter
    obstructed_aperture = make_obstructed_circular_aperture(mask_outer, central_obscuration_ratio)
    bump_mask = make_circular_aperture(bump_mask_diameter, center=bump_mask_pos)  # Generate bump cover for the MagAO-X DM
    
    if not with_spiders:
        return obstructed_aperture
	
	# spider offsets corrections based on bumpMask fits file from J. Males.
    spider1 = make_spider_infinite(spider_offset - spider1_offset_correction, 45.0, spider_width1)
    spider2 = make_spider_infinite(-spider_offset - spider2_offset_correction, -45.0, spider_width1)
    spider3 = make_spider_infinite(-spider_offset + spider3_offset_correction, 45.0 + 180.0, spider_width2)
    spider4 = make_spider_infinite(spider_offset + spider4_offset_correction, -45.0 + 180.0, spider_width2)

    def func(grid):
        return obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid) * (1 - bump_mask(grid))
    
    return func
