# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:00:03 2024

@author: warre

Create the apodized Lyot stop for use with PIAA
"""

from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

def make_unstructured_apodized_magellan_aperture(pupil_diameter, central_obstruction_diameter,spider_width1, spider_width2, rotation, spider_offset_magnitude, delta_spider_start_rotation, delta_spider_end_rotation_1, delta_spider_end_rotation_2, delta_spider_end_rotation_3, delta_spider_end_rotation_4):
    '''Make the apodized Magellan aperture.
    
    Parameters
    ----------
    pupil_diameter 
    central_obstruction_diameter
    
    spider_width1 - width at center of mask
    spider_width2 - width at edge of mask
    
    rotation - rotates entire mask geometry; orient w.r.t. pupil
    
    spider_offset_magnitude - sets distance from origin to spider intersection point
    delta_spider_start_rotation - sets rotation of spider intersection point w.r.t. pupil geometry
    
    delta_spider_end_rotation_1 - It's overkill to have four params for this
    delta_spider_end_rotation_2 - but it was the easiest way to match apodized pupil geometry 
    delta_spider_end_rotation_3
    delta_spider_end_rotation_4
    
    Returns
    -------
    Field generator
    	The apodized Magellan aperture.
    '''
    central_obstruction_ratio = central_obstruction_diameter / pupil_diameter
        
    spider_offset = [np.cos(rotation+np.pi/4 + delta_spider_start_rotation)*spider_offset_magnitude, np.sin(rotation+np.pi/4 + delta_spider_start_rotation)*spider_offset_magnitude]  # meter
    spider_offset = np.array(spider_offset)
    
    mirror_edge1 = (np.cos(rotation + delta_spider_end_rotation_1)*pupil_diameter, np.sin(rotation + delta_spider_end_rotation_1)*pupil_diameter)
    mirror_edge2 = (np.cos(rotation + delta_spider_end_rotation_2)*pupil_diameter, np.sin(rotation + delta_spider_end_rotation_2)*pupil_diameter) 
    mirror_edge3 = (np.cos(rotation + delta_spider_end_rotation_3)*pupil_diameter, np.sin(rotation + delta_spider_end_rotation_3)*pupil_diameter)
    mirror_edge4 = (np.cos(rotation + delta_spider_end_rotation_4)*pupil_diameter, np.sin(rotation + delta_spider_end_rotation_4)*pupil_diameter)

    obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obstruction_ratio)
       
    spider1 = make_spider(spider_offset, mirror_edge1, spider_width1)
    spider2 = make_spider(spider_offset, mirror_edge2, spider_width1)
    spider3 = make_spider(-spider_offset, mirror_edge3, spider_width2)
    spider4 = make_spider(-spider_offset, mirror_edge4, spider_width2)
    
    spider1_verticies = find_spider_verticies(spider_offset,mirror_edge1,spider_width1,spider_width2)
    apod_spider1 = make_obstruction(make_irregular_polygon_aperture(spider1_verticies))
    
    spider2_verticies = find_spider_verticies(spider_offset,mirror_edge2,spider_width1,spider_width2)
    apod_spider2 = make_obstruction(make_irregular_polygon_aperture(spider2_verticies))
    
    spider3_verticies = find_spider_verticies(-spider_offset,mirror_edge3,spider_width1,spider_width2)
    apod_spider3 = make_obstruction(make_irregular_polygon_aperture(spider3_verticies))
    
    spider4_verticies = find_spider_verticies(-spider_offset,mirror_edge4,spider_width1,spider_width2)
    apod_spider4 = make_obstruction(make_irregular_polygon_aperture(spider4_verticies))
        
    def func(grid):
        return obstructed_aperture(grid) * apod_spider1(grid) * apod_spider2(grid) * apod_spider3(grid) * apod_spider4(grid)
    
    return func

def find_spider_verticies(spider_offset,mirror_edge,spider_width1,spider_width2=None):
    '''Return the four coordinate points that define a trapezoidal spider
    These create spider using make_irregular_polygon_aperture
    
    Parameters
    ----------
    spider_offset: coordinates of 2D spider at center
    mirror_edge: coordinates of 2D spider at edge
    spider_width1: thickness of spider at center
    spider_width2: thickness of spider at edge
    
    Returns
    -------
    p00,p01,p11,p10: coordinates of four spider verticies
    '''
    if not spider_width2:
        spider_width2 = spider_width1.copy()
    delta = np.array(mirror_edge) - np.array(spider_offset)
    spider_angle = np.arctan2(delta[1], delta[0])
    
    delta_x = np.cos(spider_angle + np.pi/2)
    delta_y = np.sin(spider_angle + np.pi/2)
    
    p00 = [spider_offset[0] + (spider_width1/2)*delta_x, spider_offset[1] + (spider_width1/2)*delta_y]
    p01 = [spider_offset[0] - (spider_width1/2)*delta_x, spider_offset[1] - (spider_width1/2)*delta_y]    
    p10 = [mirror_edge[0] + (spider_width2/2)*delta_x, mirror_edge[1] + (spider_width2/2)*delta_y]
    p11 = [mirror_edge[0] - (spider_width2/2)*delta_x, mirror_edge[1] - (spider_width2/2)*delta_y]    

    return p00,p01,p11,p10

def make_bump(bumpDiameterX,bumpDiameterY, bumpRoe, bumpTheta, bumpTheta_2, rotation):
    '''Create a bump spot to add to the apodized pupil
    This really should be done in make_unstructured_apodized_magellan_aperture
    but the optimizer wasn't finding good solutions and it was better
    to optimize for the bump location separately'
    '''
    bumpDiameters = [bumpDiameterX,bumpDiameterY]
    bumpX = bumpRoe * np.cos(rotation + bumpTheta)
    bumpY = bumpRoe * np.sin(rotation + bumpTheta)
    bumpMask = make_obstruction(make_elliptical_aperture(bumpDiameters, center=[bumpX,bumpY], angle=rotation+bumpTheta_2))
    def func(grid):
        return bumpMask(grid)
    return func

def make_bump_mask(mask,bump):
    def func(grid):
        return mask(grid) * bump(grid)
    return func

def make_magaox_piaa_lyot_stop():
    noBump_params = [7.40778111e-03, 2.81500284e-03, 1.18611577e-04, 4.37517483e-04,
        2.83953807e+00, 3.07370887e-04, 7.30691998e-04, 1.94499730e-03,
        1.46412801e+00, 3.14131871e+00, 4.61830133e+00] #2.83953807e+00 this was element 4

    bump_params = [ 0.00067428,  0.00063527,  0.00228133,  0.07161865, -0.00016683]

    #Increase spider width for alignment tolerance
    spider_scaling = 2
    noBump_params[2]=noBump_params[2]*spider_scaling
    noBump_params[3]=noBump_params[3]*spider_scaling

    #Enlarge bump and change shape from ellipse to circle.
    #This deviated slightly from the optimized solution but was much easier to fabricate
    bump_scaling = 1.5
    bump_params[0] = bump_params[0]*bump_scaling
    bump_params[1] = bump_params[0]

    # Make grid
    # pupil_grid = make_pupil_grid(4096,Dgrid)

    #Make mask
    mask = make_unstructured_apodized_magellan_aperture(noBump_params[0],noBump_params[1],noBump_params[2],noBump_params[3],noBump_params[4],noBump_params[5],noBump_params[6],noBump_params[7],noBump_params[8],noBump_params[9],noBump_params[10])
    bump = make_bump(bump_params[0],bump_params[1],bump_params[2],bump_params[3],bump_params[4],noBump_params[4])

    return make_bump_mask(mask,bump)

#mm = 1e-3
#Dgrid = 9.5 * mm

#imshow_field(bump_mask(pupil_grid))