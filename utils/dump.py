
from hcipy import Field, Wavefront, get_refractive_index


self.pupil_diameter=8.6e-3 #m

        if self.method=='PIAA':
        radial_d0=self.piaa0_csv['Radial distance (m)'].values
        radial_d1=self.piaa1_csv['Radial distance (m)'].values
        profile0=-self.piaa0_csv['Cut depth (m)'].values #surface profile is negative of cut depth
        profile1=-self.piaa1_csv['Cut depth (m)'].values
        ###
        min_0=np.min(radial_d0)
        min_1=np.min(radial_d1)
        profile0=profile0-min_0
        profile1=profile1-min_1
        ###

            ref_ind_lenses=get_refractive_index('CAF2')
        d_pupil_piaa0=37.2e-3 #m by design (ideal)
        d_piaa0_piaa1=60.2e-3 #m by design (ideal)
        prop_pupil_piaa0=FresnelPropagator(self.aperture.grid,d_pupil_piaa0) #AngularSpectrumPropagator
        prop_piaa0_piaa1=FresnelPropagator(self.aperture.grid,d_piaa0_piaa1)
        prop_piaa1_pupil=FresnelPropagator(self.aperture.grid,d_piaa0_piaa1+d_pupil_piaa0)

        prop_LS_piaa1=FresnelPropagator(self.aperture.grid,d_pupil_piaa0)
        prop_piaa1_piaa0=FresnelPropagator(self.aperture.grid,d_piaa0_piaa1)
        prop_piaa0_LS=FresnelPropagator(self.aperture.grid,d_piaa0_piaa1+d_pupil_piaa0)

        self.prop_pupil_piaa0=prop_pupil_piaa0
        self.prop_piaa0_piaa1=prop_piaa0_piaa1
        self.prop_piaa1_pupil=prop_piaa1_pupil
        self.prop_LS_piaa1=prop_LS_piaa1
        self.prop_piaa1_piaa0=prop_piaa1_piaa0
        self.prop_piaa0_LS=prop_piaa0_LS

        sag0=Field(np.interp(self.aperture.grid.as_('polar').r,radial_d0,-profile0),self.aperture.grid)
        sag1=Field(np.interp(self.aperture.grid.as_('polar').r,radial_d1,-profile1),self.aperture.grid)

        piaa0 = SurfaceApodizer(-sag0, ref_ind_lenses)
        piaa1 = SurfaceApodizer(-sag1, ref_ind_lenses)
        self.piaa0=piaa0
        self.piaa1=piaa1
[1:47 PM]APODIZE WITH PIAA

apodized_wf_dict={}
        # apodized_wfs=[]
        current_wf=Wavefront(self.aperture*np.exp(2j*np.pi*self.separation*self.aperture.grid.x/self.pupil_diameter))
        index=-1
        for wl in self.wavelengths:
            index+=1
            current_wf.wavelength=wl
            current_wf.total_power=1
            wf_piaa0 = self.piaa0.forward(self.prop_pupil_piaa0.forward(current_wf))
            wf_piaa1 = self.piaa1.forward(self.prop_piaa0_piaa1.forward(wf_piaa0))
            apodized_wf_at_pupil = self.prop_piaa1_pupil.backward(wf_piaa1)
            # apodized_wf_at_pupil.total_power=1
            apodized_wf_dict[index]=apodized_wf_at_pupil
            # apodized_wfs[index]=apodized_wf_at_pupil
        return apodized_wf_dict
    

n_rings=10
wl=0.908e-6
F_number=69
SR=wl*F_number
expected_max_radius_norm=0.75
expected_heights=np.array([1.26196831, 0.94074998, 0.77836148, 0.6366583 , 0.7739299 , 3.19782967, 0.85539274, 0.79005693, 0.88346143, 0.87813155]) #microns


def make_rings(self):
        '''

        '''

        masks_in=dict()
        masks_out=dict()

        for i in range(len(self.radii)-1,-1,-1):
            if i==0:
                ca1=make_circular_aperture(self.radii[i]*2)
                supersampled_ca1=evaluate_supersampled(ca1,self.cor_grid,self.supersample_factor)
                masks_in[i]=1-supersampled_ca1
            else:
                ca1=make_circular_aperture(self.radii[i]*2)
                ca2=make_circular_aperture(self.radii[i-1]*2)
                supersampled_ca1=evaluate_supersampled(ca1,self.cor_grid,self.supersample_factor)
                supersampled_ca2=evaluate_supersampled(ca2,self.cor_grid,self.supersample_factor)
                masks_in[i]=1-supersampled_ca1+supersampled_ca2
            masks_out[i]=1-masks_in[i]

        return masks_out


def make_FPM_w_rings(self,thicknesses):
    '''
    Input thicknesses needs to be given in MICRONS.
    Returns FPM apodizer, where the surface sag is in METERS.
    '''

    rings=self.make_rings()

    for i in range(len(self.radii)):
        if i==0:
            FPM=rings[0]*thicknesses[0]*1e-6
        else:
            FPM+=rings[i]*thicknesses[i]*1e-6

    FPM_apodizer=SurfaceApodizer(FPM,self.n_mask)

    return FPM_apodizer


# --- FPM ring design: the attributes the coworker's class would have held ---
# CONFIRM: radii (evenly spaced to 0.75 lambda/D) and n_mask (FPM material)
radii = np.linspace(expected_max_radius_norm / n_rings, expected_max_radius_norm, n_rings) * SR  # metres
cor_grid = hp.make_focal_grid(q=100, num_airy=2, spatial_resolution=SR) 
n_mask = hp.get_refractive_index('SILICA')

def make_rings(radii, cor_grid, supersample_factor):
        '''

        '''

        masks_in=dict()
        masks_out=dict()

        for i in range(len(radii)-1,-1,-1):
            if i==0:
                ca1=make_circular_aperture(radii[i]*2)
                supersampled_ca1=evaluate_supersampled(ca1,cor_grid,supersample_factor)
                masks_in[i]=1-supersampled_ca1
            else:
                ca1=make_circular_aperture(radii[i]*2)
                ca2=make_circular_aperture(radii[i-1]*2)
                supersampled_ca1=evaluate_supersampled(ca1,cor_grid,supersample_factor)
                supersampled_ca2=evaluate_supersampled(ca2,cor_grid,supersample_factor)
                masks_in[i]=1-supersampled_ca1+supersampled_ca2
            masks_out[i]=1-masks_in[i]

        return masks_out

def make_FPM_w_rings(radii, cor_grid, supersample_factor, n_mask, thicknesses):
    '''
    Input thicknesses needs to be given in MICRONS.
    Returns FPM apodizer, where the surface sag is in METERS.
    '''

    rings=make_rings(radii, cor_grid, supersample_factor)

    for i in range(len(radii)):
        if i==0:
            FPM=rings[0]*thicknesses[0]*1e-6
        else:
            FPM+=rings[i]*thicknesses[i]*1e-6

    FPM_apodizer=SurfaceApodizer(FPM,n_mask)

    return FPM_apodizer

supersample_factor = 8
FPM_apodizer = make_FPM_w_rings(radii, cor_grid, supersample_factor, n_mask, expected_heights)

# apply the FPM to the apodized pupil at the F#=69 focal plane (gives the cell below its wf_fpm)
prop_piaa1_focal = hp.FraunhoferPropagator(piaa_pupil_grid, cor_grid, focal_length=F_number * pupil_diameter)
wf_fpm = FPM_apodizer.forward(prop_piaa1_focal.forward(wf_apodized))

# quick check of the FPM surface
hp.imshow_field(FPM_apodizer.surface_sag * 1e6, cmap='inferno', grid_units=1e-6)
plt.colorbar(label='FPM surface height [um]')
plt.title('FPM surface (from rings)')
plt.show()

