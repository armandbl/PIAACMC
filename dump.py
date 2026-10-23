
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