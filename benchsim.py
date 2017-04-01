import numpy as np
from astropy import units as u, constants as c

from solvers import *

class Spectrograph(object):
    def __init__(self, grating_props, spectrograph_props, orientation_props,
                 slit_props, det_props, telescope_props,
                 autosolve=True):
        '''
        params:
         - grating_props (dict):
             - sig: groove separation
             - delta: blaze angle
             - rf_tr: reflection or transmission grating?
         - orientation_props (dict):
             - alpha: grating normal to incident ray
             - m: diffraction order
             - lam_blaze: desired central (blaze) wavelength
         - spectrograph_props (dict)
             - f_coll: collimator focal length
             - coll_beam_width: width of a single fiber's collimated beam
             - f_cam: camera focal length
         - slit_props (dict)
             - w0_phys: physical slit width
         - det_props (dict)
             - npix: number of pixels in spectral dimension
             - lpix: size of pixels (on a side)
         - telescope_props (dict)
             - D_scope: telescope diameter
             - l_scope: telescope focal length
        '''

        for d in [grating_props, spectrograph_props, orientation_props,
                  slit_props, det_props, telescope_props]:
            for k in d:
                setattr(self, k, d[k])

        self.lpix_unbinned = self.lpix
        self.lpix = getattr(self, 'binning', 1) * self.lpix_unbinned
        self.npix = self.npix / getattr(self, 'binning', 1)

        if self.rf_tr == 'rf':
            self.sgn = 1.
        elif self.rf_tr == 'tr':
            self.sgn = -1.
        else:
            raise ValueError('Invalid grating type: reflection (rf) or transmission (tr)')

        # solve for angle corresponding to central (blaze) wavelength
        if autosolve:
            self.sol = self._solve()

    def _solve(self):
        '''
        wraps around solvers for angular and linear dispersion
        '''
        sol = {}

        # diffraction angle of blaze wavelength
        sol['beta0'] = self._solve_beta0().to(u.deg)

        # angle corresponding to each detector pixel
        # (requires knowing beta0, lam_ctr, dbeta/dl)

        # use blaze wavelength and its diffraction angle to figure out angular
        # dispersion (relationship between diffracted angle and wavelength)
        sol['ang_disp'] = AngularDispersion.from_blaze(
            alpha=self.alpha, beta0=sol['beta0'], lam_blaze=self.lam_blaze)
        sol['lin_disp'] = LinearDispersion(
            ang_disp=sol['ang_disp'], lpix=self.lpix, npix=self.npix, f_cam=self.f_cam)

        return sol

    def _solve_beta0(self):
        '''
        use grating equation to solve for diffraction angle of blaze wavelength
        '''
        beta0 = np.arcsin(self.m * self.lam_blaze / self.sig - np.sin(self.alpha))
        return beta0

    # relevant properties of wavelength solution
    @property
    def ang_disp(self):
        return self.sol['ang_disp']

    @property
    def lin_disp(self):
        return self.sol['lin_disp']

    @property
    def beta0(self):
        return self.sol['beta0']

    def lam_range_plot(self, ax):
        lr = self.lam_range
        for l in lr:
            ax.axvline(l.value, c='k', linewidth=0.5)

        ptp = (lr[1] - lr[0]).value
        ax.set_xlim([lr[0].value - 0.6 * ptp, lr[1].value + 0.6 * ptp])
        return ax

    @property
    def lam_range(self):
        '''
        wavelength limits
        '''
        return self.lin_disp.lam_range

    @property
    def wavelengths(self):
        '''
        wavelengths for all detector pixels
        '''
        return self.lin_disp.lam_pix_array

    @property
    def dwavelengths(self):
        '''
        wavelength difference subtended by a pixel on detector
        '''
        return self.lin_disp.dlam_pix_array

    @property
    def R(self):
        '''
        spectral resolution for all detector pixels (i.e., wavelengths)

        R = lam / w_lam
        '''
        lam = self.wavelengths
        R = (lam / self.w_reim_spec).to('')

        return R.to('')

    @property
    def R_vel(self):
        '''
        Spectral resolution (velocity)
        '''
        R_v = (c.c / self.R)
        return R_v.to(u.km / u.s)

    @property
    def anam(self):
        '''
        anamorphic factor for all detector pixels
        '''
        beta = self.lin_disp.p_to_beta(
            self.lin_disp.pix_array)

        return anamorphic(self.alpha, beta)

    @property
    def w0_ang(self):
        '''
        angular slit width (radians) on the sky
        '''
        return (self.w0_phys / self.l_scope).to(u.rad, equivalencies=u.dimensionless_angles())

    @property
    def w_reim_spec(self):
        '''
        spectral width of the reimaged slit for all detector pixels

        w_lam = r / (dbeta / dlam) (f_coll / w0_phys)
        '''
        lam = self.wavelengths
        r = self.anam
        gamma = self.ang_disp.dbeta_dlam(lam=lam)

        w1 = (r / gamma) * (self.w0_phys / self.f_coll).to(
            u.rad, equivalencies=u.dimensionless_angles())
        return w1.to(u.AA)

    @property
    def fiber_grasp(self):
        return (np.pi * (self.w0_ang / 2.)**2.).to(u.arcsec**2)

    @property
    def cam_coll_angle(self):
        return self.alpha - self.beta0

    @property
    def sig_face(self):
        '''
        facet size
        '''
        return self.sig * np.cos(self.delta)

    @property
    def sig_face_eff(self):
        '''
        effective (exposed) facet size, due to shadowing
        '''
        return self.sig * (np.cos(self.alpha) / np.cos(0.5 * self.cam_coll_angle))

    @property
    def N_facets(self):
        '''
        number of diffractive facets exposed to a given fiber's collimated beam
        '''
        N_facets = self.coll_beam_width / (self.sig * np.sin(self.alpha))

        return int(N_facets.value)


if __name__ == '__main__':
    grating_props = {
        'sig': 1./316 * u.mm, 'delta': 63.4 * u.deg, 'rf_tr': 'rf'}
    orientation_props = {
        'alpha': 68.234 * u.deg, 'm': 8, 'lam_blaze': 7000. * u.AA}
    spectrograph_props = {
        'f_coll': 776. * u.mm, 'f_cam': 285. * u.mm}
    slit_props = {
        'w0_phys': .100 * u.mm}
    det_props = {
        'npix': 4000, 'lpix': .012 * u.mm, 'RN': 3.4 * u.electron}
    telescope_props = {
        'D_scope': 3500. * u.mm, 'l_scope': 22004 * u.mm, 'A_scope': 9.6 * u.m**2}

    BenchSetup = Spectrograph(
        grating_props=grating_props, spectrograph_props=spectrograph_props,
        orientation_props=orientation_props, det_props=det_props,
        slit_props=slit_props, telescope_props=telescope_props)
