import numpy as np
from astropy import units as u, constants as c

class Spectrograph(object):
    def __init__(self, grating_props, spectrograph_props, slit_props, det_props, telescope_props):
        '''
        params:
         - grating_props (dict):
             - sig: groove separation
             - delta: blaze angle
             - rf_tr: reflection or transmission grating?
         - spectrograph_props (dict)
             - f_coll: collimator focal length
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

        for d in [grating_props, spectrograph_props, slit_props, det_props, telescope_props]:
            for k in d:
                setattr(self, k, d[k])

        if self.rf_tr == 'rf':
            self.sgn = 1.
        elif self.rf_tr == 'tr':
            self.sgn = -1.
        else:
            raise ValueError('Invalid grating type: reflection (rf) or transmission (tr)')

    def set_littrow(self):
        self.set_angles(self.delta, self.delta)

    def set_angles(self, alpha, beta):
        setattr(self, 'alpha', alpha)
        setattr(self, 'beta', beta)

    def set_order(self, m):
        if type(m) is not int:
            raise ValueError('order must be integer')
        setattr(self, 'm', m)

    def lam_range_plot(self, ax):
        lr = self.lam_range
        for l in lr:
            ax.axvline(l.value, c='k', linewidth=0.5)

        ptp = (lr[1] - lr[0]).value
        ax.set_xlim([lr[0].value - 0.6 * ptp, lr[1].value + 0.6 * ptp])
        return ax

    def eff(self, l):
        pass

    @property
    def w0_ang(self):
        '''
        angular slit width (radians)
        '''
        return (self.w0_phys / self.l_scope).to(u.rad, equivalencies=u.dimensionless_angles())

    @property
    def fiber_omega(self):
        return (np.pi * (self.w0_ang / 2.)**2.).to(u.arcsec**2)

    @property
    def lam_ctr(self):
        l = self.sig * (np.sin(self.beta) + self.sgn * np.sin(self.alpha)) / self.m
        return l.to(u.AA)

    @property
    def lam_range(self):
        '''
        wavelength range
        '''
        lam_ctr = self.lam_ctr

        # dx: num pixels available on each side of lam_ctr
        dx = (self.npix / 2.) * self.lpix

        # kappa: linear dispersion = (dx / dlam)
        dlam = (dx / self.kappa).to(u.AA)
        lam_ctr = self.lam_ctr
        lam_llim = lam_ctr - dlam
        lam_ulim = lam_ctr + dlam

        return (lam_llim, lam_ulim)

    @property
    def wavelength(self):
        ll, lu = self.lam_range
        return np.arange(ll.value, lu.value, self.dlam_pix.value) * u.AA

    @property
    def dlam_pix(self):
        '''
        wavelength change corresponding to one pixel
        '''
        return (self.lpix / self.kappa).to(u.AA)

    @property
    def anam(self):
        '''
        anamorphic factor
        '''
        r = np.cos(self.alpha) / np.cos(self.beta)
        return r

    @property
    def gamma(self):
        '''
        angular dispersion
        '''
        gamma = self.m / (self.sig * np.cos(self.beta))
        return gamma

    @property
    def kappa(self):
        '''
        linear dispersion
        '''
        kappa = self.f_cam * self.gamma
        return kappa

    @property
    def specres(self):
        '''
        spectral resolution at some wavelength(s)
        '''
        R = (self.f_coll / self.w0_phys) * (np.sin(self.beta) + self.sgn * np.sin(self.alpha)) / np.cos(self.alpha)
        return R

    @property
    def specres_v(self):
        return (c.c / self.specres).to(u.km / u.s)

    @property
    def theta(self):
        '''
        camera-collimator angle
        '''
        return self.alpha - self.beta

    @property
    def lam_blaze(self):
        lam_blaze = (2. * self.sig / self.m) * np.sin(self.delta) * np.cos(0.5 * self.theta)
        return lam_blaze.to(u.AA)

    @property
    def cam_coll_angle(self):
        return self.alpha - self.sgn * self.beta
