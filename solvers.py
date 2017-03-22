import numpy as np

from astropy import units as u, constants as c

def anamorphic(alpha, beta):
    '''
    compute anamorphic factor for incident and diffracted angle
    '''
    return np.cos(alpha) / np.cos(beta)

class AngularDispersion(object):
    '''
    compute diffracted angle (beta) corresponding to some incident angle (alpha)
        and wavelength---and vice-versa---calibrated to beta0 and lam_blaze

    after solving dbeta/dlam for beta, you get
        beta = -arcsin(sin(alpha) - k lam)
            and
        lam = (sin(beta) + sin(alpha)) / k

    k is calibrated from known diffracted angle (beta0) of blaze wavelength (lam_blaze)
        k [AA-1] = (sin(beta0) + sin(alpha) / lam_blaze)


    also included are some convenience methods for, e.g.:
        - dbeta/dlam
        - dlam/dbeta
    '''
    def __init__(self, alpha, k, beta0, lam_blaze):
        self.alpha = alpha
        self.k = k
        self.beta0 = beta0
        self.lam_blaze = lam_blaze

    @classmethod
    def from_blaze(cls, alpha, beta0, lam_blaze):
        k = ((np.sin(beta0) + np.sin(alpha)) / lam_blaze).to(1 / u.AA)

        return cls(alpha, k, beta0, lam_blaze)

    def beta_to_lam(self, beta):
        '''
        convert diffracted angle to wavelength
        '''

        lam = (np.sin(beta) + np.sin(self.alpha)) / self.k
        return lam

    def lam_to_beta(self, lam):
        '''
        convert wavelength to its diffracted angle
        '''

        beta = -np.arcsin(np.sin(self.alpha) - self.k * lam)
        return beta.to(u.deg)

    def _eval_dbeta_dlam(self, beta, lam):
        res = (np.sin(self.alpha) + np.sin(beta)) / (lam * np.sin(beta))
        return res

    def dbeta_dlam(self, beta=None, lam=None):

        # only one of beta and lam can be specified
        if (((beta is None) and (lam is None)) or
            ((beta is not None) and (lam is not None))):

            raise ValueError('must specify either beta or lambda, not both')

        elif beta is not None:
            # beta is given, so solve for lam
            lam = self.beta_to_lam(beta)

        elif lam is not None:
            # lambda is given, so solve for beta
            beta = self.lam_to_beta(lam)

        res = self._eval_dbeta_dlam(beta, lam)
        res = res.to(u.deg / u.AA, equivalencies=u.dimensionless_angles())
        return res

    def dlam_dbeta(beta=None, lam=None):
        # only one of beta and lam can be specified
        if (((beta is None) and (lam is None)) or
            ((beta is not None) and (lam is not None))):

            raise ValueError('must specify either beta or lambda, not both')

        elif beta is not None:
            # beta is given, so solve for lam
            lam = self.beta_to_lam(beta)

        elif lam is not None:
            # lambda is given, so solve for beta
            beta = self.lam_to_beta(lam)

        res = 1. / self._eval_dbeta_dlam(beta, lam)
        res = res.to(u.AA / u.rad, equivalencies=u.dimensionless_angles())
        return res


class LinearDispersion(object):
    '''
    compute position on detector corresponding to some diffracted wavelength

    zero-position on chip is assumed to be at blaze-wavelength (angle beta0)
        specified in self.ang_disp, and in the middle of pixel zero
    '''

    def __init__(self, ang_disp, lpix, npix, f_cam):
        self.ang_disp = ang_disp
        self.lpix = lpix
        self.npix = npix
        self.halfnpix = (npix - 1) // 2
        self.f_cam = f_cam

        self.pix_array = np.linspace(-self.halfnpix, self.halfnpix, npix)
        self.pix_borders_array = np.linspace(
            -self.halfnpix - 0.5, self.halfnpix + 0.5, npix + 1)

    def p_to_x(self, p):
        return p * self.lpix

    def x_to_p(self, x):
        return (x / lpix).to('').value

    def x_to_beta(self, x):
        dbeta = np.arcsin(x / self.f_cam).to(u.rad, equivalencies=u.dimensionless_angles())
        beta = self.ang_disp.beta0 + dbeta

        return beta

    def beta_to_x(self, beta):
        x = self.f_cam * np.sin(beta - self.ang_disp.beta0)
        return x.to(u.mm)

    def p_to_beta(self, p):
        # pixel number to linear position
        x = self.p_to_x(p)
        beta = self.x_to_beta(x)

        return beta

    def beta_to_p(self, beta):
        x = self.beta_to_x(beta)
        p = self.x_to_p(x)
        return p

    def lam_to_x(self, lam):
        '''
        transform diffracted wavelength into (physical, linear) position on detector
        '''

        beta = self.ang_disp.lam_to_beta(beta)
        x = self.beta_to_x(beta)

        return x

    def x_to_lam(self, x):
        beta = self.ang_disp.beta0 + np.arcsin(x / self.f_cam)
        lam = self.ang_disp.beta_to_lam(beta)

        return lam

    def lam_to_p(self, lam):
        '''
        transform diffracted wavelength into (pixel-wise) position on detector
        '''

        x = self.lam_to_x(lam, return_beta=True)
        p = self.x_to_p(x)

        return p.to('').value

    def p_to_lam(self, p):
        x = self.p_to_x(p)
        lam = self.x_to_lam(x)

        return lam

    def lam_is_on_chip(self, lam):
        '''
        returns True if specified wavelength is on chip, False if not
        '''

        p = self.lam_to_p(lam)
        on_chip = (p >= -self.halfnpix) * (p <= self.halfnpix)

        return on_chip

    @property
    def lam_range(self):
        '''
        find wavelength range on chip
        '''

        p_range = np.array([-self.halfnpix, self.halfnpix])
        lam_range = self.p_to_lam(p_range)

        return lam_range

    @property
    def lam_pix_array(self):
        lam = self.p_to_lam(self.pix_array)
        return lam

    @property
    def dlam_pix_array(self):
        lam = self.p_to_lam(self.pix_borders_array)
        dl = lam[1:] - lam[:-1]
        return dl

class SpectrographEfficiency(object):
    '''
    Efficiency of a spectrograph is defined as

                     I(m, lam)
    e(m, lam) = ---------------------
                 SUM_m' [I(m', lam)]

    e(m, lam) = J(m, lam) B(m, lam)

        J(nu', N) = (sin(N nu') / (N sin(nu')))^2   # Interference function

            nu'(lam, beta) = (pi sig / lam) (sin(alpha) + sin(beta))

        B(nu) = ((sin(nu)) / nu)^2   # Blaze function

            nu(lam, beta) = (pi sigface / lam) (sin(alpha) + sin(beta))

                sigface = sig cos(delta)
    '''

    def __init__(self, sig, delta, alpha, cam_coll_angle):
        self.sig = sig
        self.delta = delta
        self.alpha = alpha

        self.sigface = sig * np.cos(delta)

        self.theta = 0.5 * cam_coll_angle
        self.sigface_prime = self.sig * np.cos(alpha) / np.cos(self.theta)

        self.l_at_m = lambda m, beta: ((self.sig / m) * (
            np.sin(alpha) + np.sin(beta))).to(u.AA)

    def nu(self, lam, beta):
        f = (np.pi * self.sigface / lam).to('')
        return f * (np.sin(self.alpha) + np.sin(beta))

    def nu_prime(self):
        f = (np.pi * self.sig / lam).to('')
        return f * (np.sin(self.alpha) + np.sin(beta))

    def J(m, beta):
        '''
        interference function
        '''

        # solve grating equation (with fixed alpha & beta)
        # for wavelength at a particular order m

    def B(m, beta):
        '''
        blaze function
        '''
