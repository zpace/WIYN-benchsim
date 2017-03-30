import numpy as np
from astropy import constants as c, units as u
from astropy.io import fits

from scipy.interpolate import RegularGridInterpolator
from scipy import stats

import os, sys
if os.environ['MANGA_CONFIG_LOC'] not in sys.path:
    sys.path.append(os.environ['MANGA_CONFIG_LOC'])

import mangarc

if mangarc.tools_loc not in sys.path:
    sys.path.append(mangarc.tools_loc)

# personal
import manga_tools as m

def Flam2Fphot(drp, lamlims):
    lam = (drp['WAVE'].data * u.AA)

    illims = np.searchsorted(a=lam, v=lamlims[[0, -1]])

    lamgood = (lam >= lam[illims[0] - 1]) * (lam <= lam[illims[1] + 1])

    Flam_unit = u.Unit('1e-17 erg s-1 cm-2 AA-1')
    Flam = drp['FLUX'].data
    Flam[Flam < 0.] = 0.
    Flam = Flam * Flam_unit

    lam, Flam = lam[lamgood], Flam[lamgood, ...]

    # photon energy
    e_phot = lam.to(u.erg, equivalencies=u.spectral()) / u.photon

    Fphot = (Flam / e_phot[..., None, None]).to('photon s-1 cm-2 AA-1')

    return lam, Fphot

class IFUCubeObserver(object):
    def __init__(self, lam, Fphot, lspax):
        '''
        feed in a datacube (lam + Fphot), plus I & J grid scale

        computes an interpolated datacube (units of flam / spax, which assumes some spaxel size)
        '''
        self.II, self.JJ = tuple(map(lambda s: np.linspace(0., s - 1, s), Fphot.shape[1:]))
        self.Fphot_cube_interp = RegularGridInterpolator(
            points=(lam.value, self.II, self.JJ), values=Fphot, method='linear',
            bounds_error=False, fill_value=0.)
        self.spax_area = lspax**2.
        self.lspax = lspax

        self.lam = lam
        self.Fphot = Fphot

    @classmethod
    def from_drpall_row(cls, row, lamlims, lspax, mpl_v):
        drp = m.load_drp_logcube(plate=str(row['plate']), ifu=str(row['ifudsgn']), mpl_v=mpl_v)
        lam_, Fphot_ = Flam2Fphot(drp, lamlims=lamlims)

        return cls(lam=lam_, Fphot=Fphot_, lspax=lspax)

    def mean(self, ctr, lams, dlams, D_fib, A_scope, effs, spat_samples=(50, 50)):
        '''
        integrate the interpolated cube over a simulated fiber of diameter `D_fib` (in angular coords)
            centered at `ctr` (`X_ctr`, `Y_ctr`), after sampling the cube in grid specified
            by `spat_samples`
        '''
        if type(spat_samples) is int:
            spat_samples = (spat_samples, ) * 2

        S = (D_fib / self.lspax).decompose().value  # fiber size in terms of pixels

        Ig = np.linspace(start=(ctr[0] - 0.5 * S), stop=(ctr[0] + 0.5 * S), num=spat_samples[0])
        Jg = np.linspace(start=(ctr[1] - 0.5 * S), stop=(ctr[1] + 0.5 * S), num=spat_samples[1])

        # area of new spaxels, in units of old spaxels' area
        dI, dJ = ((Ig[1:] - Ig[:-1]).mean(), (Jg[1:] - Jg[:-1]).mean())
        frac_area = dI * dJ

        Lg = lams.value

        nLg, nIg, nJg = Lg.size, Ig.size, Jg.size
        LL_, II_, JJ_ = np.meshgrid(Lg, Ig, Jg, indexing='ij')

        mask = np.zeros_like(LL_)
        mask[np.sqrt((II_ - ctr[0])**2. + (JJ_ - ctr[1])**2.) <= (0.5 * S)] = frac_area

        coords = np.column_stack((LL_.flatten(), II_.flatten(), JJ_.flatten()))
        Fphot_subsample = (self.Fphot_cube_interp(coords).reshape((nLg, nIg, nJg)) * \
                           self.Fphot.unit * A_scope).to(u.ph / (u.AA * u.s))

        spatial_integral = (Fphot_subsample * mask).sum(axis=(1, 2))

        effs_a = aggregate_effs(effs, lams)

        S_cts_obs = spatial_integral * dlams * effs_a
        S_cts_obs = S_cts_obs.to(u.ct / u.s)

        return S_cts_obs

    def observe(self, N_obs, t_obs, RN, dark_ct_rate, sky_ct_rate,
                ctr, lams, dlams, D_fib, A_scope, effs, spat_samples):
        '''
        make mock observation(s) of a position in a datacube, assuming poisson statistics

        params:
         - N_obs: number of observations
         - t_obs: time observed
         - RN: mean read noise
        '''
        # theoretical mean counts
        src_ct_rate = self.mean(ctr=ctr, lams=lams, dlams=dlams, D_fib=D_fib,
                                A_scope=A_scope, effs=effs, spat_samples=spat_samples)
        tot_ct_rate = src_ct_rate + dark_ct_rate + sky_ct_rate
        ct_expect = (tot_ct_rate * t_obs).to(u.ct).value

        # poisson distributed read noise
        RN_dist = stats.poisson(RN)

        # array of poisson distributed mean counts
        ct_dists = [stats.poisson(s) for s in ct_expect]
        ct_real = np.column_stack([s.rvs(N_obs) for s in ct_dists])

        return ct_real * u.ct

    def observe_Flam(self, N_obs, t_obs, RN, dark_ct_rate, sky_ct_rate,
                     ctr, lams, dlams, D_fib, A_scope, effs, spat_samples):
        '''
        make mock observation(s) of a position in a datacube, assuming Poisson statistics;
            subtract dark & sky; and then turn back into a flux-density (with uncertainties)
        '''

        cts_obsvd = self.observe(N_obs=N_obs, t_obs=t_obs, RN=RN, dark_ct_rate=dark_ct_rate,
                                 sky_ct_rate=sky_ct_rate, ctr=ctr, lams=lams, dlams=dlams,
                                 D_fib=D_fib, A_scope=A_scope, effs=effs, spat_samples=spat_samples)
        cts_obsvd_sum = cts_obsvd.sum(axis=0)
        cts_unc = unc_of_cts(cts_obsvd_sum)

        # subtract expected mean sky & dark (assuming they're well-sampled)
        cts_dark_exp = dark_ct_rate * t_obs * N_obs
        cts_sky_exp = dark_ct_rate * t_obs * N_obs

        cts_src = cts_obsvd_sum - (cts_dark_exp + cts_sky_exp)  #  presumed counts from source

        # CTS to F_lambda
        # photon energy (multiply)
        e_ph = (c.h * c.c / lams) / u.ph

        # per-pixel efficiency (divide)
        effs_a = aggregate_effs(effs, lams)

        # total exposure time (divide)
        t_tot = N_obs * t_obs

        # go from counts to Flam
        cts2Flam = (e_ph) / (t_tot * effs_a * dlams * A_scope)

        Flam_src = (cts_src * cts2Flam).to(u.erg / u.s / u.cm**2 / u.AA)
        Flam_unc = (cts_unc * cts2Flam).to(u.erg / u.s / u.cm**2 / u.AA)

        return Flam_src, Flam_unc


def unc_of_cts(cts):
    return np.sqrt(cts.value) * cts.unit

def aggregate_effs(effs, lams):
    effs_a = np.ones_like(lams.value)
    for eff in effs:
        effs_a *= eff(lams)

    effs_a = effs_a * (u.ct / u.ph)
    return effs_a
