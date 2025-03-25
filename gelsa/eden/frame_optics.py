import os
import numpy as np
from scipy import interpolate
from astropy.io import fits
from astropy.wcs import WCS

import xml.etree.ElementTree as ET

# import consts
# import sir_opt_model
from .eden import psf
from .eden import elsa_spectro as es


class SkyWaveMap:
    def __init__(self, es_frame, ra, dec):
        """ """
        self.es_frame = es_frame

        wavelength = np.arange(12000, 19010, 50)
        ra0 = np.mean(ra)
        dec0 = np.mean(dec)
        x, y, detector = es_frame.get_pixel(
            ra0*np.ones(len(wavelength)), dec0*np.ones(len(wavelength)), wavelength
        )

        self._fits = {}
        for d in np.unique(detector):
            sel = np.where(detector == d)
            wave_fit = self.fit_wavelength(x[sel], y[sel], wavelength[sel])

            sky_fit = self.fit_sky(ra0, dec0, np.mean(wavelength[sel]))

            self._fits[d] = (sky_fit, wave_fit)

    def fit_sky(self, ra0, dec0, wavelength0, eps=1e-3):
        """ """
        x, y, det = self.es_frame.get_pixel(
            [ra0, ra0+eps, ra0], [dec0, dec0, dec0+eps], [wavelength0, wavelength0, wavelength0]
        )

        dx_dra = (x[1] - x[0])/eps
        dy_dra = (y[1] - y[0])/eps
        dx_ddec = (x[2] - x[0])/eps
        dy_ddec = (y[2] - y[0])/eps

        return np.array([ra0, dec0, dx_dra, dx_ddec, dy_dra, dy_ddec])


    def fit_wavelength(self, x, y, wavelength):
        """ """
        wave_interpolator_x = interpolate.interp1d(wavelength, x, bounds_error=False, fill_value='extrapolate')
        wave_interpolator_y = interpolate.interp1d(wavelength, y, bounds_error=False, fill_value='extrapolate')
        wave_min = wavelength.min()
        wave_max = wavelength.max()
        return wave_min, wave_max, wave_interpolator_x, wave_interpolator_y

    def __call__(self, ra, dec, wavelength):
        """ """
        scalar = False
        try:
            n = len(wavelength)
        except TypeError:
            scalar = True
            ra = np.array([ra])
            dec = np.array([dec])
            wavelength = np.array([wavelength])

        x = np.zeros(len(wavelength), dtype='d') - 1
        y = np.zeros(len(wavelength), dtype='d') - 1
        d = np.zeros(len(wavelength), dtype=int)
        for det, fit in self._fits.items():
            sky_fit, wavelength_fit = fit
            wave_min, wave_max, wave_interpolator_x, wave_interpolator_y = wavelength_fit
            sel = (wavelength >= wave_min)&(wavelength <= wave_max)

            ra0, dec0, dx_dra, dx_ddec, dy_dra, dy_ddec = sky_fit
            dx = (ra[sel] - ra0) * dx_dra + (dec[sel] - dec0) * dx_ddec
            dy = (ra[sel] - ra0) * dy_dra + (dec[sel] - dec0) * dy_ddec

            x[sel] = wave_interpolator_x(wavelength[sel]) + dx
            y[sel] = wave_interpolator_y(wavelength[sel]) + dy
            d[sel] = det*np.ones(np.sum(sel))

        if scalar:
            x = x[0]
            y = y[0]
            d = d[0]
        return x, y, d



class FrameOptics:
    _default_params = {
        'grism_name': 'RGS000',
        'detector_count': 16,
        'pointing_center': (0, 0),
        'pointing_pa': 0,
        'telescope_area_cm2': 1e4,
        'exptime': 550,
        'pixel_scale': 0.3,
        'det_width': 2040,
        'det_height': 2040,
        'dispersion': 13.4,
        'x_0': 0,
        'y_0': 0,
        'wavelength_0': 15200, #12500,
        'sigma2_det': 1,#1.5,
        'dispersion_angle': 4,
        'transmission_file': "SIR_Calibration_Abs_1.0.5-ON_THE_FLY-pcasenov-PLAN-000000-GJ00XYPR-20240219-152345-0-new_abs_calib-0.xml",
        'psf_amp': 0.781749,
        'psf_scale1': 0.84454,  # pixel coordinates
        'psf_scale2': 3.64980,  # pixel coordinates
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        self.params.update(kwargs)

        # self.sens = self.params['telescope_area_cm2'] * self.params['exptime']
        # self.sens /= consts.planck * consts.c

        # self.load_transmission()

        # self.psf = psf.PSF(
        #     psf_amp=self.params['psf_amp'],
        #     psf_scale1=self.params['psf_scale1'],
        #     psf_scale2=self.params['psf_scale2']
        # )
        # # self.distortion = None
        # self._interpers = {}

    def load_frame(self, filename):
        """ """
        self._frame = es.SpecFrame(filename)
        self.params['grism_name'] = self._frame.grism_name
        self.load_transmission()

    def arcsec_to_pixel(self, x):
        """Convert values in arcsec to pixel units"""
        return x / self.params['pixel_scale']

    def pixel_to_arcsec(self, x):
        """Convert values in pixels to arcsec units"""
        return x * self.params['pixel_scale']

    # def set_pointing_pa_from_wcs(self):
    #     """ """
    #     cra,cdec = self._wcs.wcs.crval
    #     eps = 1e-3
    #     x, y = self._wcs.wcs_world2pix([cra, cra], [cdec, cdec+eps], 1)
    #     dx = x[1] - x[0]
    #     dy = y[1] - y[0]
    #     self.params['pointing_pa'] = np.degrees(np.arctan2(dy, dx))

    # def load_from_fits(self, filename, i=1, j=1):
    #     """ """
    #     with fits.open(filename) as hdul:
    #         header = hdul[0].header
    #         # GWA_POS = 'RGS000  '
    #         # GWA_TILT
    #         grism_name = header['GWA_POS']
    #         grism_list = {
    #             'RGS000': 0.,
    #             'RGS180': 180.,
    #         }
    #         grism_angle = grism_list[grism_name]
    #         tilt = float(header['GWA_TILT'])

    #         self.params['dispersion_angle'] = grism_angle - tilt

    #         hdu = hdul[f"DET{i}{j}.SCI"]
    #         self._wcs = WCS(hdu)

    #         self.set_pointing_pa_from_wcs()

    #         # opt_model = sir_opt_model.SirOptModel(path='data/0.1.3.xml', pixel_size_mm=0.018)
    #         # self.distortion = opt_model.get_model(grism_name, tilt)

    def _make_wavelength_interpolator(self, ra, dec):
        """ """
        return SkyWaveMap(self._frame, ra, dec)

    def radec_to_pixel(self, ra, dec, wavelength, objid=0):
        """ """
        if objid not in self._interpers:
            self._interpers[objid] = self._make_wavelength_interpolator(ra, dec)

        return self._interpers[objid](ra, dec, wavelength)


    def lineflux_to_counts(self, flux, wavelength):
        """Convert flux to photon counts."""
        return flux * self._sensitivity(wavelength)

    def flux_to_counts(self, flux, wavelength):
        """Convert flux to photon counts."""
        step = wavelength[1] - wavelength[0]
        return flux * step * self._sensitivity(wavelength)

    def counts_to_flux(self, counts, wavelength):
        """ """
        step = wavelength[1] - wavelength[0]
        return counts / (step * self._sensitivity(wavelength))

    def sensitivity(self, wavelength):
        """ """
        step = wavelength[1] - wavelength[0]
        return step * self._sensitivity(wavelength)

    def load_transmission(self):
        """ """
        path = os.path.join('tmp', self.params['transmission_file'])
        root = ET.parse(path).getroot()
        filename = root.find("Data/DataStorage/DataContainer/FileName").text

        path = os.path.join('tmp', 'data', filename)

        grism = self.params['grism_name']
        print(f"loading sensitivity for {grism}")
        with fits.open(path) as hdul:
            sens_table = hdul[grism].data[:]

        wavelength = sens_table['Wavelength']
        # sens in e/(erg/cm2)
        sens = np.array(sens_table['Sensitivity'])
        valid = np.isfinite(sens)
        sens = sens[valid]
        wavelength = wavelength[valid]
        # multiply by exptime to get e/(erg/(cm2 s))
        # sens *= self.params['exptime']
        func = interpolate.interp1d(wavelength, sens, bounds_error=False, fill_value=0)
        self._sensitivity = func

    # def wavelength_to_pix(self, wavelength):
    #     """Compute the pixel coordinate offset from the wavelength"""
    #     d = (wavelength - self.params['wavelength_0']) / self.params['dispersion']

    #     # rotate by the dispersion angle
    #     theta = np.pi/180 * self.params['dispersion_angle']
    #     dx = d * np.cos(theta)
    #     dy = -d * np.sin(theta)

    #     return dx + self.params['x_0'], dy + self.params['y_0']


    # def footprint(self):
    #     return self.wcs.calc_footprint(axes=(self.params['det_width'], self.params['det_height']))