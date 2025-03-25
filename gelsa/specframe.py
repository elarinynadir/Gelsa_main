import os
import numpy as np

from . import utils
from .sgs import dmutils, frame_coordinates, relative_flux, psf_model

from . import spec_imager
from .spec_crop import SpecCrop


class SpecFrame:

    _default_params = {
        'grism_name': 'RGS000',
        'RA': 0,
        'DEC': 0,
        'PA': 0,
        'tilt': 0,
        'PIXSCALE': 0.3,
        'exptime_sec': 550.,
        'sigma2_det': 2.33,
    }

    dispersion_angles = {
        'RGS000': 0.,
        'RGS180': 180.,
        'BGS000': 0.
    }

    detectors = [11,21,31,41,
                12,22,32,42,
                13,23,33,43,
                14,24,34,44]

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        self.params.update(kwargs)
        self.hdu_loaded = False
        self._setup()

    def __str__(self):
        """ """
        s = f"<{self.__class__}\n"
        s += "params = {"
        for key, value in self.params.items():
            s += f"  '{key}': {value},\n"
        s += "} >"
        return s

    def _setup(self):
        """ """
        self.params['angle'] = self.params['tilt'] + \
            self.dispersion_angles[self.params['grism_name']]

        self.framecoord = frame_coordinates.FrameCoordinates(
            self.params,
        )

    def load_frame(self, frame_path=None, loctable_path=None):
        """ """
        if frame_path.endswith(".gz"):
            # check if decompressed file exists
            path_ = frame_path[:-3]
            if os.path.exists(path_):
                frame_path = path_
                print(f"found {frame_path}")

        self._hdul, self._metadata = dmutils.load_scienceframe(frame_path)

        self._header = self._hdul[0].header
        self.params['grism_name'] = self._header['GWA_POS'].strip()
        self.params['tilt'] = self._header['GWA_TILT']
        try:
            # Look for SIR adjusted pointing center
            self.params['RA'] = self._header['SIR_RA']
            self.params['DEC'] = self._header['SIR_DEC']
            self.params['PA'] = self._header['SIR_PA']
            got_adjusted_pointing = True
        except KeyError:
            # Fall back to commanded pointing with location table
            self.params['RA'] = self._header['RA']
            self.params['DEC'] = self._header['DEC']
            self.params['PA'] = self._header['PA']
            got_adjusted_pointing = False
        self.params['PIXSCALE'] = self._header['PIXSCALE']
        self.params['PTGID'] = self._header['PTGID']
        self.params['OBS_ID'] = self._header['OBS_ID']
        self.params['DITHOBS'] = self._header['DITHOBS']
        self._setup()
        if not got_adjusted_pointing:
            if loctable_path is None:
                try:
                    loctable_path = dmutils.find_location_table(frame_path)
                except ValueError:
                    print("could not find location table automatically")
            self.framecoord.update_optical_model_from_location_table(loctable_path)
        self.hdu_loaded = True


    def load_frame_h5(self, frame_path=None, loctable_path=None):
        """ """
        if frame_path.endswith(".gz"):
            # check if decompressed file exists
            path_ = frame_path[:-3]
            if os.path.exists(path_):
                frame_path = path_
                print(f"found {frame_path}")

        self._hdul, self._metadata = dmutils.load_scienceframe(frame_path)

        self._header = self._hdul[0].header
        self.params['grism_name'] = self._header['GWA_POS'].strip()
        self.params['tilt'] = self._header['GWA_TILT']
        try:
            # Look for SIR adjusted pointing center
            self.params['RA'] = self._header['SIR_RA']
            self.params['DEC'] = self._header['SIR_DEC']
            self.params['PA'] = self._header['SIR_PA']
            got_adjusted_pointing = True
        except KeyError:
            # Fall back to commanded pointing with location table
            self.params['RA'] = self._header['RA']
            self.params['DEC'] = self._header['DEC']
            self.params['PA'] = self._header['PA']
            got_adjusted_pointing = False
        self.params['PIXSCALE'] = self._header['PIXSCALE']
        self._setup()
        if not got_adjusted_pointing:
            if loctable_path is None:
                try:
                    loctable_path = dmutils.find_location_table(frame_path)
                except ValueError:
                    print("could not find location table automatically")
            self.framecoord.update_optical_model_from_h5(loctable_path)
        self.hdu_loaded = True

    def close_frame(self):
        """ """
        self._hdul.close()
        del self._header, self._hdul

    def set_detector_model(self, detector_model):
        """ """
        self.framecoord.detector_model = detector_model
        self.detector_model = detector_model

    def set_optical_model(self, optical_model):
        """ """
        self.framecoord.optical_params = optical_model.get_model(
            self.params['grism_name'], tilt=self.params['tilt']
        )
        self.optical_params = self.framecoord.optical_params

    def set_dispersion_model(self, dispersion_model):
        """ """
        ids_model = dispersion_model['ids']
        crv_model = dispersion_model['crv']
        self.framecoord.ids_params = ids_model.get_model(
            self.params['grism_name'], tilt=self.params['tilt'])
        self.framecoord.crv_params = crv_model.get_model(
            self.params['grism_name'], tilt=self.params['tilt'])

    def set_sensitivity(self, sensitivity_model):
        """ """
        self.sensitivity_params = sensitivity_model.get_model(
            self.params['grism_name'], self.params['tilt'])
        self.params['wavelength_range'] = self.sensitivity_params['bounds']

    # def set_relative_flux_model(self, relative_flux_model):
    #     """ """
    #     self.relative_flux_loss_params = None
    #     if relative_flux_model is not None:
    #         self.relative_flux_loss_params = relative_flux_model.get_model(
    #             self.params['grism_name'], self.params['tilt']
    #         )

    def set_psf_model(self, psf_model):
        """ """
        self.psf_params = None
        if psf_model is not None:
            self.psf_params = psf_model.get_model(
                self.params['grism_name']
            )

    def set_zero_order_mask(self, zero_order_mask):
        """ """
        self.zero_order_mask = zero_order_mask.create_zero_order_mask(self)

    @property
    def detector_shape(self):
        """Return detector size in pixels (nx, ny)"""
        nx = self.detector_model.params['nx_pixels']
        ny = self.detector_model.params['ny_pixels']
        return nx, ny

    def radec_to_pixel(self, ra, dec, wavelength, dispersion_order=1, **kwargs):
        """Compute detector pixel coordinates from RA, Dec sky position.

        Parameters
        ----------
        ra : float, numpy.array
            Sky position RA in degrees
        dec : float, numpy.array
            Sky position Dec in degrees
        wavelength : float, numpy.array
            Wavelength in angstrom
        dispersion_order : int
            Dispersion order (0, 1, or 2), default 1

        Returns
        -------
        x, y, detector_index
        """
        return self.framecoord.radec_to_pixel(ra, dec, wavelength,
                                              dispersion_order)

    def pixel_to_radec(self, x, y, det, wavelength, dispersion_order=1):
        """Compute sky position RA, Dec from detector pixel coordinates.

        Parameters
        ----------
        x : float, numpy.array
            detector pixel x coordinate
        y : float, numpy.array
            detector pixel y coordinate
        det : int, numpy.array
            NISP detector index starting from 0
        wavelength : float, numpy.array
            Wavelength in angstrom
        dispersion_order : int
            Dispersion order (0, 1, or 2), default 1

        Returns
        -------
        RA, Dec
        """
        return self.framecoord.pixel_to_radec( x, y, det, wavelength,
                                              dispersion_order)

    def sample_psf(self, x=None, y=None, detector=None, wavelength_ang=None,
                   **args):
        """Sample the PSF at detector coordinates x,y,detector and wavelength"""
        # if x is not None:
        #     xfov, yfov = self.detector_model.getFOVPosition(x, y, detector)
        #     pos_fov = (np.mean(xfov), np.mean(yfov))
        # else:
        if self.psf_params is None:
            return 0, 0
        pos_fov = None
        samples = psf_model.sample_psf(
            self.psf_params,
            pos_fov=pos_fov, wavelength_ang=wavelength_ang,
            **args
        )
        return samples

    def lineflux_to_counts(self, flux, wavelength):
        """Convert line flux in erg/s/cm2/s to photon counts.

        Parameters
        ----------
        flux : float
            flux in erg/s/cm2
        wavelength : float
            wavelength in angstroms

        Returns
        -------
        counts
        """
        return flux * self.sensitivity_params['func'](wavelength) * self.params['exptime_sec']

    def flux_to_counts(self, flux, wavelength):
        """Convert flux per unit wavelength in erg/s/cm2/s/A to photon counts.

        Parameters
        ----------
        flux : float
            flux in erg/s/cm2/A
        wavelength : float
            wavelength in angstroms

        Returns
        -------
        counts
        """
        step = wavelength[1] - wavelength[0]
        return flux * step * self.sensitivity_params['func'](wavelength) * self.params['exptime_sec']

    def counts_to_flux(self, counts, wavelength):
        """Convert photon counts to flux per unit wavelength in erg/s/cm2/s/A.
        Inverse of flux_to_counts

        Parameters
        ----------
        counts : float
            photon counts
        wavelength : float
            wavelength in angstroms

        Returns
        -------
        flux
        """
        step = wavelength[1] - wavelength[0]
        return counts / (step * self.sensitivity_params['func'](wavelength) * self.params['exptime_sec'])

    def sensitivity(self, wavelength):
        """Return sensitivity (or response) in units counts/(erg/cm2/s/A)

        Parameters
        -----------
        wavelength : float
            wavelength in angstrom

        Returns
        -------
        sensitivity
        """
        step = wavelength[1] - wavelength[0]
        return step * self.sensitivity_params['func'](wavelength) * self.params['exptime_sec']

    # def get_relative_flux_loss(self, ra, dec, wavelength_ang):
    #     """ """
    #     try:
    #         self.relative_flux_loss_params
    #     except AttributeError:
    #         return np.ones(len(x))

    #     if self.relative_flux_loss_params is None:
    #         return np.ones(len(x))

    #     xfov, yfov = self.framecoord.getReferencePosition(ra, dec)

    #     return relative_flux.get_flux_loss(
    #         self.detector_model,
    #         self.relative_flux_loss_params,
    #         xfov, yfov, wavelength_ang=wavelength_ang
    #     )

    def get_detector(self, detector_index):
        """Get the detector pixel arrays for signal, mask, and variance.

        Parameters
        ----------
        detector_index : int
            NISP detector index starting from 0

        Returns
        -------
        signal, mask, variance
        """
        d = self.detectors[detector_index]
        # print(f"Loading detector {d}")
        if self.hdu_loaded:
            extname = f'DET{d}.SCI'
            maskname = f'DET{d}.DQ'
            varname = f'DET{d}.VAR'
            fullimage = self._hdul[extname].data
            fullmask = self._hdul[maskname].data
            fullvar = self._hdul[varname].data
        else:
            if detector_index in self._data:
                fullimage = self._data[detector_index]
                fullmask = self._mask[detector_index]
                fullvar = self._var[detector_index]
            else:
                nx = self.framecoord.detector_model.params['nx_pixels']
                ny = self.framecoord.detector_model.params['ny_pixels']
                fullimage = np.zeros((ny, nx), dtype='d')
                fullmask = np.zeros((ny, nx), dtype=bool)
                fullvar = np.zeros((ny, nx), dtype='d')

        if self.zero_order_mask is not None:
            invalid = self.zero_order_mask[detector_index] == 0
            fullmask[invalid] = 1

        return fullimage, fullmask, fullvar

    def cutout(self, ra, dec, redshift=0, lam_step=20, wavelength_range=None, **crop_args):
        """Make a cutout around a sky position.

        Parameters
        ----------
        ra
        dec
        redshift
        lam_step

        Returns
        -------
        dict : pack_list
        """
        if wavelength_range is None:
            wavelength_range = self.params['wavelength_range']
        wave_trace = utils.intrange(*wavelength_range, lam_step)
        n = len(wave_trace)

        detx, dety, detid = self.radec_to_pixel(
            ra*np.ones(n),
            dec*np.ones(n),
            wave_trace
        )
        valid = detid >= 0
        if np.sum(valid) == 0:
            #raise ValueError(f"RA, Dec  not on detector {(ra, dec)}")
            print(f"RA, Dec  not on detector {(ra, dec)}")
            return None
        detx = detx[valid]
        dety = dety[valid]
        detid = detid[valid]
        wave_trace = wave_trace[valid]

        pack_list = {}
        for det in np.unique(detid):
            sel = detid == det
            detx_ = detx[sel]
            dety_ = dety[sel]
            crop = SpecCrop.crop_trace(self, det, detx_, dety_, **crop_args)
            crop.center = (ra, dec)
            crop.wavelength_trace = wave_trace[sel]
            crop.redshift = redshift
            pack_list[det] = crop

        return pack_list

    def add_sources(self, galaxy_list, **kwargs):
        """ """
        try:
            self._data
        except AttributeError:
            self._data = {}
            self._var = {}
            self._mask = {}
        S = spec_imager.SpectralImager(self)
        images, var_images = S.make_image(galaxy_list, return_var=True, **kwargs)

        self._data.update(images)
        self._var.update(var_images)
        for det, image in images.items():
            self._mask[det] = np.zeros(image.shape, dtype=bool)
