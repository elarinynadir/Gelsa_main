import sys
import os
import time
import h5py
import numpy as np
from astropy.wcs import WCS
import xml.etree.ElementTree as ET

from .. import utils


class FrameCoordinates:
    mm_pix_scale = 16.606974929470375

    def __init__(self, params, optical_params=None, detector_model=None):
        """ """
        self.params = params
        if optical_params is not None:
            self.optical_params = optical_params
        if detector_model is not None:
            self.detector_model = detector_model
        self._set_wcs()

    def _set_wcs(self):
        """ """
        ra = self.params['RA']
        dec = self.params['DEC']
        pa = self.params['PA']
        self._wcs = WCS(naxis=2)
        self._wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
        self._wcs.wcs.crval = [ra, dec]
        self._wcs.wcs.crpix = [1, 1]
        orientation = np.deg2rad(-pa + 90)
        cd_matrix = np.array([[np.cos(orientation), -np.sin(orientation)],
                              [np.sin(orientation), np.cos(orientation)]])
        rot_matrix = np.array([[0, -1], [1, 0]])
        cd_matrix = rot_matrix.dot(cd_matrix)
        self._wcs.wcs.cd = self.mm_pix_scale * cd_matrix/3600.

    @property
    def detector_model(self):
        """ """
        return self._detector_model

    @detector_model.setter
    def detector_model(self, detector_model):
        """ """
        self._detector_model = detector_model

    @property
    def optical_params(self):
        """ """
        return self._optical_params

    @optical_params.setter
    def optical_params(self, optical_params):
        """ """
        self._optical_params = optical_params
        self._set_extra_tilt_matrix()

    def _set_extra_tilt_matrix(self):
        """Set up the tilt rotation matrix"""
        tilt = self.optical_params['ExtraTilt']
        cos_ = np.cos(np.deg2rad(tilt))
        sin_ = np.sin(np.deg2rad(tilt))
        self._extra_tilt_matrix = np.array([[cos_, -sin_], [sin_, cos_]])

    def update_optical_model_from_location_table(self, location_table_path):
        """ """
        # print("updating WCS from location table")
        #print(location_table_path)
        tree = ET.parse(location_table_path)
        filename = tree.find('//FileName').text

        workdir = os.path.dirname(location_table_path)
        loc_path = os.path.join(workdir, 'data', filename)

        with h5py.File(loc_path) as lt:
            params = lt['OPT']
            cd_matrix = params['CD'][:]
            crpix = params['CRPIX'][:]
            crval = params['CRVAL'][:]
            ctype = params['CTYPE'][:]
        self._wcs.wcs.cd = cd_matrix
        self._wcs.wcs.crval = crval
        self._wcs.wcs.crpix = crpix
        self._wcs.wcs.ctype = ctype

    def update_optical_model_from_h5(self, h5_file_path):
        # Open the HDF5 file directly.
        with h5py.File(h5_file_path, 'r') as lt:
            params = lt['OPT']
            cd_matrix = params['CD'][:]
            crpix = params['CRPIX'][:]
            crval = params['CRVAL'][:]
            ctype = params['CTYPE'][:]
        self._wcs.wcs.cd = cd_matrix
        self._wcs.wcs.crval = crval
        self._wcs.wcs.crpix = crpix
        self._wcs.wcs.ctype = ctype

    def arcsec_to_pixel(self, x):
        """Convert values in arcsec to pixel units"""
        return x / self.params['pixel_scale']

    def pixel_to_arcsec(self, x):
        """Convert values in pixels to arcsec units"""
        return x * self.params['pixel_scale']


    def is_within_fov(self, xpos, ypos):
        """Checks if point position is withint FOV limits"""
        limits = self.detector_model.envelope
        return bool((limits[0][0] < xpos < limits[1][0]) and (limits[0][1] < ypos < limits[1][1]))

    def getUndistortedObjectPosition(self, ra, dec):
        """Get expeted position on the FOV without distortions

        @param ra: sky ra coordinate [degrees]
        @type ra: float

        @param dec: sky ra coordinate [degrees]
        @type dec: float
        """
        points = np.transpose([ra, dec])
        scalar = False
        if len(points.shape) == 1:
            scalar = True
            points = points[np.newaxis, :]
        # fov = self._wcs.wcs_world2pix([[ra, dec], ], 0)[0]
        fov = self._wcs.wcs_world2pix(points, 0)
        if scalar:
            return fov[0]
        return fov.transpose()

    def getUndistortedObjectPosition_inverse(self, x, y):
        """Get expeted position on the FOV without distortions

        @param ra: sky ra coordinate [degrees]
        @type ra: float

        @param dec: sky ra coordinate [degrees]
        @type dec: float
        """
        points = np.transpose([x, y])
        scalar = False
        if len(points.shape) == 1:
            scalar = True
            points = points[np.newaxis, :]
        radec = self._wcs.wcs_pix2world(points, 0)
        if scalar:
            return radec[0]
        return radec.transpose()

    def apply_distortion(self, xmm_undist, ymm_undist, coef_x, coef_y):
        """ """
        xmm = np.polynomial.polynomial.polyval2d(xmm_undist, ymm_undist, coef_x)
        ymm = np.polynomial.polynomial.polyval2d(xmm_undist, ymm_undist, coef_y)
        return xmm, ymm

    def getReferencePosition(self, ra=None, dec=None,
                             xmm_undist=None, ymm_undist=None,
                             order=1):
        """Return x,y position in millimeters on the FOV.
        for a given ra and dec using the WCS and the OPT coefficients

        @param ra: sky ra coordinate [degrees]
        @type ra: float

        @param dec: sky ra coordinate [degrees]
        @type dec: float

        @return: x and y position in millimeters on the FOV
        @rtype: float, float

        """
        if xmm_undist is None:
            xmm_undist, ymm_undist = self.getUndistortedObjectPosition(ra, dec)

        if order == 1:
            coef = self.optical_params['Reference']
        elif order == 0:
            coef = self.optical_params['Displacements']
        else:
            raise ValueError

        xmm, ymm = self.apply_distortion(xmm_undist, ymm_undist, coef['x'], coef['y'])


        if self.optical_params['ExtraTilt'] != 0:
            pivot = self.optical_params['pivot']
            pivot_xmm, pivot_ymm = self.apply_distortion(xmm_undist, ymm_undist, pivot['x'], pivot['y'])
            xmm, ymm = self.apply_extra_tilt_rotation([xmm, ymm], [pivot_xmm, pivot_ymm])

        return xmm, ymm

    def apply_extra_tilt_rotation(self, pos_mm, pivot_mm):
        """Rotates the reference position around the pivot center
        """
        _pos_mm = np.array(pos_mm) - np.array(pivot_mm)
        return self._extra_tilt_matrix.dot(_pos_mm) + pivot_mm

    def normalize(self, x, lim):
        """ """
        return (2*x - lim[0] - lim[1]) / (lim[1] - lim[0])

    def get_coefficients(self, x, y, params):
        """ """
        lim_x = params['global_ranges'][0]
        lim_y = params['global_ranges'][1]
        x_norm = self.normalize(x, lim_x)
        y_norm = self.normalize(y, lim_y)
        coef_list = []
        for deg in range(len(params['model'])):
            matrix = params['model'][deg]
            coef = np.polynomial.chebyshev.chebval2d(x_norm, y_norm, matrix)
            coef_list.append(coef)
        return np.array(coef_list)

    def displacement(self, x, y, wavelength, order=1):
        """ """
        try:
            _ = len(x)
            scalar = False
        except TypeError:
            x = np.array([x])
            y = np.array([y])
            scalar = True

        wavelength = np.ones(len(x))*wavelength

        ids_params = self.ids_params['Orders'][order]
        crv_params = self.crv_params['Orders'][order]

        c_ids = self.get_coefficients(x, y, ids_params)
        wavelength_ = self.normalize(wavelength, ids_params['local_ranges'])
        dx = np.zeros(len(x), dtype='d')
        for i in range(len(wavelength)):
            dx[i] = np.polynomial.chebyshev.chebval(wavelength_[i], c_ids[:, i])

        c_crv = self.get_coefficients(x, y, crv_params)

        dx_ = self.normalize(dx, crv_params['local_ranges'])
        dy = np.zeros_like(dx)
        for i in range(len(x)):
            dy[i] = np.polynomial.chebyshev.chebval(dx_[i], c_crv[:, i])

        if scalar:
            x = x[0]
            y = y[0]
            dx = dx[0]
            dy = dy[0]

        if self.params['grism_name'] == 'RGS000' or  \
           self.params['grism_name'] == 'BGS000':
            return x+dx, y+dy
        elif self.params['grism_name'] == 'RGS180':
            return x-dx, y-dy
        else:
            raise ValueError("Unknown grism!")

    def radec_to_fov(self, ra, dec, wavelength, dispersion_order=1):
        """ """
        try:
            len(ra)
            scalar = False
        except TypeError:
            scalar = True
            ra = np.array([ra])
            dec = np.array([dec])
        wavelength = np.ones(len(ra)) * wavelength
        if dispersion_order not in [0, 1]:
            raise ValueError("get_pixel not initialized for dispersion order "
                             f"{dispersion_order}")

        xmm_undist, ymm_undist = self.getUndistortedObjectPosition(ra, dec)

        ref_x, ref_y = self.getReferencePosition(
            xmm_undist=xmm_undist, ymm_undist=ymm_undist,
            order=dispersion_order
        )

        xfov, yfov = self.displacement(ref_x, ref_y, wavelength,
                                       order=dispersion_order)

        if self.optical_params['ExtraTilt'] != 0:
            pivot = self.optical_params['Pivot']
            pivot_mm = self.apply_distortion(xmm_undist, ymm_undist, pivot['x'], pivot['y'])
            xfov, yfov = self.rotate_position([xfov, yfov],  pivot_mm)

        if scalar:
            return xfov[0], yfov[0]
        return xfov, yfov

    def fov_to_radec(self, xfov, yfov, wavelength, dispersion_order=1):
        """ """
        try:
            len(xfov)
            scalar = False
            xfov = np.array(xfov)
            yfov = np.array(yfov)
        except TypeError:
            scalar = True
            xfov = np.array([xfov])
            yfov = np.array([yfov])

        wavelength = np.ones(len(xfov))*wavelength

        valid = self.detector_model.is_within_fov(xfov, yfov)

        guess_radec = self.getUndistortedObjectPosition_inverse(
            xfov[valid], yfov[valid]
        )
        guess_radec = np.transpose(guess_radec)

        ra_, dec_ = utils.inverse2d(
            self.radec_to_fov,
            xfov[valid], yfov[valid],
            wavelength[valid], dispersion_order=dispersion_order,
            guess=guess_radec
        )

        ra = np.zeros(len(xfov), dtype='d') + np.nan
        dec = np.zeros(len(xfov), dtype='d') + np.nan
        ra[valid] = ra_
        dec[valid] = dec_
        if scalar:
            return ra[0], dec[0]
        return ra, dec

    def radec_to_pixel(self, ra, dec, wavelength, dispersion_order=1):
        """Return the pixel position of the spectral element given by
        sky coordinate RA, Dec and wavelength (angstrom)

        Parameters
        -----------
        ra : float
            Right ascension angle in deg
        dec : float
            Declination angle in deg
        wavelength : float
            wavelength in angstroms
        dispersion_order : int
            Dispersion order (default 1)

        Returns
        -------
        int, int, int
            x, y pixel position
            detector number
        """
        if dispersion_order not in [0, 1]:
            raise ValueError("get_pixel not initialized for dispersion order "
                             f"{dispersion_order}")
        t0 = time.time()

        xfov, yfov = self.radec_to_fov(ra, dec, wavelength, dispersion_order)
        x, y, det_id = self.detector_model.getPixel(xfov, yfov)

        x += 0.5
        y += 0.5
        dt = time.time()-t0
        # try:
        #     print(f"time: {len(x)/dt} obj/s", file=sys.stderr)
        # except TypeError:
        #     print(f"time: {dt} s/obj", file=sys.stderr)
        return x, y, det_id

    def pixel_to_radec(self, x, y, detid, wavelength, dispersion_order=1):
        """ """
        if dispersion_order not in [0, 1]:
            raise ValueError("get_pixel not initialized for dispersion order "
                             f"{dispersion_order}")

        t0 = time.time()
        x_ = x - 0.5
        y_ = y - 0.5
        xfov, yfov = self.detector_model.getFOVPosition(x_, y_, detid)
        out = self.fov_to_radec(xfov, yfov,
                                 wavelength, dispersion_order=dispersion_order)
        dt = time.time()-t0
        # try:
        #     valid = np.isfinite(x)
        #     print(f"time: {np.sum(valid)/dt} obj/s", file=sys.stderr)
        # except TypeError:
        #     print(f"time: {dt} s/obj", file=sys.stderr)
        return out
