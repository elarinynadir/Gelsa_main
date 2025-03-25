"""
File: scienceframe_variance.py

Copyright (C) 2012-2020 Euclid Science Ground Segment

This file is part of LE3_VMSP_ID.

This library is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with this library.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import numpy as np
from scipy import ndimage

import ElementsKernel.Logging as logging

from ST_DataModelBindings.Bindings import EuclidXmlParser
from ST_DataModelBindingsXsData.dpd.sir.euc_sir_science_frame import DpdSirScienceFrame
from EL_ImageBinding import NispSurveyExposure


class ScienceFrameVariance:
    """Wrapper for SIR science frame data product
    """
    logger = logging.getLogger(__name__)
    keys = ('OBS_ID', 'DITHOBS', 'PTGID')
    median_window = 3

    grism_angle_lookup = {
        'RGS000': 0,
        'RGS180': 180,
        'RGS270': 270,
        'BGS000': 0,
    }

    def __init__(self, workdir, filename):
        """Wrapper for SIR science frame data product

        This class loads the variance images from the FITS data product.

        The variance image is processed with a median filter to account for the
        window that is used for detection.

        The filename can point to the xml data product or the fits filename.
        The file is loaded with the SIR_Utilities.DpdScienceFrame code.

        Parameters
        ----------
        workdir : str
            path to workdir
        filename : str
            dataproduct filename path relative to workdir
        """
        self._detector_cache = {}

        self.workdir = workdir
        self.load(filename)

    @property
    def metadata(self):
        """Access the metadata stored in the FITS header"""
        try:
            return self._metadata
        except AttributeError:
            self._metadata = self._scienceframe.getMetadata()
        return self._metadata

    @property
    def grism_name(self):
        """Return the grism name (keyword GWA_POS)"""
        return self.getMetadataKey('GWA_POS')

    @property
    def grism_angle(self):
        """Return the grism angle. This is determined by looking up the GWA_POS in a table."""
        return self.grism_angle_lookup[self.grism_name]

    @property
    def grism_tilt(self):
        """Return the grism tilt (keword GWA_TILT)"""
        return self.metadata.getAsInt('GWA_TILT')

    @property
    def position_angle(self):
        """Return the position angle of the pointing (keyword PA)"""
        return float(self.getMetadataKey('PA'))

    @property
    def exptime(self):
        """Return the exposure time (keyword EXPTIME)"""
        return float(self.getMetadataKey('EXPTIME'))

    @property
    def obsid(self):
        """Return the observation ID (keyword OBS_ID)"""
        return self.getMetadataKey('OBS_ID')

    @property
    def ptgid(self):
        """Return the pointing ID (keyword PTGID)"""
        return self.getMetadataKey('PTGID')

    @property
    def dithobs(self):
        """Return the dither observation number (keyword DITHOBS)."""
        return self.getMetadataKey('DITHOBS')

    def load(self, filename):
        """Load the data product.
        This runs the SIR_Utils routine
        SIR_Utilities.DpdScienceFrame.DpdScienceFrame.load

        Parameters
        ----------
        filename : str
            path relative to workdir
        """
        path = os.path.join(self.workdir, filename)
        parser = EuclidXmlParser()
        # store the data product metadata
        self.dpd = parser.parse(path, DpdSirScienceFrame)

        # load the FITS image
        mef_name = os.path.join(self.workdir, 'data',
                                self.dpd.Data.DataStorage.DataContainer.FileName)

        # read the survey exposure
        self._scienceframe = NispSurveyExposure.readFits(mef_name)

        self._identifier = {
            'ObservationId': self.dpd.Data.ObservationSequence.ObservationId,
            'DitherObservation': self.dpd.Data.ObservationSequence.DitherObservation,
            'PointingId': self.dpd.Data.ObservationSequence.PointingId
        }

    def getMetadataKey(self, key):
        """Access the metadata by keyword.

        Parameters
        ----------
        key : str
            keyword of metadata

        Returns
        -------
        str
            value
        """
        return self.metadata.getAsString(key)

    def getImage(self, detector):
        """Return the image data for detector index.

        The detector index starts from 1

        Parameters
        ----------
        detector : int
            detector index

        Returns
        -------
        np.array
            image data
        """
        # indexing starts from 0
        hdu_index = detector - 1
        frame = self._scienceframe.getDetector(hdu_index)
        return frame.getImage().getArray()

    def getVariance(self, detector):
        """Return the image data for detector index.

        The detector index starts from 1

        Parameters
        ----------
        detector : int
            detector index

        Returns
        -------
        np.array : image data
        """
        # indexing starts from 0
        hdu_index = detector - 1
        frame = self._scienceframe.getDetector(hdu_index)
        return frame.getVariance().getArray()

    def getMask(self, detector):
        """Return the image data for detector index.

        The detector index starts from 1

        Parameters
        ----------
        detector : int
            detector index

        Returns
        -------
        np.array : image data
        """
        # indexing starts from 0
        hdu_index = detector - 1
        frame = self._scienceframe.getDetector(hdu_index)
        return frame.getMask().getArray()

    def _filter_image(self, image, masked_pix, iter=1):
        """Process the image with an iterative median filter.

        Bad pixels are repaired iteratively by replacing them
        with the values from the median filtered image.

        The kernel size is set by the median_window class variable.

        Parameters
        ----------
        image : numpy.ndarray
            input image to process
        masked_pix : numpy.ndarray
            selection array indicating bad pixels
        iter : int

        Returns
        -------
        numpy.ndarray
            processed image
        """
        im_m = None
        im_copy = image.copy()
        for loop in range(iter):
            if loop > 0:
                im_copy[masked_pix] = im_m[masked_pix]
            im_m = ndimage.median_filter(im_copy, size=self.median_window)
        return im_m

    def _get_frame(self, index):
        """Access the variance and mask images for given detector.

        The variance image is processed with the median filter.

        The detecor images are stored in a cache.

        Parameters
        ----------
        index : int
            Detector index (starts at 1)

        Returns
        -------
        numpy.ndarray
            variance image
        numpy.ndarray
            masked pixels
        """
        index = int(index)

        if index in self._detector_cache:
            return self._detector_cache[index]

        # indexing starts from 0
        hdu_index = index - 1
        frame = self._scienceframe.getDetector(hdu_index)

        im_flag = frame.getMask().getArray()
        im_var = frame.getVariance().getArray()

        invalid_pix = np.logical_not(np.isfinite(im_var))
        masked_pix = (im_flag > 0) | invalid_pix
        im_var[invalid_pix] = 0
        im_var = self._filter_image(im_var, masked_pix)

        self._detector_cache[index] = im_var, masked_pix
        return im_var, masked_pix

    def _get_variance(self, xpix, ypix, detector):
        """Get the variance value at specified pixel and detector coordinates.

        Parameters
        ----------
        xpix : int
            pixel x value
        ypix : int
            pixel y value
        detector : int
            detecor index (starts at 1)

        Returns
        -------
        float
            variance value
        """
        var_image, masked_pix = self._get_frame(detector)
        if masked_pix[ypix, xpix]:
            return 0
        return var_image[ypix, xpix]

    def get_variance(self, xpix, ypix, detector):
        """Return the variance value of the pixel that corresponds to
        the position ra, dec, wavelength.

        Parameters
        ----------
        ra : list
            Right ascension angle in deg
        dec : list
            Declination angle in deg
        wavelength : list
            wavelength in angstroms
        dispersion_order : int
            Dispersion order (default 1)

        Returns
        -------
        list
            variance values
        """
        n = len(xpix)
        var = np.zeros(n, dtype='d')
        for i in range(n):
            # if the position does not fall on a detector, set variance to 0
            if detector[i] <= 0:
                continue
            var[i] = self._get_variance(ypix[i], xpix[i], detector[i])

        return var
