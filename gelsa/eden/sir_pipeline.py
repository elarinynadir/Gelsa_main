"""
File: sir_pipeline.py

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

import numpy as np

import ElementsKernel.Logging as logging

from SIR_SpectraLocation.LocationSpectrum import LocationSpectrum
from SIR_InstrumentModels.GrismModel import GrismModel


class SIRPipeline:
    """SIR Pipeline.
    This class contains methods for the SIR pipeline.

    """

    logger = logging.getLogger(__name__)

    dispersion_orders = (1, )
    arbitrary_detector = 6

    def __init__(self, sir_pointing_model):
        """Interface to SIR pipeline routines to convert RA, Dec, wavelength to coordinates on the
        detector.

        Parameters
        ----------
        sir_pointing_model : SIRPointingModel
            instance of SIRPointingModel
        """
        self.instrument = sir_pointing_model

    def check_on_detector(self, ra, dec):
        """Check if sky coordinates fall on a detector

        Parameters
        ----------
        ra : list
            Right ascension coordinate (deg)
        dec : list
            Declination coordinate (deg)

        Returns
        -------
        list : integer detector number
        """

        n = len(ra)

        detector = np.zeros(n, dtype=int)

        for i in range(n):
            # compute focal plane position in mm
            fov_position = self.instrument.optical_model.getObjectPosition(ra[i], dec[i])
            pixel = self.instrument.detector_model.getPixel(fov_position)
            detector[i] = pixel.getDetectorNumber() > 0
        return detector

    def get_pixel(self, ra, dec, wavelength, dispersion_order=1):
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
        if dispersion_order not in self.instrument.ids_model:
            raise ValueError(f"get_pixel not initialized for dispersion order {dispersion_order}")

        # get focal plane position of reference wavelength
        ref_position = self.instrument.optical_model.get1stOrderReferencePosition(ra, dec)
        # this is the position of the "virtual slit"
        slit_mm = np.array([ref_position.getPosition()])
        pivot_mm = self.instrument.optical_model.getPivotPosition(ra, dec)

        grism_pos = GrismModel.getPositionNumber(self.instrument.optical_model.gwa_pos)
        loc_spectrum = LocationSpectrum(dispersion_order, grism_pos)

        # get the detector pixel position of the slit. The detector we choose is not important.
        loc_spectrum.setDetector(self.arbitrary_detector)
        slit_pix = self.instrument.detector_model.getPixels(slit_mm, self.arbitrary_detector)
        loc_spectrum.setLambdaRefPosition(slit_pix)
        loc_spectrum.setPivot(pivot_mm[0], pivot_mm[1])

        # Set IDS and CRV models
        loc_spectrum.setIDSModel(self.instrument.ids_model[dispersion_order].getLocalModel(slit_mm))
        loc_spectrum.setCRVModel(self.instrument.crv_model[dispersion_order].getLocalModel(slit_mm))
        loc_spectrum.setDetectorModel(self.instrument.detector_model)
        loc_spectrum.setLambdaRefValue(
            self.instrument.optical_model.getReferenceLambda(dispersion_order)
        )

        loc_spectrum.setGWATilt(int(self.instrument.optical_model.gwa_tilt))
        loc_spectrum.setExtraTilt(self.instrument.extra_tilt)

        pixel = loc_spectrum.computePosition(float(wavelength), 0)
        x, y = pixel.getPosition()

        # from pixels referenced from arbitrary detector, go back to focal plane coord
        fov_position = self.instrument.detector_model.getFOVPosition(x, y, self.arbitrary_detector)

        # Now determine the actual detector and corresponding pixel coord
        pixel = self.instrument.detector_model.getPixel(fov_position)
        detector = pixel.getDetectorNumber()
        x, y = pixel.getPosition()
        x = int(np.round(x))
        y = int(np.round(y))
        return x, y, detector

    def get_pixel_list(self, ra, dec, wavelength, dispersion_order=1):
        """Return the pixel position of the spectral element given by
        sky coordinate RA, Dec and wavelength (angstrom)

        Parameters
        -----------
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
        int, int, int
            x, y pixel position
            detector number
        """
        n = len(ra)
        x = np.zeros(n, dtype=int)
        y = np.zeros(n, dtype=int)
        detector = np.zeros(n, dtype=int)

        for i in range(n):
            x[i], y[i], detector[i] = self.get_pixel(
                ra[i], dec[i], wavelength[i], dispersion_order)

        return x, y, detector
