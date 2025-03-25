"""
File: sir_pointing_model.py

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
from scipy import optimize
import h5py

import ElementsKernel.Logging as logging

from SIR_InstrumentModels.OpticalModel import OpticalModel
from SIR_InstrumentModels.DetectorModel import DetectorModel
from SIR_InstrumentModels.GrismModel import GrismModel
from SIR_InstrumentModels.GlobalCRVModel import GlobalCRVModel
from SIR_InstrumentModels.GlobalIDSModel import GlobalIDSModel
from SIR_SpectraLocation.DpdLocationTable import DpdLocationTable

from . sir_sensitivity import Sensitivity
from . import scienceframe_variance


class SIRPointingModel:
    """SIR Pointing Model.
    This class contains methods to model the SIR telescope pointing.

    """

    logger = logging.getLogger(__name__)

    def __init__(self, args, mdb, sir_scienceframe_path):
        """Initialise.
        Initialise the class.

        """

        self.args = args
        self.mdb = mdb
        self.scienceframe_path = sir_scienceframe_path

    @property
    def scienceframe(self):
        try:
            return self._scienceframe
        except AttributeError:
            self._scienceframe = scienceframe_variance.ScienceFrameVariance(
                self.args.workdir, self.scienceframe_path)
            self._identifier = self._scienceframe._identifier
            self.dpd = self._scienceframe.dpd

        return self._scienceframe

    def free_scienceframe(self):
        del self._scienceframe

    @property
    def detector_model(self):
        try:
            return self._detector_model
        except AttributeError:
            self._detector_model = DetectorModel(self.scienceframe.grism_name, self.mdb, rotate=True)
        return self._detector_model

    def update_optical_model_from_location_table(self):
        """Load the optical model from a location table"""
        if self.args.location_table.endswith("xml"):
            path = DpdLocationTable._getHDF5FileName(
                self.args.workdir, self.args.location_table)
        else:
            path = os.path.join(self.args.workdir, self.args.location_table)
        self.logger.info(f"Loading location table {path}")

        with h5py.File(path) as lt:
            params = lt['OPT']
            cd_matrix = params['CD'][:]
            crpix = params['CRPIX'][:]
            crval = params['CRVAL'][:]
            ctype = params['CTYPE'][:]
        self._optical_model.set_wcs(ctype, crpix, crval, cd_matrix)
        self.logger.info("Optical model updated from location table")

    @property
    def optical_model(self):
        try:
            return self._optical_model
        except AttributeError:
            self.logger.info(f"loading optical model: {self.args.opt_model}")
            self._optical_model = OpticalModel.load(
                self.args.workdir,
                self.args.opt_model,
                self.detector_model)
            gwa_tilt = self.scienceframe.grism_tilt
            self._optical_model.set_gwa_tilt(gwa_tilt=gwa_tilt)
            if self.args.location_table:
                self.update_optical_model_from_location_table()
            else:
                self._optical_model.setPointingFromImage(self.scienceframe._scienceframe)
        return self._optical_model

    @property
    def grism_model(self):
        try:
            return self._grism_model
        except AttributeError:
            self._grism_model = GrismModel(
                self.scienceframe.grism_name, self.scienceframe.grism_tilt)
        return self._grism_model

    @property
    def ids_model(self):
        try:
            return self._ids_model
        except AttributeError:
            self.logger.info(f"loading dispersion model: {self.args.ids_model}")
            self._ids_model = {}
            path = os.path.join(self.args.workdir, self.args.ids_model)
            for order in ('Order_0', 'Order_1'):
                dispersion_order = self.grism_model.getOrderNumber(order)
                print(f"loading {order} {dispersion_order}")
                self._ids_model[dispersion_order] = GlobalIDSModel.load(
                    path,
                    self.detector_model,
                    dispersion_order,
                    self.optical_model.gwa_tilt
                )
        return self._ids_model

    @property
    def crv_model(self):
        try:
            return self._crv_model
        except AttributeError:
            self.logger.info(f"loading CRV model: {self.args.crv_model}")
            path = os.path.join(self.args.workdir, self.args.crv_model)
            self._crv_model = {}
            for order in ('Order_0', 'Order_1'):
                dispersion_order = self.grism_model.getOrderNumber(order)
                self._crv_model[dispersion_order] = GlobalCRVModel.load(
                    path,
                    self.detector_model,
                    dispersion_order,
                    self.optical_model.gwa_tilt
                )
        return self._crv_model

    @property
    def extra_tilt(self):
        return 0

    @property
    def sensitivity_model(self):
        try:
            return self._sensitivity_model
        except AttributeError:
            self._sensitivity_model = Sensitivity(
                self.args,
                self.scienceframe.grism_name,
                self.scienceframe.grism_tilt,
                self.scienceframe.exptime
            )
        return self._sensitivity_model

    @property
    def dispersion_angle(self):
        """Return the dispersion angle with respect to north by summing the PA of the
        detector, grism angle and tilt"""
        exp_pa = self.scienceframe.position_angle
        grism_angle = self.scienceframe.grism_angle
        tilt = self.optical_model.gwa_tilt + self.extra_tilt
        return (exp_pa + grism_angle + tilt) % 360

    def pixel_to_radec(self, x_pix, y_pix, detector):
        """Pixel to RA & DEC.
        This function transforms pixels into sky position.

        """
        x, y = self.detector_model.getFOVPosition(x_pix, y_pix, detector).getPosition()
        x, y = self._apply_distortion(x, y)
        return self.optical_model.wcs.wcs_pix2world([(x, y)], 1)

    def _apply_distortion(self, x, y):
        """Apply distortion to focal plane coordinates using optical model"""
        def resid_func(xy):
            """Resid function.
            """
            dx = np.polynomial.polynomial.polyval2d(xy[0], xy[1], self.optical_model.X) - x
            dy = np.polynomial.polynomial.polyval2d(xy[0], xy[1], self.optical_model.Y) - y
            return [dx, dy]

        solution = optimize.root(resid_func, [x, y])
        if not solution.success:
            self.logger.warning(f"Distortion solver failed: {solution}")
            return x, y
        return solution.x

    @property
    def footprint(self):
        try:
            return self._footprint
        except AttributeError:
            self._footprint = []
            x_pixels = int(self.mdb.get_value(
                'SpaceSegment.Instrument.NISP.NISPDetectorScienceWindow')[0][2])
            y_pixels = int(self.mdb.get_value(
                'SpaceSegment.Instrument.NISP.NISPDetectorScienceWindow')[0][3])
            xx = [0, x_pixels, x_pixels, 0]
            yy = [0, 0, y_pixels, y_pixels]
            for detector in self.detector_model.getDetectorRange():
                poly_sky = []
                for i in range(len(xx)):
                    radec = self.pixel_to_radec(xx[i], yy[i], detector)
                    poly_sky.append(radec[0])
                self._footprint.append(poly_sky)
        return self._footprint
