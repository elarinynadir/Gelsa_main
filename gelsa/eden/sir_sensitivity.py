"""
File: sir_sensitivity.py

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
import warnings
import numpy as np
from scipy import interpolate
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

import ElementsKernel.Logging as logging
from ST_DM_DmUtils.DmUtils import read_product_metadata


class Sensitivity:
    """Sensitivity.
    This class contains methods to compute the telescope sensitivity.

    """

    logger = logging.getLogger(__name__)

    grism_names = ('RGS000', 'RGS180', 'RGS270', 'BGS000')
    extnames = ('RGS000', 'RGS000M4', 'RGS180', 'RGS180P4', 'BGS000')
    tilts = (0, 4, -4)
    on_thresh = 0.1

    def __init__(self, args, grism_pos, grism_tilt, exptime):
        """Initialise.
        Initialise the class.

        """

        self.args = args
        self.grism_pos = grism_pos.upper()
        self.grism_tilt = grism_tilt
        self.extname = self.get_extname(grism_pos, grism_tilt)
        self.exptime = exptime
        self.load_sensitivity()
        self.validate_inputs()

    def get_extname(self, grism_pos, grism_tilt):
        """Get the FITS HDU extension name from the grism position and tilt

        Parameters
        ----------
        grism_pos : str
           grism position name (RGS000, RGS180)
        grism_tilt : int
           tilt value (0, -4, 4)

        Returns
        -------
        str : extension name
        """
        if grism_tilt == 0:
            return grism_pos
        elif grism_tilt == 4:
            return f"{grism_pos}P4"
        elif grism_tilt == -4:
            return f"{grism_pos}M4"
        raise ValueError(f"Unknown extname {grism_pos=} {grism_tilt=}")

    def validate_inputs(self):
        """Validate inputs.
        This function validate the inputs from the telescope.

        """
        if self.grism_tilt not in self.tilts:
            raise ValueError(f"Grism tilt {self.grism_tilt} not one of {self.tilts}")
        if self.grism_pos not in self.grism_names:
            raise ValueError(f"Grism name not known {self.grism_pos} must be one of {self.grism_names}")
        if self.extname not in self.extnames:
            raise ValueError(f"extension name {self.extname} must be one of {self.extnames}")
        if self.exptime <= 0:
            raise ValueError(f"exposure time must be greater than 0 {self.exptime}")
        low, high = self.wavelength_limit
        if (high - low) < 100:
            self.logger.warning(f"Wavelength limits of transmission are not valid {low, high}")

    def load_sensitivity(self):
        """Load sensitivity.
        This function loads the sensitivity file.

        """

        path = os.path.join(self.args.workdir, self.args.abs_calib)
        dp = read_product_metadata(path)
        filename = dp.Data.DataStorage.DataContainer.FileName

        path = os.path.join(self.args.workdir, 'data', filename)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            with fits.open(path) as hdul:
                self.logger.info(f"Loading sensitivity {self.extname}")
                try:
                    hdu_ind = int(hdul[0].header[self.extname])
                except KeyError:
                    self.logger.critical(f"HDU {self.extname} not found in sens table {path}")
            sens_table = Table.read(path, hdu=hdu_ind)

        if not str(sens_table['Wavelength'].unit).lower() == 'angstrom':
            raise ValueError(f"Expected wavelength in angstrom, found {sens_table['Wavelength'].unit=}")

        wavelength = sens_table['Wavelength']

        # sens in e/(erg/cm2)
        sens = np.array(sens_table['Sensitivity'])
        valid = np.isfinite(sens)
        sens = sens[valid]
        wavelength = wavelength[valid]

        # multiply by exptime to get e/(erg/(cm2 s))
        sens *= self.exptime

        self._sense_func = interpolate.interp1d(
            wavelength, sens,
            bounds_error=False,
            fill_value=0
        )

        thresh = sens.max() * self.on_thresh
        i_on = sens.searchsorted(thresh)
        i_off = len(sens) - sens[::-1].searchsorted(thresh)

        self.logger.info(f"Transmission on {i_on} off {i_off} thresh {thresh} len {len(wavelength)}")

        wavelength_on = wavelength[i_on]
        wavelength_off = wavelength[i_off]

        self.logger.info(f"Transmission on {wavelength_on} off {wavelength_off}")

        self.wavelength_limit = (wavelength_on, wavelength_off)

    def flux_to_electrons(self, wavelength):
        """Flux to electrons.
        This functions transform flux into electron number counts.

        """

        return self._sense_func(wavelength)

    def wavelength_in_range(self, wavelength):
        """Check if wavelength is in range"""
        low, high = self.wavelength_limit
        return (wavelength > low) & (wavelength < high)
