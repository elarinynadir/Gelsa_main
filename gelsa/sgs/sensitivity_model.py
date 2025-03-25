import os
import numpy as np
from astropy.io import fits
from scipy import interpolate
import xml.etree.ElementTree as ET


class SensitivityModel:
    invalid_bit_mask = 1

    def __init__(self, path, datadir="."):
        """ """
        self.datadir = datadir
        self._load(path)

    def _load(self, path):
        """ """
        root = ET.parse(path).getroot()
        filename = root.find("Data/DataStorage/DataContainer/FileName").text
        data_path = os.path.join(os.path.dirname(path),
                                 self.datadir, filename)
        self.data = {}
        with fits.open(data_path) as hdul:
            for hdu in hdul:
                if 'EXTNAME' in hdu.header:
                    key = hdu.header['EXTNAME']
                    self.data[key] = self._make_interper(hdu.data[:], key)

    def _make_interper(self, sens_table, key):
        """ """
        wavelength = sens_table['Wavelength']
        # quality = sens_table['Quality']
        mask = sens_table['Mask'].astype(int)
        # sens in e/(erg/cm2/s)
        sens = np.array(sens_table['Sensitivity'])
        valid = np.bitwise_and(mask, self.invalid_bit_mask) == False
        valid &= np.isfinite(sens)
        valid &= (sens > 0)
        sens = sens[valid]
        wavelength = wavelength[valid]
        func = interpolate.interp1d(
            wavelength, sens,
            bounds_error=False, fill_value=0
        )
        bounds = wavelength.min(), wavelength.max()
        pack = {
            'bounds': bounds,
            'func': func
        }
        return pack

    def keys(self):
        """ """
        return self.data.keys()

    def get_model(self, grism='RGS000', tilt=0):
        """ """
        if tilt not in [0, -4, 4]:
            raise ValueError("Tilt must be one of (0, -4, 4)")
        if tilt == 4:
            suffix = "P4"
        elif tilt == -4:
            suffix = "M4"
        else:
            suffix = ""
        key = f"{grism}{suffix}"
        if key not in self.data:
            raise KeyError(f"Grism name missing: {grism}, tilt:{tilt}, key:{key}")
        return self.data[key]
