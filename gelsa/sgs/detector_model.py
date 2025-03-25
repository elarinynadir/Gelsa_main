import os
import numpy as np
from astropy.table import Table


class DetectorModel:
    """ """
    _default_params = {
        'workdir': 'tmp',
        'datadir': 'data',
        'rotated': True,
        'detector_slots_path': 'NISPDetectorSlots_2.1.csv',
        'nx_pixels': 2040,
        'ny_pixels': 2040
    }

    def __init__(self, detector_slots_path=None, config=None, **kwargs):
        """ """
        self.params = self._default_params.copy()
        if config:
            self.params.update(config)
        self.params.update(kwargs)
        if detector_slots_path:
            self.params['detector_slots_path'] = detector_slots_path

        self.load_detector_slots(self.params['detector_slots_path'])

    @staticmethod
    def det_pos_to_index(pos):
        """Get the detector index from the code.
        The detector code has the form {row}{col}
        """
        a = pos//10
        b = pos - a*10
        if a > 4:
            raise ValueError
        if b > 4:
            raise ValueError
        return (a-1) + (b-1)*4

    @staticmethod
    def det_index_to_pos(idx):
        """Get the detector code from index (0-15)
        The detector code has the form {row}{col}
        """
        if (idx < 0) or (idx > 16):
            raise ValueError
        a = idx % 4
        b = idx // 4
        return (a+1)*10 + b + 1

    def load_detector_slots(self, path):
        """Load the detector slots file."""
        print(f"loading {path}")
        wcs_table = Table.read(path, format='csv',
                               header_start=1, data_start=2)

        self.detector_wcs = [[]]*16
        self._detector_transforms = [[]]*16
        for det_wcs in wcs_table:
            det_pos = det_wcs['SCE_POS']
            idx = self.det_pos_to_index(det_pos)
            crpix = np.array([
                det_wcs['CRPIX1']-4,
                det_wcs['CRPIX2']-4
            ])
            crval = np.array([
                det_wcs['CRVAL1'],
                det_wcs['CRVAL2']
            ])
            cdmat = np.array([
                [det_wcs['CD1_1'], det_wcs['CD1_2']],
                [det_wcs['CD2_1'], det_wcs['CD2_2']]
            ])

            if self.params['rotated']:
                det_size = self.params['nx_pixels']
                if det_pos // 10 >= 3:
                    cdmat *= -1
                    crpix = det_size - crpix + 1
                cdmat = [
                    [cdmat[0][1], -cdmat[0][0]],
                    [cdmat[1][1], -cdmat[1][0]]
                ]
                crpix = crpix[1], det_size - crpix[0] + 1

            cdinv = np.linalg.inv(cdmat)

            self._detector_transforms[idx] = (
                np.array(crpix), np.array(crval), np.array(cdmat), cdinv
            )
        self._init_fov_geometry()

    def _init_fov_geometry(self):
        """Initialize detector polygons and envelope
        """
        nx_pixels = self.params['nx_pixels']
        ny_pixels = self.params['ny_pixels']
        corners_x = np.array([0, 1, 1, 0]) * nx_pixels - 1 + 0.5
        corners_y = np.array([0, 0, 1, 1]) * ny_pixels - 1 + 0.5
        xmin = 1e6
        xmax = -1e6
        ymin = 1e-6
        ymax = -1e6

        self.detector_poly_list = []
        for det_idx in range(16):
            xfov, yfov = self.getFOVPosition(corners_x, corners_y, det_idx)
            self.detector_poly_list.append((xfov, yfov))
            xmin = min(xmin, xfov.min())
            xmax = max(xmax, xfov.max())
            ymin = min(ymin, yfov.min())
            ymax = max(ymax, yfov.max())
        self.envelope = [[xmin, ymin], [xmax, ymax]]
        xslope = 2 / (xmax - xmin)
        xint = (1 + xmax / xmin) / (1 - xmax / xmin)
        yslope = 2 / (ymax - ymin)
        yint = (1 + ymax/ymin) / (1 - ymax/ymin)
        self._fov_scaling_params = (xslope, xint, yslope, yint)

    def is_within_fov(self, xfov, yfov, eps_mm=1):
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

        low, high = self.envelope
        valid = (xfov > low[0]-eps_mm) & (xfov < high[0]+eps_mm)
        valid &= (yfov > low[1]-eps_mm) & (yfov < high[1]+eps_mm)

        if scalar:
            return valid[0]
        return valid

    @staticmethod
    def inside_poly(x, y, poly_x, poly_y):
        """Check if a 2D cartesian point is inside a polygon.

        Parameters
        ----------
        x
        y
        poly_x
        poly_y

        Returns
        -------
        bool
        """
        inside = np.ones(len(x), dtype=bool)
        n = len(poly_x)
        for i in range(len(poly_x)):
            j = (i + 1) % n
            dx = poly_x[i] - x
            dy = poly_y[i] - y
            px = poly_x[j] - poly_x[i]
            py = poly_y[j] - poly_y[i]
            # compute cross product
            v = dx * py - dy * px
            inside = inside & (v > 0)
        return inside

    def getDetectorNumber(self, xfov, yfov):
        """Get the detector index (0 to 15) from FOV coordinates

        -1 indicates no detector.

        Parameters
        ----------
        xfov
        yfov

        Returns
        -------
        detector index
        """
        try:
            xfov[0]
            scalar_out = False
        except (TypeError, IndexError):
            xfov = np.array([xfov])
            yfov = np.array([yfov])
            scalar_out = True

        det_list = np.zeros(len(xfov), dtype=int) - 1
        for det_idx in range(16):
            corners_x, corners_y = self.detector_poly_list[det_idx]
            inside = self.inside_poly(xfov, yfov, corners_x, corners_y)
            det_list[inside] = det_idx

        if scalar_out:
            return det_list[0]
        return det_list

    def getPixel(self, xfov, yfov, det_id=None):
        """Transform FOV to detector coordinates

        Parameters
        ----------
        xfov : float, list
        yfov : float, list
        det_id : int, list
            optional

        Returns
        -------
        x, y
        """
        try:
            xfov[0]
            scalar_out = False
        except (TypeError, IndexError):
            scalar_out = True
            xfov = np.array([xfov])
            yfov = np.array([yfov])
            if det_id:
                det_id = np.array([det_id])

        if not det_id:
            det_id = self.getDetectorNumber(xfov, yfov)

        # ensure that det_id is a list
        det_id = np.ones(len(xfov), dtype=int) * det_id

        if self.params['rotated']:
            xfov, yfov = -yfov, xfov

        fov = np.array([xfov, yfov])
        x = np.zeros(len(xfov))
        y = np.zeros(len(yfov))
        for det_id_ in np.unique(det_id):
            sel = det_id == det_id_
            crpix, crval, _, cdinv = self._detector_transforms[det_id_]
            x[sel], y[sel] = np.dot(cdinv, fov[:, sel] - crval[:, np.newaxis]) + crpix[:, np.newaxis] - 1

        if scalar_out:
            return x[0], y[0], det_id[0]

        return x, y, det_id

    def getFOVPosition(self, x, y, det_id):
        """Transform detector to FOV coordinates

        Parameters
        ----------
        x : float, list
        y : float, list
        det_id : int, list

        Returns
        -------
        xfov, yfov
        """
        try:
            x[0]
            scalar_out = False
        except (TypeError, IndexError):
            scalar_out = True
            x = np.array([x])
            y = np.array([y])

        # ensure that det_id is a list
        det_id = np.ones(len(x), dtype=int) * det_id

        pix = np.array([x, y]) + 1

        xfov = np.zeros(len(x))
        yfov = np.zeros(len(y))

        for det_id_ in np.unique(det_id):
            sel = det_id == det_id_
            crpix, crval, cd, _ = self._detector_transforms[det_id_]
            xfov[sel], yfov[sel] = np.dot(cd, pix[:, sel] - crpix[:, np.newaxis]) + crval[:, np.newaxis]

        if self.params['rotated']:
            xfov, yfov = yfov, -xfov

        if scalar_out:
            return xfov[0], yfov[0]

        return xfov, yfov

    def getScaledFOV(self, xfov, yfov):
        """gets
        - the detector model which has the corners of the FOV in mm
        - the fov in mm
        returns:
        - the focal plane coordinates in normalized [-1,1] range
        """
        xslope, xint, yslope, yint = self._fov_scaling_params
        x = xslope * xfov + xint
        y = yslope * yfov + yint
        return x, y
