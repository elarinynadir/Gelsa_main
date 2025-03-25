"""
File: healpix_projection.py

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

Attributes
----------
NEST : str
    Description
RING : str
    Description
"""
import healpy
import numpy as np


# Pixel ordering modes
NEST = 'nest'
RING = 'ring'


def is_number(s):
    """
    Parameters
    ----------
    s : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    try:
        float(s)
        return True
    except TypeError:
        return False
    except ValueError:
        return False


def validate_nside(nside):
    """Validate Healpix nside parameter.

    Parameters
    ----------
    nside : int
        Healpix nside parameter.

    Raises
    ------
    TypeError
        Description
    ValueError
        Description
    TypeError : not an int type
    ValueError : not a positive factor of 2
    """
    if not isinstance(nside, (int, float)):
        raise TypeError('Healpix nside must be a number')
    if nside < 1:
        raise ValueError('Healpix nside must be greater than 0.')
    if np.abs(nside - 2**int(np.log2(nside))) > 1e-10:
        raise ValueError('Healpix nside must be a power of 2')


def validate_resolution(resolution):
    """Validate Healpix nside parameter.

    Parameters
    ----------
    resolution : int
        Healpix resolution parameter.

    Raises
    ------
    TypeError
        Description
    ValueError
        Description
    TypeError : not an int type
    ValueError : not a positive factor of 2
    """
    if not isinstance(resolution, (int,)):
        raise TypeError('Healpix nside must be an integer number')
    if resolution < 0:
        raise ValueError('Healpix resolution must be greater than or equal to 0.')


def validate_order(order):
    """Validate healpix order parameter.

    Parameters
    ----------
    order : str
        Healpix order parameter.

    Raises
    ------
    TypeError
        Description
    TypeError
    """
    if order not in (NEST, RING):
        raise TypeError('Healpix order must be one of [{}, {}]'.format(NEST, RING))


def validate_inputs(nside=None, order=None):
    """Validate healpix parameters.

    Parameters
    ----------
    nside : int
        Healpix resolution parameter.
    order : str
        Healpix order
    """
    if nside is not None:
        validate_nside(nside)
    if order is not None:
        validate_order(order)


class HealpixProjector:
    """
    Attributes
    ----------
    deg2rad : TYPE
        Description
    nest : bool
        Description
    npix : TYPE
        Description
    nside : TYPE
        Description
    order : TYPE
        Description
    pixel_size : TYPE
        Description
    query_disk : TYPE
        Description
    rad2deg : TYPE
        Description
    sphere_area : TYPE
        Description
    twothirds : TYPE
        Description
    """
    twothirds = 2. / 3
    deg2rad = np.pi / 180.
    rad2deg = 180. / np.pi
    sphere_area = 4 * np.pi

    def __init__(self, nside=64, resolution=None, order='ring'):
        """Routines to work with the Healpix projection.

        The Healpix resolution can be set either with n_side or resolution.
        The two are related by

        n_side = 2**resolution

        If resolution is specified, the nside argument is not used.

        The number of pixels on the sphere is:

        N_pix = 12*n_side**2

        Parameters
        ----------
        nside : int
            Healpix n_side parameter.  Number of pixels is 12*nside**2
        resolution : int
            Optionally specify the resolution parameter.
        order : str
            Healpix ordering scheme "ring" or "nest"
        """
        if resolution is not None:
            self.nside = int(2**resolution)
        else:
            self.nside = int(nside)
        self.npix = self.nside2npix(self.nside)
        self.order = order.lower()

        self.nest = False
        if self.order.startswith("n"):
            self.nest = True

        self.pixel_area = self.sphere_area / self.npix
        self.pixel_size = np.sqrt(self.pixel_area)

    @staticmethod
    def nside2npix(nside):
        """Compute number of pixels in a map with given nside.

        Uses the formula:
            N_pix = 12*n_side**2

        Parameters
        ----------
        nside : int
            Healpix n_side parameter.

        Returns
        -------
        number of pixels : int
        """
        if nside == 0:
            return 1
        return 12 * nside * nside

    def ang2pix(self, lon, lat):
        """Convert angle on the sky to Healpix pixel number.

        This routine wraps the Healpix ang2pix function.

        Parameters
        ----------
        lon : float, ndarray
            longitude (degr)
        lat : float, ndarray
            latitude (degr)

        Returns
        -------
        pixel number : ndarray
        """
        phi = np.array(lon) * self.deg2rad
        theta = (90 - np.array(lat)) * self.deg2rad
        return healpy.ang2pix(self.nside, theta, phi, self.nest)

    def _phitheta2xy(self, phi, theta):
        """Project points phi,theta onto the healpix plane.

        The projection is described in Gorski et al (2005) and
        Calabretta & Roukema (2007).

        Parameters
        ----------
        phi : float, ndarray
            longitude angle (radians [0,2pi]))
        theta : float, ndarray
            polar angle (radians [0,pi])

        Outputs
        -------
        x, y : coordinates in the Healpix projection.

        Returns
        -------
        TYPE
            Description
        """
        z = np.cos(theta)

        pole = np.abs(z) > self.twothirds
        equ = np.logical_not(pole)

        x = np.zeros(phi.shape)
        y = np.zeros(phi.shape)

        x[equ] = phi[equ]
        y[equ] = 3. / 8. * np.pi * z[equ]

        sig = 2 - np.sqrt(3 * (1 - np.abs(z[pole])))

        y[pole] = np.pi / 4. * sig
        south = (z < 0) & pole
        y[south] *= -1

        phi_t = phi[pole] % (np.pi / 2)
        x[pole] = phi[pole] - (np.abs(sig) - 1) * (phi_t - np.pi / 4)

        xt = x - y
        yt = x + y

        return xt, yt

    @staticmethod
    def _xy2phitheta(xt, yt):
        """Inverse projection: compute phi, theta from x, y on the Healpix plane.

        Parameters
        ----------
        xt : TYPE
            Description
        yt : TYPE
            Description

        Returns
        -------
        phi,theta (radians)

        Deleted Parameters
        ------------------
        x : ndarray
            x projected coordinate
        y : ndarray
            y projected coordinate
        """
        x = 0.5 * (xt + yt)
        y = 0.5 * (-xt + yt)

        equ = np.abs(y) < np.pi / 4
        pole = np.logical_not(equ)

        phi = np.zeros(xt.shape)
        z = np.zeros(yt.shape)

        phi[equ] = x[equ]
        z[equ] = 8 / 3. / np.pi * y[equ]

        xt = x[pole] % (np.pi / 2)
        a = np.abs(y[pole]) - np.pi / 4
        b = np.abs(y[pole]) - np.pi / 2

        nonzero = b != 0
        ab = np.zeros(len(a))
        ab[nonzero] = a[nonzero] / b[nonzero]
        if not np.all(np.isfinite(ab)):
            raise ValueError("projection error")

        phi[pole] = x[pole] - ab * (xt - np.pi / 4)

        z[pole] = (1 - 1. / 3. * (2 - 4 * np.abs(y[pole]) / np.pi)**2) * y[pole] / np.abs(y[pole])

        theta = np.arccos(z)

        return phi, theta

    def random_sample(self, pixels, n=1e5):
        """Random sample points inside a healpixel.

        Parameters
        ----------
        pixels : int, ndarray
            pixel number or list of pixel numbers
        n : n
            Total number of randoms to draw

        Returns
        -------
        lon, lat : positions of randoms (degr)
        """
        if is_number(pixels):
            pixels = [int(pixels)]

        n = int(n)

        # select pixels to sample
        if len(pixels) == 1:
            pix_i = np.zeros(n, dtype=int)
        else:
            pix_i = np.random.choice(len(pixels), n)

        # compute pixel centers
        theta, phi = healpy.pix2ang(self.nside, pixels, nest=self.nest)

        # convert to healpix projection
        xc, yc = self._phitheta2xy(phi, theta)

        # this is the size of a healpix cell in the projection
        step = np.pi / 2. / self.nside

        # generate randoms in a square
        x, y = np.random.uniform(-0.5, 0.5, (2, n)) * step

        x += xc[pix_i]
        y += yc[pix_i]

        phi_out, theta_out = self._xy2phitheta(x, y)

        lon = self.rad2deg * phi_out
        lat = 90 - self.rad2deg * theta_out

        return lon, lat
