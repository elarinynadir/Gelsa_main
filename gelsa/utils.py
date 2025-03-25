"""
File: utils.py

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

import sys
import numpy as np
from scipy import optimize


def ensurelist(x):
    """
    Parameters
    ----------
    x : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    if isinstance(x, str):
        return [x]

    try:
        len(x)
    except TypeError:
        return [x]

    return x

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


def intrange(low, high, step, limit=3):
    """ """
    if low > high:
        low, high = high, low
    n = int((high - low)/step)
    n = max(limit, n)
    return np.linspace(low, high, n)


def inverse2d(func, x, y, wavelength, *args, guess=None, tol=1e-4, **kwargs):
    """ """
    points = np.transpose([x, y])
    scalar = False
    if len(points.shape) == 1:
        scalar = True
        points = points[np.newaxis, :]
    if guess is None:
        guess = points
    else:
        guess = np.ones_like(points)*guess

    out = np.zeros(points.shape, dtype='d')
    for i in range(points.shape[0]):
        r = optimize.root(
            lambda x_: points[i] - func(x_[0], x_[1], wavelength[i], *args, **kwargs),
            guess[i],
            tol=tol
        )
        out[i] = r.x
        if not r.success:
            print("Inverse error:", r.message, file=sys.stderr)
    if scalar:
        return out[0, :]
    return np.transpose(out)


def rotate_around(wcs, x, y, theta_rad):
    """ """
    ny, nx = wcs.array_shape
    x_ = x - nx/2.
    y_ = y - ny/2.
    costheta = np.cos(theta_rad)
    sintheta = np.sin(theta_rad)
    dx = x_ * costheta + y_ * sintheta
    dy = -x_ * sintheta + y_ * costheta
    return dx + nx/2, dy + ny/2
