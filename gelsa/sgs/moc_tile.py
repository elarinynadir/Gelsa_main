"""
File: moc_tile.py

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
import healpy
import xml.etree.ElementTree as ET

from . import healpix_projection as hp


def fromNuniqToPixel(nuniq):
    """
    Conversion from NUNIQ to order, nside and index

    Copied from MER_Healpix

    Parameters
    ----------
    nuniq : numpy.ndarray, int
        Array of MOC indices
    """
    # compute the order from the NUNIQ
    order   = np.floor_divide(np.log2(np.floor_divide(nuniq, 4)), 2).astype(np.int64)

    # compute the nside
    nside   = (2**order).astype(int)

    # compute the index number
    indices = nuniq - 4*(4**order)

    # return everything
    return order, nside, indices


def fromPixelToNuniq(nside, pixel):
    """
    Conversion from NSIDE and index to NUNIQ

    Copied from MER_Healpix

    it is:
    order = int(math.log(nside, 2))

    then the definition is (https://arxiv.org/ftp/arxiv/papers/1505/1505.02937.pdf):
    nuniq = 4*(4**order)+indices
    """
    # convert from nside and index to nuniq
    nuniq = 4*(nside*nside)+pixel

    # return the sorted array
    return np.sort(nuniq)



class MOCTile:
    """Tile object
    """

    keys = (
        'Data/RaCen',
        'Data/DecCen',
        'Data/InnerNUNIQIndex'
    )

    scheme = 'MOC'
    DataNode = 'Data'
    MocNode = 'InnerNUNIQIndex'
    OuterMocNode = 'OuterNUNIQIndex'

    def __init__(self, nuniq=None, nside=None, indices=None, header=None, **kwargs):
        """Tile object
        Can be initialized with a list of NUNIQ indices
        or a list of nside and corresponding indices.

        Properties
        ----------
        nuniq : list (optional)
            list of nuniq MOC indices
        nside : list (optional)
            list of nside
        indices : list (optional)
            list of indices
        """
        if header is None:
            header = {}
        self._header = header.copy()
        self._header.update(kwargs)

        self._load_moc(nuniq, nside, indices)

    @staticmethod
    def read_xml(filename):
        """Load the MOC specification from an XML file

        Parameters
        ----------
        filename : str
            path to XML file
        """
        tree = ET.parse(filename)
        root = tree.getroot()

        data = {}
        for key in MOCTile.keys:
            node = root.find(key)
            if node is None:
                raise ValueError
            data[key] = node.text

        data_node = root.find(MOCTile.DataNode)

        header = {}
        for child in data_node:
            header[child.tag] = child.text

        moc_string = header[MOCTile.MocNode]

        nuniq = [int(v) for v in moc_string.split()]

        return MOCTile(nuniq=nuniq, header=header)

    def write_xml(self, filename, template_file):
        """ """
        tree = ET.parse(template_file)
        root = tree.getroot()

        data_node = root.find(MOCTile.DataNode)

        moc = data_node.find(MOCTile.MocNode)
        moc.text = " ".join([str(v) for v in self._nuniq])

        moc = data_node.find(MOCTile.OuterMocNode)
        moc.text = " ".join([str(v) for v in self._nuniq])

        # set additional data tags that were passed in
        for key, value in self._header.items():
            try:
                data_node.find(key).text = value
            except AttributeError:
                pass

        tree.write(filename)

    def _load_moc(self, nuniq=None, nside=None, indices=None):
        """Load MOC map from an array of indices

        Can be initialized with a list of NUNIQ indices
        or a list of nside and corresponding indices.

        Properties
        ----------
        nuniq : list (optional)
            list of nuniq MOC indices
        nside : list (optional)
            list of nside
        indices : list (optional)
            list of indices
        """
        if nside is None or indices is None:
            if nuniq is None:
                raise ValueError("MOC: Must specify nuniq or both nside and indices")
            order, nside, indices = fromNuniqToPixel(nuniq)
            self._nuniq = nuniq
        else:
            nside = np.array(nside).astype(int)
            indices = np.array(indices).astype(int)
            self._nuniq = fromPixelToNuniq(nside, indices)

        if len(nside) != len(indices):
            raise ValueError("MOC: nside and indices arrays must have same length")

        self._moc = {}
        self._area = {}

        unique_nside = np.unique(nside)

        for n in unique_nside:
            sel = nside == n
            ind_sel = indices[sel]
            self._moc[n] = ind_sel
            self._area[n] = healpy.nside2pixarea(n, degrees=True) * len(ind_sel)

        self._total_area = np.sum(list(self._area.values()))

    def __getattr__(self, key):
        """Access metadata"""
        if key in self._header:
            return self._header[key]
        raise AttributeError

    @property
    def area(self):
        """ The area of the tile in square degrees
        """
        return self._total_area

    def degrade_nside(self, pix, nside_in, nside_out):
        """ """
        if nside_out == nside_in:
            return pix
        if nside_out > nside_in:
            raise ValueError(f"nside_out={nside_out} must be less than or equal to nside_in={nside_in}")
        f = (nside_in // nside_out)**2
        return pix // f

    @property
    def enclosing_pixels(self):
        """ """
        try:
            return self._enclosing_pixels
        except AttributeError:
            pass
        nside_target = min(self._moc.keys())
        assert(nside_target>0)
        enclosing_pix = []
        for nside, pix in self._moc.items():
            enclosing_pix.append(self.degrade_nside(pix, nside, nside_target))
        enclosing_pix = np.concatenate(enclosing_pix)
        enclosing_pix = np.unique(enclosing_pix)
        self._enclosing_pixels = (nside_target, enclosing_pix)
        assert len(enclosing_pix) > 0
        return self._enclosing_pixels

    def in_tile(self, lon, lat):
        """Check if a point is inside the tile.
        True indicates that the point is in the tile and False otherwise.

        Parameters
        ----------
        lon : numpy.ndarray, float
            longitude coordinate
        lat : numpy.ndarray, float
            latitude coordinate

        Returns
        -------
        numpy.ndarray, bool
            True if the point is inside, False otherwise
        """
        inside = np.zeros(len(lon), dtype=bool)

        nside_min, indices = self.enclosing_pixels

        # first select points that are near to the
        # cell of interest by finding the enclosing
        # healpix cells at low resolution
        pix_nside_min = healpy.ang2pix(nside_min, lon, lat, nest=True, lonlat=True)
        select = np.zeros(len(pix_nside_min), dtype=bool)
        for j in indices:
            select = np.logical_or(select, pix_nside_min == j)

        # select only the points of interest
        lon_sel = lon[select]
        lat_sel = lat[select]
        inside_sel = np.zeros(len(lon_sel), dtype=bool)

        # for each point of interest, check if it is
        # inside the MOC
        nside_max = max(self._moc.keys())
        pix_nside_max = healpy.ang2pix(nside_max, lon_sel, lat_sel, nest=True, lonlat=True)

        for nside, indices in self._moc.items():
            pix = self.degrade_nside(pix_nside_max, nside_max, nside)
            for j in indices:
                inside_sel = np.logical_or(inside_sel, pix == j)

        inside[select] = inside_sel

        return inside

    def random_sample(self, count):
        """ Random sample the tile

        Parameters
        ----------
        count : int
            Number of points to draw

        Returns
        -------
        np.ndarray, np.ndarray
            Lon and lat of random points inside tile
        """
        samples = []

        pack = []
        frac_area = []
        for nside, indices in self._moc.items():
            frac_area.append(self._area[nside] / self._total_area)
            pack.append((nside, indices))

        ncells = len(pack)

        sel = np.random.choice(range(ncells), size=count, p=frac_area)

        for i in range(ncells):
            nside, indices = pack[i]
            n = np.sum(sel == i)
            proj = hp.HealpixProjector(nside=nside, order='nest')
            samples.append(proj.random_sample(indices, n=n))

        return np.hstack(samples)
