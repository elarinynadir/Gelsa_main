import os
import sys
import glob
import numpy as np
import healpy
import pickle

from . import moc_tile


class TileCollection:

    nside_coarse = 512

    def __init__(self):
        """ """
        self._coarse_map = None

    def save(self, savefile):
        """Save a pickle of the tile collection """
        pickle.dump((self.tile_map, self.filenames, self.tile_ids, self.filename_lookup), open(savefile, 'wb'))
        print("save file written to {savefile}")

    def load(self, savefile):
        """ Load the tile collection from a pickle"""
        self.tile_map, self.filenames, self.tile_ids, self.filename_lookup = pickle.load(open(savefile, "rb"))
        print("save file loaded from {savefile}")

    def read_tile_directory(self, dir, pattern='EUC_TILE_*.xml', limit=0):
        """ Load all tile files in a directory. """
        print(f"\U0001F440 Looking in directory '{dir}'")
        file_list = []
        for path in glob.glob(os.path.join(dir, pattern)):
            file_list.append(path)
            if limit > 0:
                if len(file_list) >= limit:
                    break
        self.read_filenames(file_list)

    def read_filenames(self, file_list):
        """ Load tile files from a list of filenames"""
        print(f"\U0001F440 Loading {len(file_list)} tiles ...")
        self._coarse_map = None
        self.tile_map = {}
        self.filenames = []
        self.tile_ids = []
        self.filename_lookup = {}
        tile_i = 0
        for tile_i, filename in enumerate(file_list):
            tile = moc_tile.MOCTile.read_xml(filename)
            tile_id = int(tile.TileIndex)  # assume this is unique

            self.filenames.append(filename)
            self.tile_ids.append(tile_id)
            if tile_id in self.filename_lookup:
                print(f"Warning multiple tiles have same id {tile_id} {filename} {self.filename_lookup[tile_id]}", file=sys.stderr)
            self.filename_lookup[tile_id] = filename

            for nside, pix in tile._moc.items():
                if nside not in self.tile_map:
                    self.tile_map[nside] = {}
                for p in pix:
                    if p in self.tile_map[nside]:
                        print(f"Warning! pixel found in two tiles! nside {nside} pix {p}")
                    self.tile_map[nside][p] = tile_i
        print(f"\u2714 Loaded {len(self.filenames)} tiles.")

    def read_filelist(self, path):
        """Read a file that contains a list of paths to load"""
        print(f"\U0001F440 Reading file '{path}'")

        file_list = []
        for line in open(path):
            filename = line.strip()
            if not filename:
                continue
            if filename.startswith("#"):
                continue
            if not os.path.exists(filename):
                continue
            file_list.append(filename)
        print(f"\u2714 Found {len(file_list)} files listed in '{path}'.")
        self.read_filenames(file_list)

    def get_path_from_tile_id(self, tile_id):
        """ """
        return self.filename_lookup[tile_id]

    def _which_tile(self, ra, dec):
        """Return the tile that covers a given RA, Dec position

        Returns the path to the tile file.
        Returns None if there is no overlap.

        Parameters
        ----------
        ra : float
            RA coordinate (degrees)
        dec : float
            Dec coordinate (degrees)

        Returns
        -------
        str : path to tile file or None if no overlap
        """
        for nside, map in self.tile_map.items():
            pix = healpy.ang2pix(nside, ra, dec, lonlat=True, nest=True)
            if pix in map:
                return self.tile_ids[map[pix]]
        return -1

    def which_tile(self, ra, dec):
        """ Return the integer tile ID or a list of tile IDs that
        contain the point(s) RA, Dec.

        If the input RA, Dec are floats, the output will be the
        tile ID or None.

        If the input ra, dec are lists, the output will be a list
        of tile ids (or -1). The output has the same length as
        the input.

        Parameters
        ----------
        ra : float or list
            RA coordinate
        dec : float or list
            Dec coordinate

        Returns
        -------
        int or list : tile_id
        """
        try:
            return [self._which_tile(ra[i], dec[i]) for i in range(len(ra))]
        except TypeError:
            return self._which_tile(ra, dec)

    def which_unique_tiles(self, ra, dec):
        """Return a list of unique tile ids that contain the points
        ra, dec.

        The output is a list of one or more unique IDs of tiles that
        contain the points. So the output does not necessarily have
        the same length as the input. If no tile contains the points,
        an empty list will be returned.

        Parameters
        ----------
        ra : list
            RA coordinates
        dec : float or list
            Dec coordinates

        Returns
        -------
        list : tile_id
        """
        tile_ids = [self._which_tile(ra[i], dec[i]) for i in range(len(ra))]
        # remove duplicates by putting ids in a dictionary
        cache = {t : 1 for t in tile_ids}
        # remove None
        try:
            del cache[None]
        except KeyError:
            pass
        # return unique tile ids
        return list(cache.keys())

    def check_inside_coarse(self, ra, dec, fwhm=1, thresh=.01):
        """ """
        if self._coarse_map is None:
            self._coarse_map = self.make_map(nside_map=self.nside_coarse)
            self._coarse_map = healpy.smoothing(
                self._coarse_map,
                fwhm=fwhm*np.pi/180,
                nest=True)
            self._coarse_map = self._coarse_map > thresh

        pix = healpy.ang2pix(self.nside_coarse, ra, dec,
                             nest=True, lonlat=True)
        visible = np.ones(len(pix), dtype=bool)
        for i in range(len(pix)):
            visible[i] = self._coarse_map[pix[i]] > 0
        return visible

    def fill_pixel(self, map, pix, nside):
        """ """
        nside_map = healpy.npix2nside(len(map))
        if nside_map == nside:
            # pixel is same resolution as map
            map[pix] = 1
        elif nside_map < nside:
            # pixel is higher resolution than map
            # degrade the pixel
            f = int((nside/nside_map)**2)
            map[int(pix/f)] = 1
        else:
            # pixel is lower resolution than map
            # fill in the pixel
            f = int((nside_map/nside)**2)
            for i in range(f):
                map[pix*f + i] = 1

    def make_map(self, nside_map=1024):
        """ """
        map = np.zeros(12*nside_map**2)
        for nside in self.tile_map.keys():
            for pix in self.tile_map[nside].keys():
                self.fill_pixel(map, pix, nside)
        return map
