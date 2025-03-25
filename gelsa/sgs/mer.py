import os
import glob
import warnings
import numpy as np
from astropy import units
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy import visualization
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning

from . import dputils
from . import tile_collection
from . import dataProductRetrieval


def _sign(a):
    """ Return the sign of a given number. """
    return a and (-1, 1)[a > 0] if a != 0 else 1


class MER:
    def __init__(self, metadata_dir='tmp', data_dir='data', password_file=None):
        """ """
        self.metadata_dir = metadata_dir
        self.password_file = password_file
        self.data_dir = os.path.join(metadata_dir, data_dir)
        self.load_tiles()

        self.image_cache = {}

    def load_tiles(self):
        """ """
        self.tc = tile_collection.TileCollection()
        self.tc.read_tile_directory(self.metadata_dir)

    def radec_to_tile(self, ra, dec):
        """Find tile index containing point ra, dec

        Parameters
        ----------
        ra : float
            RA coordinate
        dec : float
            Dec coordinate

        Returns
        -------
        int
            tile index
        """
        return self.tc.which_tile(ra, dec)

    @staticmethod
    def objectid_to_radec(objectID):
        """Decode RA and Dec from object ID int"""
        if not objectID:
            raise ValueError(f"Invalid ObjectID: {objectID}")
        _RA_ACCURACY = int(1e7)
        _DEC_ACCURACY = int(1e7)
        dec_sign = _sign(objectID)
        dec = abs(objectID) % (_DEC_ACCURACY * int(1e2))
        RA = (abs(objectID) - dec) / int(1e9) / _RA_ACCURACY
        Dec = dec_sign * dec / _DEC_ACCURACY
        return RA, Dec

    def load_catalog(self, product=None, path=None):
        """ """
        if product is None:
            product = dputils.DataProduct(path)
        filename = product.get_element('Data/DataStorage/DataContainer/FileName')
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            dataProductRetrieval.download(filename,
                                          self.data_dir,
                                          password_file=self.password_file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            cat = Table.read(path)
        return cat

    def get_tile_catalog(self, ra=None, dec=None, object_id=None, tile_index=None, pattern='MER_*catalog*.xml'):
        """ """
        if not tile_index:
            if not ra or not dec:
                ra, dec = self.objectid_to_radec(object_id)
            tile_index = self.radec_to_tile(ra, dec)

        product_list = []
        for path in glob.glob(os.path.join(self.metadata_dir, pattern)):
            product = dputils.DataProduct(path)
            if product.tile_index == tile_index:
                product_list.append(product)

        count = len(product_list)
        print(f"> found {count} catalogs with tile index {tile_index}")
        if count > 1:
            print("loading the first one")
        product = product_list[0]
        cat = self.load_catalog(product=product)
        return cat

    def load_image(self, product=None, path=None, hdu=0):
        """ """
        if product is None:
            product = dputils.DataProduct(path)
        filename = product.get_element('Data/DataStorage/DataContainer/FileName')
        print(f"loading {filename}")
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            dataProductRetrieval.download(filename,
                                          self.data_dir,
                                          password_file=self.password_file)
        with fits.open(path) as hdul:
            data = hdul[hdu].data[:]
            header = hdul[hdu].header
            wcs = WCS(header)
        return data, wcs, header

    def get_image(self, ra=None, dec=None, object_id=None,
                  width=51, height=51,
                  filter_name='NIR_H', pattern='MER_*mosaics*.xml', return_wcs=False):
        """ """
        if ra is None or dec is None:
            ra, dec = self.objectid_to_radec(object_id)
        tile_index = self.radec_to_tile(ra, dec)

        key = pattern+filter_name
        if key not in self.image_cache:
            self.image_cache[key] = {}
            
        if tile_index in self.image_cache[key]:
            print(f"found cached image {key} {tile_index}")
            image, wcs = self.image_cache[key][tile_index]
        else:
            product_list = []
            for path in glob.glob(os.path.join(self.metadata_dir, pattern)):
                product = dputils.DataProduct(path)
                if (product.tile_index != tile_index):
                    continue
                try:
                    if product.filter_name != filter_name:
                        continue
                except KeyError:
                    pass
                product_list.append(product)
            count = len(product_list)
            print(f"> found {count} images with {tile_index=} and {filter_name=}")
            if count > 1:
                print("loading the first one")
            product = product_list[0]
            image, wcs, _ = self.load_image(product=product)
            self.image_cache[key][tile_index] = image, wcs
            
        sky = SkyCoord(ra, dec, unit=units.deg)
        x, y = wcs.world_to_pixel(sky)
        x = int(np.round(x))
        y = int(np.round(y))
        x0 = x - width//2
        y0 = y - height//2
        x1 = x0 + width
        y1 = y0 + height
        x0_ = 0
        y0_ = 0
        x1_ = width
        y1_ = height
        if x0 < 0:
            x0_ = -x0
            x0 = 0
        if y0 < 0:
            y0_ = -y0
            y0 = 0
        if x1 > image.shape[1]-1:
            x1_ = image.shape[1] - 1 - x0
            x1 = image.shape[1] - 1
        if y1 > image.shape[0]-1:
            y1_ = image.shape[0] - 1 - y0
            y1 = image.shape[0] - 1
        image_stamp = np.zeros((height, width), dtype=image.dtype)
        image_stamp[y0_:y1_, x0_:x1_] = image[y0:y1, x0:x1]
        if return_wcs:
            wcs = wcs[y0:y1, x0:x1]
            return image_stamp, wcs
        return image_stamp

    def get_segmentation_map(self, *args, **kwargs):
        """ """
        return self.get_image(*args, pattern='*compressed_segmentation_map*', **kwargs)

    def get_color_image(self, ra=None, dec=None, object_id=None,
                        width=51, height=51,
                        filter_list=['NIR_H', 'NIR_J', 'VIS'],
                        stretch=2, Q=5, return_wcs=False, **kwargs):
        """ """
        if ra is None or dec is None:
            ra, dec = self.objectid_to_radec(object_id)
        rgb = []
        for filter_name in filter_list:
            stamp = self.get_image(ra, dec, width=width, height=height, filter_name=filter_name, return_wcs=return_wcs, **kwargs)
            if return_wcs:
                stamp, wcs = stamp
            rgb.append(stamp)
        rgb_image = visualization.make_lupton_rgb(*rgb, stretch=stretch, Q=Q)
        if return_wcs:
            return rgb_image, wcs
        return rgb_image
