import os
import sys
import glob
import shutil
import pickle
import hashlib
import warnings
import collections
from functools import wraps
import xml.etree.ElementTree as ET

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.utils.exceptions import AstropyWarning
import fitsio

from . import dataProductRetrieval as get


def spin():
    import time
    spinner = ["✶", "✸", "✹", "✺", "✹", "✷"]
    for i in range(1000):
        time.sleep(0.1)
        sys.stdout.write(f"\r{spinner[i%len(spinner)]}")
        sys.stdout.flush()
    sys.stdout.write("\n")


class DataStore:

    def __init__(self, username=None, password=None, password_file=None,
                 cachedir='cache'):
        """ """
        self.username = username
        self.password = password
        self.cachedir = cachedir
        if username is None and password is None and password_file is not None:
            self.load_password_file(password_file)
        if self.username is None or self.password is None:
            self.ask_password()
        print(f"got username `{self.username}` and password ***")
        self._param_cache = collections.defaultdict(lambda: collections.defaultdict(dict))
        self._spe_tile_cache = {}
        self._sir_pointing_cache = collections.defaultdict(dict)
        self._mer_filter_cache = {}

    def load_password_file(self, password_file):
        """ """
        with open(password_file) as inp:
            self.username = inp.readline().strip()
            self.password = inp.readline().strip()

    def ask_password(self):
        """ """
        self.username = input("Please enter your EAS username:").strip()
        import getpass
        self.password = getpass.getpass(f"Please your password (username: {self.username}):")

    def make_local_path(self, DataSetRelease, product, CatalogOrigin,  project,
                        ppo_id=None, subsample_name=None):
        if not CatalogOrigin:
            CatalogOrigin = ""
        if subsample_name:
            return os.path.join(self.cachedir, project, DataSetRelease, product, CatalogOrigin, ppo_id, subsample_name)
        return os.path.join(self.cachedir, project, DataSetRelease, product, CatalogOrigin)

    def clear_cache(self, DataSetRelease,
                    product='DpdLE3IDSELIDCatalog',
                    CatalogOrigin='UNKNOWN',  project='TEST', dryrun=True):
        """ """
        local_path = self.make_local_path(DataSetRelease, product,
                                          CatalogOrigin,  project)
        print(f"will remove {local_path}")
        if not dryrun:
            shutil.rmtree(local_path)

    def _read_tile_index(self, root):
        """ """
        try:
            elem = root.find('Data/TileIndex').text
        except AttributeError:
            elem = root.find('Data/TileIndexList').text
        return int(elem)

    def _read_pointing_id(self, root):
        """ """
        try:
            elem = root.find('Data/ObservationSequence/PointingId').text
        except AttributeError:
            elem = root.find('Data/Observation/PointingId').text
        return int(elem)

    def _read_filter(self, root):
        """ """
        return root.find("Data/Filter/Name").text

    def get_metadata(self, DataSetRelease, product='DpdLE3IDSELIDCatalog',
                     CatalogOrigin=None, project='EUCLID',
                     ppo_id=None, subsample_name=None, tile_index=None,
                     pointing_id=None, detector_id=None,
                     filter=None,
                     use_cache=True, pickle_name='cache.pickle',
                     fields=None, **kwargs):
        """ """
        local_path = self.make_local_path(DataSetRelease, product,
                                          CatalogOrigin,  project,
                                          ppo_id, subsample_name)
        print(f"> {local_path=}")

        metadata_list = []
        if use_cache and os.path.exists(local_path):
            # search the xml files
            cache = self._param_cache[product]
            if tile_index in self._spe_tile_cache:
                paths = [self._spe_tile_cache[tile_index]]
            elif pointing_id in self._sir_pointing_cache[product]:
                paths = [self._sir_pointing_cache[product][pointing_id]]
            elif filter in self._mer_filter_cache:
                paths = [self._mer_filter_cache[filter]]
            else:
                paths = glob.glob(local_path+"/*.xml")

            for path in paths:
                sys.stdout.write(f"\r collecting metadata files...{os.path.basename(path)}")
                sys.stdout.flush()
                root = ET.parse(path).getroot()
                if tile_index is not None:
                    this_tile = self._read_tile_index(root)
                    self._spe_tile_cache[this_tile] = path
                    if this_tile != tile_index:
                        continue
                if pointing_id is not None:
                    this_pointing = self._read_pointing_id(root)
                    self._param_cache[product]['PointingID'][this_pointing] = path
                    if this_pointing != pointing_id:
                        continue
                if filter is not None:
                    this_filter = self._read_filter(root)
                    self._mer_filter_cache[this_filter] = path
                    if this_filter != filter:
                        continue
                metadata_list.append(root)
            print(f"> loaded {len(metadata_list)} products from cache")

        # if nothing, submit new query
        if len(metadata_list) == 0:
            metadata_list = self._query_metadata(
                DataSetRelease, product, local_path,
                CatalogOrigin=CatalogOrigin, project=project,
                pointing_id=pointing_id, detector_id=detector_id,
                ppo_id=ppo_id, subsample_name=subsample_name,
                tile_index=tile_index, filter=filter, fields=fields
            )
        # write pickle cache
#         with open(pickle_path, "wb") as out:
#             pickle.dump(metadata_list, out)
        return metadata_list

    def _query_metadata(self, DataSetRelease, product, outdir,
                        CatalogOrigin='MEASURED_WIDE', project='TEST',
                        pointing_id=None, detector_id=None,
                        ppo_id=None, subsample_name=None, tile_index=None,
                        fields=None, filter=None):
        """ """
        print(f"> Query: {DataSetRelease=} {CatalogOrigin=} {product=}")
        query = f"Header.DataSetRelease={DataSetRelease}"
        if CatalogOrigin:
            query += f"&Data.CatalogDescription.CatalogOrigin={CatalogOrigin}"
        if ppo_id:
            query += f"&Header.PPOId=like{ppo_id}"
        if subsample_name:
            query += f"&Data.SelectionName={subsample_name}"
        if tile_index:
            query += f"&Data.TileIndex={tile_index}"
        if fields:
            query += f"&fields={fields}"
        if pointing_id and (product == "DpdSirScienceFrame"):
            query += f"&Data.ObservationSequence.PointingId={pointing_id}"
        if pointing_id and (product == "DpdSirLocationTable" or product == "DpdSirExtractedSpectraCollection"):
            query += f"&Data.Observation.PointingId={pointing_id}"
        if detector_id and (product == "DpdSirLocationTable" or product == "DpdSirExtractedSpectraCollection"):
            query += f"&Data.DetectorId={detector_id}"
        if filter and product == "DpdMerBksMosaic":
            query += f"&Data.Filter.Name={filter}"
        print(query)

        products = get.getMetadataXml(
            get.BASE_EAS_URL,
            product,
            query,
            project, self.username, self.password)
        metadata_list = []
        for prod in products:
            root = ET.fromstring(prod)
            metadata_list.append(root)
            product_id = root.find("Header/ProductId").text
            path = os.path.join(outdir, product_id+".xml")
            os.makedirs(outdir, exist_ok=True)
            with open(path, "wt") as out:
                out.write(prod)
        return metadata_list

    def _retrieve_data(self, metadata, outdir, count=0, keys=(".//FileName",)):
        """ """
        outdir = os.path.join(outdir, "data")
        os.makedirs(outdir, exist_ok=True)
        files = []
        for key in keys:
            files += [f.text for f in metadata.findall(key)]
        file_list = []
        for f in files:
            path = os.path.join(outdir, f)
            if path.endswith(".gz"):
                path_ = path[:-3]
                if os.path.exists(path_):
                    path = path_
            file_list.append(path)
            if os.path.isfile(path):
                # print("File %s exists locally." % (f))
                pass
            else:
                # print("Start retrieving " + f + " at " +
                    #   str(datetime.datetime.now())+" :")
                get.downloadDssFile(get.BASE_DSS_URL, f,
                                    self.username, self.password, count=count)
                # print("Finished retrieving " + f +
                    #   " at " + str(datetime.datetime.now()))
                os.rename(f, path)
        return file_list

    def read_fits(self, path, columns=None, hdu=1):
        """ """
        try:
            import fitsio
            try:
                cat = fitsio.read(path, columns=columns, hdu=hdu)
            except OSError:
                # print(f"\rfailed reading {path} with fitsio.")
                raise
            except (KeyError, ValueError):
                cat = Table(fitsio.read(path))
                print(f"column list:", cat.columns)
                raise
            cat = Table(cat)
        except (ModuleNotFoundError, OSError):
            cat = Table.read(path)
            if columns:
                cat = cat[columns]
        return cat

    def write_fits(self, data, path):
        """ """
        print(f"writing {path}")
        data.write(path, format='fits', overwrite=True)

    def hash(self, hasher, args):
        """ """
        if not args:
            return
        elif type(args) == tuple:
            for s in args:
                self.hash(hasher, s)
        else:
            hasher.update(args.encode("utf-8"))

    def load_sel_catalog(self, DataSetRelease, product='DpdLE3IDSELIDCatalog',
                         CatalogOrigin='UNKNOWN', project='TEST',
                         tile_index=None,
                         subsample_name=None, ppo_id=None,
                         download_only=False, columns=None, use_cache=True,
                         download_keys=('.//FileName',),
                         **kwargs):
        """ """
        print(DataSetRelease, product,
                                          CatalogOrigin,  project,
                                          ppo_id, subsample_name)
        local_path = self.make_local_path(DataSetRelease, product,
                                          CatalogOrigin,  project,
                                          ppo_id, subsample_name)

        args = (DataSetRelease, product, CatalogOrigin, project, columns, subsample_name, ppo_id)
        hasher = hashlib.sha256(usedforsecurity=False)
        self.hash(hasher, args)
        key = hasher.hexdigest()
        cachecat_path = os.path.join(local_path, f"cachecat_{key}.fits")
        if use_cache and os.path.exists(cachecat_path):
            cat = self.read_fits(cachecat_path)
            print(f"loaded {cachecat_path} total rows {len(cat)}")
            return cat

        print(f"> loading catalog {DataSetRelease=} {product=} {CatalogOrigin=} {project=} {subsample_name=} {ppo_id=}")
        metadata_list = self.get_metadata(DataSetRelease, product,
                                          CatalogOrigin=CatalogOrigin,
                                          tile_index=tile_index,
                                          project=project,
                                          ppo_id=ppo_id,
                                          subsample_name=subsample_name,
                                          use_cache=use_cache,
                                          **kwargs)
        if len(metadata_list) == 0:
            raise ValueError("No data products found")

        tile_index_cache = collections.defaultdict(int)
        concat = []
        row_count = 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            for i, metadata in enumerate(metadata_list):
                try:
                    tile_index = self._read_tile_index(metadata)
                except ValueError:
                    tile_index = 0
                tile_index_cache[tile_index] += 1
                if tile_index_cache[tile_index] > 1:
                    print("Warning! Tile More than one product found for tile {tile_index} (count: {tile_index_cache[tile_index]}).")
                path = self._retrieve_data(metadata, local_path, count=i+1, keys=download_keys)
                if not download_only:
                    cat = self.read_fits(path[0], columns=columns)
                    cat['TileIndex'] = tile_index * np.ones(len(cat))
                    row_count += len(cat)
                    concat.append(cat)
                    sys.stdout.write(f"\rLoaded tile {i+1}/{len(metadata_list)}: total rows: {row_count}         ")
                    sys.stdout.flush()
            if not download_only:
                concat = vstack(concat)
                print(f"> results: concatenated {len(metadata_list)} tiles, total rows: {len(concat)}")
                self.write_fits(concat, cachecat_path)
                return concat

    def load_subsample_catalog(self, DataSetRelease, product='DpdLE3IDSELIDSubsampledCatalog',
                         CatalogOrigin='UNKNOWN', project='TEST', subsample_name='EuclidWide_RedshiftBin0',
                         ppo_id='%FIELDB%',
                         download_only=False, columns=None, use_cache=True,
                         **kwargs):
        return self.load_sel_catalog(DataSetRelease, product, CatalogOrigin=CatalogOrigin,
                                     project=project, subsample_name=subsample_name, ppo_id=ppo_id, download_only=download_only,
                                     columns=columns, use_cache=use_cache, **kwargs)

    def load_subsample_randoms(self, DataSetRelease, product='DpdLE3IDSELIDSubsampledCatalog',
                         CatalogOrigin='RANDOM_WIDE', project='TEST', subsample_name='EuclidWide_RedshiftBin0',
                         ppo_id='%FIELDB%',
                         download_only=False, columns=None, use_cache=True,
                         **kwargs):
        return self.load_sel_catalog(DataSetRelease, product, CatalogOrigin,
                                     project=project, subsample_name=subsample_name, ppo_id=ppo_id, download_only=download_only,
                                     columns=columns, use_cache=use_cache,
                                     **kwargs)


    def load_vmsp_randoms(self, DataSetRelease, project='TEST', **kwargs):
        return self.load_sel_catalog(DataSetRelease,
                                     'DpdLE3IDVMSPRandomCatalog',
                                     CatalogOrigin='RANDOM_WIDE',
                                     project=project, **kwargs)

    def load_vmsp_noise_catalog(self, DataSetRelease, project='TEST', **kwargs):
        return self.load_sel_catalog(DataSetRelease,
                                     'DpdLE3IDVMSPRandomCatalog',
                                     CatalogOrigin='MEASURED_WIDE',
                                     project=project, **kwargs)

    def get_nisp_polygons(self, DataSetRelease, project='TEST'):
        """Load the NISP detector bounding boxes from the VMSP data product"""
        metadata_list = self.get_metadata(DataSetRelease,
                                          'DpdLE3IDVMSPRandomCatalog',
                                          'RANDOM_WIDE',
                                          project=project)
        cache = {}
        footprint_list = []
        for metadata in metadata_list:
            for footprint_meta in metadata.findall('Data/VmspDetectorFootprintList/VmspDetectorFootprint'):
                pointing_id = footprint_meta.find('PointingId').text
                if pointing_id in cache:
                    continue
                cache[pointing_id] = 1
                poly_list_meta = footprint_meta.findall('VmspPolygonList/Polygon')
                poly_list = []
                for poly_meta in poly_list_meta:
                    vert1 = poly_meta.findall('Vertex/C1')
                    vert2 = poly_meta.findall('Vertex/C2')
                    x = [float(v.text) for v in vert1]
                    y = [float(v.text) for v in vert2]
                    x.append(x[0])
                    y.append(y[0])
                    poly_list.append((x, y))
                footprint_list.append(poly_list)
        return footprint_list

    def get_area(self, DataSetRelease, project='TEST'):
        """Load the NISP detector bounding boxes from the VMSP data product"""
        metadata_list = self.get_metadata(DataSetRelease,
                                          'DpdLE3IDVMSPRandomCatalog',
                                          'RANDOM_WIDE',
                                          project=project)
        area = 0
        for metadata in metadata_list:
            area_elem = metadata.find('Data/Area')
            if area_elem is None:
                continue
            area += float(area_elem.text)
        return area

    def read_spe_fits(self, path, pack_out={}):
        """ """
        pack = {}
        with fits.open(path) as hdul:
            for hdu in hdul:
                try:
                    name = hdu.header['EXTNAME']
                except KeyError:
                    pass
                try:
                    pack[name] = hdu.data[:]
                except (AttributeError, TypeError):
                    pass
        if 'SPE_LINE_FEATURES_CAT' in pack.keys():
            table_type = 'line_table'
        elif 'SPE_QUALITY' in pack.keys():
            table_type = 'z_table'
        else:
            print(f"Unkown table type! {path} hdus: {pack.keys()}")
            table_type = 'unknown_table'
        if table_type in pack_out:
            print(f"Warning table type {table_type} already loaded.")
        pack_out[table_type] = pack
        return pack_out

    def load_spe_table(self, DataSetRelease, product='DpdSpePfOutputCatalog',
                         project='EUCLID', CatalogOrigin='UNKNOWN', tile_index=None,
                         download_only=False, use_cache=True,
                         **kwargs):
        """ """
        local_path = self.make_local_path(DataSetRelease, product,
                                          CatalogOrigin,  project,
                                          )

        print(f"> loading SPE table {DataSetRelease=} {product=} {CatalogOrigin=} {project=}")
        metadata_list = self.get_metadata(DataSetRelease, product,
                                          CatalogOrigin=CatalogOrigin,
                                          project=project,
                                          tile_index=tile_index,
                                          use_cache=use_cache,
                                          **kwargs)
        if len(metadata_list) == 0:
            raise ValueError("No data products found")

        pack_list = []
        tile_index_cache = collections.defaultdict(int)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            for i, metadata in enumerate(metadata_list):
                tile_index = int(metadata.find('Data/TileIndex').text)
                tile_index_cache[tile_index] += 1
                if tile_index_cache[tile_index] > 1:
                    print("Warning! Tile More than one product found for tile {tile_index} (count: {tile_index_cache[tile_index]}).")
                path_list = self._retrieve_data(metadata, local_path, count=i+1,
                                           keys=('Data/SpeZStorage/DataContainer/FileName',
                                                 'Data/SpeLineFeaturesStorage/DataContainer/FileName')
                                           )
                if not download_only:
                    pack = {}
                    for path in path_list:
                        pack = self.read_spe_fits(path, pack)
                    pack['TileIndex'] = tile_index
                    pack_list.append(pack)
                    sys.stdout.write(f"\rLoaded tile {i+1}/{len(metadata_list)}         ")
                    sys.stdout.flush()
            if not download_only:
                return pack_list

    def load_mer_catalog(self, DataSetRelease, product='DpdMerFinalCatalog',
                       project='EUCLID',
                       tile_index=None,
                       download_only=False, use_cache=True,
                       **kwargs):
        """ """
        return self.load_sel_catalog(DataSetRelease,
                                     product,
                                     CatalogOrigin=None,
                                     tile_index=tile_index,
                                     download_only=download_only,
                                     use_cache=use_cache,
                                     project=project,
                                     download_keys=('Data/DataStorage/DataContainer/FileName',),
                                     **kwargs)

    def load_mer_tiles(self, tile_index=None, use_cache=True):
        """ """
        return self.get_metadata('NA', 'DpdMerTile',
                                 CatalogOrigin=None,
                                 tile_index=tile_index,
                                 project='EUCLID',
                                #  fields="Data.InnerNUNIQIndex.MerHealpixList",
                                 use_cache=use_cache)

    def load_mer_stack(self, DataSetRelease, product='DpdMerBksMosaic',
                       project='EUCLID',
                       tile_index=None,
                       filter='H',
                       download_only=False, use_cache=True,
                       **kwargs):
        """ """
        local_path = self.make_local_path(DataSetRelease, product,
                                          None,  project,
                                          )

        metadata_list = self.get_metadata(DataSetRelease, product,
                                 CatalogOrigin=None,
                                 tile_index=tile_index,
                                 filter=filter,
                                 project=project,
                                 use_cache=use_cache, **kwargs)
        if len(metadata_list) == 0:
            raise ValueError("No data products found")

        pack_list = []
        for i, metadata in enumerate(metadata_list):
            path_list = self._retrieve_data(metadata, local_path, count=i+1,
                                           keys=('Data/DataStorage/DataContainer/FileName',)
                                           )
            pack_list.append(path_list)
        return pack_list

    def load_sir_location_table(self, DataSetRelease, pointing_id=None,
                               product='DpdSirLocationTable', project='EUCLID',
                               detector_id='11',
                               use_cache=True, grisms=None, **kwargs):
        """ """
        local_path = self.make_local_path(DataSetRelease, product,
                                          None,  project,
                                          )

        print(f"> loading SIR loc table {DataSetRelease=} {product=} {project=} {pointing_id=}")
        metadata_list = self.get_metadata(DataSetRelease, product=product,
                                          project=project,
                                          CatalogOrigin=None,
                                          pointing_id=pointing_id,
                                          detector_id=detector_id,
                                          use_cache=use_cache,
                                          **kwargs)
        if len(metadata_list) == 0:
            raise ValueError("No data products found")
        pack_list = []
        for i, metadata in enumerate(metadata_list):
            if grisms is not None:
                print(metadata.find("Data/GrismWheelPos").text)
                if metadata.find("Data/GrismWheelPos").text not in grisms:
                    continue
            path_list = self._retrieve_data(metadata, local_path, count=i+1,
                                           keys=('Data/CollectionFile/DataContainer/FileName',)
                                           )
            pack_list.append(path_list)
        return pack_list

    def load_sir_science_frame(self, DataSetRelease, pointing_id=None,
                               product='DpdSirScienceFrame', project='EUCLID',
                               use_cache=True, grisms=None, **kwargs):
        """ """
        local_path = self.make_local_path(DataSetRelease, product,
                                          None,  project,
                                          )

        print(f"> loading SIR frame {DataSetRelease=} {product=} {project=} {pointing_id=}")
        metadata_list = self.get_metadata(DataSetRelease, product=product,
                                          project=project,
                                          CatalogOrigin=None,
                                          pointing_id=pointing_id,
                                          use_cache=use_cache,
                                          **kwargs)
        pack_list = []
        for i, metadata in enumerate(metadata_list):
            if grisms is not None:
                print(metadata.find("Data/Grism").text)
                if metadata.find("Data/Grism").text not in grisms:
                    continue
            path_list = self._retrieve_data(metadata, local_path, count=i+1,
                                           keys=('Data/DataStorage/DataContainer/FileName',)
                                           )
            pack_list.append(path_list)
        return pack_list

    def load_sir_pack(self, DataSetRelease, pointing_id_list=None, project='EUCLID',
                               use_cache=True, grisms=None, **kwargs):
        """ """
        pack = []
        for pointing_id in pointing_id_list:
            frame_pack = self.load_sir_science_frame(
                DataSetRelease=DataSetRelease,
                pointing_id=pointing_id,
                project=project,
                use_cache=use_cache,
                grisms=grisms
            )
            if len(frame_pack) == 0:
                continue
            loctable_pack = self.load_sir_location_table(
                DataSetRelease=DataSetRelease,
                pointing_id=pointing_id,
                project=project,
                use_cache=use_cache,
                grisms=grisms
            )
            pack.append({
                'frame_path': frame_pack[0][0],
                'loctable_path': loctable_pack[0][0]
            })
        return pack


    def load_extracted_spectra(self, DataSetRelease, pointing_id=None,
                               product='DpdSirExtractedSpectraCollection', project='EUCLID',
                               detector_id=None,
                               use_cache=True, **kwargs):
        local_path = self.make_local_path(DataSetRelease, product,
                                          None,  project,
                                          )

        print(f"> loading extracted spectra {DataSetRelease=} {product=} {project=} {pointing_id=}")
        metadata_list = self.get_metadata(DataSetRelease, product,
                                          project=project,
                                          CatalogOrigin=None,
                                          pointing_id=pointing_id,
                                          detector_id=detector_id,
                                          use_cache=use_cache,
                                          **kwargs)
        pack_list = []
        for i, metadata in enumerate(metadata_list):
            path_list = self._retrieve_data(metadata, local_path, count=i+1,
                                           keys=('Data/CollectionFile/DataContainer/FileName',)
                                           )
            pack_list.append(path_list)
        return pack_list
