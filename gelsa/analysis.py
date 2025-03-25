import numpy as np
from scipy.ndimage import median_filter
import tqdm

from .spec_crop import NoExtraction
from . sgs import datastore


class Analysis:
    GRISMS = ('RGS000', 'RGS180', 'BGS000')

    def __init__(self, G, username=None, password=None, password_file=None):
        """ """
        self.G = G
        self.DS = datastore.DataStore(username=username, password=password,
                                      password_file=password_file)

    def read_science_frames(self, DataSetRelease, pointing_id_list=None,
                            grisms=None):
        """Read available SIR science frames

        Parameters
        ----------
        DataSetRelease
        pointing_id_list
        grisms
        """
        if grisms is None:
            grisms = self.GRISMS
        self.sir_pack = self.DS.load_sir_pack(
            DataSetRelease=DataSetRelease,
            pointing_id_list=pointing_id_list,
            grisms=grisms,
        )
        print(f"Loaded {len(self.sir_pack)} SIR science frames")

    def crop(self, ra, dec, redshift, padx=5, pady=25):
        """Process the crops

        Parameters
        ----------
        ra
        dec
        redshift
        """
        self.ra = ra
        self.dec = dec
        self.redshift = redshift

        self.crop_list = []

        for pack in tqdm.tqdm(self.sir_pack):
            frame = self.G.load_frame(**pack)
            try:
                det_crops = frame.cutout(ra, dec, redshift, padx=padx, pady=pady)
            except ValueError:
                continue

            frame.close_frame()

            self.crop_list.append(det_crops)

    def build_stacked_2D_spectrum(self, extraction_window=11,
                                  wave_range=(8800, 19100),
                                  wave_step=5,
                                  use_inv_var_weight=True,
                                  pixel_shrink=0.5,
                                  ndrops=100,
                                  use_rel_flux_calib=True,
                                  extraction_sigma=None,
                                  extraction_profile='gaussian',
                                  median_filter_size=0,
                                  super_sample=3):
        """Build a stacked 2D spectrum resampled on a wavelength grid.

        Parameters
        ----------
        wcs
        wave_range
        median_filter_size
        ndrops
        """
        resampled_spectra_n = 0
        resampled_spectra = 0
        resampled_spectra_var = 0
        counts = 0

        for det_crop_list in tqdm.tqdm(self.crop_list):
            for crop in det_crop_list.values():
                # if median_filter_size > 0:
                #     filtered_crop_data = np.apply_along_axis(
                #         median_filter, axis=1, arr=crop.image,
                #         size=median_filter_size
                #     )
                #     crop_filtered = crop.copy()
                #     crop_filtered.image -= filtered_crop_data
                # else:
                #     crop_filtered = crop

                try:
                    image_out, var_out, norm_out, pix_bins = crop.resample_on_wavelength(
                        self.ra, self.dec,
                        extraction_window=extraction_window,
                        wave_range=wave_range,
                        wave_step=wave_step,
                        use_inv_var_weight=use_inv_var_weight,
                        pixel_shrink=pixel_shrink,
                        ndrops=ndrops,
                        use_rel_flux_calib=use_rel_flux_calib,
                        extraction_sigma=extraction_sigma,
                        extraction_profile=extraction_profile,
                        super_sample=super_sample
                    )
                except NoExtraction:
                    continue

                if median_filter_size > 0:
                    image_out = self.median_filter(image_out, int(median_filter_size/wave_step))

                resampled_spectra = resampled_spectra + image_out
                resampled_spectra_var = resampled_spectra_var + var_out

                resampled_spectra_n = resampled_spectra_n + norm_out

                nonzero = norm_out > 0
                counts = counts + nonzero

        valid = resampled_spectra_n > 0
        out = np.zeros(resampled_spectra.shape, dtype='d')
        out_var = np.zeros(resampled_spectra.shape, dtype='d')
        out[valid] = resampled_spectra[valid] / resampled_spectra_n[valid]
        out_var[valid] = resampled_spectra_var[valid] / resampled_spectra_n[valid]

        spec1d, spec1d_var, norm, counts1d, _ = self.extract_spec2d_to_1d(
            out, out_var, resampled_spectra_n, counts, pix_bins)

        spec_pack = {
            'spec2d': out,
            'spec2d_var': out_var,
            'spec2d_norm': resampled_spectra_n,
            'spec2d_counts': counts,
            'pix_bins': pix_bins[0],
            'wave_bins': pix_bins[1],
            'spec1d': spec1d,
            'spec1d_var': spec1d_var,
            'spec1d_counts': counts1d,
            'spec1d_norm': norm
        }

        return spec_pack

    def extract_spec2d_to_1d(self, spec2d, spec2d_var, spec2d_n, spec2d_counts, spec2d_pix_bins):
        """ """
        norm = np.sum(spec2d_n, axis=0)
        norm_2 = np.sum(spec2d_n**2, axis=0)
        valid = norm > 0

        nbins = len(norm)
        spec1d = np.zeros(nbins, dtype='d')
        spec1d_var = np.zeros(nbins, dtype='d')

        spec1d[valid] = np.sum(spec2d*spec2d_n, axis=0)[valid] / norm_2[valid] * norm[valid]
        spec1d_var[valid] = np.sum(spec2d_var*spec2d_n, axis=0)[valid] / norm_2[valid] * norm[valid]

        counts1d = np.mean(spec2d_counts, axis=0)

        return spec1d, spec1d_var, norm, counts1d, spec2d_pix_bins[1]


    def build_stacked_line_map(self, wavelength, wcs,
                                  median_filter_size=40,
                                  use_inv_var_weight=True,
                                  width=50,
                                  pixel_shrink=0.5,
                                  ndrops=100,
                                  use_rel_flux_calib=True,
                                  align_dispersion_direction=False,
                                  rotation_angle=0):
        """Build a stacked 2D spectrum resampled on a wavelength grid.

        Parameters
        ----------
        wcs
        wave_range
        median_filter_size
        ndrops
        """
        resampled_spectra_n = 0
        resampled_spectra = 0
        counts = 0

        for det_crop_list in tqdm.tqdm(self.crop_list):
            for crop in det_crop_list.values():

                if median_filter_size > 0:
                    filtered_crop_data = np.apply_along_axis(
                        median_filter, axis=1, arr=crop.image,
                        size=median_filter_size
                    )
                    crop_filtered = crop.copy()
                    crop_filtered.image -= filtered_crop_data
                else:
                    crop_filtered = crop

                try:
                    image_out, image_norm, pix_bins = crop_filtered.resample_on_radec(
                        self.ra, self.dec, wavelength,
                        wcs=wcs,
                        width=width,
                        use_inv_var_weight=use_inv_var_weight,
                        pixel_shrink=pixel_shrink,
                        ndrops=ndrops,
                        use_rel_flux_calib=use_rel_flux_calib,
                        align_dispersion_direction=align_dispersion_direction,
                        rotation_angle=rotation_angle
                    )
                except NoExtraction:
                    continue

                resampled_spectra = resampled_spectra + image_out

                if image_norm.max() > 0:
                    resampled_spectra_n = resampled_spectra_n + image_norm

                nonzero = image_norm > 0
                counts = counts + nonzero

        valid = resampled_spectra_n > 0
        out = np.zeros(resampled_spectra.shape, dtype='d')
        out[valid] = resampled_spectra[valid] / resampled_spectra_n[valid]
        return out, resampled_spectra_n, counts

    def median_filter(self, image, size, axis=1):
        """ """
        m = np.apply_along_axis(
                        median_filter, axis=axis, arr=image,
                        size=size
        )
        return image - m

