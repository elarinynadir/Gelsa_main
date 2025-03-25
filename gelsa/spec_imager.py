import sys
import numpy as np
from scipy import interpolate, ndimage
from astropy.io import fits
from astropy.table import Table
from tqdm.auto import tqdm

from . import utils
from . import sample_dist
from . import consts
from . import histogram
from . import overlaps

from numba import int64
from numba.typed import Dict


class SpectralImager:

    params = {
        'wavelength_range': (9000, 19000),
        'wavelength_step': 1.,
        'nphot_max': 10000000,
        'workdir': '.',
        'datadir': '.',
    }

    def __init__(self, specframe, **kwargs):
        """ """
        self.specframe = specframe

        self.params.update(kwargs)

        self.params['wavelength_range'] = specframe.params['wavelength_range']
        print(f"spec imager wavelength range {self.params['wavelength_range']}")

        self.wave = np.arange(
            self.params['wavelength_range'][0],
            self.params['wavelength_range'][1],
            self.params['wavelength_step']
        )

        self.sigma = np.sqrt(self.specframe.params['sigma2_det'] * self.specframe.params['exptime_sec'])

        self.init_image()

    def init_image(self):
        """ """
        shape = self.specframe.detector_shape
        bin_y = np.arange(0, shape[1]+1, 1, dtype='d')
        bin_x = np.arange(0, shape[0]+1, 1, dtype='d')
        self.pixel_grid = (bin_y, bin_x)

    def sample_spectrum(self, galaxy):
        """Generate samples of wavelength"""

        flux_spectrum = galaxy.sed(self.wave)# + galaxy.emline(self.wave)

        flux_spectrum[flux_spectrum<0] = 0

#         print(flux_spectrum)
        counts_spectrum = self.specframe.flux_to_counts(flux_spectrum, self.wave)


        func = sample_dist.SampleDistribution(self.wave, counts_spectrum)


        counts = np.random.poisson(np.sum(counts_spectrum))

        if counts == 0:
            return np.array([]), 0
#         print(f"counts {counts}")

        weight = 1
        if counts > self.params['nphot_max']:
            weight = counts / self.params['nphot_max']
            counts = self.params['nphot_max']
            print(f"alert {weight=}", file=sys.stderr)

        try:
            samples = func.sample(counts)

        except ValueError:
            return np.array([]), 0

        return samples, weight

    def sample_emline(self, galaxy, line):

        redshift = galaxy.params['redshift']
        wavelength_obs = (1 + redshift) * consts.lines[line]

       # else:
        sigma_size = galaxy.params['velocity_disp']/consts.c*wavelength_obs

        if wavelength_obs<galaxy.params['obs_wavelength_range'][0] or wavelength_obs>galaxy.params['obs_wavelength_range'][1]:
            counts_emline = 0
        else:
            counts_emline = self.specframe.lineflux_to_counts(galaxy.params['fluxes_emlines'][line], wavelength_obs)
            counts_emline = np.random.poisson(counts_emline)

        weight = 1
        if counts_emline > self.params['nphot_max']:
            weight = counts_emline / self.params['nphot_max']
            counts_emline = self.params['nphot_max']
            print(f"alert {weight=}", file=sys.stderr)

        samples_emline = np.random.normal(wavelength_obs, sigma_size, counts_emline)

        return samples_emline, weight

    def sample(self, galaxy):
        """ """
        wavelength, weight = self.sample_spectrum(galaxy)

        if len(wavelength) == 0:
            return np.array([]), np.array([]), np.array([]), 0

        ra, dec = galaxy.sample_image(len(wavelength))

        x, y, detector = self.specframe.radec_to_pixel(ra, dec, wavelength)

        xpsf, ypsf = self.specframe.sample_psf(
            x, y, detector,
            wavelength_ang=wavelength
        )

        x += xpsf
        y += ypsf

        # flux_loss = self.specframe.get_relative_flux_loss(
        #     x, y, detector, wavelength
        # )
        # weight = weight * flux_loss

        return x, y, detector, weight

    def sample_line(self, galaxy, line):
        """ """
        wavelength, weight = self.sample_emline(galaxy, line)

        if len(wavelength) == 0:
            return np.array([]), np.array([]), np.array([]), 0

        ra, dec = galaxy.sample_image(len(wavelength))

        x, y, detector = self.specframe.radec_to_pixel(ra, dec, wavelength)

        xpsf, ypsf = self.specframe.sample_psf(
            x, y, detector,
            wavelength_ang=wavelength
        )

        x += xpsf
        y += ypsf

        # flux_loss = self.specframe.get_relative_flux_loss(
        #     x, y, detector, wavelength
        # )
        #weight *= flux_loss

        return x, y, detector, weight

    def make_image(self, galaxy_list, masks=None, noise=True, return_var=True, show_progress=False):
        """Builds image and variance image"""
        images = {}
        if masks is not None:
            for det, mask in masks.items():
                sel, = np.where(mask.flat > -1)
                images[det] = np.zeros((1, len(sel)), dtype='d')
        # else:
            # image = np.zeros((self.specframe.params['det_height'], self.specframe.params['det_width']), dtype='d')

        for i, g in enumerate(galaxy_list):
            x, y, detector, weights = self.sample(g)
            for d in np.unique(detector):
                if d < 0:
                    continue
                sel = detector == d
                if d not in images:
                    images[d] = np.zeros(self.specframe.detector_shape, dtype='d')

                weight = np.ones(np.sum(sel), dtype=np.float64)
                weight *=weights

                histogram.histogram2d_accumulate(
                    y[sel],
                    x[sel],
                    weight,
                    bins_y=self.pixel_grid[0],
                    bins_x=self.pixel_grid[1],
                    hist=images[d],
                    mask_dict=masks[d] if masks else None
                )

            for l in range(len(consts.lines)):
                x, y, detector, weights = self.sample_line(g, l)
                for d in np.unique(detector):
                    if d < 0:
                        continue
                    sel = detector == d
                    if not d in images:
                        images[d] = np.zeros(self.specframe.detector_shape, dtype='d')
                    
                    weight = np.ones(np.sum(sel), dtype=np.float64)
                    weight*=weights


                    histogram.histogram2d_accumulate(
                        y[sel],
                        x[sel],
                        weight,
                        bins_y=self.pixel_grid[0],
                        bins_x=self.pixel_grid[1],
                        hist=images[d],
                        mask_dict=masks[d] if masks else None
                    )
            if show_progress:
                print(f"\r spectrum {i}/{len(galaxy_list)}", end="", flush=True)

        # remove axes with length 1
        for d, image in images.items():
            images[d] = np.squeeze(image)

        if return_var:
            var_images = {}
            # poisson variance is equal to mean
            for d, image in images.items():
                var_images[d] = image + self.sigma**2

        if noise:
            # add detector noise background
            for d, image in images.items():
                image += np.random.normal(0, self.sigma, image.shape)

        if return_var:
            return images, var_images
        else:
            return images

    def make_mask(self, galaxy_list, input_mask=None, width=2, iterations_min=3):
        """ """
        galaxy_list = utils.ensurelist(galaxy_list)
        # print(f"making mask for {len(galaxy_list)} sources")

        mask_images = {}
        # mask_image = np.zeros(
        #     (
        #         self.specframe.params['det_height'],
        #         self.specframe.params['det_width']
        #     ),
        #     dtype=np.bool
        # )

        wave = np.arange(12000, 19000, 10)

        for gal_i in range(len(galaxy_list)):
            ra = galaxy_list[gal_i].params['ra']
            dec = galaxy_list[gal_i].params['dec']
            x, y, detector = self.specframe.radec_to_pixel(
                ra*np.ones(len(wave)),
                dec*np.ones(len(wave)),
                wave,
                # objid=galaxy_list[gal_i].params['id']
            )


            # print(f"detectors {np.unique(detector)}")

            for d in np.unique(detector):
                if d < 0:
                    continue
                sel = detector == d

                weight = np.ones(np.sum(sel), dtype=np.float64)

                im = histogram.histogram2d(y[sel], x[sel], weight,
                                           bins_y=self.pixel_grid[0], bins_x=self.pixel_grid[1])

                radius = galaxy_list[gal_i].halflight_radius / 0.3
                

                iterations = max(iterations_min, int(np.round(width*radius)))

                im = ndimage.binary_dilation(
                    im,
                    structure=ndimage.generate_binary_structure(2, 2),
                    iterations=iterations
                )
                if d not in mask_images:
                    mask_images[d] = im
                else:
                    mask_images[d] += im

        if input_mask is not None:
            for d in mask_images.keys():
                mask_images[d] *= input_mask[d]

        index_mask_list = {}
        for d in mask_images.keys():
            # values outside of mask are -1
            index_mask = np.zeros(mask_images[d].shape, dtype=int) - 1
            # values inside of mask are set to an index 0,1,2,3...
            sel = mask_images[d] > 0
            index_mask[sel] = np.arange(np.sum(sel))
            index_mask_list[d] = index_mask

        return index_mask_list

    def check_overlap(self, gal1, gal2, wavelength_step=20):
        """Determine if two sources overlap in the dispersed image.

        Parameters
        ----------
        ra1
        dec1
        ra2
        dec2

        Returns
        -------
        bool
        """
        wave_trace = np.arange(
            self.params['wavelength_range'][0],
            self.params['wavelength_range'][1],
            wavelength_step
        )
        x1, y1, detector1 = self.specframe.radec_to_pixel(
            gal1.params['ra']*np.ones(len(wave_trace)),
            gal1.params['dec']*np.ones(len(wave_trace)),
            wave_trace,
            objid=gal1.params['id']
        )
        x2, y2, detector2 = self.specframe.radec_to_pixel(
            gal2.params['ra']*np.ones(len(wave_trace)),
            gal2.params['dec']*np.ones(len(wave_trace)),
            wave_trace,
            objid=gal2.params['id']
        )

        separation = (gal1.halflight_radius + gal2.halflight_radius) / 0.3 * 2
        separation = max(10, separation)
        # print(f"separation {separation} pixels")
        # print(np.unique(detector1), np.unique(detector2))
        for d1 in np.unique(detector1):
            if d1 < 0:
                continue
            sel1 = detector1 == d1
            sel2 = detector2 == d1
            if np.sum(sel2) == 0:
                # obj2 not on same detector
                continue
            # print(f"on same detector {np.sum(sel1)} {np.sum(sel2)}")
            overlap = overlaps.check_segment_overlap(
                x1[sel1], y1[sel1],
                x2[sel2], y2[sel2],
                separation=separation
            )
            if overlap:
                return True
        return False

    


    # def write(self, filename, **kwargs):
    #     """Write image to a FITS file"""
    #     hdu = fits.PrimaryHDU()
    #     image_hdu = fits.ImageHDU(data=self.image, header=self.specframe.wcs.to_header())
    #     hdul = fits.HDUList([hdu, image_hdu])
    #     hdul.writeto(filename, **kwargs)
