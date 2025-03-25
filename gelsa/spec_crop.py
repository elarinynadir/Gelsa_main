import numpy as np
from scipy.interpolate import interpolate
from .fast_interp import interp3d

from .utils import intrange, rotate_around
from .histogram import histogram1d, histogram2d


class SpecCrop:
    deg_step = 0.01
    wave_step = 200.
    x_step = 200
    y_step = 200

    def __init__(self, image, mask, var,
                 detector=None, bbox=None, frame=None):
        """ """
        self.detector = detector
        self.bbox = bbox
        self.image = image
        self.mask = mask
        self.var = var
        self.shape = image.shape
        if frame is not None:
            self._setup_pixel_interpolator(frame)
            self._setup_radec_interpolator(frame)
        self.frame = frame

    def copy(self):
        """ """
        S = SpecCrop(
            self.image.copy(),
            self.mask.copy(),
            self.var.copy(),
            detector=self.detector,
            bbox=self.bbox,
        )
        S.frame = self.frame
        S.interper_ra = self.interper_ra
        S.interper_dec = self.interper_dec
        S.interper_x = self.interper_x
        S.interper_y = self.interper_y
        return S

    @staticmethod
    def crop(frame, detector, bbox):
        """ """
        x_0, x_1, y_0, y_1 = bbox
        image, mask, var = frame.get_detector(detector)
        image = image[y_0:y_1, x_0:x_1]
        mask = mask[y_0:y_1, x_0:x_1]
        var = var[y_0:y_1, x_0:x_1]
        return SpecCrop(image, mask, var,
                        detector=detector, bbox=bbox, frame=frame)

    @staticmethod
    def crop_trace(frame, detector, x, y, padx=100, pady=10):
        """ """
        image, mask, var = frame.get_detector(detector)
        nrow, ncol = image.shape

        x_0 = max(0, int(x.min() - padx))
        x_1 = min(ncol, int(x.max() + padx))
        y_0 = max(0, int(y.min() - pady))
        y_1 = min(nrow, int(y.max() + pady))

        bbox = (x_0, x_1, y_0, y_1)

        image = image[y_0:y_1, x_0:x_1]
        mask = mask[y_0:y_1, x_0:x_1]
        var = var[y_0:y_1, x_0:x_1]
        return SpecCrop(image, mask, var,
                        detector=detector, bbox=bbox, frame=frame)

    def _setup_pixel_interpolator(self, frame):
        """ """
        x_0, x_1, y_0, y_1 = self.bbox
        w_0, w_1 = frame.params['wavelength_range']

        xx = np.array([x_0, x_1, x_0, x_1]).astype(float)
        yy = np.array([y_0, y_1, y_0, y_1]).astype(float)
        wave = np.array([w_0, w_0, w_1, w_1]).astype(float)

        ra_, dec_ = frame.pixel_to_radec(xx, yy, self.detector, wave)

        ra0 = np.min(ra_)
        dec0 = np.min(dec_)
        ra1 = np.max(ra_)
        dec1 = np.max(dec_)

        mu = np.cos(np.radians(dec0))
        if mu > 0:
            ra_step = self.deg_step/mu
        else:
            print(f"Warning! at pole, declination hit 90: {dec0}")
            ra_step = self.deg_step

        ra_grid = intrange(ra0, ra1, ra_step)
        dec_grid = intrange(dec0, dec1, self.deg_step)
        wave_grid = intrange(w_0, w_1, self.wave_step)

        ra_, dec_, wave_ = np.meshgrid(ra_grid, dec_grid, wave_grid, indexing='ij')
        x, y, det_ = frame.framecoord.radec_to_pixel(ra_.flatten(), dec_.flatten(), wave_.flatten())
        off_det = det_ != self.detector
        x[off_det] = -1000
        y[off_det] = -1000

        x = x.reshape(ra_.shape)
        y = y.reshape(ra_.shape)

        x -= x_0
        y -= y_0

        self.interper_x = interp3d(
            [ra_grid[0], dec_grid[0], wave_grid[0]],
            [ra_grid[-1], dec_grid[-1], wave_grid[-1]],
            [ra_grid[1]-ra_grid[0], dec_grid[1]-dec_grid[0], wave_grid[1]-wave_grid[0]],
            x,
            k=1
        )
        self.interper_y = interp3d(
            [ra_grid[0], dec_grid[0], wave_grid[0]],
            [ra_grid[-1], dec_grid[-1], wave_grid[-1]],
            [ra_grid[1]-ra_grid[0], dec_grid[1]-dec_grid[0], wave_grid[1]-wave_grid[0]],
            y,
            k=1
        )
        # self.interper_x = RegularGridInterpolator(
        #     (ra_grid, dec_grid, wave_grid), x,
        #     bounds_error=False,
        #     fill_value=None
        # )
        # self.interper_y = RegularGridInterpolator(
        #     (ra_grid, dec_grid, wave_grid), y,
        #     bounds_error=False,
        #     fill_value=None
        # )

    def _setup_radec_interpolator(self, frame):
        """ """
        x_0, x_1, y_0, y_1 = self.bbox
        w_0, w_1 = frame.params['wavelength_range']

        x_grid = intrange(x_0, x_1, self.x_step).astype(float)
        y_grid = intrange(y_0, y_1, self.y_step).astype(float)
        wave_grid = intrange(w_0, w_1, self.wave_step).astype(float)

        x_, y_, wave_ = np.meshgrid(x_grid, y_grid, wave_grid, indexing='ij')

        det_ = np.ones(x_.shape, dtype=int) * self.detector
        ra, dec = frame.framecoord.pixel_to_radec(
            x_.flatten(), y_.flatten(), det_.flatten(), wave_.flatten())

        ra = ra.reshape(x_.shape)
        dec = dec.reshape(x_.shape)

        x_grid -= x_0
        y_grid -= y_0

        self.interper_ra = interp3d(
            [x_grid[0], y_grid[0], wave_grid[0]],
            [x_grid[-1], y_grid[-1], wave_grid[-1]],
            [x_grid[1]-x_grid[0], y_grid[1]-y_grid[0], wave_grid[1]-wave_grid[0]],
            ra,
            k=1
        )
        self.interper_dec = interp3d(
            [x_grid[0], y_grid[0], wave_grid[0]],
            [x_grid[-1], y_grid[-1], wave_grid[-1]],
            [x_grid[1]-x_grid[0], y_grid[1]-y_grid[0], wave_grid[1]-wave_grid[0]],
            dec,
            k=1
        )
        # self.interper_ra = RegularGridInterpolator(
        #     (x_grid, y_grid, wave_grid),
        #     ra,
        #     bounds_error=False,
        #     fill_value=None
        # )
        # self.interper_dec = RegularGridInterpolator(
        #     (x_grid, y_grid, wave_grid),
        #     dec,
        #     bounds_error=False,
        #     fill_value=None
        # )

    def radec_to_pixel(self, ra, dec, wavelength):
        """ """
        # points = np.transpose([ra, dec, wavelength])
        x = self.interper_x(ra, dec, wavelength)
        y = self.interper_y(ra, dec, wavelength)
        return x, y

    def pixel_to_radec(self, x, y, wavelength):
        """ """
        # points = np.transpose([x, y, wavelength])
        ra = self.interper_ra(x, y, wavelength)
        dec = self.interper_dec(x, y, wavelength)
        return ra, dec

    def pixel_to_radec_(self, x, y, wavelength):
        """ """
        ra, dec = self.frame.pixel_to_radec(
            x+self.bbox[0], y+self.bbox[2],
            self.detector, wavelength)
        return ra, dec

    def radec_to_pixel_(self, ra, dec, wavelength):
        """ """
        scalar = True
        try:
            len(ra)
            scalar = False
        except TypeError:
            scalar = True
            ra = np.array([ra])
            dec = np.array([dec])

        x, y, det = self.frame.radec_to_pixel(ra, dec, wavelength)
        x -= self.bbox[0]
        y -= self.bbox[2]
        invalid = det != self.detector
        x[invalid] = -1
        y[invalid] = -1

        if scalar:
            return x[0], y[0]
        return x, y

    def get_dispersion_direction_on_wcs(self, wcs, step=50):
        """Compute dispersion direction on the sky.
        The angle is return using the position angle convention:
        angle from North, increasing toward East.

        Angle is returned in radians.

        Parameters
        ----------
        wcs
        ra
        dec
        wavelength

        Returns
        -------
        theta : float
        dispersion direction in radians
        """
        ra, dec = self.center
        wavelength = (self.wavelength_trace.min()+self.wavelength_trace.max())/2.
        x, y = self.radec_to_pixel_(ra, dec, wavelength+step)
        sign = 1
        if x < 0:
            x, y = self.radec_to_pixel_(ra, dec, wavelength-step)
            sign = -1
            if x < 0:
                raise Exception

        ra1, dec1 = self.pixel_to_radec_(x, y, wavelength)
        im_x, im_y = wcs.wcs_world2pix(ra, dec, 0)
        im_x1, im_y1 = wcs.wcs_world2pix(ra1, dec1, 0)
        dy = im_y1 - im_y
        dx = im_x1 - im_x
        dx *= sign
        dy *= sign
        return np.arctan2(dx, dy)

    def resample_on_radec(self, ra, dec, wavelength,
                          width=None, wcs=None,
                          ndrops=100, pixel_shrink=0.5,
                          use_inv_var_weight=True,
                          use_rel_flux_calib=False,
                          align_dispersion_direction=False,
                          rotation_angle=0):
        """ """
        if align_dispersion_direction:
            theta_dispersion = self.get_dispersion_direction_on_wcs(wcs)

        shape_out = wcs.array_shape
        pix_scale_out = wcs.wcs.cd[0,0]*3600
        pix_scale_in = self.frame.params['PIXSCALE']

        if width is None:
            width = int(pix_scale_in/pix_scale_out * \
                        np.sqrt(np.sum(np.square(shape_out))))

        x, y = self.radec_to_pixel_(np.array([ra]), np.array([dec]),
                                    np.array([wavelength]))
        x = x[0]
        y = y[0]
        if x < 0 or y < 0:
            raise NoExtraction

        if x > self.shape[1] or y > self.shape[0]:
            raise NoExtraction

        x0 = int(x - width//2)
        x1 = int(x0 + width)
        y0 = int(y - width//2)
        y1 = int(y0 + width)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.shape[1]-1, x1)
        y1 = min(self.shape[0]-1, y1)

        pix_bins = (
            np.arange(wcs.array_shape[0]+1),
            np.arange(wcs.array_shape[1]+1),
        )

        image = self.image[y0:y1+1, x0:x1+1]
        mask = self.mask[y0:y1+1, x0:x1+1]
        var = self.var[y0:y1+1, x0:x1+1]
        valid = (mask == 0) & (var > 0) & np.isfinite(
            var) & np.isfinite(image) & np.isfinite(mask)

        if use_inv_var_weight:
            # inverse variance weight
            weight = 1./var[valid]
        else:
            weight = np.ones(np.sum(valid))

        if np.sum(valid)==0:
            raise NoExtraction

        signal = weight * image[valid]/ndrops

        xgrid = np.arange(x0, x1+1)
        ygrid = np.arange(y0, y1+1)
        xx, yy = np.meshgrid(xgrid, ygrid, indexing='xy')
        xx = xx[valid]
        yy = yy[valid]

        if use_rel_flux_calib:
            flux_loss = self.frame.get_relative_flux_loss(ra*np.ones(1), dec*np.ones(1),
                                                         wavelength*np.ones(1))[0]
            if flux_loss > 0:
                signal /= flux_loss
            if use_inv_var_weight:
                weight *= flux_loss**2

        image_out = np.zeros(wcs.array_shape, dtype='d')
        area_out = np.zeros(wcs.array_shape, dtype='d')

        for loop in range(ndrops):
            off_x, off_y = np.random.uniform(-0.5, 0.5, (2, len(xx))) * pixel_shrink + 0.5
            ra, dec = self.pixel_to_radec(
                xx + off_x, yy + off_y, wavelength=wavelength*np.ones(len(xx)))
            pix_x, pix_y = wcs.wcs_world2pix(ra, dec, 0)
            if align_dispersion_direction:
                pix_x, pix_y = rotate_around(wcs, pix_x, pix_y,
                                             rotation_angle - theta_dispersion + np.pi/2)
            else:
                if rotation_angle != 0:
                    pix_x, pix_y = rotate_around(wcs, pix_x, pix_y, rotation_angle)
            signal_ = histogram2d(pix_y, pix_x,
                                  weight=signal,
                                  bins_y=pix_bins[0], bins_x=pix_bins[1])
            count_ = histogram2d(pix_y, pix_x,
                                 weight=weight,
                                 bins_y=pix_bins[0], bins_x=pix_bins[1])

            image_out = image_out + signal_
            area_out = area_out + count_ * 1./ndrops

        # scale by ratio of pixel scales to correct
        # surface brightness
        # image_out *= (pix_scale_out / pix_scale_in)**2

        return image_out, area_out, pix_bins

    def resample_on_wavelength(self, ra, dec,
                               extraction_window=11,
                               wave_range=None,
                               wave_step=5.,
                               ndrops=100,
                               pixel_shrink=1,
                               seed=None,
                               use_inv_var_weight=False,
                               use_rel_flux_calib=False,
                               extraction_sigma=1,
                               extraction_profile='gaussian',
                               super_sample=3,
                               ):
        """Resample 2D spectrum on linear wavelength grid.

        Parameters
        ----------
        ra
        dec
        wave_range
        wave_step
        wcs
        ndrops
        pixel_shrink : float
        seed :

        Returns
        -------
        image, bins
        """
        extraction_profile = extraction_profile
        if not extraction_profile.lower()[0] in ['g', 'f']:
            print(f"Don't understand extraction profile `{extraction_profile}` Recognized extraction profiles include 'gaussian' or 'flat'. Will use flat.")
            extraction_profile = 'f'
        if extraction_profile.lower().startswith('g'):
            use_gaussian_profile = True
        else:
            use_gaussian_profile = False

        image = self.image.astype(float)
        mask = self.mask.astype(float)
        var = self.var.astype(float)
        valid = (mask == 0) & (var > 0) & np.isfinite(
            var) & np.isfinite(image) & np.isfinite(mask)

        var = var[valid]
        if use_inv_var_weight:
            # inverse variance weight
            weight = 1./var
        else:
            weight = np.ones(np.sum(valid))

        signal = weight * image[valid]

        shape = self.image.shape
        ygrid = np.arange(0, shape[0])
        xgrid = np.arange(0, shape[1])
        xx, yy = np.meshgrid(xgrid, ygrid, indexing='xy')
        xx = xx[valid]
        yy = yy[valid]

        trace = np.linspace(*wave_range, 100)
        trace_x, trace_y = self.radec_to_pixel_(
            ra*np.ones(len(trace)), dec*np.ones(len(trace)), trace)
        valid_trace = (trace_x > 0) & (trace_y > 0)
        if np.sum(valid_trace) == 0:
            raise NoExtraction

        trace = trace[valid_trace]
        trace_x = trace_x[valid_trace]
        trace_y = trace_y[valid_trace]

        mid_i = len(trace_x)//2

        dx = trace_x - trace_x[mid_i]
        dy = trace_y - trace_y[mid_i]

        nonzero = dx != 0
        theta = np.arctan2(dy[nonzero], dx[nonzero])
        positive = theta > 0
        theta[positive] -= np.pi

        theta = np.median(theta)
        cos_trace = np.cos(theta)
        sin_trace = np.sin(theta)

        # rotate the trace
        trace_par = dx * cos_trace + dy * sin_trace
        trace_perp = -dx * sin_trace + dy * cos_trace

        wavelength_map = interpolate.interp1d(
            trace_par, trace,
            bounds_error=False, fill_value=np.nan
        )

        curvature_map = interpolate.interp1d(
            trace_par, trace_perp,
            bounds_error=False, fill_value=np.nan
        )

        if use_rel_flux_calib:
            pix_dx = xx - trace_x[mid_i]
            pix_dy = yy - trace_y[mid_i]
            # rotate trace
            t_par = pix_dx * cos_trace + pix_dy * sin_trace
            flux_loss = self.get_relative_flux_loss(ra*np.ones(len(t_par)),
                                                    dec*np.ones(len(t_par)),
                                                    wavelength_map(t_par))
            valid_loss = flux_loss > 0
            signal[valid_loss] /= flux_loss[valid_loss]
            if use_inv_var_weight:
                weight[valid_loss] *= flux_loss[valid_loss]**2

        pix_step = 1./super_sample
        pix_bins = (
            np.arange(0, extraction_window + pix_step, pix_step) - extraction_window/2.,
            np.arange(*wave_range, wave_step),
        )
        out_shape = (len(pix_bins[0])-1, len(pix_bins[1])-1)
        image_out = np.zeros(out_shape, dtype='d')
        var_out = np.zeros(out_shape, dtype='d')
        norm_out = np.zeros(out_shape, dtype='d')
        area_out = np.zeros(out_shape, dtype='d')
        for loop in range(ndrops):
            off_x, off_y = np.random.uniform(-0.5, 0.5, (2, len(xx))) * pixel_shrink + 0.5

            pix_dx = xx + off_x - trace_x[mid_i]
            pix_dy = yy + off_y - trace_y[mid_i]

            # rotate trace
            t_par = pix_dx * cos_trace + pix_dy * sin_trace
            t_perp = -pix_dx * sin_trace + pix_dy * cos_trace

            wavelength = wavelength_map(t_par)

            curve = curvature_map(t_par)
            t_perp -= curve

            valid_wavelength = np.isfinite(wavelength)
            wavelength = wavelength[valid_wavelength]
            if len(wavelength) == 0:
                continue

            t_perp = t_perp[valid_wavelength]

            window = np.abs(t_perp) < extraction_window/2.
            if use_gaussian_profile:
                extraction_weight = np.exp(-t_perp[window]**2/2./extraction_sigma**2)
            else:
                extraction_weight = 1

            t_perp = t_perp[window]

            signal_ = histogram2d(t_perp, wavelength[window],
                                  weight=signal[valid_wavelength][window]*extraction_weight/ndrops,
                                  bins_y=pix_bins[0], bins_x=pix_bins[1])
            var_ = histogram2d(t_perp, wavelength[window],
                                 weight=var[valid_wavelength][window]*weight[valid_wavelength][window]*extraction_weight*2/ndrops,
                                 bins_y=pix_bins[0], bins_x=pix_bins[1])
            norm_ = histogram2d(t_perp, wavelength[window],
                                weight=weight[valid_wavelength][window]*extraction_weight/ndrops,
                                bins_y=pix_bins[0], bins_x=pix_bins[1])
            area_ = histogram2d(t_perp, wavelength[window],
                                weight=np.ones(np.sum(window))/ndrops,
                                bins_y=pix_bins[0], bins_x=pix_bins[1])

            image_out = image_out + signal_
            var_out = var_out + var_
            norm_out = norm_out + norm_
            area_out = area_out + area_

        nonzero = area_out > 0
        norm_tmp = norm_out*0
        norm_tmp[nonzero] = norm_out[nonzero]/area_out[nonzero]
        norm_out = norm_tmp

        return image_out, var_out, norm_out, pix_bins


class NoExtraction(Exception):
    pass
