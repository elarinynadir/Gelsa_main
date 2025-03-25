from SIR_Utilities.Mdb import Mdb as SirMdb
from .sgs import sir_pipeline, scienceframe_variance, sir_pointing_model
import os
import re
import time
import numpy as np
import tqdm
import h5py
import glob
import xml.etree.ElementTree as ET
from astropy.io import fits
from scipy import interpolate


from SIR_SpectraLocation.LocationSpectrum import LocationSpectrum
from SIR_InstrumentModels.GrismModel import GrismModel
from SIR_InstrumentModels.DetectorModel import DetectorModel


from matplotlib import pyplot as plt


line_list = {
    'Hb_4862': 4862.68,
    'O3_4960': 4960.295,
    'O3_5008': 5008.240,
    # 'N2_6549': 6549.86,
    'Ha_6564': 6564.61,
    # 'N2_6585': 6585.27,
    # 'S2_6718': 6718.29,
    # 'S2_6732': 6732.67,
    'S3_9069': 9068.6,
    'S3_9530': 9530.6,
    'Pae': 9548.6,
    'Pad': 10052.1,
    'HeI': 10830.0,
    'Pag': 10941.1,
    'Pab':  12821.6,
    'Paa':  18756.1,


}


def load_scienceframe(xml_path, workdir='.'):
    """ """
    print(f"loading {xml_path}")
    tree = ET.parse(os.path.join(xml_path))
    root = tree.getroot()
    datafile = root.find('Data/DataStorage/DataContainer/FileName').text

    hdul = fits.open(os.path.join(workdir, 'data', datafile))
    return hdul


def update_optical_model_from_location_table(optical_model, location_table_path):
    """ """
    print("updating WCS from location table")
    with h5py.File(location_table_path) as lt:
        params = lt['OPT']
        cd_matrix = params['CD'][:]
        crpix = params['CRPIX'][:]
        crval = params['CRVAL'][:]
        ctype = params['CTYPE'][:]
    optical_model.set_wcs(ctype, crpix, crval, cd_matrix)


def find_location_table(pointing_id, glob_pattern='*loc_tables*.xml', workdir='tmp'):
    """ """
    file_list = glob.glob(os.path.join(workdir, glob_pattern))
    pattern = re.compile(f"<PointingId>(\s*){pointing_id}(\s*)</PointingId>")

    for filename in file_list:
        with open(filename, 'rt') as inp:
            chars = inp.read()
            result = re.search(pattern, chars)
            if result:
                return os.path.basename(filename)
    print(f"loc table not found: {pointing_id=}")


def get_pointing_id(filename, workdir='tmp'):
    """ """
    path = os.path.join(workdir, filename)
    pattern = re.compile(f"<PointingId>(\s*)(?P<id>\d+)(\s*)</PointingId>")

    with open(path, 'rt') as inp:
        chars = inp.read()
        result = re.search(pattern, chars)
        if result:
            return result.group('id')
    print("not found")


class SkyWaveMap:
    wavelength0 = 15000
    dlam = 100

    def __init__(self, x, y, wavelength):
        """ """
        self.fit_wavelength(x, y, wavelength)

    def fit_wavelength(self, x, y, wavelength):
        """ """
        self._wave_interpolator_x = interpolate.interp1d(wavelength, x, bounds_error=False, fill_value='extrapolate')
        self._wave_interpolator_y = interpolate.interp1d(wavelength, y, bounds_error=False, fill_value='extrapolate')
        self._pix_interpolator_x = interpolate.interp1d(x, wavelength, bounds_error=False, fill_value='extrapolate')

    def wavelength_to_pixel(self, wavelength):
        """ """
        x = self._wave_interpolator_x(wavelength)
        y = self._wave_interpolator_y(wavelength)
        return x, y

    def pixel_to_wavelength(self, x, y):
        """ """
        return self._pix_interpolator_x(x)

    def fit_sky(self, ra0, dec0, sky_to_pixel, bbox, dra=0.02, ddec=0.02):
        """ """
        self.center = (ra0, dec0)
        dra /= np.cos(np.radians(dec0))
        self.dra = dra
        self.ddec = ddec

        print(f"{dra=} {ddec=}")
        alpha = np.linspace(ra0-dra, ra0+dra, 3)
        delta = np.linspace(dec0-ddec, dec0+ddec, 3)
        wavelengths = np.linspace(12000, 19000, 3)

        grid_ra, grid_dec, grid_wave = np.meshgrid(alpha, delta, wavelengths, indexing='ij')

        x, y, detector = sky_to_pixel(grid_ra.flat, grid_dec.flat, grid_wave.flat)

        x = x.reshape(grid_ra.shape)
        y = y.reshape(grid_ra.shape)
        x -= bbox[0]
        y -= bbox[2]

        self._sky_interp_x = interpolate.RegularGridInterpolator((alpha, delta, wavelengths), x, bounds_error=False, fill_value=None)
        self._sky_interp_y = interpolate.RegularGridInterpolator((alpha, delta, wavelengths), y, bounds_error=False, fill_value=None)

    def fit_in_projection(self, ra0, dec0, sky_to_pixel, bbox, dra=0.02, ddec=0.02):
        """ """
        self.center = (ra0, dec0)
        dra /= np.cos(np.radians(dec0))
        self.dra = dra
        self.ddec = ddec

        print(f"{dra=} {ddec=}")
        alpha = np.linspace(ra0-dra, ra0+dra, 3)
        delta = np.linspace(dec0-ddec, dec0+ddec, 3)


        grid_ra, grid_dec = np.meshgrid(alpha, delta, indexing='ij')

        wavelengths = np.ones(len(grid_dec.flat))*self.wavelength0
        x, y, detector = sky_to_pixel(grid_ra.flat, grid_dec.flat, wavelengths)

        x = x.reshape(grid_ra.shape)
        y = y.reshape(grid_ra.shape)
        x -= bbox[0]
        y -= bbox[2]

        self._sky_interp_ra = interpolate.RegularGridInterpolator((x, y), x, bounds_error=False, fill_value=None)
        self._sky_interp_dec = interpolate.RegularGridInterpolator((x, y), y, bounds_error=False, fill_value=None)


    def sky_to_pixel(self, ra, dec, wavelength):
        """ """
        coord = np.transpose([ra, dec, wavelength])
        x = self._sky_interp_x(coord)
        y = self._sky_interp_y(coord)
        return x, y

    @property
    def dispersion_angle(self):
        """ """
        try:
            return self._dispersion_angle
        except AttributeError:
            pass
        center = self.center
        wavelength0 = self.wavelength0
        dlam = self.dlam
        dra = self.dra/10
        ddec = self.ddec/10
        print(f"{center=}")

        ra_ = [center[0], center[0], center[0]+dra, center[0]]
        dec_ = [center[1], center[1], center[1], center[1]+ddec]
        wavelength_ = [wavelength0, wavelength0+dlam, wavelength0, wavelength0]
        x_, y_ = self.sky_to_pixel(ra_, dec_, wavelength_)
        dx = x_[1] - x_[0]
        dy = y_[1] - y_[0]

        # inverse transform at fixed wavelength
        dra_dx = dra / (x_[2] - x_[0])
        dra_dy = dra / (y_[2] - y_[0])
        ddec_dx = ddec / (x_[3] - x_[0])
        ddec_dy = ddec / (y_[3] - y_[0])
        print(f"{dra_dx=} {dra_dy=}")
        print(f"{ddec_dx=} {ddec_dy=}")
        print(f"{dx=} {dy=}")

        dra = dra_dx * dx + dra_dy * dy
        ddec = ddec_dx * dx + ddec_dy * dy

        self._dispersion_angle = np.degrees(np.arctan2(ddec, dra))
        return self._dispersion_angle





class SpecFrame:

    _default_config = {
        'workdir': 'tmp',
        'mdb_file': "EUC_MDB_MISSIONCONFIGURATION_PVPHASE_2023-12-15-1.xml",
        'abs_file': "SIR_Calibration_Abs_1.0.5-ON_THE_FLY-pcasenov-PLAN-000000-GJ00XYPR-20240219-152345-0-new_abs_calib-0.xml",
        'opt_file': 'SIR_Calib_Opt_1.0.8-ON_THE_FLY-pcasenov-PLAN-000000-6YI2Z9UY-20240213-154829-0-new_opt_cal-0.xml',
        'ids_file': 'DpdSirIdsModel__SIR_Calib_Ids_EUCLID_1.0.5-ON_THE_FLY-pcasenov-PLAN-000001-67PH88PO-20240803-211109-0-new_ids_calib-0.xml',
        # 'ids_file': 'SIR_Calib_Ids_1.0.5-ON_THE_FLY-pcasenov-PLAN-000000-4S9TBW7A-20240215-080942-0-new_ids_calib-0.xml',
        # 'crv_file': 'SIR_Calib_Crv_1.0.6-ON_THE_FLY-pcasenov-PLAN-000000-US9LP4ZZ-20240214-082238-0-new_crv_cal-0.xml',
        'crv_file': 'SIR_Calib_Crv_1.0.6-ON_THE_FLY-pcasenov-PLAN-000000-US9LP4ZZ-20240214-082238-0-new_crv_cal-0.xml',
        'location_table': None
    }

    detectors = [11,21,31,41,
                 12,22,32,42,
                 13,23,33,43,
                 14,24,34,44]

    def __init__(self, frame_path, config=None, **kwargs):
        """ """
        self.config = {} if config is None else config
        self.config.update(self._default_config)
        self.config.update(kwargs)
        self.config['frame_path'] = frame_path

        pointing_id = get_pointing_id(
            frame_path
        )

        # self.config['location_table'] = find_location_table(
            # pointing_id,
            # workdir=self.config['workdir']
        # )

        class Args:
            mdb_file=self.config['mdb_file']
            abs=self.config['abs_file']
            workdir=self.config['workdir']
            opt_model=self.config['opt_file']
            ids_model=self.config['ids_file']
            crv_model=self.config['crv_file']
            location_table=self.config['location_table']

        args = Args()
        self._mdb = SirMdb.load(args.workdir, args.mdb_file)
        self._PM = sir_pointing_model.SIRPointingModel(args, self._mdb, frame_path)

        self._PM._detector_model = DetectorModel(self._PM.scienceframe.grism_name, self._mdb, rotate=True)
        print(f"CRVAL(11) = {self._PM._detector_model.getCRVal('11')}")
        self._SP = sir_pipeline.SIRPipeline(self._PM)

    @property
    def center(self):
        """ """
        ra = float(self._PM.scienceframe.getMetadataKey('RA'))
        dec = float(self._PM.scienceframe.getMetadataKey('DEC'))
        return ra, dec

    @property
    def pa(self):
        return float(self._PM.scienceframe.getMetadataKey('PA'))

    @property
    def grism(self):
        return self._PM.scienceframe.grism_name

    @property
    def tilt(self):
        return self._PM.scienceframe.grism_tilt

    @property
    def grism_name(self):
        if self.tilt == 4:
            s = "P4"
        elif self.tilt == -4:
            s = "M4"
        else:
            s = ""
        return self.grism + s

    @property
    def hdul(self):
        """ """
        if not hasattr(self, '_hdul'):
            print(f"Loading {self.config['frame_path']}")
            self._hdul = load_scienceframe('tmp/'+self.config['frame_path'], 'tmp')
        return self._hdul

    def close(self):
        """ """
        self._hdul.close()
        del self._hdul

    def plot_bounding_box(self, **plotparams):
        """ """
        ax = plt.subplot(111, aspect=1./np.cos(np.radians(self.center[1])))
        for xy in self._PM.footprint:
            x, y = np.transpose(xy)
            x = np.concatenate([x, [x[0]]])
            y = np.concatenate([y, [y[0]]])
            ax.plot(x, y, **plotparams)

    def get_detector(self, detector_index):
        """ """
        d = self.detectors[detector_index-1]
        print(f"Loading detector {d}")
        extname = f'DET{d}.SCI'
        maskname = f'DET{d}.DQ'
        varname = f'DET{d}.VAR'
        fullimage = self.hdul[extname].data[:]
        fullmask = self.hdul[maskname].data[:]
        fullvar = self.hdul[varname].data[:]
        return fullimage, fullmask, fullvar

    def get_pixel_(self, ra, dec, wavelength, dispersion_order=1, arbitrary_detector=6):
        """Return the pixel position of the spectral element given by
        sky coordinate RA, Dec and wavelength (angstrom)

        Parameters
        -----------
        ra : float
            Right ascension angle in deg
        dec : float
            Declination angle in deg
        wavelength : float
            wavelength in angstroms
        dispersion_order : int
            Dispersion order (default 1)

        Returns
        -------
        int, int, int
            x, y pixel position
            detector number
        """
        if dispersion_order not in self._SP.instrument.ids_model:
            print(self._SP.instrument.ids_model.keys())
            raise ValueError(f"get_pixel not initialized for dispersion order {dispersion_order}")

        # get focal plane position of reference wavelength
        if dispersion_order == 1:
            ref_position = self._SP.instrument.optical_model.get1stOrderReferencePosition(ra, dec)
        elif dispersion_order == 0:
            ref_position = self._SP.instrument.optical_model.get0thOrderReferencePosition(ra, dec)

        ref_mm = np.array([ref_position.getPosition()])
        # print(f"{ref_mm=}")
        pivot_mm = self._SP.instrument.optical_model.getPivotPosition(ra, dec)
        # print(f"{pivot_mm=}")

        grism_pos = GrismModel.getPositionNumber(self._SP.instrument.optical_model.gwa_pos)
        loc_spectrum = LocationSpectrum(dispersion_order, grism_pos)

        # get the detector pixel position of the slit. The detector we choose is not important.
        loc_spectrum.setDetector(arbitrary_detector)
        slit_pix = self._SP.instrument.detector_model.getPixels(ref_mm, arbitrary_detector)
        # print(f"{slit_pix=}")
        loc_spectrum.setLambdaRefPosition(slit_pix)
        loc_spectrum.setPivot(pivot_mm[0], pivot_mm[1])

        # Set IDS and CRV models
        loc_spectrum.setIDSModel(self._SP.instrument.ids_model[dispersion_order].getLocalModel(ref_mm))
        loc_spectrum.setCRVModel(self._SP.instrument.crv_model[dispersion_order].getLocalModel(ref_mm))
        loc_spectrum.setDetectorModel(self._SP.instrument.detector_model)
        loc_spectrum.setLambdaRefValue(
            self._SP.instrument.optical_model.getReferenceLambda(dispersion_order)
        )

        loc_spectrum.setGWATilt(int(self._SP.instrument.optical_model.gwa_tilt))
        loc_spectrum.setExtraTilt(self._SP.instrument.extra_tilt)

        pixel = loc_spectrum.computePosition(float(wavelength), 0)

        # print(f"{loc_spectrum.getExtraTilt()=}")
        self.loc_spectrum = loc_spectrum
        x, y = pixel.getPosition()

        # from pixels referenced from arbitrary detector, go back to focal plane coord
        fov_position = self._SP.instrument.detector_model.getFOVPosition(x, y, arbitrary_detector)

        # Now determine the actual detector and corresponding pixel coord
        pixel = self._SP.instrument.detector_model.getPixel(fov_position)
        detector = pixel.getDetectorNumber()
        x, y = pixel.getPosition()

        x += 0.5
        y += 0.5

        return x, y, detector

    def get_pixel(self, ra, dec, wavelength, dispersion_order=1):
        """Return the pixel position of the spectral element given by
        sky coordinate RA, Dec and wavelength (angstrom)

        Parameters
        -----------
        ra : list
            Right ascension angle in deg
        dec : list
            Declination angle in deg
        wavelength : list
            wavelength in angstroms
        dispersion_order : int
            Dispersion order (default 1)

        Returns
        -------
        int, int, int
            x, y pixel position
            detector number
        """
        try:
            n = len(ra)
        except TypeError:
            return self.get_pixel_(ra, dec, wavelength, dispersion_order)

        x = np.zeros(n, dtype=float)
        y = np.zeros(n, dtype=float)
        detector = np.zeros(n, dtype=int)

        # t0 = time.time()
        for i in range(n):
            x[i], y[i], detector[i] = self.get_pixel_(
                ra[i], dec[i], wavelength[i], dispersion_order)
        # dt = time.time() - t0
        # print(f"get_pixel {n} calls: {1000*dt/n:0.3} msec/call")
        return x, y, detector

    def cutout_detector(self, det, x, y, wavelength, ra, dec, wspace=50):
        """ """
        fullimage, fullmask, fullvar = self.get_detector(det)

        nrow, ncol = fullimage.shape

        x_low = max(0, int(x.min() - wspace))
        x_high = min(ncol, int(x.max() + wspace))
        y_low = max(0, int(y.min() - wspace))
        y_high = min(nrow, int(y.max() + wspace))

        bbox = (x_low, x_high, y_low, y_high)

        im = fullimage[y_low:y_high, x_low:x_high]
        mask = fullmask[y_low:y_high, x_low:x_high]
        var = fullvar[y_low:y_high, x_low:x_high]

        # valid = (var > 0)&(mask > 0)&np.isfinite(var)&(np.isfinite(mask))&(np.isfinite(im))

        # print(f"cutout detector {det}, corner, {bbox[0], bbox[2]}, shape {im.shape}")

        wave_map = SkyWaveMap(
            x - bbox[0],
            y - bbox[2],
            wavelength
        )

        wave_map.fit_sky(ra, dec, self.get_pixel, bbox)

        pack = dict(
            image=im,
            mask=mask,
            var=var,
            map=wave_map,
            bbox=bbox,
            # detector_name=det,
            detector_index=det
        )
        return pack


    def cutout(self, ra, dec, redshift, lam_min=12000, lam_max=19000, lam_step=20):
        """ """
        n = int((lam_max - lam_min)*1. / lam_step)
        wave_trace = np.linspace(lam_min, lam_max, n)

        detx, dety, detid = self.get_pixel(
            ra*np.ones(n),
            dec*np.ones(n),
            wave_trace
        )
        valid = detid > 0
        detx = detx[valid]
        dety = dety[valid]
        detid = detid[valid]
        wave_trace = wave_trace[valid]

        pack_list = {}
        for det in np.unique(detid):
            if det == 0:
                continue
            print(f"Detector {det}")
            sel = detid == det
            detx_ = detx[sel]
            dety_ = dety[sel]
            wave_trace_ = wave_trace[sel]

            pack = self.cutout_detector(det, detx_, dety_, wave_trace_, ra, dec)

            pack['redshift'] = redshift
            pack['ra'] = ra
            pack['dec'] = dec
            # pack['filename'] = frame_path

            pack_list[det] = pack

        return pack_list



def label_wavelength(map, wavelength, label, h=20, c='c', **plot_params):
    """ """
    linx, liny = map.wavelength_to_pixel(wavelength)
    plt.text(linx, liny-h, label, c=c, rotation=90, ha='center', va='top', **plot_params)

    plt.plot([linx, linx], [liny-h, liny-h/2], c=c)
    plt.plot([linx, linx], [liny+h, liny+h/2], c=c)


def plot_box(map, h=10, wave_min=12000, wave_max=19000, npoints=100, c='c', **plot_params):
    """ """
    x, y = map.wavelength_to_pixel(np.linspace(wave_min, wave_max, npoints))
    # print(x)
    # print(y)
    xx = np.concatenate([x, x[::-1], [x[0]]])
    yy = np.concatenate([y-h, y[::-1]+h, [y[0]-h]])
    plt.plot(xx, yy, c=c)


def plot_cutout(image=None, mask=None, redshift=None, map=None, c='c', **kwargs):
    """ """
    valid = (mask==0) & np.isfinite(mask) & np.isfinite(image)
    # image[valid==False] = 0
    # low, high = np.percentile(image[valid], [5,99.9])
    low = -20
    high = 300
    print(f"range {low}, {high}")
    plt.imshow(image, vmin=low, vmax=high, cmap='plasma', origin='lower')
    # plt.imshow(valid, vmin=0, vmax=1, cmap='plasma', origin='lower')

    wavelength = []
    names = []
    for name, wave in line_list.items():
        wave_obs = wave * (1 + redshift)
        if (wave_obs>12000)&(wave_obs<19000):
            label_wavelength(map,  wave_obs, name, c=c)

    plot_box(map, c='c', lw=1)



def plot(pack_list):
    plt.figure(figsize=(10,10))
    n = 0
    for p in pack_list:
        n += len(p)

    i = 0
    for p in pack_list:
        for key in p.keys():
            plt.subplot(n, 1, i+1, aspect='equal')
            plot_cutout(**p[key])
            i += 1


def fit_gaussian(image, mask, var, x0, y0, fwhm=1, nphot=1e4, bias=2, offx=0, offy=0):
    """ """
    nrow, ncol = image.shape
    biny = np.arange(nrow+1) + offx
    binx = np.arange(ncol+1) + offy

    sigma = fwhm/2.355
    x, y = np.random.normal(0, sigma, (2, int(nphot)))
    x += x0
    y += y0

    h, ey, ex = np.histogram2d(y, x, bins=(biny, binx))
    if np.sum(h) < nphot/2:
        return 0, 0
    h = h *1./ np.sum(h)

    valid = (mask==0) & np.isfinite(image) & np.isfinite(var) & (var > 0)

    if np.sum(valid) == 0:
        return 0, 0

    invvar = 1./var[valid] / bias
    top = np.sum(h[valid] * image[valid] * invvar)
    bot = np.sum(h[valid]**2*invvar)

    return top, bot



def extract(pack_list, wavelengths, fwhm, offx=0, offy=0):
    """ """
    top_sum = np.zeros(len(wavelengths))
    bot_sum = np.zeros(len(wavelengths))

    fit_list = []
    for pack in pack_list:
        x, y = pack['map'].wavelength_to_pixel(wavelengths)
        top = np.zeros(len(wavelengths))
        bot = np.zeros(len(wavelengths))

        for i, wave in enumerate(wavelengths):
            top[i], bot[i] = fit_gaussian(pack['image'], pack['mask'], pack['var'], x[i], y[i], fwhm, offx=offx, offy=offy)

        top_sum += top
        bot_sum += bot
        fit_list.append((top, bot))

    valid = bot_sum > 0
    amp = top_sum / bot_sum
    err = 1./np.sqrt(bot_sum)
    return amp, err, fit_list


def plot_spec(wavelengths, amp, sig_amp, redshift=None):
    """ """
    plt.fill_between(wavelengths, amp-sig_amp, amp+sig_amp, fc='orange')
    plt.plot(wavelengths, amp, c='k', lw=1)

    plt.plot(wavelengths, sig_amp)

    for name, wave in line_list.items():
        wave_obs = wave*(1+redshift)
        if (wave_obs > 12000)&(wave_obs < 19000):
            plt.axvline(wave_obs, dashes=[4,1], c='grey', zorder=0, lw=1)
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Signal")

def trace_profile(image, mask, x, y, h=10):
    """ """
    stack = 0
    count = 0
    # h = 10
    done_already = {}
    for i in range(len(x)):
        xx = int(x[i])
        if xx in done_already:
            continue
        done_already[xx] = 1
        yy = int(y[i])
        col = image[yy-h:yy+h, xx].copy()
        mcol = mask[yy-h:yy+h, xx].copy()
        valid = mcol > 0
        col[valid==False] = 0
        stack = stack + col
        count = count + valid
    # print(f"{count=}")
    # print(f"{stack=}")
    valid = count > 0
    out = np.zeros(len(stack))
    out[valid] = stack[valid]/count[valid]
    return out


def cross_profile(image, model, mask, galaxy, frame, background=0, h=50):
    """ """
    wave = np.arange(12000, 19000, 10)
    x, y, detector = frame.optics.radec_to_pixel(
            galaxy.params['ra']*np.ones(len(wave)),
            galaxy.params['dec']*np.ones(len(wave)),
            wave,
            objid=galaxy.params['id']
    )
    prof_im = {}
    prof_mod = {}
    prof_resid = {}
    for d in image.keys():
        mask_ = mask[d] > -1
        sel = detector == d

        x_ = x[sel]
        y_ = y[sel]
        if len(x_) == 0:
            continue
        prof_im[d] = trace_profile(image[d], mask_, x_, y_, h=h)
        prof_mod[d] = trace_profile(model[d], mask_, x_, y_, h=h)
        prof_resid[d] = trace_profile(image[d] - model[d], mask_, x_, y_, h=h)
    return prof_im, prof_mod, prof_resid
