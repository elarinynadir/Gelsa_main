import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter

from . import sample_dist
from . import profile_sampler
from . import consts

# from . import gravlens

class Galaxy:
    _default_params = {
        'ra': 0,
        'dec': 0,
        'redshift': 1.0,
        'fwhm_arcsec': 1.,
        'profile': 'gaussian',
        'disk_r50': 1.,
        'bulge_r50': 1.,
        'bulge_fraction': 0.2,
        'pa': 0,
        'axis_ratio': 1,
        'obs_wavelength_step': 1,
        'obs_wavelength_range': (12000., 19000.),
        'continuum_params': (15000, -1e-5, -18),
        'fluxes_emlines' : np.zeros(11),
        'velocity_disp': 1.2e15, #angstrom/s
        'nlines': 11,
        'id': 0,
        'lens': False,
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Unknown galaxy parameter {key}")

        self.wavelength = np.arange(
            self.params['obs_wavelength_range'][0],
            self.params['obs_wavelength_range'][1],
            self.params['obs_wavelength_step']
        )

        self.init_sed()

        # if self.params['lens']:
            # self.lens_imager = gravlens.LensImager()


    def copy(self):
        params = self.params.copy()
        return type(self)(**params)


 #   def init_emline(self):

 #       redshift = self.params['redshift']
 #       wavelength_obs = (1 + self.params['redshift']) * consts.rest_wavelength_ha
 #       sigma_size = self.params['velocity_disp']/consts.c*wavelength_obs
 #       ampl = self.params['fha']/np.sqrt(2 * np.pi * sigma_size**2)

 #       emline_range=(wavelength_obs-5*sigma_size, wavelength_obs+5*sigma_size)
 #       self.emline_wave = np.arange(
 #           *emline_range,
 #           sigma_size/10.)

 #       n=500

 #       self.emline = ampl * np.random.normal(wavelength_obs, sigma_size, n)
        #self.emline = ampl * np.exp(-(self.emline_wave - wavelength_obs)**2/(2 * sigma_size**2))

    def init_sed(self):
        """Take care of units"""
        x, a, b = self.params['continuum_params']
        self.sed = 10**((self.wavelength - x) * a + b)

    @property
    def profile_sampler(self):
        # try:
            # return self._profile_sampler
        # except AttributeError:
        if self.params['profile'][0].lower() == 'g': # profile name starts with g for gaussian
            # profile is gaussian
            # print("initalizing gaussian profile")
            self._profile_sampler = self._sample_gaussian
        else:
            # profile is bulgydisk
            # print("initalizing bulgy disk profile")
            self._profile_sampler = self._sample_bulgy_disk
        return self._profile_sampler

    @property
    def sed(self):
        return self._sed

    @sed.setter
    def sed(self, y):
        self.sed_params = y
        self._sed = sample_dist.SampleDistribution(self.wavelength, y)

    @property
    def halflight_radius(self):
        """Return the half-light radius in arcsec"""
        try:
            return self._halflight_radius
        except AttributeError:
            pass
        # sample the profile to compute the half light radius
        coord_xy = self.profile_sampler(1e4)
        r = np.sqrt(coord_xy[:, 0]**2 + coord_xy[:, 1]**2)
        self._halflight_radius = np.median(r)

        return self._halflight_radius

    def set_sed(self, wavelength, flux):
        """ """
        wave_obs = wavelength * (1 + self.params['redshift'])
        spec_interpolator = interpolate.interp1d(wave_obs, flux)
        self.sed = spec_interpolator(self.wavelength)

    def set_flux_line(self, **kwargs):
        """ """
        for name, flux in kwargs.items():
            i = consts.line_indices[name]
            self.params['fluxes_emlines'][i] = flux

    def set_flux_Ha(self, flux_Ha=1e-15, NII_Ha_ratio=0.2):
        """ """
        self.set_flux_line(Ha=flux_Ha)
        flux_N2 = flux_Ha * NII_Ha_ratio
        flux_N2a = flux_N2 * 1./4
        flux_N2b = flux_N2 * 3./4
        self.set_flux_line(N2_6549=flux_N2a)
        self.set_flux_line(N2_6585=flux_N2b)

    def set_flux_HbO3(self, flux_Hb=1e-16, flux_O3_5008=1e-16):
        """ """
        self.set_flux_line(Hb=flux_Hb)
        flux_O3_4960 = flux_O3_5008 / 3.
        self.set_flux_line(O3_5008=flux_O3_5008)
        self.set_flux_line(O3_4960=flux_O3_4960)

 #   @property
 #   def emline(self):
 #       return self._emline

 #   @emline.setter
 #   def emline(self,y):

 #       redshift = self.params['redshift']
 #       wavelength_obs = (1 + self.params['redshift']) * consts.rest_wavelength_ha
 #       sigma_size = self.params['velocity_disp']/consts.c*wavelength_obs
 #       ampl = self.params['fha']/np.sqrt(2 * np.pi * sigma_size**2)

 #       emline_range=(wavelength_obs-5*sigma_size, wavelength_obs+5*sigma_size)
 #       self.emline_wave = np.arange(
 #           *emline_range,
 #           sigma_size/10.)

 #       n=500

 #       self.emline = ampl * np.random.normal(wavelength_obs, sigma_size, n)
 #       self._emline = interpolate.interp1d(self.emline_wave,y)

    def _sample_gaussian(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.fwhm_gaussian_sampler(n) * self.params['fwhm_arcsec']

    def _sample_bulge(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.bulge_sampler(n) * self.params['bulge_r50']

    def _sample_disk(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.disk_sampler(n) * self.params['disk_r50']

    def _sample_bulgy_disk(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))

        n_bulge = int(np.round(self.params['bulge_fraction'] * n))
        n_disk = n - n_bulge
        samples = []
        if n_bulge > 0:
            samples.append(self._sample_bulge(n_bulge))
        if n_disk > 0:
            samples.append(self._sample_disk(n_disk))

        return np.vstack(samples)

    def sample_image(self, n):
        """Draw samples from the galaxy image"""
        coord_xy = self.profile_sampler(n)

        root_axis_ratio = np.sqrt(self.params['axis_ratio'])
        coord_xy *= np.array([root_axis_ratio, 1./root_axis_ratio])

        # rotate to position angle
        angle = np.deg2rad(self.params['pa'])
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        rotation_mat = np.array([[cosangle, -sinangle], [sinangle, cosangle]])
        coord_xy = coord_xy @ rotation_mat

        # convert arcsec to deg
        coord_xy /= 3600

        x, y = coord_xy.transpose()

        if self.params['lens']:
            print("Running lens imager...", len(x))
            x, y = self.lens_imager(x, y)
            print("Done", len(x))

        # rotate to RA, Dec on sky
        x = self.params['ra'] + x / np.cos(np.deg2rad(self.params['dec']))
        y = self.params['dec'] + y

        return x, y



class GalaxyComponent:
    _default_params = {
        'ra': 0,
        'dec': 0,
        'xmean' : None,
        'ymean' : None,
        'fwhm': None,
        'fwhm_arcsec': 1.,
        'profile': 'gaussian',
        'disk_r50': 1.,
        'bulge_r50': 1.,
        'bulge_fraction': 0.2,
        'pa': 0,
        'axis_ratio': 1,
        'weight': 1,
        'id': 0,
        'wcs': None
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Unknown galaxy component parameter {key}")

    def copy(self):
        params = self.params.copy()
        return type(self)(**params)

    @property
    def profile_sampler(self):
        """ """
        if self.params['profile'][0].lower() == 'g':  # Gaussian profile
            self._profile_sampler = self._sample_gaussian
        else:  # Bulge-disk profile
            self._profile_sampler = self._sample_bulgy_disk
        return self._profile_sampler

    @property
    def halflight_radius(self):
        """"""
        try:
            return self._halflight_radius
        except AttributeError:
            pass
        coord_xy = self.profile_sampler(1e4)
        r = np.sqrt(coord_xy[:, 0]**2 + coord_xy[:, 1]**2)
        self._halflight_radius = np.median(r)
        return self._halflight_radius

    def gaussian_sampler(self,n):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.fwhm_gaussian_sampler(n) * self.params['fwhm']

    def _sample_gaussian(self, n):
        """"""
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.fwhm_gaussian_sampler(n) * self.params['fwhm_arcsec']

    def _sample_bulge(self, n):
        """"""
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.bulge_sampler(n) * self.params['bulge_r50']

    def _sample_disk(self, n):
        """"""
        if n == 0:
            return np.zeros((n, 2))
        return profile_sampler.disk_sampler(n) * self.params['disk_r50']

    def _sample_bulgy_disk(self, n):
        """"""
        if n == 0:
            return np.zeros((n, 2))

        n_bulge = int(np.round(self.params['bulge_fraction'] * n))
        n_disk = n - n_bulge
        samples = []
        if n_bulge > 0:
            samples.append(self._sample_bulge(n_bulge))
        if n_disk > 0:
            samples.append(self._sample_disk(n_disk))

        return np.vstack(samples)

    def sample_image(self, n):
        '''Uses Ra-Dec coordinates'''
        coord_xy = self.profile_sampler(n)

        root_axis_ratio = np.sqrt(self.params['axis_ratio'])
        coord_xy *= np.array([root_axis_ratio, 1./root_axis_ratio])

        # Rotate to position angle
        angle = np.deg2rad(self.params['pa'])
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        rotation_mat = np.array([[cosangle, -sinangle], [sinangle, cosangle]])
        coord_xy = coord_xy @ rotation_mat

        # Convert arcsec to degrees
        coord_xy /= 3600

        x, y = coord_xy.transpose()

        # Rotate to RA, Dec on sky
        x = self.params['ra'] + x / np.cos(np.deg2rad(self.params['dec']))
        y = self.params['dec'] + y

        return x, y

    def sample_image2(self, n):
        '''Uses Pixel coordinates'''
        coord_xy = self.gaussian_sampler(n)

        root_axis_ratio = np.sqrt(self.params['axis_ratio'])
        coord_xy *= np.array([root_axis_ratio, 1./root_axis_ratio])

        # Rotate to position angle
        angle = np.deg2rad(self.params['pa'])
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        rotation_mat = np.array([[cosangle, -sinangle], [sinangle, cosangle]])
        coord_xy = coord_xy @ rotation_mat
        x, y = coord_xy.transpose()
        x+= self.params['xmean']
        y+= self.params['ymean']

        return x, y




class Galaxy_v2:
    _default_params = {
        'ra': 0,
        'dec': 0,
        'redshift': 1.0,
        'obs_wavelength_step': 100,
        'obs_wavelength_range': (12000., 19000.),
        'continuum_params': (15000, -1e-5, -18),
        'fluxes_emlines': np.zeros(8),
        'velocity_disp': 1.2e15,  # angstrom/s
        'id': 0,
        'lens': False,
        'component_number': 1, # Number of components,
        'nlines': 8, 
        'wcs' : None
    }

    def __init__(self, components=None, **kwargs):
        """"""
        self.params = self._default_params.copy()
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Unknown galaxy parameter {key}")

        self.wavelength = np.arange(
            self.params['obs_wavelength_range'][0],
            self.params['obs_wavelength_range'][1],
            self.params['obs_wavelength_step']
        )

        self.init_sed()

        if components is not None:
            self.components = components
        else:
            []
        if len(self.components) != self.params['component_number']:
            raise ValueError("Number of components provided does not match 'components' parameter.")

    @property
    def profile_sampler(self, n):
        """ """
        total_samples = np.zeros((n, 2))
        for component in self.components:
            component_samples = component.profile_sampler(n*component.params['weight'])
            total_samples += component_samples
        return total_samples

    @property
    def halflight_radius(self):
        """"""
        try:
            return self._halflight_radius
        except AttributeError:
            pass

        ra, dec = self.sample_image(100000)
        ra-=self.params['ra']
        dec-=self.params['dec']


        pix_scale =0.1

        x = ra * 3600 / pix_scale
        y = dec * 3600 / pix_scale

        bin = np.arange(-26,26,1)

        h, ey, ex = np.histogram2d(x, y, (bin, bin))

        h_smooth = gaussian_filter(h, sigma=1)

        total_light = np.sum(h_smooth)
        half_light_threshold = total_light / 2
        cumsum_image = np.cumsum(h_smooth.flatten())
        cumsum_image = cumsum_image.reshape(h_smooth.shape)


        self._halflight_radius = np.sqrt(np.mean(np.where(cumsum_image >= half_light_threshold)[0])) * pix_scale

        return self._halflight_radius


    def copy(self):
        params = self.params.copy()
        components_copy = [comp.copy() for comp in self.components]
        return type(self)(components=components_copy, **params)

    def init_sed(self):
        """Initialize the SED based on the continuum parameters."""
        x, a, b = self.params['continuum_params']
        self.sed = 10**((self.wavelength - x) * a + b)

    @property
    def sed(self):
        return self._sed

    @sed.setter
    def sed(self, y):
        self.sed_params = y
        self._sed = sample_dist.SampleDistribution(self.wavelength, y)

    def sample_image(self, n):
        '''Uses Ra-Dec coordinates'''
        total_x, total_y = np.array([]), np.array([])
        n_sampled = 0
        norm_sum = 0

        for comp in self.components:
            sigma = comp.params['fwhm_arcsec']*10/(2*np.sqrt(2*np.log(2)))
            norm_sum += comp.params['weight']*(sigma**2)

        sigma_0 =  self.components[0].params['fwhm_arcsec']*10/(2*np.sqrt(2*np.log(2)))
        n_0 = (n* self.components[0].params['weight']*(sigma_0**2))/norm_sum


        for comp in self.components:
            s = comp.params['fwhm_arcsec']*10/(2*np.sqrt(2*np.log(2)))
            n_ = int(n_0 * comp.params['weight']*(s**2) / (self.components[0].params['weight']*(sigma_0**2)))
            n_sampled +=n_
            x, y = comp.sample_image(n_)
            total_x = np.concatenate([total_x, x])
            total_y = np.concatenate([total_y, y])

        x, y =  self.components[0].sample_image(n-n_sampled)
        total_x = np.concatenate([total_x, x])
        total_y = np.concatenate([total_y, y])
        return total_x, total_y

    def sample_image2(self, n):
        '''Uses Pixel coordinates'''
        total_x, total_y = np.array([]), np.array([])
        n_sampled = 0
        norm_sum = 0

        for comp in self.components:
            sigma = comp.params['fwhm']/(2*np.sqrt(2*np.log(2)))
            norm_sum += comp.params['weight']*(sigma**2)

        sigma_0 =  self.components[0].params['fwhm']/(2*np.sqrt(2*np.log(2)))
        n_0 = (n* self.components[0].params['weight']*(sigma_0**2))/norm_sum


        for comp in self.components:
            s = comp.params['fwhm']/(2*np.sqrt(2*np.log(2)))
            n_ = int(n_0 * comp.params['weight']*(s**2) / (self.components[0].params['weight']*(sigma_0**2)))
            n_sampled +=n_
            x, y = comp.sample_image2(n_)
            total_x = np.concatenate([total_x, x])
            total_y = np.concatenate([total_y, y])

        x, y =  self.components[0].sample_image2(n-n_sampled)
        total_x = np.concatenate([total_x, x])
        total_y = np.concatenate([total_y, y])
        return total_x, total_y