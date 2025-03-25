import os

from . import specframe
from . import zero_order_mask
from .sgs import (
    detector_model,
    optical_model,
    displacement_model,
    sensitivity_model,
    relative_flux,
    psf_model
)


class Gelsa:
    """ """
    _default_config = {
        'workdir': '.',
        'datadir': '.',
        'abs_file': "SIR_Calibration_Abs_1.0.5-ON_THE_FLY-pcasenov-PLAN-000000-GJ00XYPR-20240219-152345-0-new_abs_calib-0.xml",
        'opt_file': 'SIR_Calib_Opt_1.0.8-ON_THE_FLY-pcasenov-PLAN-000000-6YI2Z9UY-20240213-154829-0-new_opt_cal-0.xml',
        # 'ids_file': 'SIR_Calib_Ids_1.0.5-ON_THE_FLY-pcasenov-PLAN-000000-4S9TBW7A-20240215-080942-0-new_ids_calib-0.xml',
        'ids_file': 'DpdSirIdsModel__SIR_Calib_Ids_EUCLID_1.0.5-ON_THE_FLY-pcasenov-PLAN-000001-67PH88PO-20240803-211109-0-new_ids_calib-0.xml',
        'crv_file': 'SIR_Calib_Crv_1.0.6-ON_THE_FLY-pcasenov-PLAN-000000-US9LP4ZZ-20240214-082238-0-new_crv_cal-0.xml',
        'location_table': None,
        'detector_slots_path': 'EUC_SIR_DETMODEL_REAL_DATA_OP_01.csv',
        'rel_flux_file': 'EUC_SIR_W-RELATIVEFLUX-SCALE_1_20240805T191055.394545Z.fits',
        'psf_file': 'psf_wavelength.json',
        # options
        'use_psf': True,
        'use_chromatic_psf': False,
        'use_relative_flux_loss': False,
        'zero_order_catalog': None
    }

    def __init__(self, config={}, **kwargs):
        """ """
        self.config = self._default_config.copy()
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"Warning! unknown option: {key}={value}")
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"Warning! unknown option: {key}={value}")

    @property
    def detector_model(self):
        """ """
        try:
            return self._detector_model
        except AttributeError:
            pass
        path = os.path.join(self.config['workdir'], self.config['detector_slots_path'])
        self._detector_model = detector_model.DetectorModel(
            detector_slots_path=path
        )
        return self._detector_model

    @property
    def optical_model(self):
        """ """
        try:
            return self._optical_model
        except AttributeError:
            pass
        path = os.path.join(self.config['workdir'], self.config['opt_file'])
        self._optical_model = optical_model.OpticalModel(
            path=path
        )
        return self._optical_model

    @property
    def dispersion_model(self):
        """ """
        try:
            return self._dispersion_model
        except AttributeError:
            pass
        path = os.path.join(self.config['workdir'], self.config['ids_file'])
        ids_model = displacement_model.DisplacementModel(
            path=path
        )
        path = os.path.join(self.config['workdir'], self.config['crv_file'])
        crv_model = displacement_model.DisplacementModel(
            path=path
        )
        self._dispersion_model = {
            'ids': ids_model,
            'crv': crv_model
        }
        return self._dispersion_model

    @property
    def sensitivity_model(self):
        """ """
        try:
            return self._sensitivity_model
        except AttributeError:
            pass
        path = os.path.join(self.config['workdir'], self.config['abs_file'])
        self._sensitivity_model = sensitivity_model.SensitivityModel(
            path=path,
        )
        return self._sensitivity_model

    # @property
    # def relative_flux_model(self):
    #     """ """
    #     try:
    #         return self._relative_flux_model
    #     except AttributeError:
    #         pass
    #     if self.config['use_relative_flux_loss']:
    #         path = os.path.join(self.config['workdir'], self.config['rel_flux_file'])
    #         self._relative_flux_model = relative_flux.RelativeFluxCalibration(
    #             path=path,
    #         )
    #     else:
    #         self._relative_flux_model = None
    #     return self._relative_flux_model

    @property
    def psf_model(self):
        """ """
        try:
            return self._psf_model
        except AttributeError:
            pass
        if not self.config['use_psf']:
            self._psf_model = None
        elif self.config['use_chromatic_psf']:
            path = os.path.join(self.config['workdir'], self.config['psf_file'])
            self._psf_model = psf_model.PSFModel(
                path=path,
            )
        else:
            self._psf_model = psf_model.DefaultPSFModel()
        return self._psf_model

    @property
    def zero_order_mask(self):
        """ """
        try:
            return self._zero_order_mask
        except AttributeError:
            self._zero_order_mask = zero_order_mask.ZeroOrderMask(self.config['zero_order_catalog'])
        return self._zero_order_mask


    def load_frame(self, frame_path, loctable_path=None):
        """ """
        frame = specframe.SpecFrame()

        if loctable_path is not None:
            loctable_path = os.path.join(self.config['workdir'], loctable_path)

        frame.load_frame(
            os.path.join(self.config['workdir'], frame_path),
            loctable_path=loctable_path
        )
        frame.set_optical_model(self.optical_model)
        frame.set_detector_model(self.detector_model)
        frame.set_dispersion_model(self.dispersion_model)
        frame.set_sensitivity(self.sensitivity_model)
        # frame.set_relative_flux_model(self.relative_flux_model)
        frame.set_psf_model(self.psf_model)
        frame.set_zero_order_mask(self.zero_order_mask)
        return frame



    
    def load_frame_h5(self, frame_path, loctable_path=None):
        """ """
        frame = specframe.SpecFrame()

        if loctable_path is not None:
            loctable_path = os.path.join(self.config['workdir'], loctable_path)

        frame.load_frame_h5(
            os.path.join(self.config['workdir'], frame_path),
            loctable_path=loctable_path
        )
        frame.set_optical_model(self.optical_model)
        frame.set_detector_model(self.detector_model)
        frame.set_dispersion_model(self.dispersion_model)
        frame.set_sensitivity(self.sensitivity_model)
        #frame.set_relative_flux_model(self.relative_flux_model)
        frame.set_psf_model(self.psf_model)
        frame.set_zero_order_mask(self.zero_order_mask)

        return frame

    def new_frame(self, ra, dec, pa, **params):
        """ """
        frame = specframe.SpecFrame(RA=ra, DEC=dec, PA=pa, **params)
        frame.set_optical_model(self.optical_model)
        frame.set_detector_model(self.detector_model)
        frame.set_dispersion_model(self.dispersion_model)
        frame.set_sensitivity(self.sensitivity_model)
        # frame.set_relative_flux_model(self.relative_flux_model)
        frame.set_psf_model(self.psf_model)
        return frame
