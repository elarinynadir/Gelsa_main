GELSA: Advanced tools for Euclid grism spectroscopy
===================================================

Overview
--------



Contributors
------------

Ben Granett (benjamin.granett@inaf.it), Louis Gabarra, Francesca Passalacqua, Federico Lepri, Nadir El Ariny



Install
-------

Dependencies:

* numpy
* pandas
* scipy
* astropy
* matplotlib
* tqdm
* scikit-image
* numba
* healpy
* h5py (for reading SIR location table data product)

Installation can be done with `pip`:
```
pip install .
```

For development, use the `-e` option to install in editable mode:
```
pip install -e .
```

Usage
-----

The following code loads a SIR calibrated frame, cuts out the image of the spectrum, and displays it.

```
import gelsa

config = {
    'workdir': '.',
    'abs_file': "calib/DpdSirAbsoluteFluxScaling__SIR_Calibration_Abs_EUCLID_1.0.5-ON_THE_FLY-pcasenov-PLAN-000001-93KUXBON-20240806-071439-0-new_abs_calib-0.xml",
    'opt_file': 'calib/SIR_Calib_Opt_EUCLID_1.0.8-ON_THE_FLY-pcasenov-PLAN-000001-54BOM20F-20240802-145647-0-new_opt_cal-0.xml',
    'ids_file': 'calib/SIR_Calib_Ids_EUCLID_1.0.5-ON_THE_FLY-pcasenov-PLAN-000001-67PH88PO-20240803-211109-0-new_ids_calib-0.xml',
    'crv_file': 'calib/SIR_Calib_Crv_EUCLID_1.0.6-ON_THE_FLY-pcasenov-PLAN-000001-YGK279V1-20240803-100506-0-new_crv_cal-0.xml',
    'location_table': None,
    'detector_slots_path': 'calib/EUC_SIR_DETMODEL_REAL_DATA_OP_01.csv',
    'rel_flux_file': 'calib/EUC_SIR_W-RELATIVEFLUX-SCALE_1_20240805T191055.394545Z.fits',
    'zero_order_catalog': 'zero_order_cat.fits'
    }  
G = gelsa.Gelsa(config=config)

frame = G.load_frame(
    'data/EUC_SIR_W-SCIFRM_0-145_20240810T121359.070371Z.fits.gz')

ra = 150.11455921612844
dec = 2.284824605042879
z = 0.4486


pack_list = frame.cutout(ra, dec, z)

crop = pack_list[9]['crop']

visu.show(crop.image, crop.mask, crop.var, levels=(1, 99))
```
