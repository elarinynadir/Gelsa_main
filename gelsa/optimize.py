import numpy as np
from scipy import optimize, sparse, linalg
from tqdm.auto import tqdm

from . import utils
from . import galaxy
from . import consts

def make_gaussian_kernel(shape, nlines, n=3, sigma=1):
    offsets = np.arange(-n, n+1)
    values = np.exp(-0.5*offsets**2/sigma**2)
    values /= np.sum(values)
    #print("v", np.sum(values))
    d = []
    for i in range(len(offsets)):
        d.append(np.ones(shape[0]) * values[i])
    m = sparse.dia_array((d, offsets), shape)
    m = m.tolil()
    '''m[shape[0]-1,:] = 0
    m[:, shape[1]-1] = 0
    m[-1,-1] = 1'''
    for i in range(nlines):
        m[-(i+1),:] = 0
        m[:, -(i+1)] = 0

    for i in range(nlines):
        m[-(i+1),-(i+1)] = 1

    m= m.tocsr()
    return m


def pseudoinv(Q, r=1e-15, k=None):
    """ """
    if k is None:
        k = Q.shape[0]-1

    if k >= Q.shape[0]:
        k = Q.shape[0]-1

    u, s, v = sparse.linalg.svds(Q, k=k)

    sinv = np.zeros(len(s))

    threshold = s.max()*r

    nonzero = s > threshold
    print(f"number of singular values {np.sum(nonzero)} of {Q.shape[0]}")

    sinv[nonzero] = 1./s[nonzero]

    Qinv = np.dot(v.T*sinv, u.T)

    return Qinv


def solve(A, b, b_var, nlines=1, k=None, smooth=False, lam=1e-3, return_cov=False):
    """ """
    inv_var = 1./b_var

    Q = (A.T * inv_var) @ A
    print(type(Q))
    print(Q.shape)

    if lam > 0:
        # print(f"Ridge regression lam={lam}")
        Q += np.eye(Q.shape[0])*lam

    # Qinv = pseudoinv(Q, k=k)
    # print(f"Qinv {type(Qinv)} {Qinv.shape}")
    try:
        print(f"inverting {Q.shape}")
        Qinv = linalg.inv(Q)
    except:
        raise
        print("inv error")
        return np.zeros(A.shape[1])

    if smooth:
        kernel = make_gaussian_kernel(Q.shape, nlines, 3)
        QinvA = kernel @ Qinv @ (A.T * inv_var)
    else:
        print("Qinv @ (A.T * inv_var)")
        QinvA = Qinv @ (A.T * inv_var)

    print("QinvA @ b")
    x = QinvA @ b

    if return_cov:
        cov = (QinvA*b_var) @ QinvA.T
        return x, QinvA*np.sqrt(b_var)
    else:
        return x


def evaluate_model(galaxy, frame_list, mask_list, sed_params=None, flux_scale=None):
    """ """
    if sed_params is not None:
        galaxy.sed = sed_params*1e-18
    # if flux_scale is not None:
        # sed = galaxy.sed_params
        # galaxy.sed = sed * flux_scale
    
    images = []
    for i, frame in enumerate(frame_list):
        images.append(frame.make_image([galaxy], masks=mask_list[i], noise=False, return_var=False))

    # if flux_scale is not None:
        # galaxy.sed = sed
        
    return images


def evaluate_emline_model(sed_params, fluxes, galaxy, frame_list, mask_list):
    """ """
    galaxy.sed = sed_params*1e-18
    galaxy.params['fluxes_emlines']=fluxes*1e-16
    images = []
    for i, frame in enumerate(frame_list):
        images.append(frame.make_image([galaxy], masks=mask_list[i], noise=False, return_var=False))
    return images


def get_data_vec(images, var_images, mask_list):
    """ """
    data_vec = []
    var_vec = []
    for i in range(len(images)):
        d = []
        v = []
        for det in mask_list[i].keys():
            sel = mask_list[i][det].flatten() > -1
            d.append(images[i][det].flatten()[sel])
            v.append(var_images[i][det].flatten()[sel])
        if len(mask_list[i].keys()) > 0 :
            d = np.concatenate(d)
            v = np.concatenate(v)
        if len(d) > 0:
            data_vec = np.concatenate([data_vec, d])
            var_vec = np.concatenate([var_vec, v])
    return data_vec, var_vec


def super_concat(models):
    """ """
    data = []
    for model in models:
        keys = list(model.keys())
        keys.sort()
        for key in keys:
            data.append(model[key])
    return np.concatenate(data)
        

def super_concat_diff(models0, models1):
    """ """
    data = []
    for i, model in enumerate(models0):
        keys = list(model.keys())
        keys.sort()
        for key in keys:
            data.append(model[key] - models1[i][key])
    return np.concatenate(data)


def compute_jacobian(galaxy_list, frame_list, mask_list, dx=1000, fit_background=True):
    """ """
    # print(f"Compute jacobian, gals:{len(galaxy_list)}, frames:{len(frame_list)}, {dx=}")
    galaxy_list = utils.ensurelist(galaxy_list)

    npix = 0
    for mask in mask_list:
        for maskarr in mask.values():
            npix += np.sum(maskarr > -1)

    jac_list = []

    for galaxy in galaxy_list:

        jac = sparse.lil_array(
            (npix, len(galaxy.wavelength)),
            dtype='d'
        )

        galaxy.params['fluxes_emlines'] = np.zeros(galaxy.params['nlines'])
        test_sed = np.zeros(len(galaxy.wavelength), dtype='d')

        for i in range(len(galaxy.wavelength)):
            test_sed *= 0
            test_sed[i] = dx

            model = evaluate_model(galaxy, frame_list, mask_list, sed_params=test_sed)
            concat_model = super_concat(model)

            jac[:, i] = concat_model / dx

        jac_list.append(jac)

    if fit_background:
        jac_ext = []
        pix_count = []
        for i in range(len(mask_list)):
            npix_ = 0
            for d in mask_list[i].keys():
                npix_ += np.sum(mask_list[i][d] > -1)
            pix_count.append(npix_)
        print(f"{pix_count=}")
        jac_bg = sparse.lil_array(
                (npix, len(pix_count)),
                dtype='d'
        )
        a = 0
        b = a+pix_count[0]
        for i in range(len(pix_count)-1):
            print(f"i, a,b {i, a,b} {npix}")
            jac_bg[a:b, i] = 1
            a = b
            b += pix_count[i+1]
        print(f"i, a,b {i+1, a,b} {npix}")
        jac_bg[a:b, 3] = 1
    
    jac = sparse.hstack(jac_list)
    if fit_background:
        jac = sparse.hstack([jac, jac_bg])

    jac = sparse.csr_array(jac)

    # print(f"Computing Jacobian done {jac.shape}")

    return jac


def compute_jacobian_emline(galaxy, frame_list, mask_list, dx=100):
    """ """

    test_sed = np.zeros(len(galaxy.wavelength), dtype='d')

    # perturb the emission line by dx
    test_fluxes = np.zeros(galaxy.params['nlines'])

    jac_list = []

    for i in range(galaxy.params['nlines']):

        test_fluxes *= 0
        test_fluxes[i] = dx

        model = evaluate_emline_model(test_sed, test_fluxes, galaxy, frame_list, mask_list)
        concat_model = super_concat(model)

        jac = concat_model / dx
        jac_list.append(jac)

    jac = np.array(jac_list).T
    jac = sparse.csr_array(jac)

    return jac


def compute_jacobian_emline_2(galaxy, frame_list, mask_list, coeff, dx=50):
    """ """
    test_sed = np.zeros(len(galaxy.wavelength), dtype='d')

    # perturb the emission line by dx
    test_fluxes = np.zeros(galaxy.params['nlines'])

    for i in range(len(test_fluxes)):
        wavelength_obs = (1+galaxy.params['redshift'])*consts.lines[i]
        if wavelength_obs < galaxy.params['obs_wavelength_range'][0] or wavelength_obs > galaxy.params['obs_wavelength_range'][1]:
            test_fluxes[i] = 0
        else:
            test_fluxes[i] = dx*coeff[i]

    model = evaluate_emline_model(test_sed, test_fluxes, galaxy, frame_list, mask_list)
    concat_model = super_concat(model)

    jac = concat_model / dx

    jac = np.array(jac).T
    jac = jac.reshape(len(jac), 1)
    jac = sparse.csr_array(jac)

    return jac


def compute_jacobian_fwhm(galaxy_list, frame_list, mask_list, dfwhm=0.1, fluxboost=1e-14):
    """ """
    print(f"Compute jacobian, gals:{len(galaxy_list)}, frames:{len(frame_list)}, {dfwhm=}")
    galaxy_list = utils.ensurelist(galaxy_list)

    npix = 0
    for mask in mask_list:
        for maskarr in mask.values():
            npix += np.sum(maskarr > -1)

    jac_list = []
    for galaxy in galaxy_list:
        flux_scale = 1#fluxboost / np.mean(galaxy.sed_params)
        print(f"{flux_scale=}")
        
        fwhm_orig = galaxy.params['fwhm_arcsec']

        model0 = evaluate_model(galaxy, frame_list, mask_list, flux_scale=flux_scale)
        galaxy.params['fwhm_arcsec'] += dfwhm

        model = evaluate_model(galaxy, frame_list, mask_list, flux_scale=flux_scale)

        dmodel = super_concat_diff(model, model0)
        
        jac_list.append(dmodel / dfwhm / flux_scale)
        galaxy.params['fwhm_arcsec'] = fwhm_orig

    jac = np.transpose(jac_list)
    print(f"Computing Jacobian done {jac.shape}")

    return jac

    

def measure_everything(images, var_images, frame_list, galaxies, pixmask_list=None, k=None, jac=None, lam=1e-3, smooth=False, fit_background=False, mask_list=None, dx=10000):
    """Measure all sources"""
    wave = galaxies[0].wavelength

    if mask_list is None:
        # print(f"Building masks")
        mask_list = []
        for frame_i in range(len(frame_list)):
            mask_list.append(frame_list[frame_i].make_mask(galaxies, input_mask=pixmask_list[frame_i]))

    if jac is None:
        print('Building jacobian...')
        jac = compute_jacobian(galaxies, frame_list, mask_list, fit_background=fit_background, dx=dx)

    data, var = get_data_vec(images, var_images, mask_list)

    print('Start solving...')

    x = solve(jac, data, var, k=k, lam=lam, smooth=smooth, return_cov=False)

    print("np.dot(jac, x)")
    yfit = jac @ x
    print("chi2")
    chi2 = np.sum((yfit - data)**2 / var)
    dof = len(x)
    print(f"{chi2=} {dof=}")

    del jac
    jac = None

    print('...Finished solving')

    if fit_background:
        backgrounds = x[-4:]
        extractions = x[:-4].reshape((len(galaxies), len(galaxies[0].wavelength)))
    else:
        backgrounds = np.zeros(4)
        extractions = x.reshape((len(galaxies), len(galaxies[0].wavelength)))

    galaxies_out = []
    for i in range(len(galaxies)):
        new_gal = galaxies[i].copy()
        new_gal.sed = extractions[i] * 1e-18
        galaxies_out.append(new_gal)

    fit_info = dict(yfit=yfit, chi2=chi2, dof=dof, jac=jac, mask_list=mask_list, backgrounds=backgrounds)

    return galaxies_out, fit_info


def measure_fwhm(images, var_images, frame_list, galaxies, pixmask_list=None, k=None, jac=None, lam=1e-3, smooth=False, dfwhm=0.1, fluxboost=1e-14):
    """ """
    print(f"Building masks")
    masks_list = []
    for frame_i in range(len(frame_list)):
        masks_list.append(frame_list[frame_i].make_mask(galaxies, input_mask=pixmask_list[frame_i]))

    if jac is None:
        print('Making jacobian...')
        jac = compute_jacobian_fwhm(galaxies, frame_list, masks_list, dfwhm=dfwhm, fluxboost=fluxboost)

    resid = []
    for i in range(len(images)):
        model = frame_list[i].make_image(galaxies, noise=False, return_var=False)
        r = {}
        for d in model.keys():
            r[d] = images[i][d] - model[d]
        resid.append(r)
        
    data, var = get_data_vec(resid, var_images, masks_list)

    print('Start solving...')

    x = solve(jac, data, var, k=k, lam=lam, smooth=smooth, return_cov=False)

    # yfit = np.dot(jac, x)
    # chi2 = np.sum((jac @ x - data)**2 / var)
    # dof = len(x)
    # print(f"{chi2=} {dof=}")

    print('...Finished solving')


    for i in range(len(galaxies)):
        print(i, x[i])
        fwhm = galaxies[i].params['fwhm_arcsec']
        fwhm = max(0.1, fwhm + x[i])
        galaxies[i].params['fwhm_arcsec'] = fwhm
        
    return galaxies, jac, masks_list


# def grid_search_fwhm():
#     """ """
#     fit_index = 0
#     fwhm_grid = np.arange(0.5, 4.5, 0.5)
    
#     chi2_list = []
#     for fwhm in fwhm_grid:
#         print(f"----- FWHM {fwhm} -----")
#         overlap_gal_list[fit_index].params['fwhm_arcsec'] = fwhm
#         gal_fits_single, fit_info_single = optimize.measure_everything(
#             image_list, var_list, frame_list, [overlap_gal_list[fit_index]], pixmask_list=pixmask_list, 
#             lam=5e-7, 
#             smooth=False, 
#             fit_background=False
#         )
#         chi2_list.append(fit_info_single['chi2'])
#     return fwhm_grid, chi2_list

def evaluate_fit(image_list, var_list, frame_list, pixmask_list, gal, fwhm_grid, cache=None):
    """ """
    if cache is None:
        cache = {}
    chi2_out = []
    for fwhm in fwhm_grid:
        key = int(fwhm*1000)
        if key in cache:
            continue
        print(f"----- gal {gal.params['id']} FWHM {fwhm} -----")
        gal.params['fwhm_arcsec'] = fwhm
        gal_fits_single, fit_info_single = measure_everything(
            image_list, var_list, frame_list, [gal], pixmask_list=pixmask_list, 
            lam=5e-7, 
            smooth=False, 
            fit_background=False
        )
        cache[key] = fit_info_single['chi2']
        chi2_out.append(fit_info_single['chi2'])
    fwhm_grid = np.array(fwhm_grid)
    chi2_out = np.array(chi2_out)
    return fwhm_grid, chi2_out

def halving_optimizer(image_list, var_list, frame_list, pixmask_list, gal, fwhm_bounds=[0.1, 1., 2.0, 3, 4.0], max_iterations=10):
    """ """
    cache = {}
    fwhm, chi2 = evaluate_fit(image_list, var_list, frame_list, pixmask_list, gal, fwhm_bounds, cache=cache)
    fwhm_ = fwhm[-1]
    delta_fwhm=1
    for loop in range(max_iterations):
        best = np.argmin(chi2)
        left = max(0, best-1)
        right = min(len(chi2)-1,best+1) 
        next_left = (fwhm[best] + fwhm[left])/2.
        next_right = (fwhm[best] + fwhm[right])/2.
        next = [next_left, next_right]
        delta_fwhm = np.abs(next_right - next_left)
        fwhm_, chi2_ = evaluate_fit(image_list, var_list, frame_list, pixmask_list, gal, next, cache=cache)
        fwhm = np.concatenate([fwhm, fwhm_])
        chi2 = np.concatenate([chi2, chi2_])
        o = np.argsort(fwhm)
        fwhm = fwhm[o]
        chi2 = chi2[o]
        if delta_fwhm < 0.4:
            break
    best = np.argmin(chi2)
    fwhm_best = fwhm[best]
    print(f"done fwhm:{fwhm_best}, chi2: {chi2[best]}")
    gal.params['fwhm_arcsec'] = fwhm[best]
    return gal, (fwhm, chi2)
        
