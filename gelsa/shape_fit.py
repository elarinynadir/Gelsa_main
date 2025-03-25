import numpy as np
from scipy.optimize import least_squares

from skimage.measure import regionprops
from skimage.measure import label
from . import galaxy


def pixel_in_box(x_pixel, y_pixel, x_arr, y_arr, m , width):

    idx = (np.abs(x_arr - x_pixel)).argmin()
    
    x_low_val = x_arr[idx] - width
    x_up_val = x_arr[idx] + width
    
    y_low_val = y_arr[idx] - np.abs(m*width)
    y_up_val = y_arr[idx] + np.abs(m*width)

    
    if x_low_val <= x_pixel <= x_up_val:
    
        if y_low_val <= y_pixel <= y_up_val:
            return True  
    
    return False  


def elliptical_fit(segmap):
    non_zero_coords = np.argwhere(segmap > 0)

    min_y, min_x = np.min(non_zero_coords, axis=0)
    max_y, max_x = np.max(non_zero_coords, axis=0)

    width = max_x - min_x + 1
    height = max_y - min_y + 1



    labeled_array = label(segmap > 0)
    regions = regionprops(labeled_array)
    region = regions[0]

    semi_major = region.major_axis_length/2
    semi_minor = region.minor_axis_length/2

    max_sigx = semi_major/2
    max_sigy = semi_minor/2
    
    return width/2, height/2, max_sigx, max_sigy


def gaussian(amp, mux, muy, sig_x, sig_y, theta, xx, yy):
    """ """
    xx_ = (xx - mux) 
    yy_ = (yy - muy) 
    a = np.cos(theta)**2 /(2*(sig_x**2)) + np.sin(theta)**2 /(2*(sig_y**2))
    b = np.sin(2 * theta)/(4*(sig_x**2)) - np.sin(2 * theta)/(4*(sig_y**2))
    c = np.sin(theta)**2 /(2*(sig_x**2))+ np.cos(theta)**2 /(2*(sig_y**2))
    return amp * np.exp(-(a * xx_**2 + 2 * b * xx_ * yy_ + c * yy_**2))
    
def model(params, xx, yy):
    """ """
    image = np.zeros(xx.shape, dtype='d')
    for i in range(0, len(params), 6):
        x,y, amp, sig_x, sig_y, theta = params[i:i+6]
        # print((x,y))
        image+= gaussian(amp, x, y, sig_x, sig_y, theta, xx, yy)
    return image 
        

def resid(params, xx, yy, image):
    """ """
    image_mod = model(params, xx, yy)
    r = image_mod - image
    return r.flatten()

def iterative_fit(initial_image, initial_params, bound_x, bound_y, mid_x, mid_y, max_sigx, max_sigy):
    xx, yy = np.meshgrid(np.arange(initial_image.shape[1]), np.arange(initial_image.shape[0])) 
    xx =  xx +0.5
    yy =  yy +0.5
    
    params = initial_params
    
    lower_bounds = np.zeros(len(params))
    upper_bounds = np.zeros(len(params))

    for i in range(0, len(params), 6):
        lower_bounds[i] = (mid_x - bound_x)        # x lower bound
        lower_bounds[i+1] = (mid_y - bound_y)       # y lower bound
        lower_bounds[i+3] = .5        # sigx lower bound
        lower_bounds[i+4] = .5        # sigy lower bound
        upper_bounds[i] = (mid_x + bound_x)     # x upper bound
        upper_bounds[i+1] = (mid_y + bound_y)     # y upper bound
        upper_bounds[i+2] = np.inf 
        upper_bounds[i+3] = max_sigx  # sig_x upper bound
        upper_bounds[i+4] = max_sigy    # sig_y upper bound
        upper_bounds[i+5] = np.pi  # pa upper bound

    fit = least_squares(resid, params, bounds=(lower_bounds, upper_bounds), args=(xx, yy, initial_image))
    image_fit = model(fit.x, xx, yy)

    for i in range(0, len(params), 6):
       fit.x[i+5] = np.pi/2 - fit.x[i+5] 


    return image_fit, fit.x

def generate_params(x_center, y_center, amplitude, max_sigx , max_sigy, n_components=8):
    params = []
    for i in range(n_components):
        scale = amplitude/ (2 ** (i+1)) 
        sig_x = max_sigx/ (2 ** (n_components-i-1)) 
        if sig_x <= .5: sig_x = .5
        sig_y = max_sigy/(2 ** (n_components-i-1)) 
        if sig_y <= .5: sig_y = .5
        
        angle = (i * np.pi / n_components) 
        
       
        params.extend([
            x_center, y_center, scale, sig_x, sig_y, angle
        ])
    
   
    initial_params = tuple(params)
    
    return initial_params
    
def pixel_to_radec(wcs, x, y):
    sky = wcs.pixel_to_world(x, y)
    ra = sky.ra.value
    dec = sky.dec.value
    return ra, dec


def gal_list_fit(gal_list, M, n_components, width, height, wave_step):
    gal_results = []
    
    for g in gal_list:
        print('Fitting the next galaxy...')
        stamp, wcs = M.get_image(g.params['ra'], g.params['dec'], filter_name='NIR_H', return_wcs=True, width=width, height=height)
        segmap = M.get_segmentation_map(g.params['ra'], g.params['dec'], width=width, height=height)

        midx, midy = stamp.shape[0]//2 , stamp.shape[1]//2 

        for i,x in enumerate(np.unique(segmap)):
            sel = np.where(segmap==x)
            sel1 = np.where(segmap != x)
            segmap[sel] = i+1
            
        value = segmap[midx][midy]
        sel = segmap == value
        sel1 = segmap != value

        segmap[sel] = 1
        segmap[sel1] = 0
        stamp[sel1] = 0

        final_image, gal = gal_shape_fit(stamp, segmap, wcs, n_components= 8, width= width, height=height, wave_step=wave_step, ra_obj = g.params['ra'], dec_obj = g.params['dec'])

        gal_results.append(gal)
    return gal_results



def gal_shape_fit(image, segmap, wcs, n_components, width, height, wave_step, ra_obj , dec_obj ):
    
    bound_x, bound_y, max_sigx, max_sigy = elliptical_fit(segmap)

    sel = segmap != 0
    max_value = np.max(image[sel])

    max_index_2d = np.argwhere(image == max_value)[0] 
    x, y = max_index_2d
    
    norm = image.max()
    image/=norm

    initial_params = generate_params(x, y, 1, max_sigx, max_sigy, 8)
    print(initial_params)
    
    final_image, final_fit = iterative_fit(image, initial_params, bound_x, bound_y, x,y, max_sigx, max_sigy)
    
    final_image*=norm
    image*=norm

    xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    gauss_list =[]
    weight_list = []
    max_list = []
    
    for i in range(n_components):
        if i * 6 < len(final_fit):
            #Extract parameters for the Gaussian
            gauss = gaussian(final_fit[i*6+2], final_fit[i*6], final_fit[i*6+1], final_fit[i*6+3], final_fit[i*6+4], np.pi/2 - final_fit[i*6+5], xx, yy)
            
            if (gauss.max() - final_fit[i*6+2]) >= gauss.max()/10 :
                weight_list.append(gauss.max())
            else:
                weight_list.append(final_fit[i*6+2])
                
            gauss*=norm
            gauss_list.append(gauss)
            max_list.append(gauss.max())
            
        else:
            
            break
    
    weight_list /= sum(weight_list)

    fit_list = []
    for i in range(0, len(final_fit), 6):
        x, y, amp, sig_x, sig_y, theta = final_fit[i:i+6]
        fit_list.append(dict(ra= None, dec = None, xmean=x, ymean=y, amp = amp, sigx=sig_x, sigy=sig_y, pa=theta))
    
    
    par_list=[]
    for i, fit in enumerate(fit_list):
        #print(fit)
        ra, dec = pixel_to_radec(wcs, fit['xmean'],fit['ymean'])
        par_list.append(dict( xmean = fit['xmean'], ymean = fit['ymean'], ra = ra, dec = dec, amp = fit['amp'], sigma = np.sqrt(fit['sigy']*fit['sigx']), axis_ratio = fit['sigy']/fit['sigx'], pa = fit['pa'], weight = weight_list[i]))
    
    j = 0
    galcomp_list = []
    for par in par_list:
        g = galaxy.GalaxyComponent(
            ra=par['ra'],
            dec=par['dec'],
            xmean = par['xmean'],
            ymean = par['ymean'],
            pa =par['pa']*180/np.pi,
            fwhm = par['sigma']*2*np.sqrt(2*np.log(2)),
            fwhm_arcsec=par['sigma']*2*np.sqrt(2*np.log(2))/10,
            axis_ratio = par['axis_ratio'],
            profile='gaussian',
            weight=par['weight'],
            # profile='gaussian',
            id=j,
            wcs = wcs
        )
        galcomp_list.append(g)
        j += 1
    
    Target_Galaxy = galaxy.Galaxy_v2( 
        components = galcomp_list, 
        ra=ra_obj,
        dec=dec_obj,
        obs_wavelength_step = wave_step,
        component_number = len(galcomp_list)
    )
    return final_image, Target_Galaxy










