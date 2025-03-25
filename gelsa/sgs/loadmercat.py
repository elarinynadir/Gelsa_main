import glob
from astropy import table
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import numpy as np
import os

columns = ("OBJECT_ID","RIGHT_ASCENSION", "DECLINATION",'POSITION_ANGLE', 'ELLIPTICITY', 'SEMIMAJOR_AXIS', 'FLUX_VIS_PSF', 'FLUX_Y_TEMPLFIT','FLUX_J_TEMPLFIT','FLUX_H_TEMPLFIT', 'SPURIOUS_FLAG')

def load_data(filename):
    path = os.path.join("tmp", "data", filename)
    cat = table.Table.read(path)

    cat = cat[columns]

    valid = np.isfinite(cat['FLUX_H_TEMPLFIT']) & (cat['FLUX_H_TEMPLFIT']>0) #& \
            # (cat['SPURIOUS_FLAG']==0) 
    cat = cat[valid]
    hmag = -2.5*np.log10(cat['FLUX_H_TEMPLFIT'].value*1e-6) + 8.9
    valid = np.isfinite(hmag)&(hmag>15)&(hmag<22)
    cat = cat[valid]
    # hmag = -2.5*np.log10(cat['FLUX_H_TEMPLFIT'].value*1e-6) + 8.9
    # print(np.percentile(hmag, [10,50, 90]))
    return cat

def find_obj_id (id , cat):
    #x1 & x2 are the bounds used to reduce the search sample
    obj_id = id
    index, ra_obj, dec_obj = None, None, None
    index = int(np.where(cat["OBJECT_ID"] == obj_id)[0])
    dec_obj = cat["DECLINATION"][index]
    ra_obj = cat["RIGHT_ASCENSION"][index] 
    
    valid = np.isfinite(cat['FLUX_H_TEMPLFIT']) & (cat['FLUX_H_TEMPLFIT']>0) #& \
            # (cat['SPURIOUS_FLAG']==0) 
    cat = cat[valid]
    hmag = -2.5*np.log10(cat['FLUX_H_TEMPLFIT'].value*1e-6) + 8.9
    valid = np.isfinite(hmag)&(hmag>15)&(hmag<22)
    cat = cat[valid]
    
    if cat is not None:
        print("Object found.")
        cat_info = dict(obj_id=obj_id, cat=cat, index=index, ra_obj=ra_obj, dec_obj=dec_obj)
        return cat_info
    else:
        print("Object not found.")
        return None

def find_obj(ra, dec , cat):
    #x1 & x2 are the bounds used to reduce the search sample
    ra_obj, dec_obj = ra, dec
    target_index = None
    j = 0
    for i in range(len(cat)):
        dra = cat['RIGHT_ASCENSION'][i] - ra_obj
        ddec = cat['DECLINATION'][i] - dec_obj
        if (np.abs(dra)>1/60): 
            continue
        if (np.abs(ddec)>3/60): 
            continue
        if (np.abs(dra)<.0001) & (np.abs(ddec)<.0001):
            target_index = j
            break 
        j+=1
            
    valid = np.isfinite(cat['FLUX_H_TEMPLFIT']) & (cat['FLUX_H_TEMPLFIT']>0) #& \
            # (cat['SPURIOUS_FLAG']==0) 
    cat = cat[valid]
    hmag = -2.5*np.log10(cat['FLUX_H_TEMPLFIT'].value*1e-6) + 8.9
    valid = np.isfinite(hmag)&(hmag>15)&(hmag<22)
    cat = cat[valid]
    
    if target_index is not None:
        print("Object found.")
        cat_info = dict(obj_id=cat["OBJECT_ID"][target_index], cat=cat, index=target_index, ra_obj=ra_obj, dec_obj=dec_obj)
        return cat_info
    else:
        print("Object not found.")
        cat_info = dict(obj_id=None, cat=cat, index=None, ra_obj=ra_obj, dec_obj=dec_obj)
        return cat_info


