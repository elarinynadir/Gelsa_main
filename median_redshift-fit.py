#!/usr/bin/env python3
import os
import sys
import glob
import copy
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from scipy.ndimage import median_filter 
from scipy.ndimage import rotate
#from tqdm.notebook import tqdm
from tqdm import tqdm


import argparse


parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import gelsa
from gelsa import galaxy, specframe, spec_imager, shape_fit, utils, visu, opt_redshift
from gelsa.sgs import mer
from gelsa.spec_crop import SpecCrop
from gelsa.spec_crop import NoExtraction

print(f"{gelsa.version.version=}")


#wrong estimation:0, 2,3, 4 6,8,9, 13, 16 18,19,21, 25, 26(negative), 29 , 31(negative), 33


def crop_trace(image, mask, var ,frame, detector, x, y, padx=100, pady=10):
        """ """
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


def process_galaxy(gal_index, cat, M, G, sir_pack):
    objid = cat['OBJECT_ID'][gal_index]
    ra = cat['RIGHT_ASCENSION_DEEP'][gal_index]
    dec = cat['DECLINATION_DEEP'][gal_index]
    redshift = cat['SPE_Z_DEEP'][gal_index]

    stamp_H, wcs = M.get_image(ra=ra, dec=dec, filter_name='NIR_H', return_wcs=True, width=100, height=100)
    segmap = M.get_segmentation_map(ra=ra, dec=dec, width=100, height=100)
    norm = stamp_H.max()

    # Process segmentation map
    midx, midy = stamp_H.shape[0] // 2, stamp_H.shape[1] // 2
    for i, x in enumerate(np.unique(segmap)):
        sel = np.where(segmap == x)
        sel1 = np.where(segmap != x)
        segmap[sel] = i + 1
        if segmap[midx, midy] == i + 1:
            segmap[sel1] = 0
            stamp_H[sel1] = 0

    # Load frames
    frame_list = []
    #frame_list_copy = []
    for entry in sir_pack:
        loc_path = os.path.join("tmp/data", entry["loctable_name"])
        path = os.path.join("tmp/data", entry["frame_name"])
        if not os.path.exists(path) or not os.path.exists(loc_path):
            frame_list.append(None)
            #frame_list_copy.append(None)

            continue
        f = G.load_frame_h5(path, loc_path)
        S = spec_imager.SpectralImager(f)
        frame_list.append(S)
        #frame_list_copy.append(S)

    #frame_list = [frame_list_copy[0], frame_list_copy[1], frame_list_copy[2], frame_list_copy[3], frame_list_copy[4], frame_list_copy[5],frame_list_copy[6],frame_list_copy[7], frame_list_copy[8], frame_list_copy[9],frame_list_copy[10],frame_list_copy[11]]

    # Shape fitting
    final_image, Target_Galaxy = shape_fit.gal_shape_fit(
        stamp_H, segmap, wcs, 8, stamp_H.shape[0], stamp_H.shape[1], 100, ra, dec
    )

    # Extract cutouts
    pack_original_list = []
    for frame in frame_list:
        if frame is None:
            pack_original_list.append(None)

        else:
            pack_original_list.append(frame.specframe.cutout(ra, dec, redshift))

    mask_list = []
    for frame in frame_list:
        if frame is None:
            mask_list.append(None)
        else:
            mask_list.append(frame.make_mask([Target_Galaxy]))
            print(mask_list[-1].keys())

    filter_size = 50
    image_list, pixmask_list, original_mask_list, var_list, tilt_list = [], [], [], [], []
    for i, S in enumerate(frame_list):
        if S is None:
            image_list.append(None)
            original_mask_list.append(None)
            pixmask_list.append(None)
            var_list.append(None)
            tilt_list.append(None)
        else:
            image, pixmask, original_mask, var = {}, {}, {}, {}
            tilt = S.specframe.params["tilt"]

            for d in mask_list[i]:
                im, pm, v = S.specframe.get_detector(d)
                image[d] = im.astype(np.float64)
                pixmask[d] = pm
                original_mask[d] = pm
                var[d] = v
                if tilt not in [0, 180]:
                    filtered_image = np.apply_along_axis(median_filter, axis=1, arr=image[d], size=filter_size)
                    image[d] -= filtered_image

                    valid = np.isfinite(image[d]) & np.isfinite(var[d]) & (pixmask[d] == 0)
                    image[d][~valid] = 0
                    pixmask[d] = np.zeros(image[d].shape, dtype=bool)
                    pixmask[d][valid] = 1
                else:
                    rotated_image = rotate(image[d], tilt, reshape=False, order=3, mode='constant')
                    if np.isnan(rotated_image).sum() / rotated_image.size > 0.5:
                        rotated_image = rotate(image[d], tilt, reshape=False, order=1, mode='constant')
                    filtered_image = np.apply_along_axis(median_filter, axis=1, arr=rotated_image, size=filter_size)
                    rotated_back_image = rotate(filtered_image.astype(np.float64), -tilt, reshape=False, order=3, mode='constant')
                    if np.isnan(rotated_back_image).sum() / rotated_back_image.size > 0.5:
                        rotated_back_image = rotate(filtered_image.astype(np.float64), -tilt, reshape=False, order=1, mode='constant')

                    image[d] -= rotated_back_image
                    valid = np.isfinite(image[d]) & np.isfinite(var[d]) & (pixmask[d] == 0)
                    image[d][~valid] = 0
                    pixmask[d] = np.zeros(image[d].shape, dtype=bool)
                    pixmask[d][valid] = 1
            image_list.append(image)
            pixmask_list.append(pixmask)
            original_mask_list.append(original_mask)
            var_list.append(var)
            tilt_list.append(tilt)


    pack_list = []

    for i, frame in enumerate(frame_list):
        if frame is None:
            pack_list.append(None)

        else:
            wavelength_range = frame.specframe.params['wavelength_range']
            wave_trace = utils.intrange(*wavelength_range, 20)
            n = len(wave_trace)

            detx, dety, detid = frame.specframe.radec_to_pixel(
                ra*np.ones(n),
                dec*np.ones(n),
                wave_trace
            )
            valid = detid >= 0
            if np.sum(valid) == 0:
                #raise ValueError(f"RA, Dec  not on detector {(ra, dec)}")
                print(f"RA, Dec  not on detector {(ra, dec)}")
                pack_list.append(None)
            detx = detx[valid]
            dety = dety[valid]
            detid = detid[valid]
            wave_trace = wave_trace[valid]

            pack_dict = {}
            for det in np.unique(detid):
                sel = detid == det
                detx_ = detx[sel]
                dety_ = dety[sel]
                crop = crop_trace(image_list[i][det], original_mask_list[i][det], var_list[i][det], frame.specframe, det, detx_, dety_)
                crop.center = (ra, dec)
                crop.wavelength_trace = wave_trace[sel]
                crop.redshift = redshift
                pack_dict[det] = crop
            pack_list.append(pack_dict)

        
    redshift_grid = np.arange(0.9, 1.8, 4.5e-3)

    wave_list, pz_list, new_pz_list, new_redshifts_list, measured_redshift_list = [], [], [], [], []

    n_frames = len(frame_list)
    n_groups = n_frames // 4

    for group_idx in tqdm(range(n_groups), desc="Inner", position=1):
        start_idx, end_idx = group_idx * 4, (group_idx * 4) + 4
        group_images = image_list[start_idx:end_idx]
        group_var_images = var_list[start_idx:end_idx]
        group_frame_list = frame_list[start_idx:end_idx]
        group_pixmask_list = pixmask_list[start_idx:end_idx]

        if all(frame is None for frame in group_frame_list):
            wave_list.append(None)
            pz_list.append(None)
            new_pz_list.append(None)
            new_redshifts_list.append(None)
            measured_redshift_list.append(None)
            continue

        wave, pz, new_pz, new_redshifts, z_est = opt_redshift.MeasureSource(
            images=group_images,
            var_images=group_var_images,
            frame_list=group_frame_list,
            test_galaxy=Target_Galaxy.copy(),
            redshift_grid=redshift_grid,
            pixmask_list=group_pixmask_list,
        )

        wave_list.append(wave)
        pz_list.append(pz)
        new_pz_list.append(new_pz)
        new_redshifts_list.append(new_redshifts)
        measured_redshift_list.append(z_est)

    results = []
    true_z = redshift
    for group_idx, z_measured in enumerate(measured_redshift_list):
        if z_measured is None:
            results.append({
                "Galaxy": gal_index,
                "Group": group_idx,
                "z_true": true_z,
                "z_measured": -1,
                "dz": -1
            })
            continue

        sigma = np.abs(z_measured - true_z) 
        results.append({
            "Galaxy": gal_index,
            "Group": group_idx,
            "z_true": round(true_z, 5),
            "z_measured": round(z_measured, 5),
            "dz": round(sigma, 5)
        })

    # Ensure the Results directory exists
    os.makedirs("Results", exist_ok=True)

    # Define the output PDF path
    output_pdf_path = os.path.join("Results", f"Results-{gal_index}.pdf")

    # Create a multipage PDF
    with PdfPages(output_pdf_path) as pdf:
        # ---------------------- Image Pair (Stamp & Segmap) ----------------------
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        im1 = axs[0].imshow(stamp_H, origin='lower')
        axs[0].set_title(f"Stamp - Galaxy {gal_index}")
        fig.colorbar(im1, ax=axs[0], orientation='vertical')

        im2 = axs[1].imshow(segmap, origin='lower')
        axs[1].set_title(f"Segmap - Galaxy {gal_index}")
        axs[1].axis('off')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------- Cutouts (Original) ----------------------
        for i, crop in enumerate(pack_original_list):
            if crop is None:
                continue
            for det in crop.keys():
                fig, ax = plt.subplots(figsize=(20, 4))
                _ = visu.show(crop[det], levels=(10, 95), infill=False)
                ax.set_xlim(12000, 19000)     

                ax.set_title(f"{objid}, {i//4}")
   
                pdf.savefig(fig)
                plt.close(fig)

        # ---------------------- Cutouts (Median Filtered) ----------------------
        for i, crop in enumerate(pack_list):
            if crop is None:
                continue
            for det in crop.keys():
                fig, ax = plt.subplots(figsize=(20, 4))
                _ = visu.show(crop[det], levels=(10, 95), infill=False)
                ax.set_xlim(12000, 19000)     

                ax.set_title(f"{objid}, {i//4}")
   
                pdf.savefig(fig)
                plt.close(fig)
         # ---------------------- Images (Zero-order Filter) ----------------------
        # for i, image_dict in enumerate(image_list):
        #     if image_dict is None:
        #         continue
        #     for d, image in image_dict.items():
        #         height, width = image.shape
        #         fig, ax = plt.subplots(figsize=(width / 100, height / 100))  # Dynamically set size
        #         ax.imshow(image, cmap='inferno', origin='lower') 
        #         ax.set_title(f"Image {i}, Detector {d}")
        #         ax.axis('off')

        #         pdf.savefig(fig)  # This was missing, adding it here
        #         plt.close(fig)
        # ---------------------- Redshift Groups Plots ----------------------
        for group_idx in range(len(measured_redshift_list)):  
            if measured_redshift_list[group_idx] is None:
                continue

            fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=200)

            # Full grid plot
            full_grid_pz = pz_list[group_idx]
            axs[0].plot(redshift_grid, full_grid_pz)
            axs[0].set_title(f'Galaxy {gal_index}, Group {group_idx + 1}: Full grid')
            axs[0].set_xlabel(r'Redshift $z$')
            axs[0].set_ylabel(r'Log-likelihood')

            # Refined grid plot
            refined_grid_pz = new_pz_list[group_idx]
            refined_redshifts = new_redshifts_list[group_idx]
            axs[1].plot(refined_redshifts, refined_grid_pz)
            axs[1].axvline(measured_redshift_list[group_idx], linestyle='--', color='red')
            axs[1].set_title(f'Galaxy {gal_index}, Group {group_idx + 1}: Refined grid')
            axs[1].set_xlabel(r'Redshift $z$')
            axs[1].set_ylabel(r'Log-likelihood')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        # ---------------------- Redshift Table  ----------------------
        df = pd.DataFrame(results)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [df.columns.tolist()] + df.values.tolist()
        table = ax.table(cellText=table_data, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)  # Scale table for better fit
        
        plt.title(f"Redshift Table - Galaxy {gal_index}")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Final merged results saved to {output_pdf_path}")



def main():
    parser = argparse.ArgumentParser(description="Run parallel galaxy processing")
    parser.add_argument("--sirpack", type=str, default="sir_pack.pickle", help="Path to sir_pack file")
    parser.add_argument("--gold_sample", type=str, default="gold_sample.fits", help="Path to Galaxy FITS file")
    parser.add_argument("--index", type=int, default=0, help="galaxy index")

    args = parser.parse_args()

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
    M = mer.MER(password_file='password')
    cat = Table.read("gold_sample.fits")
    index = args.index

    with open(args.sirpack, "rb") as input_file:
        sir_pack_paths = pickle.load(input_file)

    sir_pack = []
    for entry in sir_pack_paths:
        frame_name = os.path.basename(entry['frame_path'])
        loctable_name = os.path.basename(entry['loctable_path'])
        sir_pack.append({
            'frame_name': frame_name,
            'loctable_name': loctable_name
        })

    process_galaxy(index, cat, M, G, sir_pack)

if __name__ == '__main__':
    main()






