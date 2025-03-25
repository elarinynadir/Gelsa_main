importlib.reload(es)

vmin = -10
vmax = 200

for im_i, S in enumerate(frame_list):
    model_ = S.make_image(gal_fits, noise=False, return_var=False)
    model = S.make_image(gal_fits, noise=False, return_var=False)

    image = image_list[im_i]
    var_im = var_list[im_i]

    prof_im, prof_mod, prof_resid = es.cross_profile(image, model, mask_list_out[im_i], gal_fits[0], S, background=0, h=25)

    for d in image.keys():
        print(f"detector {d}")
        if not d in pack[im_i]:
            continue

        (x_low, x_high, y_low, y_high) = pack[im_i][d]['bbox']
        mid = (y_low + y_high)//2
        y_low = mid-50
        y_high = mid+50
        
        mask = mask_list_out[im_i][d] > -1
        data = image[d] * mask
        error = np.sqrt(var_im[d])
        m = model_[d]
        
        # plt.figure(figsize=(12,20))
        fig, axes = plt.subplots(2, 4, figsize=(13, 20), height_ratios=(0.9, 0.1), layout='constrained')

        for gal in gal_fits:
            xx,yy,dd = S.optics.radec_to_pixel(
                    gal.params['ra']*np.ones(len(gal.wavelength)),
                    gal.params['dec']*np.ones(len(gal.wavelength)),
                    gal.wavelength,
                    objid=gal.params['id']
                )
            axes[0,0].plot(yy-y_low, xx-x_low, alpha=0.5, lw=5)
        axes[0,0].grid()
        axes[0,0].set_xlim(0,y_high-y_low)
        axes[0,0].set_ylim(0,x_high-x_low)

        d_ = np.ma.array(data, mask=1-mask)
        pltim0 = axes[0,1].imshow(d_[y_low:y_high,x_low:x_high].T, origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest', cmap='plasma')
        axes[0,1].grid()
        fig.colorbar(pltim0, ax=axes[0,1], shrink=0.8, location='top', pad=0.0, orientation='horizontal', label="Data")
        axes[0,1].yaxis.set_ticklabels([])

        # ax2 = plt.subplot(1,3,2)
        # axes[0,1].text(0.01,0.99,"Model", va='top', c='c', fontsize=24, transform=axes[0,1].transAxes)
        m_ = np.ma.array(m, mask=1-mask)
        pltim1 = axes[0,2].imshow(m_[y_low:y_high,x_low:x_high].T, origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest', cmap='plasma')
        axes[0,2].grid()
        fig.colorbar(pltim1, ax=axes[0,2],shrink=0.8, location='top', pad=0.0, orientation='horizontal', label="Model")
        axes[0,2].yaxis.set_ticklabels([])
        
        # ax3 = plt.subplot(1,3,3)
        # axes[0,2].text(0.01,0.99,"Residual", va='top', c='c', fontsize=24, transform=axes[0,2].transAxes)
        r = np.ma.array(data-m)
        r.mask = 1-mask
        pltim2 = axes[0,3].imshow(r[y_low:y_high,x_low:x_high].T, origin='lower', vmin=-100, vmax=100, interpolation='nearest', cmap='bwr')
        axes[0,3].grid()
        fig.colorbar(pltim2, ax=axes[0,3],shrink=0.8, location='top', pad=0.0, orientation='horizontal', label="Residual Data-Model", )
        axes[0,3].yaxis.set_ticklabels([])

        axes[1,1].plot(prof_im[d], c='firebrick', label="Data")
        axes[1,1].legend()

        axes[1,2].plot(prof_im[d], c='grey', dashes=[4,1], label="data")
        axes[1,2].plot(prof_mod[d], c='firebrick', label="Model")
        axes[1,2].yaxis.set_ticklabels([])
        axes[1,2].legend()
        axes[1,2].set_xlabel("Cross-dispersion pixel")

        axes[1,3].plot(prof_im[d], c='grey', dashes=[4,1], label="data")
        axes[1,3].plot(prof_resid[d], dashes=[4,1], c='firebrick', label="Residual")
        axes[1,3].yaxis.set_ticklabels([])
        axes[1,3].legend()
        
        # ax4 = plt.subplot(5,1,4)
        # plt.text(0,0,"Error", c='hotpink', fontsize=24, transform=ax4.transAxes)
        # plt.imshow(error[y_low:y_high,x_low:x_high], origin='lower', vmin=35, vmax=55)
        # plt.grid()
        # plt.colorbar(pad=0.0)
        
        # ax5 = plt.subplot(4,1,4)
        # plt.text(0,0,"Mask", c='k', fontsize=24, transform=ax5.transAxes)
        # plt.imshow(mask[y_low:y_high,x_low:x_high], origin='lower', vmin=0, vmax=1, cmap='Greys', interpolation='nearest')
        # plt.grid()
        # plt.colorbar(pad=0.0)
        # plt.subplots_adjust(wspace=0.01, hspace=0, left=0.05, right=0.99, top=0.99, bottom=0.05)
        plt.savefig(f"fits_{im_i}_{d}.png")
    break
        # 
        # plt.tight_layout()
