import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

from .. import galaxy


def reduce_gal_list(cat_info):
    gal_list = []
    j = 0
    for i in range(len(cat_info["cat"])):
        dra = cat_info["cat"]['RIGHT_ASCENSION'][i] - cat_info["ra_obj"]
        ddec =cat_info["cat"]['DECLINATION'][i] -  cat_info["dec_obj"]
        if (np.abs(dra)>1/60):
            continue
        if (np.abs(ddec)>3/60):
            continue

        if (np.abs(dra)<.0001) & (np.abs(ddec)<.0001):
            target_index = j
        g = galaxy.Galaxy(
            ra=cat_info["cat"]['RIGHT_ASCENSION'][i],
            dec=cat_info["cat"]['DECLINATION'][i],
            fwhm_arcsec=cat_info["cat"]['SEMIMAJOR_AXIS'][i]*0.1,
            bulge_r50=cat_info["cat"]['SEMIMAJOR_AXIS'][i]*0.1/2,
            disk_r50=cat_info["cat"]['SEMIMAJOR_AXIS'][i]*0.1/2,
            bulge_fraction=0.1,
            profile='bulgydisk',
            # profile='gaussian',
            obs_wavelength_step=50,
            id=j
        )
        gal_list.append(g)
        j += 1
    return target_index, gal_list

def new_image_widget(array= np.random.normal(0, 1, (51, 600)), width=51, height=51, zoom=4, label=None, colormap="plasma"):
    """ """
    f = go.FigureWidget()
    im = go.Heatmap(z=array,
                    showscale=False,
                   colorscale=colormap,
                   colorbar=dict(
                       x=0.85,
                       y=0.25,
                       xpad=0,
                       thickness=10,
                       len=0.5,
                       xref="container",
                       bgcolor='white'
                    )
    )
    f.add_trace(im)
    f.update_layout(height=height*zoom, width=width*zoom,
                    margin=dict(l=0.01, r=0.01, t=0.01, b=0.01, pad=0.01))
    f.update_yaxes(showticklabels=False)
    f.update_xaxes(showticklabels=False)

    if label:
        f.add_annotation(
            x=0,
            y=1,
            text=label,
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(
                family="Arial",
                size=16,
                color="#000000"
            ),
            bgcolor="#ffffff",
            opacity=0.6,
        )

    return widgets.HBox([f], layout={'border':'solid 1px black', 'width': f'{width*zoom}px', 'height': f'{height*zoom}px'})



def new_spec_widget(array = np.random.normal(0, 1, (51, 600)),  mask=None, map=None, width=600, height=51, zoom=1, label=None, colormap="plasma", lmargin=0.1, rmargin=0.1):
    """ """
    masked_array = array
    if mask is not None:
        bad_pixels = mask > 0
        masked_array = array.copy()
        masked_array[bad_pixels] = 0

    xx, yy = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    wavelengths = xx
    # if map is not None:
        # wavelengths = map.pixel_to_wavelength(xx, yy).astype(int)

    f = go.FigureWidget()
    im = go.Heatmap(z= masked_array,
                    showscale=False,
                    zmin=-20, zmax=300,
                   colorscale=colormap,
                   colorbar=dict(
                       x=0.85,
                       y=0.25,
                       xpad=0,
                       thickness=10,
                       len=0.5,
                       xref="container",
                       bgcolor='white'
                    ),
                    customdata=wavelengths,
                    hovertemplate="(%{x},%{y})<br>wavelength: %{customdata} <br>flux:%{z}",
    )
    f.add_trace(im)
    f.update_layout(height=height*zoom, width=width*zoom,
                    margin=dict(l=lmargin, r=rmargin, t=0.01, b=0.01, pad=0.01))
    f.update_yaxes(showticklabels=True)
    f.update_xaxes(showticklabels=True)

    if label:
        f.add_annotation(
            x=0,
            y=1,
            text=label,
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(
                family="Arial",
                size=16,
                color="#000000"
            ),
            bgcolor="#ffffff",
            opacity=0.6,
        )

    return f




def new_color_image(array, width=51, height=51, zoom=4, label=None, colormap="plasma"):
    """ """
    # Create a Plotly FigureWidget
    fig = px.imshow(array, origin = 'lower', color_continuous_scale=colormap)

    # Update layout for Plotly figure
    fig.update_layout(
        height=height*zoom,
        width=width*zoom,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title=None,
        yaxis_title=None
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    if label:
        fig.add_annotation(
            x=0,
            y=1,
            text=label,
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(
                family="Arial",
                size=16,
                color="#000000"
            ),
            bgcolor="#ffffff",
            opacity=0.6,
        )

    # Wrap the Plotly FigureWidget in an Output widget
    output = widgets.Output()
    with output:
        fig.show()

    # Return the HBox with Output widget
    return widgets.HBox([output], layout={'border':'solid 1px black', 'width': f'{width*zoom}px', 'height': f'{height*zoom}px'})



def truncate(x,y, start,end):
    xy = np.column_stack((x, y))
    xy = xy[xy[:, 0].argsort()] #sortying respect to the x
    x = xy[:, 0]
    y = xy[:, 1]
    start_idx = next((i for i, x_val in enumerate(x) if x_val >= start), 0)
    end_idx = next((i-1 for i, x_val in enumerate(x) if x_val >= end), len(x))

    x = x[start_idx:end_idx]
    y = y[start_idx:end_idx]
    return x,y



def plot_box(fig, map, start , end, h=10, wave_min=12000, wave_max=19000, npoints=100, c='cyan', **plot_params):
    """
    """

    x, y = map.wavelength_to_pixel(np.linspace(wave_min, wave_max, npoints))
    x,y = truncate(x,y,start,end)

    if len(x)!=0 :
        xx = np.concatenate([x, x[::-1], [x[0]]])
        yy = np.concatenate([y-h, y[::-1]+h, [y[0]-h]])

        fig.add_trace(go.Scatter(
        x=xx,
        y=yy,
        mode='lines',
        line=dict(color=c),
        hoverinfo='skip',
        **plot_params
        ))
        return fig
    else:
        return fig


def label_wavelength(fig, map, wavelength, label, start, end, h=20, c='cyan', **plot_params):
    """
    Add a label and lines indicating the wavelength
    """
    linx, liny = map.wavelength_to_pixel(wavelength)
    linx,liny = truncate(linx,liny,start,end)

    if len(linx)!=0:
        linx = linx[0]
        liny = liny[0]
        fig.add_annotation(
            x=linx,
            y=liny - 3*h,
            text=label,
            showarrow=False,
            font=dict(color=c),
            textangle=270,
            **plot_params
        )

        fig.add_trace(
            go.Scatter(
                x=[linx, linx],
                y=[liny - h, liny - h / 2],
                mode='lines',
                line=dict(color=c),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[linx, linx],
                y=[liny + h, liny + h / 2],
                mode='lines',
                line=dict(color=c),
                showlegend=False
            )
        )
        return fig
    else:
        return fig
