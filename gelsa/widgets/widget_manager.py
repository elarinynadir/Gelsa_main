import numpy as np
from astropy.visualization import make_lupton_rgb
import ipywidgets as widgets

from ..sgs import loadmercat
from . import run_image as run


class ImageWidget:
    def __init__(self):
        self.create_widgets()

    def create_widgets(self):
        self.vis_image = run.new_image_widget(label='VIS')
        self.nispy_image = run.new_image_widget(label='NIR Y')
        self.nispj_image = run.new_image_widget(label='NIR J')
        self.nisph_image = run.new_image_widget(label='NIR H')
        self.color_image = run.new_color_image(np.random.uniform(0, 255, (51, 51, 3)), label='VIS,J,H')
        self.images_row = widgets.HBox([ self.vis_image, self.nispy_image, self.nispj_image, self.nisph_image, self.color_image],
                                       layout=widgets.Layout(width='100%', border='1px dashed black'))


class SpecWidget:
    def __init__(self):
        self.create_widgets()

    def create_widgets(self):
        self.spec_image = run.new_spec_widget(width=600, height=101, zoom=1)
        self.spec_row = widgets.VBox([self.spec_image],
                                     layout=widgets.Layout(width='100%', border='1px dashed red'))


class TabManager:
    def __init__(self, tab_labels=[]):

        self.tab = widgets.Tab()
        self.children = []
        self.tab.children = self.children
        self.tab_labels = tab_labels

    def set_tabs(self, children):

        self.children = children
        self.tab.children = self.children
        for i, label in enumerate(self.tab_labels):
            self.tab.set_title(i, label)

    def get_tab_widget(self):
        return self.tab

class Load_Data:
    def __init__(self, frame_list, M):

        self.M = M
        self.frame_list = frame_list

        self.tab_labels = ['Data', 'Mask', 'Fit']
        self.tab_manager = TabManager(tab_labels=self.tab_labels)

        self.image_widget = ImageWidget()
        self.spec_widget = SpecWidget()

        self.load_button = widgets.Button(description="Load")
        self.load2_button = widgets.Button(description="Load")
        self.clear_button = widgets.Button(description="Clear")
        self.objid_box = widgets.Combobox(
            description='Object ID:',
            placeholder='Object ID',
            options=('1501145500022848246', '1503320507021042193', '1501948067020536519', '1502548751023699819', '1502190945020590005'),
            layout={'width': '250px'}
        )
        self.ra_box = widgets.Combobox(
            description='RA:',
            value='150.11455921612844',
            layout={'width': '250px'}
        )
        self.dec_box = widgets.Combobox(
            description='DEC:',
            value='2.284824605042879',
            layout={'width': '250px'}
        )

        self.title_row = widgets.HBox([widgets.Label(value="ELSA visualize")], layout=widgets.Layout(border='1px dashed blue'))
        self.infobox = widgets.Textarea(value='0', disabled=True, layout={'width': '100%', 'height': '200px'})
        self.info_row = widgets.HBox([self.infobox])
        self.images_row = self.image_widget.images_row
        self.descr_box = widgets.VBox([self.objid_box,self.ra_box, self.dec_box, self.load_button, self.clear_button, self.info_row])
        self.main_row = widgets.HBox([self.descr_box, self.images_row ])

        self.tab_manager.set_tabs([self.spec_widget.spec_row, self.spec_widget.spec_row, self.spec_widget.spec_row])

        self.load_button.on_click(self.load_clicked)
        self.clear_button.on_click(self.clear_clicked)

        display(widgets.VBox([self.title_row, self.main_row, self.tab_manager.tab]))


    def load_clicked(self, b=None):
        self.clear_clicked()
        try:
            bool, obj_id, ra, dec = validate_input(self.objid_box, self.ra_box, self.dec_box)
        except ValueError as e:
            print(f"Validation Error: {e}")
            return

        if bool:  # Input is via Object ID
            ra, dec = get_ra_dec_from_obj_id(obj_id)
            cat_info = search_catalog(ra, dec,  self.M, obj_id=obj_id)
        else:  # Input is via RA-DEC
            cat_info = search_catalog(ra, dec, self.M)

        print("Searching the Object")
        self.infobox.value = f'ra = { cat_info["ra_obj"]} \ndec = {cat_info["dec_obj"]}'
        image_list, mask_list, map_list = process_images_and_spectra(self.frame_list, cat_info)
        spec_box = display_spectra(image_list, mask_list, map_list)

        self.tab_manager.set_tabs([widgets.VBox(spec_box), self.spec_widget.spec_row, self.spec_widget.spec_row])

        stamp_vis, stamp_Y, stamp_J, stamp_H, rgb = display_images(ra, dec, self.M)
        self.image_widget.vis_image = run.new_image_widget(stamp_vis, width=101, height=101, label='VIS', zoom=2)
        self.image_widget.nispy_image = run.new_image_widget(stamp_Y, width=101, height=101, label='NIR Y', zoom=2)
        self.image_widget.nispj_image = run.new_image_widget(stamp_J, width=101, height=101, label='NIR J', zoom=2)
        self.image_widget.nisph_image = run.new_image_widget(stamp_H, width=101, height=101, label='NIR H', zoom=2)
        self.image_widget.color_image = run.new_color_image(rgb, width=101, height=101,label='VIS,J,H', zoom=2)

        self.image_widget.images_row.children = [
            self.image_widget.vis_image,
            self.image_widget.nispy_image,
            self.image_widget.nispj_image,
            self.image_widget.nisph_image,
            self.image_widget.color_image
        ]

        print("Done")



    def clear_clicked(self, b=None):
        self.image_widget.vis_image = run.new_image_widget(label='VIS')
        self.image_widget.nispy_image = run.new_image_widget(label='NIR Y')
        self.image_widget.nispj_image = run.new_image_widget(label='NIR J')
        self.image_widget.nisph_image = run.new_image_widget(label='NIR H')
        self.image_widget.color_image = run.new_color_image(np.random.uniform(0, 255, (51, 51, 3)), label='VIS,J,H')
        self.image_widget.images_row.children = [
            self.image_widget.vis_image,
            self.image_widget.nispy_image,
            self.image_widget.nispj_image,
            self.image_widget.nisph_image,
            self.image_widget.color_image
        ]
        self.infobox.value = "0"

        self.spec_widget.create_widgets()
        self.tab_manager.set_tabs([self.spec_widget.spec_row, self.spec_widget.spec_row, self.spec_widget.spec_row])
        self.main_row = widgets.HBox([self.descr_box,self.images_row ])



def validate_input(objid_box, ra_box, dec_box):
    """
    Returns:
        (bool, obj_id, ra, dec):
        - bool: True if input is via Object ID, False if via RA-DEC.
    Raises:
    - ValueError: If input is invalid.
    """

    if objid_box.value != "" and (ra_box.value != "" or dec_box.value != ""):
        raise ValueError("Input only the Object Id or the Ra-Dec coordinates.")

    if objid_box.value != "":
        try:
            obj_id = int(objid_box.value)
            return True, obj_id, None, None  # True indicates input is via Object ID
        except ValueError:
            raise ValueError("Invalid Object ID. Please enter a valid number.")

    elif ra_box.value != "" or dec_box.value != "":
        try:
            ra = float(ra_box.value)
            dec = float(dec_box.value)
            if ra <= 0 or dec <= 0:
                raise ValueError
            return False, None, ra, dec  # False indicates input is via RA and DEC
        except ValueError:
            raise ValueError("Invalid RA or DEC value. Please enter a valid number.")

    raise ValueError("No valid input provided.")


def get_ra_dec_from_obj_id(obj_id):
    id_str = str(obj_id)
    ra_int = int(id_str[:11])
    dec_int = int(id_str[-8:])
    ra = ra_int / 10**8
    dec = dec_int / 10**7
    return ra, dec

def search_catalog(ra, dec,  M, obj_id=None):
    cat = M.get_tile_catalog(ra, dec)
    if obj_id:
        return loadmercat.find_obj_id(obj_id, cat)
    else:
        return loadmercat.find_obj(ra, dec, cat)


def process_images_and_spectra(frame_list, cat_info):
    pack = []
    z_obj = 1
    for frame in frame_list:
        pack.append(frame.cutout(
            cat_info["ra_obj"], cat_info["dec_obj"], z_obj
        ))

    image_list, mask_list, map_list = [], [], []
    for p in pack:
        image_pack, mask_pack, map_pack = [], [], []
        for key in p.keys():
            crop = p[key]["crop"]
            image_pack.append(crop.image)
            mask_pack.append(crop.mask)
            map_pack.append(crop)
        image_list.append(image_pack)
        mask_list.append(mask_pack)
        map_list.append(map_pack)

    return image_list, mask_list, map_list

def display_spectra(image_list, mask_list, map_list):
    spec_box = []
    for i in range(len(image_list)):
        spec_images = []
        for j in range(len(image_list[i])):
            num_rows = len(image_list[i][j])
            num_columns = len(image_list[i][j][0])
            im = run.new_spec_widget(image_list[i][j], mask_list[i][j], map_list[i][j], 2*num_columns, 2*num_rows, zoom = 1, lmargin =5,rmargin=5 )
            # im = run.plot_box(im, map_list[i][j], 0, num_columns)
            box = widgets.HBox([im])
            spec_images.append(box)
        spec_box.append(widgets.HBox(spec_images, layout=widgets.Layout(margin='10px', padding='10px', width='100%', border='1px dashed black')))
    return spec_box

def display_images(ra, dec, M):
    stamp_vis = M.get_image(ra, dec, filter_name='VIS')
    stamp_Y = M.get_image(ra, dec, filter_name='NIR_Y')
    stamp_J = M.get_image(ra, dec, filter_name='NIR_J')
    stamp_H = M.get_image(ra, dec, filter_name='NIR_H')
    rgb = make_lupton_rgb(stamp_H, stamp_J, stamp_vis)

    return stamp_vis, stamp_Y, stamp_J, stamp_H, rgb

