import xml.etree.ElementTree as ET


class DataProduct:

    def __init__(self, path):
        """ """
        self.root = ET.parse(path).getroot()

    @property
    def tile_index(self):
        """Read the tile index from the data product"""
        node = self.root.find('Data/TileIndex')
        if node is None:
            raise KeyError(f"Data/TileIndex does not exist")
        tile_index = int(node.text)
        return tile_index

    @property
    def filter_name(self):
        """Read the tile index from the data product"""
        node = self.root.find('Data/Filter/Name')
        if node is None:
            raise KeyError(f"Data/Filter/Name does not exist")
        filter_name = node.text
        return filter_name

    def get_element(self, key):
        """ """
        return self.root.find(key).text
