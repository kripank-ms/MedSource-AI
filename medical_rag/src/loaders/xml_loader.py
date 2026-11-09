import xml.etree.ElementTree as ET

def load_xml(file_path):
    tree = ET.parse(file_path)      
    root = tree.getroot()

    return root