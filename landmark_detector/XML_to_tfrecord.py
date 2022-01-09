import xmltodict
import json
from tqdm import tqdm
import re
from glob import glob


def get_class(file_path):
    class_name = file_path.split('/')[1]
    class_name = re.sub('[\d\.]', '', class_name)
    return class_name


xml_path = './data/all_dogs_labeled.xml'
f = open(xml_path).read()
merge_xml = xmltodict.parse(f)

merge_xml = json.dumps(merge_xml)
merge_xml = json.loads(merge_xml)


images = merge_xml['dataset']['images']['image']

# '@file', '@width', '@height', 'box'
# '@top', '@left', '@width', '@height', 'label', 'part'

for image in tqdm(images):
    path = image['@file']

    image_class = get_class(path)       # dog species
    image_file = open('./data/' + path).read()     # image
