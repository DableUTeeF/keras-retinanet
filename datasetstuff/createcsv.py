from xml.etree import cElementTree as ET
import os

if __name__ == '__main__':
    path = '/media/palm/data/ppa/v6/anns/val/'
    ipath = '/media/palm/data/ppa/v6/images/val/'
    csv = []
    classid = []
    for file in os.listdir(path):
        tree = ET.parse(os.path.join(path, file))
        impath = ''
        bndbox = [0, 0, 0, 0]
        classname = ''
        for elem in tree.iter():
            if 'filename' in elem.tag:
                impath = ipath + elem.text
            elif elem.tag == 'name':
                classname = elem.text
                if classname == 'Boot':
                    classname = 'goodshoes'
                classid.append(classname)
            elif 'xmin' in elem.tag:
                bndbox[0] = int(elem.text)
            elif 'ymin' in elem.tag:
                bndbox[1] = int(elem.text)
            elif 'xmax' in elem.tag:
                bndbox[2] = int(elem.text)
            elif 'ymax' in elem.tag:
                bndbox[3] = int(elem.text)
                csv.append(f'{impath},{bndbox[0]},{bndbox[1]},{bndbox[2]},{bndbox[3]},{classname}')
    with open('v6/val_annotations', 'w') as wr:
        for element in csv:
            wr.write(element)
            wr.write('\n')
    classid = list(set(classid))
    x = 0
    with open('v6/classes', 'w') as wr:
        for i, e in enumerate(classid):
            wr.write(str(e))
            wr.write(str(','))
            wr.write(str(x))
            wr.write(str('\n'))
            x += 1
