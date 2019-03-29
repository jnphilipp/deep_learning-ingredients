# -*- coding: utf-8 -*-

import math
import numpy as np
import os

from datetime import datetime
from decorators import runtime
from ingredients import models
from ingredients.datasets.h5py import save
from ingredients.datasets.images import load
from keras import backend as K
from keras.callbacks import BaseLogger, CallbackList, History, ProgbarLogger
from keras.preprocessing.image import array_to_img
from lxml import etree as ET

from . import ingredient
from .image import func


@ingredient.command
def boxes(images, threshold, rescale, output=None, output_class=None,
          data_format=None, _log=None, _run=None):
    """Genereate bounding boxes from image predictions."""
    if isinstance(images, str):
        images = [images]

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format:', data_format)

    model = models.load()
    if output is None:
        if len(model.outputs) == 1:
            if not output_class:
                if data_format == 'channels_first':
                    nb_classes = model.outputs[0].shape[1]
                elif data_format == 'channels_last':
                    nb_classes = model.outputs[0].shape[-1]

                if nb_classes != 1:
                    raise RuntimeError('The output has more than one class. ' +
                                       'Need to set "output_class".')
                else:
                    output_class = 0
        else:
            raise RuntimeError('No output specified and more than one output' +
                               ' found in model.')
    else:
        for o in model.outputs:
            if output in o.name:
                if not output_class:
                    if data_format == 'channels_first':
                        nb_classes = model.outputs[0].shape[1]
                    elif data_format == 'channels_last':
                        nb_classes = model.outputs[0].shape[-1]

                    if nb_classes != 1:
                        raise RuntimeError('The output has more than one ' +
                                           'class. Need to set ' +
                                           '"output_class".')

    base_dir = os.path.join(_run.observers[0].run_dir, 'boxes')
    os.makedirs(base_dir, exist_ok=True)

    page_dir = os.path.join(_run.observers[0].run_dir, 'boxes', 'page')
    os.makedirs(page_dir, exist_ok=True)

    pages = []
    for image in images:
        name, ext = os.path.splitext(os.path.basename(image))
        _log.info(f'Image: {name}')

        history, outputs = func(model, image)

        if output is None:
            if data_format == 'channels_first':
                p = outputs[0]['img'][output_class, :, :]
                r = outputs[0]['img'].shape[1]
                c = outputs[0]['img'].shape[2]
            elif data_format == 'channels_last':
                p = outputs[0]['img'][:, :, output_class]
                r = outputs[0]['img'].shape[1]
                c = outputs[0]['img'].shape[2]
        else:
            for i in range(len(outputs)):
                if outputs[i]['name'] == output:
                    if data_format == 'channels_first':
                        p = outputs[0]['img'][output_class, :, :]
                        r = outputs[0]['img'].shape[1]
                        c = outputs[0]['img'].shape[2]
                    elif data_format == 'channels_last':
                        p = outputs[0]['img'][:, :, output_class]
                        r = outputs[0]['img'].shape[1]
                        c = outputs[0]['img'].shape[2]
                    break

        if data_format == 'channels_first':
            o = np.repeat(p.reshape((1,) + p.shape), 3, axis=0)
        elif data_format == 'channels_last':
            o = np.repeat(p.reshape(p.shape + (1,)), 3, axis=2)
        o *= load(image, False, rescale)
        array_to_img(o * 255, scale=False).save(
            os.path.join(base_dir, os.path.basename(image)))
        boxes = generate_boxes(p)
        pages.append(page_xml(page_dir, os.path.basename(image), c, r, boxes))
    mets_xml(base_dir, pages)


@ingredient.capture
def generate_boxes(p, threshold, pixel_distance=9, _log=None):
    def surrounding_points(point):
        for i in range(pixel_distance * -1, pixel_distance + 1):
            for j in range(pixel_distance * -1, pixel_distance + 1):
                if i == 0 and j == 0:
                    continue
                yield (point[0] + i, point[1] + j)

    def same(p1, p2):
        return (p1[0] - pixel_distance > p2[1] + pixel_distance or
                p1[1] + pixel_distance < p2[0] - pixel_distance) or \
            (p1[2] - pixel_distance > p2[3] + pixel_distance or
             p1[3] + pixel_distance < p2[2] - pixel_distance)

    _log.info('Calculating bounding boxes...')

    boxes = []
    current_box = None
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i, j] >= threshold:
                if current_box is None:
                    boxes.append([(i, j)])
                    current_box = len(boxes) - 1
                else:
                    boxes[current_box].append((i, j))
            else:
                current_box = None

    # i = 0
    # while i < len(boxes):
    #     j = i + 1
    #     while j < len(boxes):
    #         for point in boxes[j]:
    #             merged = False
    #             for o in surrounding_points(point):
    #                 if o in boxes[i]:
    #                     boxes[i] += boxes[j]
    #                     del boxes[j]
    #                     merged = True
    #                     break
    #             if merged:
    #                 break
    #         j += 1
    #     i += 1

    for i in range(len(boxes)):
        min_x = None
        max_x = None
        min_y = None
        max_y = None
        for j in range(len(boxes[i])):
            if j == 0:
                min_x = boxes[i][j][0]
                max_x = boxes[i][j][0]
                min_y = boxes[i][j][1]
                max_y = boxes[i][j][1]
            else:
                min_x = min(min_x, boxes[i][j][0])
                max_x = max(max_x, boxes[i][j][0])
                min_y = min(min_y, boxes[i][j][1])
                max_y = max(max_y, boxes[i][j][1])
        boxes[i] = [min_x, max_x, min_y, max_y]

    i = 0
    while i < len(boxes):
        j = i + 1
        while j < len(boxes):
            if same(boxes[i], boxes[j][1]):
                j += 1
            else:
                boxes[i] = [min(boxes[i][0], boxes[j][0]),
                            max(boxes[i][1], boxes[j][1]),
                            min(boxes[i][2], boxes[j][2]),
                            max(boxes[i][3], boxes[j][3])]
                del boxes[j]
                j = i + 1
        boxes[i] = [(boxes[i][0], boxes[i][2]), (boxes[i][0], boxes[i][3]),
                    (boxes[i][1], boxes[i][3]), (boxes[i][1], boxes[i][2])]
        i += 1

    return boxes


@ingredient.capture
def page_xml(page_dir, name, width, height, boxes, _log, _run):
    _log.info('Generating PAGE XML...')

    page = {
        'name': name,
        'created': f'{datetime.utcnow().isoformat(timespec="seconds")}+00:00',
    }

    attr_qname = ET.QName('http://www.w3.org/2001/XMLSchema-instance',
                          'schemaLocation')
    attrib = {
        attr_qname: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/' +
                    '2013-07-15 http://schema.primaresearch.org/PAGE/gts/' +
                    'pagecontent/2013-07-15/pagecontent.xsd'
    }
    nsmap = {
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        None: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
    }
    pcgts = ET.Element('PcGts', attrib, nsmap)

    metadata = ET.Element('Metadata')
    pcgts.append(metadata)

    creator = ET.Element('Creator')
    creator.text = f'{_run.experiment_info['name']}[{_run._id}]'
    metadata.append(creator)

    created = ET.Element('Created')
    created.text = page['created']
    metadata.append(created)

    attrib = {'imageFilename': page['name'], 'imageWidth': str(width),
              'imageHeight': str(height)}
    epage = ET.Element('Page', attrib)
    pcgts.append(epage)

    for i in range(len(boxes)):
        text_region = ET.Element('TextRegion', {'id': f'r{i:d}'})
        epage.append(text_region)

        points = ' '.join([f'{p[1]:d},{p[0]:d}' for p in boxes[i]])
        coords = ET.Element('Coords', {'points': points})
        text_region.append(coords)

    name, ext = os.path.splitext(page['name'])
    page['xml'] = 'page/%s.xml' % name
    page['ext'] = ext
    with open(os.path.join(page_dir, f'{name}.xml'), 'bw') as f:
        f.write(ET.tostring(pcgts, pretty_print=True, xml_declaration=True,
                            encoding='UTF-8', standalone=True))

    return page


@ingredient.capture
def mets_xml(base_dir, pages, _log, _run):
    _log.info('Generating mets.xml ...')

    attr_qname = ET.QName('http://www.w3.org/2001/XMLSchema-instance',
                          'schemaLocation')
    attrib = {
        attr_qname: 'http://www.loc.gov/METS/ http://www.loc.gov/standards/' +
                    'mets/mets.xsd',
        'LABEL': f'{_run.experiment_info['name']}[{_run._id}]',
        'PROFILE': f'{_run.experiment_info['name']}[{_run._id}]'
    }
    nsmap = {
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'ns2': 'http://www.w3.org/1999/xlink',
        'ns3': 'http://www.loc.gov/METS/'
    }

    mets = ET.Element('{http://www.loc.gov/METS/}mets', attrib, nsmap)

    attrib = {
        'CREATEDATE': pages[0]['created'],
        'LASTMODDATE': pages[-1]['created']
    }
    metshdr = ET.Element('{http://www.loc.gov/METS/}metsHdr', attrib)
    mets.append(metshdr)

    agent = ET.Element('{http://www.loc.gov/METS/}agent', {'ROLE': 'CREATOR'})
    metshdr.append(agent)

    name = ET.Element('{http://www.loc.gov/METS/}name')
    name.text = f'{_run.experiment_info['name']}[{_run._id}]'
    agent.append(name)

    amdsec = ET.Element('{http://www.loc.gov/METS/}amdSec', {'ID': 'SOURCE'})
    mets.append(amdsec)

    sourcemd = ET.Element('{http://www.loc.gov/METS/}sourceMD',
                          {'ID': 'MD_ORIG'})
    amdsec.append(sourcemd)

    mdwrap = ET.Element('{http://www.loc.gov/METS/}mdWrap',
                        {'ID': 'TRP_DOC_MD', 'MDTYPE': 'OTHER'})
    sourcemd.append(mdwrap)

    xmldata = ET.Element('{http://www.loc.gov/METS/}xmlData')
    mdwrap.append(xmldata)

    trpdocmetadata = ET.Element('trpDocMetadata')
    xmldata.append(trpdocmetadata)

    title = ET.Element('title')
    title.text = f'{_run.experiment_info['name']}[{_run._id}]'
    trpdocmetadata.append(title)

    nrofpages = ET.Element('nrOfPages')
    nrofpages.text = str(len(pages))
    trpdocmetadata.append(nrofpages)

    status = ET.Element('status')
    status.text = "0"
    trpdocmetadata.append(status)

    filesec = ET.Element('{http://www.loc.gov/METS/}fileSec')
    mets.append(filesec)

    filegrp = ET.Element('{http://www.loc.gov/METS/}fileGrp', {'ID': 'MASTER'})
    filesec.append(filegrp)

    imggrp = ET.Element('{http://www.loc.gov/METS/}fileGrp', {'ID': 'IMG'})
    filegrp.append(imggrp)

    xmlgrp = ET.Element('{http://www.loc.gov/METS/}fileGrp', {'ID': 'PAGEXML'})
    filegrp.append(xmlgrp)

    for i, page in enumerate(pages):
        attrib = {
            'ID': f'IMG_{i:d}',
            'SEQ': str(i),
            'MIMETYPE': f'image/{page["ext"]}',
            'CREATED': page['created']
        }
        file = ET.Element('{http://www.loc.gov/METS/}file', attrib)
        imggrp.append(file)

        attrib = {
            'LOCTYPE': 'OTHER',
            'OTHERLOCTYPE': 'FILE',
            '{http://www.w3.org/1999/xlink}type': 'simple',
            '{http://www.w3.org/1999/xlink}href': page['name']
        }
        FLocat = ET.Element('{http://www.loc.gov/METS/}FLocat', attrib)
        file.append(FLocat)

        attrib = {
            'ID': f'PAGEXML_{i:d}',
            'SEQ': str(i),
            'MIMETYPE': 'application/xml',
            'CREATED': page['created'],
            'CHECKSUM': '',
            'CHECKSUMTYPE': 'MD5'
        }
        file = ET.Element('{http://www.loc.gov/METS/}file', attrib)
        xmlgrp.append(file)

        attrib = {
            'LOCTYPE': 'OTHER',
            'OTHERLOCTYPE': 'FILE',
            '{http://www.w3.org/1999/xlink}type': 'simple',
            '{http://www.w3.org/1999/xlink}href': page['xml']
        }
        FLocat = ET.Element('{http://www.loc.gov/METS/}FLocat', attrib)
        file.append(FLocat)

    attrib = {
        'ID': 'TRP_STRUCTMAP',
        'TYPE': 'MANUSCRIPT'
    }
    structmap = ET.Element('{http://www.loc.gov/METS/}structMap', attrib)
    mets.append(structmap)

    attrib = {
        'ID': 'TRP_DOC_DIV',
        'ADMID': 'MD_ORIG'
    }
    div = ET.Element('{http://www.loc.gov/METS/}div', attrib)
    structmap.append(div)

    for i, page in enumerate(pages):
        attrib = {
            'ID': f'PAGE_{i:d}',
            'ORDER': str(i),
            'TYPE': 'SINGLE_PAGE'
        }
        pdiv = ET.Element('{http://www.loc.gov/METS/}div', attrib)
        div.append(pdiv)

        ifptr = ET.Element('{http://www.loc.gov/METS/}fptr')
        pdiv.append(ifptr)

        attrib = {'FILEID': f'IMG_{i:d}'}
        iarea = ET.Element('{http://www.loc.gov/METS/}area', attrib)
        ifptr.append(iarea)

        xmlfptr = ET.Element('{http://www.loc.gov/METS/}fptr')
        pdiv.append(xmlfptr)

        attrib = {'FILEID': f'PAGEXML_{i:d}'}
        xmlarea = ET.Element('{http://www.loc.gov/METS/}area', attrib)
        xmlfptr.append(xmlarea)

    with open(os.path.join(base_dir, 'metadata.xml'), 'bw') as f:
        f.write(ET.tostring(trpdocmetadata, pretty_print=True, standalone=True,
                            xml_declaration=True, encoding='UTF-8'))
    with open(os.path.join(base_dir, 'mets.xml'), 'bw') as f:
        f.write(ET.tostring(mets, pretty_print=True, standalone=True,
                            xml_declaration=True, encoding='UTF-8'))
