import cv2
import pytesseract
import pandas as pd
import numpy as np

class ImageMeta:
    def __init__(self, img):
        raws = pytesseract.image_to_boxes(img).split('\n')[:-1]
        matrix = [[int(el) for el in raw.split(' ')[1:-1]] for raw in raws]

        metainfo = (pd.DataFrame(np.array(matrix)).rename({
            0: 'x1',
            1: 'y1',
            2: 'x2',
            3: 'y2' 
            }, axis=1)
            .astype({
                    'x1': int,
                    'x2': int,
                    'y1': int,
                    'y2': int,
                    }))

        metainfo['x_centroid'] = metainfo['x1'] + (metainfo['x2'] - metainfo['x1'] /2)
        metainfo['y_centroid'] = metainfo['y1'] + (metainfo['y2'] - metainfo['y1'] /2)
        metainfo['zero'] = 0
        metainfo['char_shape'] = abs(metainfo['y2'] - metainfo['y1']) * abs(metainfo['x2'] - metainfo['x1'])
        metainfo['width'] = abs(metainfo['x1'] - metainfo['x2'])
        metainfo['height'] = abs(metainfo['y1'] - metainfo['y2'])

        self.table = metainfo
        
    def calculate(self, calc_type, **kwargs):
        if calc_type == 'character_avg_size':
            return (self.table.width.mean(), self.table.height.mean())


def black(img, threshold):
    img[img >= threshold] = 255
    img[img < threshold] = 0
    return img 

def inverse(img):
    return 255 - img

def paragraph_kernel_selection(char_size_tuple):
    mean = min(*char_size_tuple)

    floor  = np.floor(mean)
    odd_floor = floor if floor % 2 == 1 else floor - 1

    ceil  = np.ceil(mean)
    odd_ceil = ceil if ceil % 2 == 1 else ceil + 1

    _kernel_dim =  np.uint8(odd_floor) if abs(mean - odd_floor) < abs(mean - odd_ceil) else np.uint8(odd_ceil)
    _kernel_dim = _kernel_dim - 2 if _kernel_dim >= 9 else _kernel_dim

    return np.ones((_kernel_dim, _kernel_dim), np.uint8)