import numpy as np
import matplotlib
from matplotlib import path
matplotlib.use('agg')


def _spaced_points(low, high,n):
    """ We want n points between low and high, but we don't want them to touch either side"""
    padding = (high-low)/(n*2)
    return np.linspace(low + padding, high-padding, num=n)

def make_mask(mask_size, box, polygons_list):
    """
    Mask size: int about how big mask will be
    box: [x1, y1, x2, y2, conf.]
    polygons_list: List of polygons that go inside the box
    """
    mask = np.zeros((mask_size, mask_size), dtype=np.bool)
    
    xy = np.meshgrid(_spaced_points(box[0], box[2], n=mask_size),
                     _spaced_points(box[1], box[3], n=mask_size)) 
    xy_flat = np.stack(xy, 2).reshape((-1, 2))

    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((mask_size, mask_size))
    return mask.astype(np.float32)
#
#from matplotlib import pyplot as plt
#
#
#with open('XdtbL0dP0X0@44.json', 'r') as f:
#    metadata = json.load(f)
#from time import time
#s = time()
#for i in range(100):
#    mask = make_mask(14, metadata['boxes'][3], metadata['segms'][3])
#print("Elapsed {:3f}s".format(time()-s))
#plt.imshow(mask)