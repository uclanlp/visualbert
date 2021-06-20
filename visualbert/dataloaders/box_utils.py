import os
import random
import numpy as np
import scipy
import warnings
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional
USE_IMAGENET_PRETRAINED = True


##### Image
def load_image(img_fn):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    return default_loader(img_fn)
    # # Load image
    # image = skimage.io.imread(img_fn)
    # # If grayscale. Convert to RGB for consistency.
    # if image.ndim != 3:
    #     image = skimage.color.gray2rgb(image)
    # # If has an alpha channel, remove it for consistency
    # if image.shape[-1] == 4:
    #     image = image[..., :3]
    # return image


# Let's do 16x9
# Two common resolutions: 16x9 and 16/6 -> go to 16x8 as that's simple
# let's say width is 576. for neural motifs it was 576*576 pixels so 331776. here we have 2x*x = 331776-> 408 base
# so the best thing that's divisible by 4 is 384. that's
def resize_image(image, desired_width=768, desired_height=384, random_pad=False):
    """Resizes an image keeping the aspect ratio mostly unchanged.

    Returns:
    image: the resized image
    window: (x1, y1, x2, y2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [left, top, right, bottom]
    """
    # Default window (x1, y1, x2, y2) and default scale == 1.
    w, h = image.size

    width_scale = desired_width / w
    height_scale = desired_height / h
    scale = min(width_scale, height_scale)

    # Resize image using bilinear interpolation
    if scale != 1:
        image = functional.resize(image, (round(h * scale), round(w * scale)))
    w, h = image.size
    y_pad = desired_height - h
    x_pad = desired_width - w
    top_pad = random.randint(0, y_pad) if random_pad else y_pad // 2
    left_pad = random.randint(0, x_pad) if random_pad else x_pad // 2

    padding = (left_pad, top_pad, x_pad - left_pad, y_pad - top_pad)
    assert all([x >= 0 for x in padding])
    image = functional.pad(image, padding)
    window = [left_pad, top_pad, w + left_pad, h + top_pad]

    return image, window, scale, padding


if USE_IMAGENET_PRETRAINED:
    def to_tensor_and_normalize(image):
        return functional.normalize(functional.to_tensor(image), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
else:
    # For COCO pretrained
    def to_tensor_and_normalize(image):
        tensor255 = functional.to_tensor(image) * 255
        return functional.normalize(tensor255, mean=(102.9801, 115.9465, 122.7717), std=(1, 1, 1))
