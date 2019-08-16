# Modified by Harold. Courtesy of the author of VCR
"""
Detect the images from a dataframe, saving masks to a json.
"""

from collections import defaultdict
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
from detectron.utils.vis import convert_from_cls_format, kp_connections, get_class_string
import detectron.utils.keypoints as keypoint_utils
from tqdm import tqdm
from detectron.utils.colormap import colormap
import pycocotools.mask as mask_util
import numpy as np
import json
import pickle as pkl

# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
import detectron.utils.env as envu

envu.set_up_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

THRESHOLD = 0.7

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
logger = logging.getLogger('__main__')

dummy_coco_dataset = dummy_datasets.get_coco_dataset()


def get_model(use_keypoints=False):
    """
    Obtain model
    :param use_keypoints: whether to use keypoints or mask rcnn
    :return:
    """
    if use_keypoints:
        MODEL_CONFIG = '/home/rowan/tools/Detectron/configs/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml'
        MODEL_WEIGHTS = 'https://s3-us-west-2.amazonaws.com/detectron/37732318/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_55_09.Lx8H5JVu/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
    else:
        MODEL_CONFIG = '/local/harold/vqa/trained_detectron/e2e_mask_rcnn_X-101-64x4d-FPN_1x.yaml'
        MODEL_WEIGHTS = '/local/harold/vqa/trained_detectron/e2e_mask_rcnn_X-101-64x4d-FPN_1x.pkl'

    merge_cfg_from_file(MODEL_CONFIG)
    cfg.NUM_GPUS = 1
    cfg.MODEL.KEYPOINTS_ON = use_keypoints
    cfg.MODEL.MASK_ON = not use_keypoints
    weights_arg = cache_url(MODEL_WEIGHTS, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    assert not cfg.MODEL.RPN_ONLY, 'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, 'Models that require precomputed proposals are not supported'
    model = infer_engine.initialize_model_from_cfg(weights_arg)
    return model


def detect_from_img(model, im, dets_pkl_fn=None, dets_json_fn=None, debug_img_fn=None):
    """
    Detect the boxes and segmentations in an image. Currently doesn't do segmentation.

    :param im: Image
    :param dets_pkl_fn: We'll back up the detections to here
    :param dets_json_fn: We'll save detections here (above THRESHOLD) for turking
    :param debug_img_fn: We'll backup the detections in a nice image, to this file
    :return: boxes, obj names, classes if successful, otherwise NONE NONE NONE.
    """
    # logger.info('Processing {}'.format(img_fn))
    # im = cv2.imread(img_fn)
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    #logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    #for k, v in timers.items():
    #    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

    if not isinstance(cls_boxes, list) or not any([x.size > 0 for x in cls_boxes if hasattr(x, 'size')]):
        print("Skip because of other things")
        return None, None, None

    # Get the mask for visualization. #TODO do keypoints
    boxes, segms, keypoints, classes = convert_from_cls_format(
        cls_boxes, cls_segms, cls_keyps)

    inds = np.where(boxes[:, -1] > THRESHOLD)[0]
    if inds.size == 0:
        print("Skip because of harsh threshhold")
        return None, None, None

    if dets_pkl_fn is not None:
        with open(dets_pkl_fn, 'wb') as f:
            pkl.dump({'boxes': cls_boxes, 'segms': cls_segms, 'keyps': cls_keyps, 'im_shape': im.shape}, f)

    boxes = boxes[inds]
    segms = [segms[i] for i in inds.tolist()] if segms is not None else None
    classes = np.array([classes[i] for i in inds.tolist()])
    keypoints = [keypoints[i].tolist() for i in inds.tolist()] if keypoints is not None else None

    contours = []
    if segms is not None:
        masks = mask_util.decode(segms).transpose((2, 0, 1))
        for mask_slice in masks:
            contour, hier = cv2.findContours(
                mask_slice.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            contours.append([c.squeeze(1).tolist() for c in contour])

    # get the names
    obj_names = []
    for object_counter, obj_id in enumerate(classes):
        obj_names.append('{} ({})'.format(object_counter+1, dummy_coco_dataset.classes[obj_id].replace(' ', '')))

    if dets_json_fn is not None:
        with open(dets_json_fn, 'w') as f:
            json.dump({
                'boxes': boxes.tolist(),  # [num_boxes, dims]
                'segms': contours,  # [num_boxes, num_segms, num_points, 2]
                'names': obj_names,
                'width': int(im.shape[1]),
                'height': int(im.shape[0]),
                'keyps': keypoints,
            }, f)


    if debug_img_fn is not None:
        vis_one_image(im[:, :, ::-1], debug_img_fn, boxes, contours, obj_names, keypoints,
                      dpi=200, box_alpha=0.3)

    return {'boxes': boxes.tolist(),  # [num_boxes, dims]
                'segms': contours,  # [num_boxes, num_segms, num_points, 2]
                'names': obj_names,
                'width': int(im.shape[1]),
                'height': int(im.shape[0]),
                'keyps': keypoints}     
    #return boxes, obj_names, classes


def vis_one_image(
        im, im_name, boxes, segm_contours, obj_names, keypoints=None,
        kp_thresh=2, dpi=200, box_alpha=0.0, show_class=True):
    """Visual debugging of detections. We assume that there are detections"""
    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    assert boxes is not None
    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for mask_color_id, i in enumerate(sorted_inds):
        bbox = boxes[i, :4]
        score = boxes[i, -1]

        # show box
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor=color_list[mask_color_id % len(color_list)],
                          linewidth=3, alpha=box_alpha))
        if show_class:
            # TODO: Make some boxes BIGGER if they are far from other things
            y_coord = bbox[1] - 2
            fontsize = max(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 40, 5)
            if fontsize * 2 > y_coord:
                y_coord += fontsize * 2 + 2

            ax.text(
                bbox[0], y_coord,
                obj_names[i] + ' {:0.2f}'.format(score).lstrip('0'),
                fontsize=fontsize,
                family='serif',
                bbox=dict(
                    facecolor=color_list[mask_color_id % len(color_list)],
                    alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if len(segm_contours) > 0:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]

            for segm_part in segm_contours[i]:
                polygon = Polygon(
                    np.array(segm_part),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = np.array(keypoints[i])
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

                if kps[2, i2] > kp_thresh:
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                                   kps[:2, dataset_keypoints.index('right_shoulder')] +
                                   kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                              kps[:2, dataset_keypoints.index('right_hip')] +
                              kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)

    ext = im_name.split('.')[-1]
    rest_of_the_fn = im_name[:-(len(ext) + 1)]
    ext2use = 'png' if ext == 'jpg' else ext
    output_name = rest_of_the_fn + '.' + ext2use
    fig.savefig(output_name, dpi=dpi)
    plt.close('all')

    # Convert to JPG manually... ugh
    if ext == 'jpg':
        assert os.path.exists(output_name)
        png_img = cv2.imread(output_name)
        cv2.imwrite(rest_of_the_fn + '.' + ext, png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        os.remove(output_name)


def convert_detections(im_file, dets_pkl_fn, dets_json_fn=None, debug_img_fn=None):
    """
    Update format for detections....

    :param im: Image
    :param dets_pkl_fn: We'll back up the detections to here
    :param dets_json_fn: We'll save detections here (above THRESHOLD) for turking
    :param debug_img_fn: We'll backup the detections in a nice image, to this file
    :return: boxes, obj names, classes if successful, otherwise NONE NONE NONE.
    """
    with open(dets_pkl_fn, 'rb') as f:
        pkl_dict = pkl.load(f)

    cls_boxes = pkl_dict['boxes']
    cls_segms = pkl_dict['segms']
    cls_keyps = pkl_dict['keyps']
    im_shape = pkl_dict['im_shape']

    if not isinstance(cls_boxes, list) or not any([x.size > 0 for x in cls_boxes if hasattr(x, 'size')]):
        return None, None, None

    # Get the mask for visualization. #TODO do keypoints
    boxes, segms, keypoints, classes = convert_from_cls_format(
        cls_boxes, cls_segms, cls_keyps)

    inds = np.where(boxes[:, -1] > THRESHOLD)[0]
    if inds.size == 0:
        return None, None, None


    boxes = boxes[inds]
    segms = [segms[i] for i in inds.tolist()] if segms is not None else None
    classes = np.array([classes[i] for i in inds.tolist()])
    keypoints = [keypoints[i].tolist() for i in inds.tolist()] if keypoints is not None else None

    contours = []
    if segms is not None:
        masks = mask_util.decode(segms).transpose((2, 0, 1))
        for mask_slice in masks:
            contour, hier = cv2.findContours(
                mask_slice.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            contours.append([c.squeeze(1).tolist() for c in contour])

    # get the names
    obj_names = []
    for object_counter, obj_id in enumerate(classes):
        obj_names.append('{} ({})'.format(object_counter+1, dummy_coco_dataset.classes[obj_id].replace(' ', '')))


    # object_counter = defaultdict(int)
    # obj_names = []
    # for obj_id in classes:
    #     object_counter[obj_id] += 1
    #     obj_names.append('[{}{}]'.format(dummy_coco_dataset.classes[obj_id].replace(' ', ''),
    #                                      object_counter[obj_id]))


    if dets_json_fn is not None:
        with open(dets_json_fn, 'w') as f:
            json.dump({
                'boxes': boxes.tolist(),  # [num_boxes, dims]
                'segms': contours,  # [num_boxes, num_segms, num_points, 2]
                'names': obj_names,
                'width': int(im_shape[1]),
                'height': int(im_shape[0]),
                'keyps': keypoints,
            }, f)

    if debug_img_fn is not None:
        im = cv2.imread(im_file)
        vis_one_image(im[:, :, ::-1], debug_img_fn, boxes, contours, obj_names, keypoints,
                      dpi=200, box_alpha=0.3)

    return boxes, obj_names, classes

if __name__ == "__main__":
    model = get_model()
    return_dict = detect_from_img(model, im)