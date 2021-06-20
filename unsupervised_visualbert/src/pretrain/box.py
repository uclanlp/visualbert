import torch
import numpy
import numpy as np
def heuristic_filter(box_a, box_b, image_size, threshhold = 0.15):
    # center_mass
    box_a_x_center = (box_a[0] + box_a[2]) / 2
    box_b_x_center = (box_b[0] + box_b[2]) / 2

    box_a_y_center = (box_a[1] + box_a[3]) / 2
    box_b_y_center = (box_b[1] + box_b[3]) / 2

    # X non overlap
    if box_a[0] > box_b[2] or box_b[0] > box_a[2]:
        if min(abs(box_a[0] - box_b[2]), abs(box_b[0] - box_a[2])) / image_size[0] > threshhold:
            return False
    if box_a[1] > box_b[3] or box_b[1] > box_a[3]:
        if min(abs(box_a[1] - box_b[3]), abs(box_b[1] - box_a[3])) / image_size[1] > threshhold:
            return False
    
    '''print(abs(box_b_x_center - box_a_x_center) / image_size[0])

    if abs(box_b_x_center - box_a_x_center) / image_size[0] > threshhold:
        return False
    
    if abs(box_b_y_center - box_a_y_center) / image_size[1] > threshhold:
        return False'''
    
    return True

def determine_box_position_type(box_a, box_b, image_size):
    if box_a[0] > box_b[2] or box_b[0] > box_a[2]:  # No overlap
        # Then calculate their distance

        if box_a[1] > box_b[3] or box_b[1] > box_a[3]: # y not overlap

            return ( "x, y not overlap",
            (min(abs(box_a[0] - box_b[2]), abs(box_b[0] - box_a[2])) / image_size[0]).item(),
            
            (min(abs(box_a[0] - box_b[2]), abs(box_b[0] - box_a[2])) / min(abs(box_a[0] - box_a[2]), abs(box_b[0] - box_b[2]))).item(),

            (min(abs(box_a[0] - box_a[2]), abs(box_b[0] - box_b[2])) / image_size[0]).item()
            )
        else:
            overlap_length = min(abs(box_a[1] - box_b[3]), abs(box_b[1] - box_a[3]))
            overlap_ratio = overlap_length / min(abs(box_a[1] - box_a[3]), abs(box_b[1] - box_b[3]))
            return ("x not overlap, y overlap", min(overlap_ratio.item(), 1))
    else:
        # there is overlap, calculate how much they overlap
        overlap_length = min(abs(box_a[0] - box_b[2]), abs(box_b[0] - box_a[2]))
        overlap_ratio = overlap_length / min(abs(box_a[0] - box_a[2]), abs(box_b[0] - box_b[2]))
        return min(overlap_ratio.item(), 1)


def add_to_the_left_to_the_right_relation(box_a, box_b, image_size, y_overlap_ratio_thresh, x_overlap_ratio_thresh):
    if box_a[0] > box_b[2] or box_b[0] > box_a[2]:  # No overlap
        '''distance_ratio = min(abs(box_a[0] - box_b[2]), abs(box_b[0] - box_a[2])) / image_size[0]
        if distance_ratio < no_overlap_thresh:
            return (True, box_a[0] > box_b[2]) # a is to the right of b, if box_a[0] > box_b[2]
        else:
            return (False, box_a[0] > box_b[2]) '''
        if box_a[1] > box_b[3] or box_b[1] > box_a[3]:  # y not overlap
            return (False, box_a[0] > box_b[2])
        else:
            overlap_length = min(abs(box_a[1] - box_b[3]), abs(box_b[1] - box_a[3]))
            overlap_ratio = overlap_length / min(abs(box_a[1] - box_a[3]), abs(box_b[1] - box_b[3]))
            if overlap_ratio > y_overlap_ratio_thresh:
                return (True, box_a[0] > box_b[0])
            else:
                return (False, box_a[0] > box_b[0])
    else:
        # there is overlap, calculate how much they overlap
        overlap_length = min(abs(box_a[0] - box_b[2]), abs(box_b[0] - box_a[2]))
        overlap_ratio = overlap_length / min(abs(box_a[0] - box_a[2]), abs(box_b[0] - box_b[2]))
        if overlap_ratio < x_overlap_ratio_thresh:
            return (True, box_a[0] > box_b[0])
        else:
            return (False, box_a[0] > box_b[0])

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


###########################################################################
### Torch Utils, creds to Max de Groot
###########################################################################

def bbox_intersections(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_intersections_np(box_a, box_b)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def bbox_overlaps(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        return bbox_overlaps_np(box_a, box_b)

    inter = bbox_intersections(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
