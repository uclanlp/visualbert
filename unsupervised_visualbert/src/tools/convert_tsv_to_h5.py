import sys
import csv
import base64
import time
import torch
import numpy as np

from src.utils import load_obj_tsv_save_to_h5

load_obj_tsv_save_to_h5(
    "data/mscoco_imgfeat/train2014_obj36.tsv", 
    "data/mscoco_imgfeat/train2014_obj36.h5", 
    "data/mscoco_imgfeat/train2014_obj36.json",
    82783
)


load_obj_tsv_save_to_h5(
    "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv", 
    "data/vg_gqa_imgfeat/vg_gqa_obj36.h5", 
    "data/vg_gqa_imgfeat/vg_gqa_obj36.json",
    148854
)

load_obj_tsv_save_to_h5(
    "data/mscoco_imgfeat/val2014_obj36.tsv", 
    "data/mscoco_imgfeat/val2014_obj36.h5", 
    "data/mscoco_imgfeat/val2014_obj36.json",
    40504
)

'''
load_obj_tsv_save_to_h5(
    "data/nlvr2_imgfeat/train_obj36.tsv", 
    "data/nlvr2_imgfeat/train_obj36.h5", 
    "data/nlvr2_imgfeat/train_obj36.json",
    103170
)

load_obj_tsv_save_to_h5(
    "data/nlvr2_imgfeat/valid_obj36.tsv", 
    "data/nlvr2_imgfeat/valid_obj36.h5", 
    "data/nlvr2_imgfeat/valid_obj36.json",
    8102
)'''

'''
load_obj_tsv_save_to_h5(
    "data/nlvr2_imgfeat/test_obj36.tsv", 
    "data/nlvr2_imgfeat/test_obj36.h5", 
    "data/nlvr2_imgfeat/test_obj36.json",
    8082
)'''
