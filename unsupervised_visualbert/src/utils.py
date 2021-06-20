# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
import torch
import numpy as np
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(tqdm(reader)):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

def load_obj_tsv_save_to_h5(fname, save_h5_name, save_json_name, all_examples):
    import h5py
    import json
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)

    metadata = []
    
    import h5py
    h5_file = h5py.File(save_h5_name, 'w')
    h5_features = h5_file.create_dataset('features', (all_examples, 36, 2048), dtype=np.float32)
    h5_boxes = h5_file.create_dataset('boxes', (all_examples, 36, 4), dtype=np.float32)
    h5_objects_id = h5_file.create_dataset('objects_id', (all_examples,36), dtype=np.int64)
    h5_objects_conf = h5_file.create_dataset('objects_conf', (all_examples,36), dtype=np.float32)
    h5_attrs_id = h5_file.create_dataset('attrs_id', (all_examples,36), dtype=np.int64)
    h5_attrs_conf = h5_file.create_dataset('attrs_conf', (all_examples,36), dtype=np.float32)

    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(tqdm(reader)):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            metadata.append(
                {
                    "img_id": item["img_id"],
                    "img_h": item["img_h"],
                    "img_w": item['img_w']
                }
            )
            h5_features[i] = item["features"]
            h5_boxes[i] = item["boxes"]
            h5_objects_id[i] = item["objects_id"]
            h5_objects_conf[i] = item["objects_conf"]
            h5_attrs_id[i] = item["attrs_id"]
            h5_attrs_conf[i] = item["attrs_conf"]


    with open(save_json_name, "w") as f:
        json.dump(metadata, f)
    return data

def create_slim_h5(fname, save_h5_name, save_json_name, all_examples, img_ids_to_keep):
    import h5py
    import json
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)

    metadata = []
    
    import h5py
    h5_file = h5py.File(save_h5_name, 'w')
    h5_features = h5_file.create_dataset('features', (all_examples, 36, 2048), dtype=np.float32)
    h5_boxes = h5_file.create_dataset('boxes', (all_examples, 36, 4), dtype=np.float32)
    h5_objects_id = h5_file.create_dataset('objects_id', (all_examples,36), dtype=np.int64)
    h5_objects_conf = h5_file.create_dataset('objects_conf', (all_examples,36), dtype=np.float32)
    h5_attrs_id = h5_file.create_dataset('attrs_id', (all_examples,36), dtype=np.int64)
    h5_attrs_conf = h5_file.create_dataset('attrs_conf', (all_examples,36), dtype=np.float32)
    i = 0
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for index, item in enumerate(tqdm(reader)):
            #continue
            if item["img_id"] not in img_ids_to_keep:
                continue

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            metadata.append(
                {
                    "img_id": item["img_id"],
                    "img_h": item["img_h"],
                    "img_w": item['img_w']
                }
            )
            h5_features[i] = item["features"]
            h5_boxes[i] = item["boxes"]
            h5_objects_id[i] = item["objects_id"]
            h5_objects_conf[i] = item["objects_conf"]
            h5_attrs_id[i] = item["attrs_id"]
            h5_attrs_conf[i] = item["attrs_conf"]
            i += 1
    with open(save_json_name, "w") as f:
        json.dump(metadata, f)
    return data

def load_lxmert_sgg(path, model):
    print("Load rel pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load(path)["model"]
    model_state_dict = model.state_dict()
    '''print("loaded_state_dict", loaded_state_dict["model"].keys())
    print("\n\n\n\n\n")
    print("model_state_dict", model_state_dict.keys())
    assert(0)'''

    new_loaded_state_dict = {}
    for key in list(loaded_state_dict.keys()):
        if "lxrt" in key:
            new_loaded_state_dict[key.split("lxrt.")[-1]] = loaded_state_dict[key]
            # module.rel_heads.rel_predictor.lxrt.encoder.r_layers.3.output.LayerNorm.weight -> encoder.r_layers.3.output.LayerNorm.weight

    load_state_dict_flexible(model.lxrt_encoder.model.bert, new_loaded_state_dict)
    
def load_lxmert_sgg_pretrain(path, model):
    print("Load rel pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load(path)["model"]
    model_state_dict = model.state_dict()
    '''print("loaded_state_dict", loaded_state_dict.keys())
    print("\n\n\n\n\n")
    print("model_state_dict", model_state_dict.keys())
    assert(0)'''

    new_loaded_state_dict = {}
    for key in list(loaded_state_dict.keys()):
        if "lxrt" in key:
            new_loaded_state_dict[key.split("lxrt.")[-1]] = loaded_state_dict[key]
            # module.rel_heads.rel_predictor.lxrt.encoder.r_layers.3.output.LayerNorm.weight -> encoder.r_layers.3.output.LayerNorm.weight

    load_state_dict_flexible(model.bert, new_loaded_state_dict)
    
def load_lxmert_to_sgg(path, model):
    print("Load rel pre-trained LXMERT from %s " % path)
    loaded_state_dict = torch.load(path)["model"]
    model_state_dict = model.state_dict()
    '''print("loaded_state_dict", loaded_state_dict.keys())
    print("\n\n\n\n\n")
    print("model_state_dict", model_state_dict.keys())
    assert(0)'''

    new_loaded_state_dict = {}
    for key in list(loaded_state_dict.keys()):
        if "lxrt" in key:
            new_loaded_state_dict[key.split("lxrt.")[-1]] = loaded_state_dict[key]
            # module.rel_heads.rel_predictor.lxrt.encoder.r_layers.3.output.LayerNorm.weight -> encoder.r_layers.3.output.LayerNorm.weight

    load_state_dict_flexible(model.bert, new_loaded_state_dict)
    

def load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        print("Full loading failed!! Try partial loading!!")

    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print("Skipped: " + name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print("Successfully loaded: "+name)
        except:
            print("Part load failed: " + name)