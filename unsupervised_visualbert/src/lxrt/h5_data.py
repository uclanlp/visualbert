import h5py
from copy import deepcopy
import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
from param import args
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gc
from src.tools import sharearray
import os
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        if len(data_index) == n: # Do not do any incomplete batch
            all_.append(data_index)

    return all_


class CustomBatchSampler():
    # We upsample certain datasets
    def __init__(self, datasets, batch_size, upsample_ratios = [1, 1, 1], reduce_to_non_batch_sampler = False):
        self.datasets = datasets
        self.batch_size = batch_size

        self.lengths = [len(i) for i in self.datasets]
        self.upsample_ratios = upsample_ratios
        self.rotate_index = [0] * len(self.upsample_ratios)
        self.reduce_to_non_batch_sampler = reduce_to_non_batch_sampler

        _flag = False
        for i in self.upsample_ratios:
            if i < 1:
                _flag = True

        self.all_indexes = [torch.randperm(i).tolist() for i in self.lengths]
        assert(not args.get("old_sampler", False))

        if args.get("gradient_accumulation_steps", None):
            self.batch_size = batch_size * args.gradient_accumulation_steps
        self.prepare_indexes()
    
    def prepare_indexes(self):
        self.all_batched_indexes = []
        current_index = 0
        for index, i in enumerate(self.lengths):
            #if args.get("debug", False):
            #    random_indexes = list(range(i))
            #else:
            tmp_indexes = []

            if self.upsample_ratios[index] < 1:
                sample_num = int(1 / self.upsample_ratios[index])
                random_indexes = self.all_indexes[index][self.rotate_index[index]:][::sample_num]
                
                self.rotate_index[index] = self.rotate_index[index] + 1 #% sample_num
                if self.rotate_index[index] == sample_num:
                    self.all_indexes[index] = torch.randperm(i).tolist()
                    self.rotate_index[index] = 0 # Reset rotate index 

                random.shuffle(random_indexes)
                random_indexes = [j + current_index for j in random_indexes]
                random_indexes = chunks(random_indexes, self.batch_size)
                #self.all_batched_indexes.extend(random_indexes)
            else:
                random_indexes = torch.randperm(i).tolist()
                random_indexes = [j + current_index for j in random_indexes]
                random_indexes = chunks(random_indexes, self.batch_size)
                #self.all_batched_indexes.extend(random_indexes)
            
            random.shuffle(random_indexes)
            self.all_batched_indexes.append(random_indexes)

            if self.upsample_ratios[index] > 1:
                for k in range(self.upsample_ratios[index] - 1):
                    #if args.get("debug", False):
                    #    random_indexes = list(range(i))
                    #else:
                    random_indexes = torch.randperm(i).tolist()

                    random_indexes = [j + current_index for j in random_indexes]

                    random_indexes = chunks(random_indexes, self.batch_size)
                    #self.all_batched_indexes.extend(random_indexes)
                    
                    random.shuffle(random_indexes)
                    self.all_batched_indexes[index].extend(random_indexes)

            current_index += i

        all_flatterned_indexes = []
        original_recorder = [len(i) for i in self.all_batched_indexes]
        original_recorder = [i / sum(original_recorder) for i in original_recorder]
        index_recorder = np.array([len(i) - 1 for i in self.all_batched_indexes])
        
        while np.any(index_recorder >= 0):
            choosed_index = np.random.choice(len(original_recorder), p=original_recorder)
            if index_recorder[choosed_index] >= 0:
                all_flatterned_indexes.append(self.all_batched_indexes[choosed_index][index_recorder[choosed_index]])
                index_recorder[choosed_index] -= 1

        self.all_batched_indexes = all_flatterned_indexes

        if self.reduce_to_non_batch_sampler:
            new_ = []
            for i in self.all_batched_indexes:
                for j in i:
                    new_.append([j])
            self.all_batched_indexes = new_
        
        if args.get("gradient_accumulation_steps", None):
            flattened_indexes = []
            for indexes in self.all_batched_indexes:
                flattened_indexes.extend(indexes)
            self.all_batched_indexes = chunks(flattened_indexes, self.batch_size // args.gradient_accumulation_steps)
        return current_index

    def __iter__(self):
        self.prepare_indexes()
        
        return iter(self.all_batched_indexes)

    def __len__(self):
        return len(self.all_batched_indexes)

class ConcateDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        

    def __getitem__(self, index):
        #return self.datasets[1][0]   
        # 
        len_of_datasets = [len(i) for i in self.datasets]
     
        for i in range(0, len(len_of_datasets)):
            '''if i == len(self.len_of_datasets) - 1 and index >= self.len_of_datasets[i]:
                index = index % self.len_of_datasets[i]'''

            if index < len_of_datasets[i]:
                return self.datasets[i][index]
            else:
                index -= len_of_datasets[i]

    def __len__(self):
        return sum([len(i) for i in self.datasets])

class ConcateH5():
    def __init__(self, list_of_h5):
        self.list_of_h5 = list_of_h5
        self.len_of_h5 = [len(i) for i in list_of_h5]
        self.current_copy_index = None
        self.current_copy = None

    def __getitem__(self, index):
        for i in range(0, len(self.len_of_h5)):
            if index < self.len_of_h5[i]:
                return self.list_of_h5[i][index]
            else:
                index -= self.len_of_h5[i]
    def __len__(self):
        return sum(self.len_of_h5)

class ImageFeatureDataset():
    def __init__(self, h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf, h5_wh, ids_to_index, h5_num_boxes = None, version_3 = False):
        self.h5_features = h5_features
        self.h5_boxes = h5_boxes
        self.h5_objects_id = h5_objects_id
        self.h5_objects_conf = h5_objects_conf
        self.h5_attrs_id = h5_attrs_id
        self.h5_attrs_conf = h5_attrs_conf
        self.h5_wh = h5_wh
        self.ids_to_index = ids_to_index
        self.h5_num_boxes = h5_num_boxes
        self.all_indexes = None
        self.version_3 = version_3

    def __getitem__(self, img_id):
        image_index = self.ids_to_index[img_id]
        if self.h5_num_boxes is not None:
            obj_num = self.h5_num_boxes[image_index]
        else:
            obj_num = 36
        feats = self.h5_features[image_index]
        boxes = self.h5_boxes[image_index]
        img_h = self.h5_wh[image_index][1]
        img_w = self.h5_wh[image_index][0]
        
        # For VCR, we did not keep the labels rather we kept the confidence
        if self.version_3:
            obj_confs = np.array(self.h5_objects_conf[image_index][:, 1:])
            attr_confs = np.array(self.h5_attrs_conf[image_index][:, 1:])
            obj_labels = np.argmax(obj_confs, axis=1)
            attr_labels = np.argmax(attr_confs, axis=1)
            obj_confs = np.max(obj_confs, axis=1)
            attr_confs = np.max(attr_confs, axis = 1) 
        else:
            obj_labels = self.h5_objects_id[image_index]
            obj_confs = self.h5_objects_conf[image_index]
            attr_labels = self.h5_attrs_id[image_index]
            attr_confs = self.h5_attrs_conf[image_index]

        return image_index, obj_num, feats, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs
    
    def get_everything_except_features(self, img_id):
        image_index = self.ids_to_index[img_id]
        obj_num = 36
        #feats = self.h5_features[image_index]
        boxes = self.h5_boxes[image_index]
        img_h = self.h5_wh[image_index][1]
        img_w = self.h5_wh[image_index][0]
        obj_labels = self.h5_objects_id[image_index]
        obj_confs = self.h5_objects_conf[image_index]
        attr_labels = self.h5_attrs_id[image_index]
        attr_confs = self.h5_attrs_conf[image_index]

        return image_index, obj_num, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs
    
    @classmethod
    def create(cls, 
        sources, 
        Split2ImgFeatPath_h5, 
        load_custom_h5_version2=False, load_custom_h5_version3=False, 
        text_only = False, on_memory=False):

        current_counter = 0
        ids_to_index = {}
        h5_features_list = []
        h5_boxes_list = []
        h5_objects_id_list = []
        h5_objects_conf_list = []
        h5_attrs_id_list = []
        h5_attrs_conf_list = []
        h5_wh_list = []
        h5_num_boxes_list = []

        for split in sources:
            if load_custom_h5_version2:
                h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf, wh_list, h5_num_boxes = cls.load_custom_h5_version2(Split2ImgFeatPath_h5[split], text_only = text_only, on_memory = on_memory)
            elif load_custom_h5_version3:
                h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf, wh_list, h5_num_boxes = cls.load_custom_h5_version3(Split2ImgFeatPath_h5[split], on_memory = on_memory)
            else:
                h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf = cls.load_custom_h5(Split2ImgFeatPath_h5[split], on_memory=on_memory, text_only=text_only)
                h5_num_boxes = [36] * len(h5_features)
            print(Split2ImgFeatPath_h5[split], len(h5_boxes))
            h5_features_list.append(h5_features)
            h5_boxes_list.append(h5_boxes)
            h5_objects_id_list.append(h5_objects_id)
            h5_objects_conf_list.append(h5_objects_conf)
            h5_attrs_id_list.append(h5_attrs_id)
            h5_attrs_conf_list.append(h5_attrs_conf)
            h5_num_boxes_list.append(h5_num_boxes)

            if load_custom_h5_version2 or load_custom_h5_version3:
                with open(Split2ImgFeatPath_h5[split].replace("h5", "txt").replace('no_features', "image_ids"), "r") as f:
                    image_ids = f.readlines()
                for index, i in enumerate(image_ids):
                    # we will skip images with no boxes, might need some sanity check
                    if h5_num_boxes[index] == 0:
                        continue
                    ids_to_index[i.replace("\n", "")]  = index + current_counter
                current_counter += len(image_ids)
            else:
                with open(Split2ImgFeatPath_h5[split].replace("h5", "json"), "r") as f:
                    metadata = json.load(f)
                wh_list = []
                for index, i in enumerate(metadata):
                    ids_to_index[i["img_id"]] = index + current_counter
                    wh_list.append((i['img_w'], i['img_h']))
                current_counter += len(metadata)
    
            h5_wh_list.append(wh_list)
        
        print("Created {}".format(sources))
        h5_features = ConcateH5(h5_features_list)
        h5_boxes = ConcateH5(h5_boxes_list)
        h5_objects_id = ConcateH5(h5_objects_id_list)
        h5_objects_conf = ConcateH5(h5_objects_conf_list)
        h5_attrs_id = ConcateH5(h5_attrs_id_list)
        h5_attrs_conf = ConcateH5(h5_attrs_conf_list)
        h5_wh = ConcateH5(h5_wh_list)
        h5_num_boxes_list = ConcateH5(h5_num_boxes_list)
        return cls(h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf, h5_wh, ids_to_index, h5_num_boxes = h5_num_boxes_list, version_3 = load_custom_h5_version3)

    @staticmethod
    def load_custom_h5(h5_file_name, on_memory=False, text_only = False):
        h5_file = h5py.File(h5_file_name, "r")

        if on_memory:
            print("Reading h5 {}".format(h5_file))
            h5_features = sharearray.cache(h5_file_name.split("/")[-1], lambda: h5_file['features'])
            gc.collect()
        else:
            h5_features = h5_file['features']
        
        h5_boxes = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "boxes"), np.array(h5_file['boxes']))
        h5_objects_id = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "objects_id"), np.array(h5_file['objects_id']))
        h5_objects_conf = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "objects_conf"), np.array(h5_file['objects_conf']))
        h5_attrs_id = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "attrs_id"), np.array(h5_file['attrs_id']))
        h5_attrs_conf = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "attrs_conf"), np.array(h5_file['attrs_conf']))

        for index in range(len(h5_attrs_id)):
            assert( np.all(h5_attrs_id[index] ==  np.array(h5_file['attrs_id'][index])))

        if on_memory:
            h5_file.close()
            del h5_file
            gc.collect()
        return h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf
    
    @staticmethod
    def load_custom_h5_version2(h5_file_name, on_memory=False, text_only=False):  # This version used in the conceptual caption
        if not text_only:
            h5_file_feature = h5py.File(h5_file_name.replace("no_features", "features"), "r")
        h5_file = h5py.File(h5_file_name, "r")

        if on_memory:
            print("Reading h5 {}".format(h5_file_name.replace("no_features", "features")))
            h5_features = sharearray.cache(h5_file_name.replace("no_features", "features").split("/")[-1], lambda: h5_file_feature['image_features'])
            gc.collect()
        else:
            if not text_only:
                h5_features = h5_file_feature['image_features']
        
        h5_boxes = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "boxes"), lambda: h5_file['boxes'])
        h5_num_boxes = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "num_boxes"), lambda: h5_file['num_boxes'])

        if not args.get("kl_divergence", False):
            h5_objects_id = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "object_ids"), lambda: np.array(h5_file['object_ids'])[:, :, 0]) #deepcopy(np.array(h5_file['object_ids'])[:, :, 0])
            h5_objects_conf = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "object_pro"), lambda: np.array(h5_file['object_pro'])[:, :, 0]) #deepcopy(np.array(h5_file['object_pro'])[:, :, 0])
            h5_attrs_id = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "attribute_ids"), lambda: np.array(h5_file['attribute_ids'])[:, :, 0]) #deepcopy(np.array(h5_file['attribute_ids'])[:, :, 0])
            h5_attrs_conf = sharearray.cache("{}_{}".format(h5_file_name.split("/")[-1], "attribute_pro"), lambda: np.array(h5_file['attribute_pro'])[:, :, 0]) #deepcopy(np.array(h5_file['attribute_pro'])[:, :, 0])
        else:
            h5_objects_id = deepcopy(np.array(h5_file['object_ids']))
            h5_objects_conf = deepcopy(np.array(h5_file['object_pro']))
            h5_attrs_id = deepcopy(np.array(h5_file['attribute_ids']))
            h5_attrs_conf = deepcopy(np.array(h5_file['attribute_pro']))
        gc.collect()
        
        img_h = deepcopy(np.array(h5_file['img_h'])).tolist()
        img_w = deepcopy(np.array(h5_file['img_w'])).tolist()
        wh_list = []
        for i in range(len(img_h)):
            wh_list.append((img_w[i], img_h[i]))
        
        h5_file.close()
        del h5_file
        gc.collect()

        if text_only:
            h5_features = [0] * len(h5_num_boxes)

        return h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf, wh_list, h5_num_boxes
    
    @staticmethod
    def load_custom_h5_version3(h5_file_name, on_memory=False, keep_top_1=True):  # This version used in the conceptual caption 
        h5_file_feature = h5py.File(h5_file_name.replace("no_features", "features"), "r")
        h5_file = h5py.File(h5_file_name, "r")

        if on_memory:
            print("Reading h5 {}".format(h5_file_name.replace("no_features", "features")))
            h5_features = sharearray.cache(h5_file_name.replace("no_features", "features").split("/")[-1], lambda: h5_file_feature['image_features'])
            gc.collect()
        else:
            h5_features = h5_file_feature['image_features']
        
        h5_boxes = deepcopy(np.array(h5_file['boxes']))
        h5_num_boxes = deepcopy(np.array(h5_file['num_boxes']))
        h5_objects_conf = h5_file['object_pro']
        h5_attrs_conf = h5_file['attribute_pro']
        
        img_h = deepcopy(np.array(h5_file['img_h'])).tolist()
        img_w = deepcopy(np.array(h5_file['img_w'])).tolist()
        wh_list = []
        for i in range(len(img_h)):
            wh_list.append((img_w[i], img_h[i]))
        
        h5_objects_id = np.zeros(len(wh_list)) # Place holder 
        h5_attrs_id = np.zeros(len(wh_list)) # Place holder 

        return h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf, wh_list, h5_num_boxes
