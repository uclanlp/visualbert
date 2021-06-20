# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from copy import deepcopy

from param import args
from utils import load_obj_tsv
from pretrain.tag_data_utilis import create_tags
from lxrt.tokenization import BertTokenizer
from lxrt.h5_data import ImageFeatureDataset


# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}
Split2ImgFeatPath = {
    'train': 'data/mscoco_imgfeat/train2014_obj36.h5',
    'valid': 'data/mscoco_imgfeat/val2014_obj36.h5',
    'minival': 'data/mscoco_imgfeat/val2014_obj36.h5',
    'nominival': 'data/mscoco_imgfeat/val2014_obj36.h5',
    "test": 'data/mscoco_imgfeat/test2015_obj36.h5',
}

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

class ConcateH5():
    def __init__(self, list_of_h5):
        self.list_of_h5 = list_of_h5
        self.len_of_h5 = [len(i) for i in list_of_h5]
        
    def __getitem__(self, index):
        for i in range(0, len(self.len_of_h5)):
            if index < self.len_of_h5[i]:
                return self.list_of_h5[i][index]
            else:
                index -= self.len_of_h5[i]
    
    def __len__(self):
        return sum(self.len_of_h5)

"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
mapping_rawdataset_name_to_json = {
    "train": "train",
    "nominival": "val",
    "minival": "val"
}

class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset, args):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        
        self.limit_to_symbolic_split = args.get("limit_to_symbolic_split", False)
        if self.limit_to_symbolic_split:
            dataDir = "/local/harold/ubert/bottom-up-attention/data/vg/"
            coco_ids = set()
            self.mapping_cocoid_to_imageid = {}
            with open(os.path.join(dataDir, 'image_data.json')) as f:
                metadata = json.load(f)
                for item in metadata:
                    if item['coco_id']:
                        coco_ids.add(int(item['coco_id']))
                        self.mapping_cocoid_to_imageid[int(item['coco_id'])] = item["image_id"]

            from lib.data.vg_gqa import vg_gqa
            self.vg_gqa = vg_gqa(None, split = "val" if self.raw_dataset.name == "minival" else "train", transforms=None, num_im=-1)

        self.custom_coco_data = args.get("custom_coco_data", False)
        self.use_h5_file = args.get("use_h5_file", False)
        if self.use_h5_file:
            self.image_feature_dataset = ImageFeatureDataset.create(dataset.splits, Split2ImgFeatPath, on_memory = args.get("on_memory", False))
            self.ids_to_index = self.image_feature_dataset.ids_to_index

            # Screen data
            used_data = []
            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.ids_to_index:
                    used_data.append(datum)
        else:
            # Loading detection features to img_data
            img_data = []
            for split in dataset.splits:
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 if (split == 'minival' and topk is None) else topk
                img_data.extend(load_obj_tsv(
                    os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                    topk=load_topk))

            # Convert img list to dict
            self.imgid2img = {}
            for img_datum in img_data:
                self.imgid2img[img_datum['img_id']] = img_datum
            
            used_data = self.raw_dataset.data

        used_data = used_data[::args.get("partial_dataset", 1)]
        self.data = used_data

        # Only kept the data with loaded image features
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

        if args.get("add_tags", False):
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased",
                do_lower_case=True
            )
            from lxrt.symbolic_vocabulary import SymbolicVocab
            self.symbolic_vocab = SymbolicVocab(args.objects_vocab, args.attributes_vocab)

    def load_custom_h5(self, h5_file):
        h5_features = h5_file['features']
        h5_boxes = deepcopy(np.array(h5_file['boxes']))
        h5_objects_id = deepcopy(np.array(h5_file['objects_id']))
        h5_objects_conf = deepcopy(np.array(h5_file['objects_conf']))
        h5_attrs_id = deepcopy(np.array(h5_file['attrs_id']))
        h5_attrs_conf = deepcopy(np.array(h5_file['attrs_conf']))
        return h5_features, h5_boxes, h5_objects_id, h5_objects_conf, h5_attrs_id, h5_attrs_conf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        if self.custom_coco_data:
            image_index = self.ids_to_index[img_id]
            obj_num = None
            feats = self.h5_features[image_index]
            boxes = self.h5_boxes[image_index]
            img_h = self.h5_wh[image_index][1]
            img_w = self.h5_wh[image_index][0]
            obj_confs = None
            attr_labels = None
            attr_confs = None
        elif self.use_h5_file:
            '''image_index = self.ids_to_index[img_id]
            obj_num = 36
            feats = self.h5_features[image_index]
            boxes = self.h5_boxes[image_index]
            img_h = self.h5_wh[image_index][1]
            img_w = self.h5_wh[image_index][0] '''
            image_index, obj_num, feats, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs = self.image_feature_dataset[img_id]
        else:
            # Get image info
            img_info = self.imgid2img[img_id]
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
            assert obj_num == len(boxes) == len(feats)
            img_h, img_w = img_info['img_h'], img_info['img_w']

        # Normalize the boxes (to 0 ~ 1)
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        if args.get("add_tags", False):
            tags = create_tags(obj_labels=obj_labels, attr_labels=attr_labels, obj_confs=None, attr_confs=None, tokenizer=self.tokenizer, symbolic_vocab = self.symbolic_vocab, visual_tags_box = boxes, use_bert_input=True)
        else:
            tags = None

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, tags, target
        else:
            return ques_id, feats, boxes, ques, tags


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


