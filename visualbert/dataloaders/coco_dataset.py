import os
import random
import json
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import numpy
import torch
from torch.utils.data import Dataset
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from dataloaders.bert_field import BertField
import h5py
from copy import deepcopy

from torch.utils.data.dataloader import default_collate
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from pytorch_pretrained_bert.fine_tuning import _truncate_seq_pair, random_word
from dataloaders.bert_field import IntArrayField
from allennlp.data.fields import ListField

from .bert_data_utils import *
from visualbert.pytorch_pretrained_bert.tokenization import BertTokenizer

from pycocotools.coco import COCO
class COCODataset(Dataset):
    def __init__(self, args, visual_genome_chunk = False):
        super(COCODataset, self).__init__()
        self.args = args
        self.coco = COCO(args.annots_path)
        self.annots_path = args.annots_path
        self.split_name = args.split_name
        self.data_root = args.data_root
        self.visual_genome_chunk = visual_genome_chunk
        self.masks = args.masks

        self.image_feature_type = args.image_feature_type
        self.text_only = args.get("text_only", False)
        self.add_spatial_features = args.get("add_spatial_features", False)
        self.expanded = False
        ########## Loading Annotations
        self.items = self.coco.loadAnns(self.coco.getAnnIds())

        print("{} of captions in total.".format(len(self.items)))

        self.image_feat_reader = faster_RCNN_feat_reader()

        if args.get("chunk_path", None) is not None and self.image_feature_type == "nlvr":
            print("Loading images...")
            self.chunk = torch.load(args.chunk_path)
            average = 0.0
            counter = 0
            new_chunk = {}
            for image_id in self.chunk.keys():
                image_feat_variable, image_boxes, confidence  = self.chunk[image_id]
                if ".npz" in image_id:
                    new_chunk[image_id] = screen_feature(image_feat_variable, image_boxes,confidence, args.image_screening_parameters)
                    average += new_chunk[image_id][2]
                else:
                    new_chunk[image_id+".npz"] = screen_feature(image_feat_variable, image_boxes,confidence, args.image_screening_parameters)
                    average += new_chunk[image_id+".npz"][2]
            print("{} features on average.".format(average/len(self.chunk)))
            self.chunk = new_chunk


        self.do_lower_case = args.do_lower_case
        self.bert_model_name = args.bert_model_name
        self.max_seq_length = args.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name, do_lower_case=self.do_lower_case)
        self.pretraining = args.pretraining
        self.masked_lm_prob = args.get("masked_lm_prob", 0.15)

        with open(os.path.join('./cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}


        if self.image_feature_type == "r2c":
            items = []
            counter = 0
            for i in self.items:
                if self.expanded and index >= self.train_size:
                    image_file_name = "COCO_val2014_{:0>12d}.jpg".format(i['image_id'])
                else:
                    image_file_name = "COCO_{}2014_{:0>12d}.jpg".format(self.split_name, i['image_id'])
                if isinstance(self.masks[image_file_name], dict):
                    items.append(i)
                else:
                    # For some images, the detector seems to have Null output. Thus we just skip them. This will not affect much.
                    counter += 1
            print("Discarded {} instances in {}.".format(counter, self.split_name))
            self.items = items

    def get_image_features_by_training_index(self, index):
        item = self.items[index]

        if self.args.image_feature_type == "flickr":
            v_item = self.visual_genome_chunk[item['image_id']]
            image_feat_variable = v_item["features"]
            image_boxes = None
            image_dim_variable = image_feat_variable.shape[0]
            if self.add_spatial_features:
                image_w = float(v_item['image_w'])
                image_h = float(v_item['image_h'])

                bboxes = v_item["boxes"]
                box_width = bboxes[:, 2] - bboxes[:, 0]
                box_height = bboxes[:, 3] - bboxes[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bboxes[:, 0] / image_w
                scaled_y = bboxes[:, 1] / image_h
                box_width = box_width[..., np.newaxis]
                box_height = box_height[..., np.newaxis]
                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                     scaled_y,
                     scaled_x + scaled_width,
                     scaled_y + scaled_height,
                     scaled_width,
                     scaled_height),
                    axis=1)

                image_feat_variable = np.concatenate((image_feat_variable, spatial_features), axis=1)
            return image_feat_variable, image_boxes, image_dim_variable

        if self.args.image_feature_type == "vqa_fix_100":
            if self.expanded and index >= self.train_size:
                image_file_name = "COCO_val2014_{:0>12d}.npy".format(item['image_id'])
            else:
                image_file_name = "COCO_{}2014_{:0>12d}.npy".format(self.split_name, item['image_id'])

            if "train" in image_file_name:
                folder = os.path.join(self.data_root, "data/detectron_fix_100/fc6/vqa/train2014")
            elif "val" in image_file_name:
                folder = os.path.join(self.data_root, "data/detectron_fix_100/fc6/vqa/val2014")
            image_feat_variable = np.load(os.path.join(folder, image_file_name))
            image_dim_variable = image_feat_variable.shape[0]
            return image_feat_variable, None, image_dim_variable

        if self.expanded and index >= self.train_size:
            image_file_name = "COCO_val2014_{:0>12d}.jpg.npz".format(item['image_id'])
            return self.chunk_val[image_file_name]
        else:
            image_file_name = "COCO_{}2014_{:0>12d}.jpg.npz".format(self.split_name, item['image_id'])

        if self.args.get("chunk_path", None) is not None:
            return self.chunk[image_file_name]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if self.image_feature_type == "r2c":
           return self.__getitem_detector__(index)

        item = self.items[index]
        sample = {}
        if not self.text_only:
            image_feat_variable, image_boxes, image_dim_variable = self.get_image_features_by_training_index(index)
            image_feat_variable = ArrayField(image_feat_variable)
            image_dim_variable = IntArrayField(np.array(image_dim_variable))
            sample["image_feat_variable"] = image_feat_variable
            sample["image_dim_variable"] = image_dim_variable
            sample["label"] = image_dim_variable
        else:
            sample["label"] = IntArrayField(np.array([0]))

        caption_a = item["caption"]
        imageID = item["image_id"]

        if self.expanded and index >= self.train_size:
            coco = self.coco_val
        else:
            coco = self.coco

        rest_anns = coco.loadAnns([i for i in coco.getAnnIds(imgIds=imageID) if i != item['id']])

        if self.args.get("two_sentence", True):
            if random.random() > 0.5:
                item_b = self.items[random.randint(0, len(self.items) - 1)]
                while item_b["image_id"] == imageID:
                    item_b = self.items[random.randint(0, len(self.items) - 1)]
                flag = False
            else:
                item_b = rest_anns[random.randint(0, len(rest_anns) - 1)]
                flag = True

            caption_b = item_b["caption"]
            subword_tokens_a = self.tokenizer.tokenize(caption_a)
            subword_tokens_b = self.tokenizer.tokenize(caption_b)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_a, text_b = subword_tokens_b, is_correct=flag, max_seq_length = self.max_seq_length)
        elif not self.args.get("no_next_sentence", False):
            if random.random() < self.args.false_caption_ratio:
                item_b = self.items[random.randint(0, len(self.items) - 1)]
                while item_b["image_id"] == imageID:
                    item_b = self.items[random.randint(0, len(self.items) - 1)]
                flag = False
            else:
                item_b = item
                flag = True

            caption_b = item_b["caption"]
            subword_tokens_b = self.tokenizer.tokenize(caption_b)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_b, text_b = None, is_correct=flag, max_seq_length = self.max_seq_length)
        else:
            caption_b = item["caption"]
            subword_tokens_b = self.tokenizer.tokenize(caption_b)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_b, text_b = None, is_correct=None, max_seq_length = self.max_seq_length)

        bert_feature = InputFeatures.convert_one_example_to_features_pretraining(
                    example = bert_example,
                    tokenizer=self.tokenizer,
                    probability = self.masked_lm_prob)
        bert_feature.insert_field_into_dict(sample)

        return Instance(sample)

    def __getitem_detector__(self, index):
        item = self.items[index]
        sample = {}
        if self.expanded and index >= self.train_size:
            image_file_name = "COCO_val2014_{:0>12d}.jpg".format(item['image_id'])
        else:
            image_file_name = "COCO_{}2014_{:0>12d}.jpg".format(self.split_name, item['image_id'])

        image_info = self.masks[image_file_name]
        if "train" in image_file_name:
            image_file_path = os.path.join(self.data_root, "train2014", image_file_name)
        elif "val" in image_file_name:
            image_file_path = os.path.join(self.data_root, "val2014", image_file_name)

        ###################################################################
        # Most of things adapted from VCR
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        image = load_image(image_file_path)
        image, window, img_scale, padding = resize_image(image, random_pad=self.is_train)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape
        ###################################################################
        metadata = self.masks[image_file_name] # Get the metadata
        # Load boxes.
        # We will use all detections
        dets2use = np.arange(len(metadata['boxes']))
        # [nobj, 14, 14]
        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata['segms'][i]) for i in dets2use])

        # Chop off the final dimension, that's the confidence
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        # Possibly rescale them if necessary
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]
        
        try:
            metadata['names'] = [i.split(" ")[1][1:-1] for i in metadata["names"]]
        except:
            pass
        obj_labels = [self.coco_obj_to_ind[metadata['names'][i]] for i in dets2use.tolist()]
        boxes = np.row_stack((window, boxes))
        segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
        obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels

        sample['segms'] = ArrayField(segms, padding_value=0)
        sample['objects'] = ListField([LabelField(x, skip_indexing=True) for x in obj_labels])

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        sample['boxes'] = ArrayField(boxes, padding_value=-1)

        caption_a = item["caption"]
        imageID = item["image_id"]
        
        sample["label"] = sample['objects'] # This is an useless field. Just so that they know the batch size.

        if self.expanded and index >= self.train_size:
            coco = self.coco_val
        else:
            coco = self.coco

        rest_anns = coco.loadAnns([i for i in coco.getAnnIds(imgIds=imageID) if i != item['id']])

        if self.args.get("two_sentence", True):
            if random.random() > 0.5:
                item_b = self.items[random.randint(0, len(self.items) - 1)]
                while item_b["image_id"] == imageID:
                    item_b = self.items[random.randint(0, len(self.items) - 1)]
                flag = False
            else:
                item_b = rest_anns[random.randint(0, len(rest_anns) - 1)]
                flag = True # is next sentence

            caption_b = item_b["caption"]
            subword_tokens_a = self.tokenizer.tokenize(caption_a)
            subword_tokens_b = self.tokenizer.tokenize(caption_b)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_a, text_b = subword_tokens_b, is_correct=flag, max_seq_length = self.max_seq_length)
        elif not self.args.get("no_next_sentence", False):
            if random.random() < self.args.false_caption_ratio:
                item_b = self.items[random.randint(0, len(self.items) - 1)]
                while item_b["image_id"] == imageID:
                    item_b = self.items[random.randint(0, len(self.items) - 1)]
                flag = False
            else:
                item_b = item
                flag = True # is next sentence

            caption_b = item_b["caption"]
            subword_tokens_b = self.tokenizer.tokenize(caption_b)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_b, text_b = None, is_correct=flag, max_seq_length = self.max_seq_length)
        else:
            subword_tokens_a = self.tokenizer.tokenize(caption_a)
            bert_example = InputExample(unique_id = index, text_a = subword_tokens_a, text_b = None, is_correct=None, max_seq_length = self.max_seq_length)

        bert_feature = InputFeatures.convert_one_example_to_features_pretraining(
                    example = bert_example,
                    tokenizer=self.tokenizer,
                    probability = self.masked_lm_prob)
        bert_feature.insert_field_into_dict(sample)

        return image, Instance(sample)

    @classmethod
    def splits(cls, args):
        data_root = args.data_root

        if args.image_feature_type == "r2c":
            # For r2c, the masks are pre-computed from a larger detector. Thus, when pre-training on COCO, we follow the same procedure.
            masks = torch.load(os.path.join(data_root, "mask_train.th"))
            mask_val = torch.load(os.path.join(data_root, "mask_val.th"))
            for i in mask_val:
                masks[i] = mask_val[i]
        else:
            masks = None

        if args.image_feature_type == "flickr":
            import base64
            import csv
            import sys
            import zlib
            import time
            import mmap
            csv.field_size_limit(sys.maxsize)
            FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
            infiles = [
            os.path.join(data_root, "trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv"),
            os.path.join(data_root, "trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0"),
            os.path.join(data_root, "trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1"),
            os.path.join(data_root, "trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv")
            ]
            chunk = {}
            chunk_file = os.path.join(data_root, "trainval/resnet101_genome.th")
            if not os.path.exists(chunk_file):
                print("Loading COCO files for Flickr30K for the first time...")
                for infile in infiles:
                    with open(infile, "r+") as tsv_in_file:
                        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                        for item in tqdm(reader):
                            item['image_id'] = int(item['image_id'])
                            item['image_h'] = float(item['image_h'])
                            item['image_w'] = float(item['image_w'])   
                            item['num_boxes'] = int(item['num_boxes'])
                            for field in ['boxes', 'features']:
                                # Hope the python2/3 b64decode does not mess things up.
                                item[field] = np.frombuffer(base64.b64decode(item[field]), 
                                      dtype=np.float32).reshape((item['num_boxes'],-1))
                            item["features"] = torch.from_numpy(item["features"])
                            item["boxes"] = torch.from_numpy(item["boxes"])
                            chunk[item['image_id']] = item
                torch.save(chunk, chunk_file)
            else:
                chunk = torch.load(chunk_file)
        else:
            chunk = None

        copy_args = deepcopy(args)
        copy_args.split_name = "train"
        copy_args.annots_path = os.path.join(data_root, "annotations/captions_{}2014.json".format(copy_args.split_name))

        if args.image_feature_type == "nlvr":
            copy_args.chunk_path = os.path.join(data_root, "coco_features_{}_150.th".format(copy_args.split_name))

        copy_args.data_root = data_root
        copy_args.masks = masks

        trainset = cls(copy_args, chunk)
        trainset.is_train = True


        copy_args = deepcopy(args)
        copy_args.split_name = "val"
        copy_args.annots_path = os.path.join(data_root, "annotations/captions_{}2014.json".format(copy_args.split_name))
        if args.image_feature_type == "nlvr":
            copy_args.chunk_path = os.path.join(data_root, "coco_features_{}_150.th".format(copy_args.split_name))
        copy_args.data_root = data_root
        copy_args.masks = masks

        validationset = cls(copy_args, chunk)
        validationset.is_train = False



        if args.get("expand_coco", False):
            # This is to expand the COCO train 
            trainset.expanded = True
            trainset.train_size = len(trainset.items)

            trainset.items.extend(validationset.items)

            trainset.coco_val = validationset.coco

            if args.image_feature_type != "r2c" and args.image_feature_type != "vqa_fix_100" and args.image_feature_type != "flickr": # For NLVR, we pre-load features so we need to expand the chunk as well
                trainset.chunk_val = validationset.chunk

            imdb = np.load(os.path.join(data_root, "data/imdb/imdb_minival2014.npy"), allow_pickle = True)[1:]
            image_names_mini_val = set([i["image_name"] + ".jpg" for i in imdb])

            if args.get("exclude_minival", False):
                trainset.items = [i for i in trainset.items if "COCO_val2014_{:0>12d}.jpg".format(i['image_id']) not in image_names_mini_val]

            validationset.items = [i for i in validationset.items if "COCO_val2014_{:0>12d}.jpg".format(i['image_id']) in image_names_mini_val]
            print("After expanding, train has {} items, val has {} items".format(len(trainset.items), len(validationset.items)))

        testset = validationset # Testset will not be used so this is just a placeholder
        return trainset, validationset, testset

    @staticmethod
    def collate_fn(data):
        if isinstance(data[0], Instance):
            batch = Batch(data)
            td = batch.as_tensor_dict()
            return td
        else:
            images, instances = zip(*data)
            images = torch.stack(images, 0)

            batch = Batch(instances)
            td = batch.as_tensor_dict()
            td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
            td['images'] = images
            return td
