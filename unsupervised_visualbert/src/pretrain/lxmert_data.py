# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random

import numpy as np
from torch.utils.data import Dataset
import torch
from param import args
from src.pretrain.qa_answer_table import AnswerTable
from src.utils import load_obj_tsv
from copy import deepcopy
import h5py
from lxrt.h5_data import ImageFeatureDataset
from lxrt.tokenization import BertTokenizer

from src.pretrain import tag_data_utilis
from tqdm import tqdm
from src.tools import sharearray
import os

TINY_IMG_NUM = 500
FAST_IMG_NUM = 5000

Split2ImgFeatPath = {
    'mscoco_train': '/local/harold/ubert/lxmert/data/mscoco_imgfeat/train2014_obj36.tsv',
    'mscoco_minival': '/local/harold/ubert/lxmert/data/mscoco_imgfeat/val2014_obj36.tsv',
    'mscoco_nominival': '/local/harold/ubert/lxmert/data/mscoco_imgfeat/val2014_obj36.tsv',
    'vgnococo': '/local/harold/ubert/lxmert/data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
}
Split2ImgFeatPath_h5 = {
    'mscoco_train': 'data/mscoco_imgfeat/train2014_obj36.h5',
    'mscoco_minival': 'data/mscoco_imgfeat/val2014_obj36.h5',
    'mscoco_nominival': 'data/mscoco_imgfeat/val2014_obj36.h5',
    'vgnococo': 'data/vg_gqa_imgfeat/vg_gqa_obj36.h5',
    "nlvr_for_pretrain_train": "data/nlvr2_imgfeat/train_obj36.h5",
    "nlvr_for_pretrain_valid": "data/nlvr2_imgfeat/valid_obj36.h5",
    "flickr_train": 'data/flickr30k/fixed36_no_features_split_0_of_1_splits.h5'
}

class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None, sent_b=None,
                 use_visual_tag_flag=False,
                mlm_labels=None,token_ids=None,max_seq_len=96):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label
        self.sent_b = sent_b
        self.use_visual_tag_flag = use_visual_tag_flag

        # The following attributes are used for the bookcorpus/wikipedia pre-training
        self.mlm_labels = mlm_labels
        self.token_ids = token_ids
        self.max_seq_len = max_seq_len


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids,
                 visual_feats, obj_labels,
                 is_matched, ans,
                visual_tags = None,
                visual_tags_objective = None,
                visual_tags_mask = None,
                visual_tags_box=None,
                visual_tags_mismatch=None,
                obj_labels_transformed_mismatch=None,
                visual_tags_box_mismatch=None,
                use_visual_tag_flag=False,
                visual_tags_segment_ids=None,
                visual_feats_seg_ids=None
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.is_matched = is_matched
        self.ans = ans
        self.visual_tags = visual_tags
        self.visual_tags_objective = visual_tags_objective
        self.visual_tags_mask = visual_tags_mask
        self.visual_tags_box = visual_tags_box
        self.visual_tags_mismatch = visual_tags_mismatch

        self.obj_labels_transformed_mismatch = obj_labels_transformed_mismatch
        self.visual_tags_box_mismatch = visual_tags_box_mismatch
        self.use_visual_tag_flag = use_visual_tag_flag
        self.visual_tags_segment_ids = visual_tags_segment_ids
        self.visual_feats_seg_ids = visual_feats_seg_ids

class LXMERTDataset:
    def __init__(self, splits: str, qa_sets=None):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        self.name = splits
        self.sources = splits.split(',')

        # Loading datasets to data
        self.data = []
        for source in self.sources:
            try:
                self.data.extend(json.load(open("data/lxmert/%s.json" % source)))
            except:
                self.data.extend(json.load(open("/local/harold/ubert/lxmert/data/lxmert/%s.json" % source))) # hacky 
        print("Load %d data from %s" % (len(self.data), self.name))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)
        print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        # Modify the answers
        for datum in self.data:
            labelf = datum['labelf']
            for cat, labels in labelf.items():
                for label in labels:
                    for ans in list(label.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                label[new_ans] = label.pop(ans)
                        else:
                            label.pop(ans)

    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),


def load_vocabs():
    attributes = []
    with open(args.attributes_vocab) as f:
        for line in f:
            attr = line.strip("\n")
            if len(attr) != 0:
                attributes.append(attr)
    assert (len(attributes) == 400)

    objects = []
    with open(args.objects_vocab) as f:
        for line in f:
            attr = line.strip("\n")
            if len(attr) != 0:
                objects.append(attr)
    assert (len(objects) == 1600)
    
    return objects, attributes


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = args.word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label
"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
mapping_rawdataset_name_to_json = {
    "mscoco_train,mscoco_nominival,vgnococo": "train",
    "mscoco_minival": "val"
}
from lxrt.symbolic_vocabulary import SymbolicVocab
global symbolic_vocab
symbolic_vocab = SymbolicVocab(args.objects_vocab, args.attributes_vocab)

class LXMERTTorchDataset(Dataset):
    def __init__(self, dataset: LXMERTDataset, topk=-1, sgg_dataset = None, image_only = False, text_only = False, use_visual_tag_flag = False, limit_source = [], available_split_for_cc = None):
        super().__init__()
        self.raw_dataset = dataset
        self.name = '_'.join(self.raw_dataset.sources)
        if args.get('disable_mismatch_for_other_dataset', False):
            # Do not resample for datasets such as BookCorpus
            self.task_matched = args.task_matched if "book_corpus" in self.raw_dataset.sources else False
        else:
            self.task_matched = args.task_matched

        print(self.raw_dataset.sources)
        print(self.task_matched)
        print("\n\n\n")
        self.sgg_dataset = sgg_dataset
        self.image_only = image_only
        self.text_only = text_only
        self.use_visual_tag_flag = use_visual_tag_flag
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.task_nlvr2 = args.get("task_nlvr2", False)

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM

        #self.fake_data = args.get("fake_data", False)
        self.custom_coco_data = args.get("custom_coco_data", False)
        self.use_h5_file = args.get("use_h5_file", False)
        if self.use_h5_file:
            if "google_cc_train" in dataset.sources:
                if args.get('change_split', False):
                    available_split_for_cc = [39]
                else:
                    available_split_for_cc = args.get("available_split_for_cc", [0])
                sources = []
                split_map = {}
                for i in available_split_for_cc:
                    sources.append("google_cc_{}".format(i))
                    split_map["google_cc_{}".format(i)] = "data/google_concetual/butd_feat/train_no_features_split_{}_of_40_splits.h5".format(i)
                self.image_feature_dataset = ImageFeatureDataset.create(sources, split_map, load_custom_h5_version2=True, text_only = self.text_only, on_memory = False)
            elif "open_images_train" in dataset.sources:
                available_split_for_open_image = args.get("available_split_for_open_image", [0])
                sources = []
                split_map = {}
                for split_i, split_j, total_split in available_split_for_open_image:
                    sources.append("open_image_{}_{}".format(split_i, split_j))
                    split_map["open_image_{}_{}".format(split_i, split_j)] = "data/open_image/butd_feat/train_{}_no_features_split_{}_of_{}_splits.h5".format(split_i, split_j, total_split)
                self.image_feature_dataset = ImageFeatureDataset.create(sources, split_map, load_custom_h5_version2=True, on_memory = False)
            else:
                self.image_feature_dataset = ImageFeatureDataset.create(dataset.sources, Split2ImgFeatPath_h5, text_only = self.text_only, load_custom_h5_version2 = True if "flickr_train" in dataset.sources else False, on_memory = args.get("on_memory", False))
            self.ids_to_index = self.image_feature_dataset.ids_to_index

            # Screen data
            used_data = []
            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.ids_to_index:
                    used_data.append(datum)
        else:
            # Original LXMERT. Load the dataset
            img_data = []
            for source in self.raw_dataset.sources:
                img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))

            self.imgid2img = {}
            for img_datum in img_data:
                self.imgid2img[img_datum['img_id']] = img_datum

            # Filter out the dataset
            used_data = []
            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.imgid2img:
                    used_data.append(datum)
        
        used_data = used_data[::args.get("partial_dataset", 1)]


        if sgg_dataset is not None:
            used_data = [datum for datum in used_data if str(datum["img_id"]) in self.sgg_dataset.imageids_to_index]

        # Flatten the dataset (into one sent + one image entries)
        self.data = []

        record_img_id = set()

        remaining_set = set()
        for datum in used_data:
            # datum: {'img_id': 'COCO_train2014_000000318556', 'labelf': {'vqa': [{'no': 1}, {'yes': 1}, {'no': 1}, {'blue': 1, 'blue and white': 0.3}]}, 'sentf': {'mscoco': ['A very clean and well decorated empty bathroom', 'A blue and white bathroom with butterfly themed wall tiles.', 'A bathroom with a border of butterflies and blue paint on the walls above it.', 'An angled view of a beautifully decorated bathroom.', 'A clock that blends in with the wall hangs in a bathroom. '], 'vqa': ['Is the sink full of water?', 'Are there any butterflies on the tiles?', 'Is this bathroom in a hotel?', 'What color are the walls?']}}

            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in limit_source:
                    continue
                
                remaining_set.add(sents_cat)

                if sents_cat in datum['labelf']:
                    labels = datum['labelf'][sents_cat]
                else:
                    labels = None
                for sent_idx, sent in enumerate(sents):
                    new_datum = {
                        'uid': make_uid(datum['img_id'], sents_cat, sent_idx) if args.task_qa else None,
                        'img_id': datum['img_id'], # if not self.text_only else "",
                        'sent': sent #if not self.image_only else ""
                    }
                    if image_only: # If we only use image, make sure one image only appears one time
                        if datum["img_id"] in record_img_id:
                            continue
                        record_img_id.add(datum["img_id"])

                    if labels is not None and args.task_qa:
                        new_datum['label'] = labels[sent_idx]

                    if self.task_nlvr2:
                        new_datum['match_label'] = datum["label"]
                        new_datum['img_id_1'] = datum["img_id_1"]

                    self.data.append(new_datum)
        
        if image_only:
            dataset_str = "image_only"
        elif text_only:
            dataset_str = "text_only"
        else:
            dataset_str = "vision and language"
        
        if self.image_only and args.get("screen_image", False):
            counter = 0
            from tqdm import tqdm
            _data = []
            for data_item in tqdm(self.data):
                img_id = data_item["img_id"]
                image_index = self.image_feature_dataset.ids_to_index[img_id]
                img_h = self.image_feature_dataset.h5_wh[image_index][1]
                img_w = self.image_feature_dataset.h5_wh[image_index][0]
                if img_h == 0 or img_w == 0:
                    counter += 1
                else:
                    _data.append(data_item)
            
            print("Screened {} images with zero heights and weidths, {} in total".format(counter, len(_data)))
            self.data = _data


        print("Use {} data in {} torch dataset, {}, limit_source {}".format(len(self.data), dataset_str, remaining_set, limit_source))


        if text_only:
            del self.image_feature_dataset

        if text_only or image_only:
            del self.raw_dataset.data
            del self.raw_dataset
        
        self.compress_memory = False
        if args.get("compress_memory", False):
            # Move some data to shared memory so the memory will not explode when using multi-process for data loading
            self.compress()
        print("\n\n\n")

    
    def compress(self):

        print("image_only", self.image_only)
        print("text_only", self.text_only)
       
        self._img_ids_shared_array, self._img_ids_record_position = self.compress_list_of_strings([i["img_id"] for i in self.data], "data_imonly_img_id_{}".format(self.name))
        self.compress_memory = True
        self._sent_shared_array, self._sent_record_position = self.compress_list_of_strings([i["sent"] for i in self.data], "data_txtonly_sent_{}".format(self.name))
        self.compress_memory = True

    def compress_list_of_strings(self, list_of_string, name):
        record_position = []
        all_text = []
        current_length = 0
        for index, string in enumerate(list_of_string):
            array = [ord(c) for c in string]
            all_text.extend(array)
            current_length += len(array)
            record_position.append(current_length)

        shared_array = sharearray.cache(name, lambda: np.array(all_text, dtype=np.int32))
        del all_text
        return shared_array, record_position
    
    def decompress_string_index(self, index, shared_array, record_position):
        string_array = shared_array[0 if index == 0 else record_position[index - 1]:record_position[index]]
        return ''.join([chr(c) for c in string_array])
    
    def decompress_getitem__(self, index):
        if self._sent_shared_array is not None:
            sent = self.decompress_string_index(index, self._sent_shared_array, self._sent_record_position)
        else:
            sent = ""
        if self._img_ids_shared_array is not None:
            img_id = self.decompress_string_index(index, self._img_ids_shared_array, self._img_ids_record_position)
        else:
            img_id = None
        return {"sent": sent, "img_id": img_id, "uid": None}

    
    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        if self.compress_memory:
            datum = self.decompress_getitem__(random.randint(0, len(self.data) - 1))
        else:
            datum = self.data[random.randint(0, len(self.data) - 1)]
        img_id = datum['img_id']
        if self.use_h5_file:
            image_index = self.ids_to_index[img_id]
            feat = self.image_feature_dataset.h5_features[image_index]
            feat = feat[random.randint(0, 35)]
        else:
            img_info = self.imgid2img[img_id]
            feat = img_info['features']
            feat = feat[random.randint(0, 35)]
        return feat
    
    def random_tags(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data)-1)]
        img_id = datum['img_id']
        image_index, obj_num, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs = self.image_feature_dataset.get_everything_except_features(img_id)
        boxes = boxes.copy()        
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        return image_index, obj_num, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs

    def __getitem__(self, item: int):
        if self.compress_memory:
            datum = self.decompress_getitem__(item)
        else:
            datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']
        sent=datum['sent'].lower()

        if not self.text_only:
            # Get image info
            if self.use_h5_file:
                image_index, obj_num, feats, boxes, img_h, img_w, obj_labels, obj_confs, attr_labels, attr_confs = self.image_feature_dataset[img_id]
            else:
                img_info = self.imgid2img[img_id]
                obj_num = img_info['num_boxes']
                feats = img_info['features'].copy()
                boxes = img_info['boxes'].copy()
                obj_labels = img_info['objects_id'].copy()
                obj_confs = img_info['objects_conf'].copy()
                attr_labels = img_info['attrs_id'].copy()
                attr_confs = img_info['attrs_conf'].copy()
                assert obj_num == len(boxes) == len(feats)
                # Normalize the boxes (to 0 ~ 1)
                img_h, img_w = img_info['img_h'], img_info['img_w']
                #print(item, img_info, img_h, img_w)
            boxes = boxes.copy()  
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched=None
        if args.get('task_nlvr2', False):
            match_label = datum["match_label"]
            is_matched = match_label
            second_image_index, second_obj_num, second_feats, second_boxes, second_img_h, second_img_w, second_obj_labels, second_obj_confs, second_attr_labels, second_attr_confs = self.image_feature_dataset[datum["img_id_1"]]
            second_boxes = second_boxes.copy()        
            second_boxes[:, (0, 2)] /= second_img_w
            second_boxes[:, (1, 3)] /= second_img_h
            np.testing.assert_array_less(second_boxes, 1+1e-5)
            np.testing.assert_array_less(-second_boxes, 0 + 1e-5)

            feats=np.concatenate((feats, second_feats))
            boxes=np.concatenate((boxes, second_boxes))
            obj_labels=np.concatenate((obj_labels, second_obj_labels))
            obj_confs=np.concatenate((obj_confs, second_obj_confs))
            #obj_confs=np.concatenate((obj_confs, second_obj_confs))
            attr_labels = np.concatenate((attr_labels, second_attr_labels))
            attr_confs = np.concatenate((attr_confs, second_attr_confs))

        elif self.task_matched :
            if random.random() < 0.5:
                is_matched = 0
                if self.compress_memory:
                    other_datum = self.decompress_getitem__(random.randint(0, len(self.data) - 1))
                else:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]
                while other_datum['img_id'] == img_id:
                    if self.compress_memory:
                        other_datum = self.decompress_getitem__(random.randint(0, len(self.data) - 1))
                    else:
                        other_datum = self.data[random.randint(0, len(self.data)-1)]
                sent = other_datum['sent']
            else:
                is_matched = 1
        
        # Label, convert answer to id
        if 'label' in datum and args.task_qa:
            label = datum['label'].copy()
            for ans in list(label.keys()):
                label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None
        
        if self.image_only:
            sent = None
        if self.text_only:
            feats = None
            boxes = None
            obj_labels = None
            obj_confs = None 
            attr_labels = None
            attr_confs = None

        # Create target
        example = InputExample(
            uid, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label,
            use_visual_tag_flag=self.use_visual_tag_flag
        )

        #if args.get("faster_loading", False):
        return self.convert_example_to_features(example, args.get("max_seq_length", 20), self.tokenizer)
            
    def random_mask_features(self, feats, boxes = None):
        mask_feats = deepcopy(feats) #.copy()
        feat_mask = np.zeros(len(feats), dtype=np.float32)

        for i in range(len(feats)):
            prob = random.random()
            # mask token with probability
            if prob < args.obj_mask_rate:
                feat_mask[i] = 1.

                prob /= args.obj_mask_rate

                # 80% randomly change token to zero feat
                if prob < 0.8:
                    mask_feats[i, :] = 0.

                # 10% randomly change token to random feat
                elif prob < 0.9:
                    if not args.get("disable_random_feat", False) and not args.get("inbatch_random", False):
                        mask_feats[i,:] = self.random_feat()
                    if args.get("inbatch_random", False):
                        feat_mask[i] = 2.0 # special mark
                # -> rest 10% randomly keep current feat

                # Need to predict this feat
        return mask_feats, feat_mask

    def convert_example_to_features(self, example: InputExample, max_seq_length, tokenizer):

        if example.mlm_labels is not None:  # The data is already pre-masked
            input_ids = example.token_ids
            lm_label_ids = example.mlm_labels
            max_seq_len = example.max_seq_len + 2
            # Add [CLS] and [SEP]
            input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + input_ids + tokenizer.convert_tokens_to_ids(["[SEP]"])
            lm_label_ids = [-1] + lm_label_ids + [-1]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                lm_label_ids.append(-1)

            features = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                lm_label_ids=lm_label_ids,
                visual_feats=(None, None),
                obj_labels={
                    'obj': (None, None),
                    'attr': (None, None),
                    'feat': (None, None),
                },
                is_matched=None,
                ans=-1,
                visual_tags = None,
                visual_tags_objective = None,
                visual_tags_mask = None,
                visual_tags_box=None,
                visual_tags_mismatch=None
            )
            return features

        if example.sent is not None:

            tokens = tokenizer.tokenize(example.sent.strip())

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]

            # Ge random words
            masked_tokens, masked_label = random_word(tokens, tokenizer)

            # concatenate lm labels and account for CLS, SEP, SEP
            masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

            # Mask & Segment Word
            lm_label_ids = ([-1] + masked_label + [-1])
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                lm_label_ids.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(lm_label_ids) == max_seq_length
        elif args.get("insert_cls", False):
            masked_tokens = ["[CLS]"]
            input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            lm_label_ids = [-1]
        else:
            input_ids = None
            input_mask = None
            segment_ids = None
            lm_label_ids = None
        
        if example.use_visual_tag_flag and example.visual_feats[0] is not None:  # Let's do a hybrid embedding
            feat, boxes = example.visual_feats
            obj_labels, obj_confs = example.obj_labels
            attr_labels, attr_confs = example.attr_labels

            # Mask Image Features:
            masked_feat, feat_mask = self.random_mask_features(feat, boxes=boxes)

            assert(args.non_exclusive_tags)
            assert(args.use_bert_input_for_tags)
            visual_tags, visual_tags_objective, visual_tags_mask, visual_tags_box, visual_tags_segment_ids = tag_data_utilis.create_tags_pretrain(
                        obj_labels=obj_labels,
                        attr_labels=attr_labels,
                        obj_confs=obj_confs,
                        attr_confs=attr_confs,
                        tokenizer=self.tokenizer,
                        symbolic_vocab=symbolic_vocab,
                        visual_tags_box = boxes,
                        feat_mask = feat_mask,
                        use_bert_input=True
                    )
        elif example.visual_feats[0] is not None:
            feat, boxes = example.visual_feats
            obj_labels, obj_confs = example.obj_labels
            attr_labels, attr_confs = example.attr_labels
            # Mask Image Features:
            masked_feat, feat_mask = self.random_mask_features(feat, boxes=boxes)
            visual_tags = None
            visual_tags_objective = None
            visual_tags_mask = None
            visual_tags_box = None
            visual_mismatch_label = None
            obj_labels_transformed_mismatch = None
            visual_tags_box_mismatch = None
        else:
            masked_feat = None
            boxes = None
            obj_labels = None
            obj_confs = None
            attr_labels = None
            attr_confs = None
            feat_mask = None
            feat = None
            visual_tags = None
            visual_tags_objective = None
            visual_tags_mask = None
            visual_tags_box = None
            visual_mismatch_label = None
            obj_labels_transformed_mismatch = None
            visual_tags_box_mismatch = None
        
        # QA answer label
        if example.label is None or len(example.label) == 0 or example.is_matched != 1:
            # 1. No label 2. Label is pruned 3. unmatched visual + language pair
            ans = -1
        else:
            keys, values = zip(*example.label.items())
            if len(keys) == 1:
                ans = keys[0]
            else:
                value_sum = sum(values)
                prob = [value / value_sum for value in values]
                choice = np.random.multinomial(1, prob).argmax()
                ans = keys[choice]

        features = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            lm_label_ids=lm_label_ids,
            visual_feats=(masked_feat, boxes),
            obj_labels={
                'obj': (obj_labels, obj_confs),
                'attr': (attr_labels, attr_confs),
                'feat': (feat, feat_mask),
            },
            is_matched=example.is_matched,
            ans=ans,
            visual_tags = visual_tags,
            visual_tags_objective = visual_tags_objective,
            visual_tags_mask = visual_tags_mask,
            visual_tags_box=visual_tags_box,
            visual_tags_mismatch=None if not args.get('use_tag_mismatch', None) else visual_mismatch_label,
            obj_labels_transformed_mismatch=None if not args.get("use_tag_mismatch", None) else obj_labels_transformed_mismatch,
            visual_tags_box_mismatch=None if not args.get('use_tag_mismatch', None) else visual_tags_box_mismatch,
            use_visual_tag_flag=example.use_visual_tag_flag        )
        return features

    def create_in_batch_random_feat(self, example, example_index, all_examples):
        if args.get("inbatch_random", False) and example.visual_feats[0] is not None:
            feats, _ = example.visual_feats
            feat_mask = example.obj_labels["feat"][1]
            #original_feats = example.obj_labels["feat"][0]
            for i in range(len(feat_mask)):
                if feat_mask[i] == 2:
                    feat_mask[i] = 1
                    select_index = random.randint(0, len(all_examples) - 1)
                    while select_index == example_index:
                        select_index = random.randint(0, len(all_examples) - 1)
                    select_index_j = random.randint(0, len(feat_mask) - 1)
                    while select_index_j == i:
                        select_index_j = random.randint(0, len(feat_mask) - 1)
                    feats[i] = all_examples[select_index].obj_labels["feat"][0][select_index_j]
        return example

    def custom_collact_fn(self, examples):
        
        hybrid_num = random.randint(args.get("hybrid_min", 2), args.get("hybrid_max", 34))
        
        train_features = [self.create_in_batch_random_feat(example, example_index, all_examples = examples) for example_index, example in enumerate(examples)]

        if train_features[0].input_ids is not None:
            # language Inputs
            input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

            # Language Prediction
            lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long)
        else:
            input_ids = None
            input_mask = None
            segment_ids = None
            lm_labels = None

        if train_features[0].visual_feats[0] is not None:
            # Visual Inputs
            if isinstance(train_features[0].visual_feats[0], torch.FloatTensor):
                feats = torch.stack([f.visual_feats[0] for f in train_features])
            else:
                feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features]))
            pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features]))
            # Visual Prediction
            obj_labels = {}
            for key in args.visual_losses.split(","):#('obj', 'attr', 'feat'):
                visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features]))
                #if self.custom_coco_data:
                #    visn_mask = torch.ones(visn_labels.size(0), visn_labels.size(1)).float().cuda()
                #else:
                visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features]))
                assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
                obj_labels[key] = (visn_labels, visn_mask)
            if args.get('task_nlvr2', False):
                visual_feats_seg_ids = []
                for i in range(feats.size(0)):
                    visual_feats_seg_ids.append([0] * 36 + [1] * 36)
                visual_feats_seg_ids= torch.tensor(visual_feats_seg_ids, dtype=torch.int64)
            else:
                visual_feats_seg_ids = None
        else:
            feats = None
            pos = None
            obj_labels = None
            visual_feats_seg_ids = None
        
        if train_features[0].visual_tags is not None:
            # do padding
            tag_max_length = max([len(f.visual_tags) for f in train_features])
            for f in train_features:
                current_tag_length = len(f.visual_tags)
                if current_tag_length < tag_max_length:
                    f.visual_tags = f.visual_tags + [0] * (tag_max_length - current_tag_length)
                    f.visual_tags_objective = f.visual_tags_objective + [-1] * (tag_max_length - current_tag_length)
                    f.visual_tags_mask = f.visual_tags_mask + [0] * (tag_max_length - current_tag_length)
                    f.visual_tags_box = f.visual_tags_box + [ np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32) ] * (tag_max_length - current_tag_length)
                    f.visual_tags_box = np.stack(f.visual_tags_box)
                    if f.visual_tags_segment_ids is not None:
                        f.visual_tags_segment_ids = f.visual_tags_segment_ids + [0] * (tag_max_length - current_tag_length)

            visual_tags = torch.tensor([f.visual_tags for f in train_features], dtype=torch.long)
            visual_tags_mask = torch.tensor([f.visual_tags_mask for f in train_features], dtype=torch.long)
            visual_tags_box = torch.from_numpy(np.stack([f.visual_tags_box for f in train_features]))
            visual_tags_objective = torch.tensor([f.visual_tags_objective for f in train_features], dtype=torch.long)
            if train_features[0].visual_tags_mismatch is not None:
                visual_tags_mismatch = torch.tensor([f.visual_tags_mismatch for f in train_features], dtype=torch.long)
            else:
                visual_tags_mismatch = None
            if train_features[0].visual_tags_segment_ids is not None:
                visual_tags_segment_ids = torch.tensor([f.visual_tags_segment_ids for f in train_features], dtype=torch.long)
            else:
                visual_tags_segment_ids = None
            
            if args.get("tag_hard_max_length", None) is not None and tag_max_length > args.tag_hard_max_length:
                # truncate the tag sequence
                visual_tags = visual_tags[:, :args.tag_hard_max_length].contiguous()
                visual_tags_mask = visual_tags_mask[:, :args.tag_hard_max_length].contiguous()
                visual_tags_box = visual_tags_box[:, :args.tag_hard_max_length].contiguous()
                visual_tags_objective = visual_tags_objective[:, :args.tag_hard_max_length].contiguous()
                if visual_tags_mismatch is not None:
                    visual_tags_mismatch = visual_tags_mismatch[:, :args.tag_hard_max_length].contiguous()
                if visual_tags_segment_ids is not None:
                    visual_tags_segment_ids = visual_tags_segment_ids[:, :args.tag_hard_max_length].contiguous()

        else:
            visual_tags = None
            visual_tags_mask = None
            visual_tags_box = None
            visual_tags_objective = None
            visual_tags_mismatch = None
            visual_tags_segment_ids = None

        if train_features[0].is_matched is not None:
            matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long)
        else:
            matched_labels = None
        ans = torch.from_numpy(np.stack([f.ans for f in train_features]))


        if args.get("lxmert_style_nlvr", False):
            # Reorganize the inputs
            input_ids = input_ids.unsqueeze(1).expand(input_ids.size(0), 2, input_ids.size(-1)).contiguous().view(-1, input_ids.size(-1)).contiguous()
            lm_labels = lm_labels.unsqueeze(1).expand(lm_labels.size(0), 2, lm_labels.size(-1)).contiguous().view(-1, lm_labels.size(-1)).contiguous()
            input_mask = input_mask.unsqueeze(1).expand(input_mask.size(0), 2, input_mask.size(-1)).contiguous().view(-1, input_mask.size(-1)).contiguous()

            visual_feats_seg_ids = None
            feats = feats.view(-1, feats.size(1)//2, feats.size(-1)).contiguous()
            pos = pos.view(-1, pos.size(1) // 2, pos.size(-1)).contiguous()
            if args.get("use_visual_tag_flag", False):
                visual_tags = visual_tags.view(-1, visual_tags.size(1) // 2).contiguous()
                visual_tags_box = visual_tags_box.view(-1, visual_tags_box.size(1) // 2, visual_tags_box.size(-1)).contiguous()
                visual_tags_objective = visual_tags_objective.view(-1, visual_tags_objective.size(1) // 2).contiguous()
                visual_tags_mask = visual_tags_mask.view(-1, visual_tags_mask.size(1)//2).contiguous()
        return [input_ids, segment_ids, input_mask, lm_labels, feats, pos, obj_labels, matched_labels, ans, visual_feats_seg_ids, visual_tags, visual_tags_mask, visual_tags_box, visual_tags_objective, visual_tags_mismatch, visual_tags_segment_ids]

class LXMERTEvaluator:
    def __init__(self, dataset: LXMERTDataset):
        self.raw_dataset = dataset

        # Create QA Eval Data
        self.data = []
        for datum in self.raw_dataset.data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplemented
