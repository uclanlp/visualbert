# Functions to convert raw strings into BERT input feature (InputFeatures' class method)
# Some functions for reading image features

# To take care of padding, we will use AllenNLP's Field;
# Caveat: we pad sequences with zero with one exception: BERT's pre-training language model objective mask's padding should be -1.

import os
from torch.utils.data import Dataset
import numpy as np
import random
import json
from collections import defaultdict
from tqdm import tqdm
import os
import numpy
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
import h5py
from copy import deepcopy
import copy
from torch.utils.data.dataloader import default_collate
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import ListField

from pytorch_pretrained_bert.fine_tuning import _truncate_seq_pair, random_word
from .box_utils import load_image, resize_image, to_tensor_and_normalize
from .mask_utils import make_mask
from .bert_field import BertField, IntArrayField, IntArrayTensorField, ArrayTensorField

class InputExample(object):
    def __init__(self, unique_id=None, text_a=None, text_b=None, is_correct=True, lm_labels=None, max_seq_length = None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.is_correct = is_correct # This sort of serves as the correct label as well as the is_next label

        # This should always be None. Right? 
        assert(lm_labels is None)
        self.lm_labels = lm_labels  # masked words for language model

        if max_seq_length is not None:
            self.perform_truncate(max_seq_length)

    def perform_truncate(self, max_seq_length):

        if self.text_b is None:
            len_total = len(self.text_a) + 2
            self.text_a = self.text_a[:max_seq_length - 2]
        else:
            len_total = len(self.text_a) + len(self.text_b) + 3
            if len_total > max_seq_length:
                take_away_from_ctx = min((len_total - max_seq_length + 1) // 2, max(len(self.text_a) - 32, 0))
                take_away_from_answer = len_total - max_seq_length + take_away_from_ctx
                # Follows VCR, perform truncate from the front...
                self.text_a = self.text_a[take_away_from_ctx:]
                self.text_b = self.text_b[take_away_from_answer:]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, is_correct, lm_label_ids=None):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.is_correct = is_correct
        self.lm_label_ids = lm_label_ids

        # For compatiblity with Huggingface Models:
        self.segment_ids = input_type_ids
        self.is_next = is_correct

    # Convert one sentence_a + sentence_b to pre-training example
    @classmethod
    def convert_one_example_to_features(cls, example, tokenizer):
        # note, this is different because weve already tokenized
        tokens_a = example.text_a
        # tokens_b = example.text_b
        tokens_b = None
        if example.text_b:
            tokens_b = example.text_b

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        return cls(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                is_correct=example.is_correct)

    @classmethod
    def convert_examples_to_features(cls, examples, tokenizer):
        # This one does not pad
        max_len = 0
        features = []
        for (ex_index, example) in enumerate(examples):
            feature = cls.convert_one_example_to_features(example, tokenizer)

            if max_len < len(feature.input_ids):
                max_len = len(feature.input_ids)
            features.append(feature)

        for i in features:
            # Zero-pad up to the sequence length.
            while len(i.input_ids) < max_len:
                i.input_ids.append(0)
                i.input_mask.append(0)
                i.input_type_ids.append(0)

            assert len(i.input_ids) == max_len
            assert len(i.input_mask) == max_len
            assert len(i.input_type_ids) == max_len

        return features

    @classmethod
    def convert_one_example_to_features_pretraining(cls, example, tokenizer, probability):
        ############ Modifed by Harold
        # This function does not care about padding, and we leave it to AllenNLP's field to take care of that.
        # But we need to be extra carefule about the padding index.
        # Not everything is padded with zero.

        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        tokens_a = example.text_a

        tokens_b = None
        if example.text_b:
            tokens_b = example.text_b
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens_a, t1_label = random_word(tokens_a, tokenizer, probability)
        if tokens_b:
            tokens_b, t2_label = random_word(tokens_b, tokenizer, probability)
        # concatenate lm labels and account for CLS, SEP, SEP
        if tokens_b:
            lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
        else:
            lm_label_ids = ([-1] + t1_label + [-1])

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            assert len(tokens_b) > 0
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        return cls(unique_id=example.unique_id,
                     tokens=tokens,
                     input_ids=input_ids,
                     input_mask=input_mask,
                     input_type_ids=segment_ids,
                     lm_label_ids=lm_label_ids,
                     is_correct=example.is_correct)

    def convert_to_allennlp_feild(self):
        self.input_ids_field = IntArrayField(np.array(self.input_ids, dtype="int"), padding_value = 0)
        self.input_mask_field = IntArrayField(np.array(self.input_mask, dtype="int"))
        self.input_type_ids_field = IntArrayField(np.array(self.segment_ids, dtype="int"))
        # Padding Value = -1
        if self.lm_label_ids is not None:
            self.masked_lm_labels_field = IntArrayField(np.array(self.lm_label_ids, dtype="int"), padding_value = -1)
        else:
            self.masked_lm_labels_field = None
        if self.is_next is not None:
            self.is_random_next_field = IntArrayField(np.array(self.is_next, dtype="int"))
        else:
            self.is_random_next_field = None

    def convert_to_pytorch_tensor(self):
        # For multi-process efficiency, we will use pytorch tensor instead of ...
        self.input_ids_field = torch.tensor(self.input_ids, dtype=torch.int64)
        self.input_mask_field = torch.tensor(self.input_mask, dtype=torch.int64)
        self.input_type_ids_field = torch.tensor(self.segment_ids, dtype=torch.int64)
        # Padding Value = -1
        if self.lm_label_ids is not None:
            self.masked_lm_labels_field = torch.tensor(self.lm_label_ids, dtype=torch.int64)
        else:
            self.masked_lm_labels_field = None
        if self.is_next is not None:
            self.is_random_next_field = torch.tensor([int(self.is_next)], dtype=torch.int64)
        else:
            self.is_random_next_field = None

    def insert_field_into_dict(self, instance_dict):
        self.convert_to_allennlp_feild()
        instance_dict["bert_input_ids"] = self.input_ids_field
        instance_dict["bert_input_mask"] = self.input_mask_field
        instance_dict["bert_input_type_ids"] = self.input_type_ids_field
        if self.masked_lm_labels_field is not None:
            instance_dict["masked_lm_labels"] = self.masked_lm_labels_field
        if self.is_random_next_field is not None:
            instance_dict["is_random_next"] = self.is_random_next_field
    
    def insert_tensor_into_dict(self, instance_dict):
        self.convert_to_pytorch_tensor()
        instance_dict["bert_input_ids"] = self.input_ids_field
        instance_dict["bert_input_mask"] = self.input_mask_field
        instance_dict["bert_input_type_ids"] = self.input_type_ids_field
        if self.masked_lm_labels_field is not None:
            instance_dict["masked_lm_labels"] = self.masked_lm_labels_field
        if self.is_random_next_field is not None:
            instance_dict["is_random_next"] = self.is_random_next_field

    @staticmethod
    def convert_list_features_to_allennlp_list_feild(list_features, instance_dict):
        input_ids_list = []
        input_mask_list = []
        input_type_ids_list = []
        masked_lm_labels_list = []
        is_random_next_list = []
        # Every element in the list_features is a feature instance
        for i in list_features:
            i.convert_to_allennlp_feild()
            input_ids_list.append(i.input_ids_field)
            input_mask_list.append(i.input_mask_field)
            input_type_ids_list.append(i.input_type_ids_field)
            masked_lm_labels_list.append(i.masked_lm_labels_field)
            is_random_next_list.append(i.is_random_next_field)

        input_ids_list = ListField(input_ids_list)
        input_mask_list = ListField(input_mask_list)
        input_type_ids_list = ListField(input_type_ids_list)
        if masked_lm_labels_list[0]:
            masked_lm_labels_list = ListField(masked_lm_labels_list)
            is_random_next_list = ListField(is_random_next_list)
        else:
            masked_lm_labels_list = None
            is_random_next_list = None

        instance_dict["bert_input_ids"] = input_ids_list
        instance_dict["bert_input_mask"] = input_mask_list
        instance_dict["bert_input_type_ids"] = input_type_ids_list

        if masked_lm_labels_list:
            instance_dict["masked_lm_labels"] = masked_lm_labels_list
            instance_dict["is_random_next"] = is_random_next_list
        return

class faster_RCNN_feat_reader:
    def read(self, image_feat_path):
        return np.load(image_feat_path)

class CHW_feat_reader:
    def read(self, image_feat_path):
        feat = np.load(image_feat_path)
        assert (feat.shape[0] == 1), "batch is not 1"
        feat = feat.squeeze(0)
        return feat

class dim_3_reader:
    def read(self, image_feat_path):
        tmp = np.load(image_feat_path)
        _, _, c_dim = tmp.shape
        image_feat = np.reshape(tmp, (-1, c_dim))
        return image_feat

class HWC_feat_reader:
    def read(self, image_feat_path):
        tmp = np.load(image_feat_path)
        assert (tmp.shape[0] == 1), "batch is not 1"
        _, _, _, c_dim = tmp.shape
        image_feat = np.reshape(tmp, (-1, c_dim))
        return image_feat

class padded_faster_RCNN_feat_reader:
    def __init__(self, max_loc):
        self.max_loc = max_loc

    def read(self, image_feat_path):
        image_feat = np.load(image_feat_path)
        image_loc, image_dim = image_feat.shape
        tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat[0:image_loc, ] = image_feat
        image_feat = tmp_image_feat
        return (image_feat, image_loc)

class padded_faster_RCNN_with_bbox_feat_reader:
    def __init__(self, max_loc):
        self.max_loc = max_loc

    def read(self, image_feat_path):
        image_feat_bbox = np.load(image_feat_path)
        image_boxes = image_feat_bbox.item().get('image_bboxes')
        tmp_image_feat = image_feat_bbox.item().get('image_feat')
        image_loc, image_dim = tmp_image_feat.shape
        tmp_image_feat_2 = np.zeros((self.max_loc, image_dim),
                                    dtype=np.float32)
        tmp_image_feat_2[0:image_loc, ] = tmp_image_feat
        tmp_image_box = np.zeros((self.max_loc, 4), dtype=np.int32)
        tmp_image_box[0:image_loc] = image_boxes

        return (tmp_image_feat_2, image_loc, tmp_image_box)

    def read(self, image_feat_path):
        image_feat = np.load(image_feat_path)
        image_loc, image_dim = image_feat.shape
        tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
        tmp_image_feat[0:image_loc, ] = image_feat
        image_feat = tmp_image_feat
        return (image_feat, image_loc)



def parse_npz_img_feat(feat):
    return feat['x']


def get_image_feat_reader(ndim, channel_first, image_feat, max_loc=None):
    if ndim == 2 or ndim == 0:
        if max_loc is None:
            return faster_RCNN_feat_reader()
        else:
            if isinstance(image_feat.item(0), dict):
                return padded_faster_RCNN_with_bbox_feat_reader(max_loc)
            else:
                return padded_faster_RCNN_feat_reader(max_loc)
    elif ndim == 3 and not channel_first:
        return dim_3_reader()
    elif ndim == 4 and channel_first:
        return CHW_feat_reader()
    elif ndim == 4 and not channel_first:
        return HWC_feat_reader()
    else:
        raise TypeError("unkown image feature format")


def compute_answer_scores(answers, num_of_answers, unk_idx):
    scores = np.zeros((num_of_answers), np.float32)
    for answer in set(answers):
        if answer == unk_idx:
            scores[answer] = 0
        else:
            answer_count = answers.count(answer)
            scores[answer] = min(np.float32(answer_count)*0.3, 1)
    return scores

def read_in_image_feats(image_dirs, image_readers, image_file_name):
    image_feats = []
    for i, image_dir in enumerate(image_dirs):
        image_feat_path = os.path.join(image_dir, image_file_name)
        tmp_image_feat = image_readers[i].read(image_feat_path)
        image_feats.append(tmp_image_feat)

    return image_feats

def get_one_image_feature(path, reader, image_feature_cap):
        image_feat = reader.read(path)
        image_loc = image_feat[1]
        if len(image_feat) == 3:
            image_boxes = image_feat[2]
        else:
            image_boxes = None

        returned_feat = image_feat[0]
        if image_feature_cap != -1:
            if image_feature_cap < image_loc:
                returned_feat = returned_feat[:image_feature_cap, :]
                if image_boxes is not None:
                    image_boxes = image_boxes[:image_feature_cap]
                image_loc = image_feature_cap

        return returned_feat, image_boxes, image_loc

def get_one_image_feature_npz_screening_parameters(path, reader, image_screening_parameters, return_confidence = False):
        result = reader.read(path)

        image_feat = result["box_features"]
        max_conf = result["max_conf"]
        cls_boxes = result["cls_boxes"]

        confidence_cap = image_screening_parameters.get("confidence_cap", None)
        image_feature_cap = image_screening_parameters.get("image_feature_cap", None)

        if confidence_cap:
            keep_boxes = np.where(max_conf >= confidence_cap)[0]

            if keep_boxes.shape[0] == 0:
                image_feat = image_feat[:1] # Just keep one feature...
                cls_boxes = cls_boxes[:1]
                max_conf = max_conf[:1]
            else:
                image_feat = image_feat[keep_boxes]
                cls_boxes = cls_boxes[keep_boxes]
                max_conf = max_conf[keep_boxes]

        if image_feature_cap:
            image_loc = image_feat.shape[0]
            if image_feature_cap < image_loc:
                image_feat = image_feat[:image_feature_cap, :]
                cls_boxes = cls_boxes[:image_feature_cap]
                max_conf = max_conf[:image_feature_cap]

        image_loc = image_feat.shape[0]

        if return_confidence:
            return image_feat, cls_boxes, max_conf
        else:
            return image_feat, cls_boxes, image_loc

def screen_feature(image_feat, cls_boxes, max_conf, image_screening_parameters, mandatory_keep = None):

    confidence_cap = image_screening_parameters.get("confidence_cap", None)
    image_feature_cap = image_screening_parameters.get("image_feature_cap", None)

    min_cap = image_screening_parameters.get("min_cap", 1)
    max_cap = image_screening_parameters.get("max_cap", 300)

    keep_boxes = np.arange(image_feat.shape[0])

    if confidence_cap:
        keep_boxes = np.where(max_conf >= confidence_cap)[0]
        if keep_boxes.shape[0] < min_cap:
            keep_boxes = np.arange(min_cap)
            #image_feat = image_feat[:min_cap]
            #cls_boxes = cls_boxes[:min_cap]

    if image_feature_cap:
        if image_feature_cap < keep_boxes.shape[0]:
            keep_boxes = np.arange(image_feature_cap)
    if max_cap:
        if max_cap < keep_boxes.shape[0]:
            keep_boxes = np.arange(max_cap)

    if mandatory_keep is not None:
        keep_boxes = np.union1d(keep_boxes, mandatory_keep)

    image_feat = image_feat[keep_boxes]
    cls_boxes = cls_boxes[keep_boxes]
    image_loc = image_feat.shape[0]

    return image_feat, cls_boxes, image_loc


