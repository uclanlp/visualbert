import os
from torch.utils.data import Dataset
import numpy as np

import random

import json
from collections import defaultdict
from tqdm import tqdm
import json
import os

import numpy as np
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
import numpy as np
from allennlp.data.fields import ListField

import os
import json
import _pickle as cPickle
import numpy as np
import utils
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset

import itertools
import re

COUNTING_ONLY = False


from .flickr_ban.dataset import _load_flickr30k, _load_flickr30k_our
from .bert_data_utils import *
from .vcr_data_utils import retokenize_with_alignment

class Flickr30kFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary = None, data_root='data/flickr30k/', chunk = None, entries = None):
        super(Flickr30kFeatureDataset, self).__init__()

        self.add_spatial_features = args.add_spatial_features
        self.dictionary = dictionary
        self.use_visual_genome = args.get("use_visual_genome", True)

        if self.use_visual_genome:
            self.img_id2idx = cPickle.load(
                open(os.path.join(data_root, '%s_imgid2idx.pkl' % name), 'rb'))

            h5_path = os.path.join(data_root, '%s.hdf5' % name)
            with h5py.File(h5_path, 'r') as hf:
                self.features = np.array(hf.get('image_features'))
                self.spatials = np.array(hf.get('spatial_features'))
                self.bbox = np.array(hf.get('image_bb'))
                self.pos_boxes = np.array(hf.get('pos_boxes'))
            self.entries = _load_flickr30k(data_root, self.img_id2idx, self.bbox, self.pos_boxes, limit = None, cache_name = name)
        else:
            self.features_chunk = chunk
            self.entries = entries

        self.pretraining = args.pretraining
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        self.do_lower_case = args.do_lower_case
        self.bert_model_name = args.bert_model_name
        self.max_seq_length = args.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name, do_lower_case=self.do_lower_case)
        self.masked_lm_prob = args.get("masked_lm_prob", 0.15)

    @classmethod
    def splits(cls, args):
        data_root = args.data_root

        if args.get("use_visual_genome", True):
            chunk = None
            train_entries = None
            val_entries = None
            test_entries = None
        else:
            assert(0)
            '''
            chunk = torch.load(chunk_data)
            image_screening_parameters = args.image_screening_parameters
            for image_id in chunk.keys():
                image_feat_variable, image_boxes, confidence, image_h, image_w = chunk[image_id]
                image_feat_variable, image_boxes, confidence = screen_feature(image_feat_variable, image_boxes, confidence, image_screening_parameters)
                chunk[image_id] = (image_feat_variable, image_boxes, confidence, image_h, image_w)

            train_ids = cPickle.load(
                open(os.path.join(data_root, '%s_imgid2idx.pkl' % "train"), 'rb'))
            val_ids = cPickle.load(
                open(os.path.join(data_root, '%s_imgid2idx.pkl' % "val"), 'rb'))
            test_ids = cPickle.load(
                open(os.path.join(data_root, '%s_imgid2idx.pkl' % "test"), 'rb'))
            val_ids = list(val_ids.keys())
            train_ids = list(train_ids.keys())
            test_ids = list(test_ids.keys())
            entities_data_path = os.path.join(data_root, args.entries_path)

            if not os.path.exists(entities_data_path):
                entries = _load_flickr30k_our(data_root, chunk)
                with open(entities_data_path, 'wb') as f:
                    cPickle.dump(entries, f)
            else:
                entries = cPickle.load(open(entities_data_path, "rb"))

            train_entries = [i for i in entries if int(i["image"][:-4]) in train_ids]
            val_entries = [i for i in entries if int(i["image"][:-4]) in val_ids]
            test_entries = [i for i in entries if int(i["image"][:-4]) in test_ids]
            '''

        train = cls(name = "train", args = args, data_root = data_root, chunk = chunk, entries = train_entries)
        train.is_train = True

        val = cls(name = "val", args = args, data_root = data_root, chunk = chunk, entries = val_entries)
        val.is_train = False

        test = cls(name = "test", args = args, data_root = data_root, chunk = chunk, entries = test_entries)

        test.is_train = False

        return train, val, test

    def tokenize(self, max_length=82):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['sentence'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['p_token'] = tokens

    def tensorize(self, max_box=100, max_entities=16, max_length=82):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            phrase = torch.from_numpy(np.array(entry['p_token']))
            entry['p_token'] = phrase

            assert len(entry['target_indices']) == entry['entity_num']
            assert len(entry['entity_indices']) == entry['entity_num']

            target_tensors = []
            for i in range(entry['entity_num']):
                target_tensor = torch.zeros(1, max_box)
                if len(entry['target_indices'][i]) > 0:
                    target_idx = torch.from_numpy(np.array(entry['target_indices'][i]))
                    target_tensor = torch.zeros(max_box).scatter_(0, target_idx, 1).unsqueeze(0)
                target_tensors.append(target_tensor)
            assert len(target_tensors) <= max_entities, '> %d entities!' % max_entities
            for i in range(max_entities - len(target_tensors)):
                target_tensor = torch.zeros(1, max_box)
                target_tensors.append(target_tensor)
                entry['entity_ids'].append(0)
            # padding entity_indices with non-overlapping indices
            entry['entity_indices'] += [x for x in range(max_length) if x not in entry['entity_indices']]
            entry['entity_indices'] = entry['entity_indices'][:max_entities]
            entry['target'] = torch.cat(target_tensors, 0)
            # entity positions in (e) tensor
            entry['e_pos'] = torch.LongTensor(entry['entity_indices'])
            entry['e_num'] = torch.LongTensor([entry['entity_num']])
            entry['entity_ids'] = torch.LongTensor(entry['entity_ids'])
            entry['entity_types'] = torch.LongTensor(entry['entity_types'])

    def __getitem__(self, index):
        entry = self.entries[index]
        sentence = entry['sentence']
        e_pos = entry['entity_indices']
        e_num = entry['entity_num']
        target = entry['target_indices']
        entity_ids = entry['entity_ids']
        entity_types = entry['entity_types']
        #v, b, p, e, n, a, idx, types

        if self.use_visual_genome:
            features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        else:
            image_id = entry["image"]
            features, cls_boxes, max_conf, image_h, image_w = self.features_chunk[image_id]

        if self.add_spatial_features:
            features = np.concatenate((features, spatials), axis=1)
        else:
            spatials = None

        sample = {}

        image_feat_variable = ArrayField(features)
        image_dim_variable = IntArrayField(np.array(len(features)))
        sample["image_feat_variable"] = image_feat_variable
        sample["image_dim_variable"] = image_dim_variable

        tokenized_sentence, alignment = retokenize_with_alignment(sentence.split(" "), self.tokenizer)

        e_pos_after_subword = []
        current_index = 0
        for position in e_pos:
            for index, i in enumerate(alignment):
                if i == position:
                    if index == len(alignment) - 1 or alignment[index+1] != i:
                        e_pos_after_subword.append(index + 1) # Because the added [CTX] token

        if len(e_pos_after_subword) != len(e_pos) or len(e_pos_after_subword) != len(target):
            assert(0)
        
        # Need to convert target into soft scores:
        target_len = features.shape[0]
        new_target = []
        for i in target:
            new_i = [0.0] * target_len
            if len(i) != 0:
                score = 1.0  / len(i)
                for j in i:
                    new_i[j] = score
            new_target.append(new_i)

        # target = entity_num x v_feature_size
        target = ArrayField(np.array(new_target, dtype="float"), padding_value = 0.0)

        original_position = IntArrayField(np.array(e_pos_after_subword, dtype="int"), padding_value = -1)
        sample["label"] = target # Remember that sometimes that label is empty for certain entities, that's because the boxes we provided do not have a match.
        sample["flickr_position"] = original_position


        bert_example = InputExample(unique_id = -1, text_a = tokenized_sentence, text_b = None, is_correct = None, max_seq_length = self.max_seq_length)

        if self.pretraining:
            bert_feature = InputFeatures.convert_one_example_to_features_pretraining(
                        example = bert_example,
                        tokenizer=self.tokenizer,
                        probability = self.masked_lm_prob)
            bert_feature.insert_field_into_dict(sample)
        else:
            bert_feature = InputFeatures.convert_one_example_to_features(
                    example = bert_example,
                    tokenizer=self.tokenizer)
            bert_feature.insert_field_into_dict(sample)

        return Instance(sample)

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def collate_fn(data):
        batch = Batch(data)
        td = batch.as_tensor_dict()
        return td

