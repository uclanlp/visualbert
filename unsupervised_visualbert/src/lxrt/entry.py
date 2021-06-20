# coding=utf-8
# Copyright 2021 Project Unsupervised VisualBERT
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.nn as nn
import numpy as np
import numpy
from lxrt.tokenization import BertTokenizer
from collections import defaultdict
def get_padding_lengths(list_of_np_array):
    return_dict = defaultdict(int)
    for array in list_of_np_array:
        for i, shape in enumerate(array.shape):
            if return_dict["dimension_{}".format(i)] < shape:
                return_dict["dimension_{}".format(i)] = shape
    return return_dict

def pad_np_arrays(list_of_np_array, padding_value, dtype, cuda = True):
    if isinstance(list_of_np_array[0], list):
        list_of_np_array = [np.array(i, dtype=dtype) for i in list_of_np_array]
    
    if list_of_np_array[0] is None:
        return None

    padding_lengths = get_padding_lengths(list_of_np_array)

    max_shape = [padding_lengths["dimension_{}".format(i)]
                    for i in range(len(padding_lengths))]

    # Convert explicitly to an ndarray just in case it's an scalar (it'd end up not being an ndarray otherwise)
    final_list = []

    for array_index, array in enumerate(list_of_np_array):
        return_array = numpy.asarray(numpy.ones(max_shape, dtype = dtype) * padding_value)
        # If the tensor has a different shape from the largest tensor, pad dimensions with zeros to
        # form the right shaped list of slices for insertion into the final tensor.
        slicing_shape = list(array.shape)
        #if len(array.shape) < len(max_shape):
        #    slicing_shape = slicing_shape + [0 for _ in range(len(max_shape) - len(array.shape))]
        slices = tuple([slice(0, x) for x in slicing_shape])

        return_array[slices] = array
        final_list.append(return_array)
    final_list = np.stack(final_list, 0)
    tensor = torch.from_numpy(final_list)
    if cuda:
        return tensor.cuda()
    else:
        return tensor

#from param import args

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features

def convert_sents_to_features_tensors(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).cuda()
    return input_ids, input_mask, segment_ids

def convert_tags_to_tensorts(tags, cuda = True):
    if tags[0] is None:
        return None, None, None, None, None
    visual_tags, visual_tags_mask, visual_tags_box, visual_tags_type, visual_tags_segment_ids = zip(*tags)
    
    visual_tags = pad_np_arrays(visual_tags, padding_value=0, dtype=np.int64, cuda = cuda)
    visual_tags_mask = pad_np_arrays(visual_tags_mask, padding_value=0, dtype=np.int64, cuda = cuda)
    visual_tags_box = pad_np_arrays(visual_tags_box, padding_value=0, dtype=np.float32, cuda = cuda)
    visual_tags_type = pad_np_arrays(visual_tags_type, padding_value=0, dtype=np.int64, cuda = cuda)
    visual_tags_segment_ids = pad_np_arrays(visual_tags_segment_ids, padding_value=0, dtype=np.int64, cuda = cuda)
    return visual_tags, visual_tags_mask, visual_tags_box, visual_tags_type, visual_tags_segment_ids


def convert_sent_features_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


def set_visual_config(args, VISUAL_CONFIG):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers

class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length

       
        from lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG

        set_visual_config(args, VISUAL_CONFIG)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased",
            mode=mode
        )

        if args.from_scratch:
            print("Re-initializing all the weights")
            self.model.apply(self.model.init_bert_weights)
        
        self.load_pretrain_head = args.get("load_pretrain_head", False)
        if self.load_pretrain_head:
            from lxmert.src.lxrt.modeling import BertPreTrainingHeads
            self.pretrained_head = BertPreTrainingHeads(self.model.config, self.model.bert.embeddings.word_embeddings.weight)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None, input_already_tokenized=False, visual_feats_seg_ids = None):
        if not input_already_tokenized:
            train_features = convert_sents_to_features(
                sents, self.max_seq_length, self.tokenizer)
        else:
            train_features = sents

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        output = self.model(input_ids, segment_ids, input_mask,
                            visual_feats=feats,
                            visual_attention_mask=visual_attention_mask,
                            visual_feats_seg_ids = visual_feats_seg_ids)
        return output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)




