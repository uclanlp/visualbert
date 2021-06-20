# coding=utf-8
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
"""PyTorch LXRT model."""

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
from copy import deepcopy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss

from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except Importtokenization:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualConfig(object):
    VISUAL_LOSSES = ['obj', 'attr', 'feat']
    def __init__(self,
                 l_layers=12,
                 x_layers=5,
                 r_layers=0):
        from param import args

        self.l_layers = args.llayers
        self.x_layers = args.xlayers
        self.r_layers = args.rlayers

        self.visual_feat_dim = 2048
        self.visual_pos_dim = 4

        '''if args.get("kl_divergence", False):
            self.obj_id_num = 1601
            self.attr_id_num = 401
        else:'''
        self.obj_id_num = 1600
        self.attr_id_num = 400

        self.visual_losses = self.VISUAL_LOSSES
        weight = 1 / 0.15
        if args.get("weight_disable", False):
            weight = 1.0
        
        ce_or_kl = "kl" if args.get("kl_divergence", False) else "ce"
        self.visual_loss_config = {
            'obj': (self.obj_id_num, ce_or_kl, (-1,), weight),
            'attr': (self.attr_id_num, ce_or_kl, (-1,), weight),
            'feat': (2048, 'l2', (-1, 2048), weight),
        }

        try:
            from param import args
            self.visualbert_style = args.get('visualbert_style', False)
            self.symbolic_vocab_size = args.get('symbolic_vocab_size', 2632)
            self.multi_choice = args.get("multi_choice", 0)
        except:
            self.visualbert_style = False
        
    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim


VISUAL_CONFIG = VisualConfig()


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
    
        if VISUAL_CONFIG.visualbert_style:
            self.symbolic_embedding = nn.Embedding(VISUAL_CONFIG.symbolic_vocab_size + 1, config.hidden_size) # The first is reserved for masking

    def forward(self, input_ids, token_type_ids=None, attribute_ids=None, symbolic_embedding=False):
        if symbolic_embedding:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            
            words_embeddings = self.symbolic_embedding(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = words_embeddings + token_type_embeddings
            if attribute_ids is not None:
                attribute_mask = (attribute_ids != 0).float()
                attribute_embedding = self.symbolic_embedding(attribute_ids)
                # Need to average along the latter lines 
                attribute_embedding = attribute_embedding * attribute_mask.unsqueeze(-1)  # mask out paddings
                attribute_embedding = attribute_embedding.sum(2)
                length_attribute = attribute_mask.sum(2)
                length_attribute[length_attribute == 0] = 1
                attribute_embedding = attribute_embedding / length_attribute.unsqueeze(-1)
                embeddings = embeddings + attribute_embedding

            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def special_embedding(self, tokenized_words):
        with torch.no_grad():
            all_embeddings = []
            for subwords in tokenized_words:
                subwords = torch.LongTensor(subwords)
                embedding = self.word_embeddings(subwords)
                embedding = embedding.mean(dim=0)
                all_embeddings.append(embedding)
            all_embeddings = torch.stack( [torch.zeros_like(all_embeddings[0])] + all_embeddings, dim=0)
            self.symbolic_embedding.weight = torch.nn.Parameter(all_embeddings)


class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if args.get("output_attention", False):
            return context_layer, attention_probs
        return context_layer

class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        if args.get("output_attention", False):
            attention_output, attention_weights = self.attention(hidden_states, attention_mask)
        else:
            attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if args.get("output_attention", False):
            return layer_output, attention_weights
        return layer_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""

class BertEmbeddingsWithVisualEmbedding(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddingsWithVisualEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.symbolic_embedding = nn.Embedding(2003, config.hidden_size)

        #### Below are for encoding visual features

        # Segment and position embedding for image features
        self.token_type_embeddings_visual = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        feat_dim = VISUAL_CONFIG.visual_feat_dim
        pos_dim = VISUAL_CONFIG.visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
        
        self.add_segment_embedding_to_visual = args.get("add_segment_embedding_to_visual", False)
        self.add_segment_embedding_to_visual_tags=args.get("add_segment_embedding_to_visual_tags", False)
        self.add_position_embedding_to_visual_tags=args.get("add_position_embedding_to_visual_tags", False)

        self.tag_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.joint_layer_norm=args.get("joint_layer_norm", False)
        self.use_segment_embedding_for_vision_and_tag=args.get("use_segment_embedding_for_vision_and_tag", False)
        self.use_bert_input_for_tags=args.get('use_bert_input_for_tags', False)
        self.disable_divide_2 = args.get("disable_divide_2", False)

    def initialize_visual_position_type_embeddings(self):
        ### This is a bit unorthodox. The better way might be to add an inititilizer to AllenNLP.
        # This function is used to initialize the token_type_embeddings_visual and positiona_embedding_visual, just incase.
        self.token_type_embeddings_visual.weight = torch.nn.Parameter(deepcopy(self.token_type_embeddings.weight.data), requires_grad = True)
        self.position_embeddings_visual.weight = torch.nn.Parameter(deepcopy(self.position_embeddings.weight.data), requires_grad = True)
        return
    
    def initialize_symbolic_embeddings(self, tokenized_words):
        with torch.no_grad():
            all_embeddings = []
            for subwords in tokenized_words:
                subwords = torch.LongTensor(subwords)
                embedding = self.word_embeddings(subwords)
                embedding = embedding.mean(dim=0)
                all_embeddings.append(embedding)
            all_embeddings = torch.stack(all_embeddings, dim=0)
        self.symbolic_embedding = nn.Embedding.from_pretrained(deepcopy(all_embeddings), freeze = False)

    def forward(self, input_ids, token_type_ids=None, visual_embeddings=None, visual_embeddings_type=None, position_embeddings_visual=None, image_text_alignment=None, confidence=None, position_ids=None, boxes=None, visual_tags=None, visual_tags_box=None, visual_tags_type=None, visual_tags_segment_ids=None):
        if input_ids is not None:
            seq_length = input_ids.size(1)
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)

            words_embeddings = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            text_embeddings = words_embeddings + position_embeddings + token_type_embeddings
            if not self.joint_layer_norm:
                text_embeddings = self.LayerNorm(text_embeddings)
        else:
            text_embeddings = None

        if visual_tags is not None:
            if self.use_bert_input_for_tags:
                tag_embeddings = self.word_embeddings(visual_tags)
            else:
                tag_embeddings = self.symbolic_embedding(visual_tags)

            if args.get("oscar_style", False):
                tag_position_ids = torch.arange(visual_tags.size(1), dtype=torch.long, device=visual_tags.device)
                tag_position_ids = tag_position_ids.unsqueeze(0).expand_as(visual_tags)
                tag_type_ids = torch.ones_like(visual_tags)
                
                tag_position_embeddings = self.position_embeddings_visual(tag_position_ids)
                tag_type_embeddings = self.token_type_embeddings_visual(tag_type_ids)
                tag_embeddings = tag_embeddings + tag_position_embeddings + tag_type_embeddings
            else:
                y = self.box_fc(visual_tags_box)
                if not self.joint_layer_norm:
                    y = self.box_layer_norm(y)
                    tag_embeddings = self.tag_layer_norm(tag_embeddings)
                if not self.disable_divide_2:
                    tag_embeddings = (tag_embeddings + y) / 2  # + token_type_embeddings
                else:
                    tag_embeddings = tag_embeddings + y

                if visual_tags_segment_ids is not None:
                    assert(self.use_segment_embedding_for_vision_and_tag)

                if self.use_segment_embedding_for_vision_and_tag:
                    if visual_tags_segment_ids is not None:
                        tag_type_ids = visual_tags_segment_ids
                    else:
                        tag_type_ids = torch.zeros_like(visual_tags) # Temporary
                    tag_type_embeddings = self.token_type_embeddings_visual(tag_type_ids)
                    tag_embeddings += tag_type_embeddings
        else:
            tag_embeddings = None

        if visual_embeddings is not None:
            x = self.visn_fc(visual_embeddings)
            #x = self.visn_layer_norm(x)
            y = self.box_fc(boxes)
            #y = self.box_layer_norm(y)
            if not self.joint_layer_norm:
                x = self.visn_layer_norm(x)
                y = self.box_layer_norm(y)
            if not self.disable_divide_2:
                v_embeddings = (x + y) / 2
            else:
                v_embeddings = x + y
            
            #if visual_embeddings_type is not None:
            #    assert(self.use_segment_embedding_for_vision_and_tag)
            
            if self.use_segment_embedding_for_vision_and_tag:
                if visual_embeddings_type is None:
                    visual_embeddings_type = torch.zeros(*visual_embeddings.size()[:-1], dtype=torch.long).cuda()
                token_type_embeddings_visual = self.token_type_embeddings_visual(visual_embeddings_type)
                v_embeddings += token_type_embeddings_visual
        else:
            v_embeddings = None

        if args.get("joint_layer_norm", False):
            # Concate the two:
            embeddings = torch.cat([i for i in [text_embeddings, tag_embeddings, v_embeddings] if i is not None] , dim = 1) # concat the visual embeddings after the attentions
            embeddings = self.LayerNorm(embeddings)
        else:
            embeddings = torch.cat([i for i in [text_embeddings, tag_embeddings, v_embeddings] if i is not None], dim=1)  # concat the visual embeddings after the attentions

        embeddings = self.dropout(embeddings)
        return embeddings
    
    def unfreeze_obj_feat(self):
        all_modules = [        
        self.token_type_embeddings_visual,
        self.position_embeddings_visual,

        # Object feature encoding
        self.visn_fc,
        self.visn_layer_norm,

        # Box position encoding
        self.box_fc,
        self.box_layer_norm,
        self.dropout]
        for submodule in all_modules:
            for p in submodule.parameters():
                p.requires_grad = True

class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output


class VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim
        pos_dim = VISUAL_CONFIG.visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        # This is when we do not have the box_fc yet
        if isinstance(visn_input, tuple) or isinstance(visn_input, list):
            feats, boxes = visn_input
            x = self.visn_fc(feats)
            x = self.visn_layer_norm(x)
            y = self.box_fc(boxes)
            y = self.box_layer_norm(y)
            output = (x + y) / 2
            output = self.dropout(output)
            return output
        else:
            assert(0)
            x = self.visn_fc(visn_input)
            x = self.visn_layer_norm(x)
            return x


def _cat_with_none(feat_1, feat_2, dim):
    if feat_1 is None:
        return feat_2
    if feat_2 is None:
        return feat_1
    return torch.cat((feat_1, feat_2), dim=dim)

def _split_with_none(lang_feats, visn_feats, joint_feats):
    if lang_feats is None:
        assert(visn_feats.size(1) == joint_feats.size(1))
        return None, joint_feats
    if visn_feats is None:
        assert(lang_feats.size(1) == joint_feats.size(1))
        return joint_feats, None
    return joint_feats[:, :lang_feats.size(1), :].contiguous(), joint_feats[:, lang_feats.size(1):, :].contiguous()

class LXRTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers
        print("LXRT encoder with %d l_layers, %d x_layers, and %d r_layers." %
              (self.num_l_layers, self.num_x_layers, self.num_r_layers))
        self.multi_choice = VISUAL_CONFIG.multi_choice
        self.visualbert_style = VISUAL_CONFIG.visualbert_style
        if self.visualbert_style:
            layers = [BertLayer(config) for _ in range(self.num_l_layers)]
            self.layer = nn.ModuleList(layers)
            if args.get("additional_attention_layer", False):
                _config = copy.deepcopy(config)
                _config.intermediate_size = 768
                _config.num_attention_heads = 1
                #layers += [BertLayer(_config)]
                self.additional_layer = BertLayer(_config)
            
                print("\n\n!! Has {} layers".format(len(self.layer) + 1))
            else:
                print("\n\n!! Has {} layers".format(len(self.layer)))
            return
        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        '''self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        ) '''
        layers = [BertLayer(config) for _ in range(self.num_l_layers)]
        print(args.additional_attention_layer)
        assert(0)
        if args.get("additional_attention_layer", False):
            
            _config = copy.deepcopy(config)
            _config.intermediate_size = 768
            layers += [BertLayer(_config)]

        self.layer = nn.ModuleList(layers)
        print("\n\n!! Has {} layers".format(len(self.layer)))

        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        )
        self.multi_choice = VISUAL_CONFIG.multi_choice
        self.config = config
    
    def forward(self,
                lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None,
                bypass_visual_feat=None, bypass_mask=None,
                layer_limit = -1):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        if not args.get("hybrid_embedding", False):
            if args.get("symbolic", False):
                visn_feats, adj = visn_feats
            elif visn_feats[0] is not None:
                visn_feats = self.visn_fc(visn_feats)
            else:
                visn_feats = None
        
        if self.multi_choice != 0:
            visn_feats = visn_feats.unsqueeze(1).expand( visn_feats.size(0), self.multi_choice, visn_feats.size(1), visn_feats.size(2))
            visn_attention_mask = visn_attention_mask.unsqueeze(1).expand(visn_attention_mask.size(0), self.multi_choice, visn_attention_mask.size(1), visn_attention_mask.size(2), visn_attention_mask.size(3))
            #print(visn_feats.size())
            visn_feats = visn_feats.reshape((-1, visn_feats.size(2), visn_feats.size(3)))
            visn_attention_mask = visn_attention_mask.reshape((-1, visn_attention_mask.size(-3), visn_attention_mask.size(-2), visn_attention_mask.size(-1)))

        if self.visualbert_style:
            if args.get("bypass_visual_feat", False):
                joint_feats = _cat_with_none(lang_feats, visn_feats, dim=1) #torch.cat((lang_feats, visn_feats), dim=1)
                joint_mask = _cat_with_none(lang_attention_mask, visn_attention_mask, dim=-1)  #torch.cat((lang_attention_mask, visn_attention_mask), dim=-1)
                if args.get("include_additional_layer", True):
                    for layer_module in self.layer[:-1]:
                        joint_feats = layer_module(joint_feats, joint_mask)
                    joint_feats = torch.cat((joint_feats, bypass_visual_feat), dim=1)
                    joint_feats = self.layer[-1](joint_feats, bypass_mask)
                    return _split_with_none(joint_feats, visn_feats, joint_feats)
                else:
                    for layer_module in self.layer:
                        joint_feats = layer_module(joint_feats, joint_mask)
                    return torch.cat((joint_feats, bypass_visual_feat), dim = 1), None
            if args.get("seperate_modeling", False):
                #assert (args.get("additional_attention_layer", False))
                joint_feats = _cat_with_none(lang_feats, visn_feats, dim=1) #torch.cat((lang_feats, visn_feats), dim=1)
                joint_mask = _cat_with_none(lang_attention_mask, visn_attention_mask, dim=-1)  #torch.cat((lang_attention_mask, visn_attention_mask), dim=-1)
                if layer_limit != -1:
                    for layer_module in self.layer[:layer_limit]:
                        joint_feats = layer_module(joint_feats, joint_mask)
                else:
                    for layer_module in self.layer:
                        joint_feats = layer_module(joint_feats, joint_mask)
                return _split_with_none(lang_feats, visn_feats, joint_feats) #joint_feats[:, :lang_feats.size(1), :].contiguous(), joint_feats[:, lang_feats.size(1):, :].contiguous()

            joint_feats = _cat_with_none(lang_feats, visn_feats, dim=1) #torch.cat((lang_feats, visn_feats), dim=1)
            joint_mask = _cat_with_none(lang_attention_mask, visn_attention_mask, dim=-1)  #torch.cat((lang_attention_mask, visn_attention_mask), dim=-1)
            all_attention_weights = []
            for layer_module in self.layer:
                if args.get("output_attention", False):
                    joint_feats, attention_weights = layer_module(joint_feats, joint_mask)
                    all_attention_weights.append(attention_weights)
                else:
                    joint_feats = layer_module(joint_feats, joint_mask)
            if args.get("additional_attention_layer", False):
                joint_feats = self.additional_layer(joint_feats, joint_mask)
            if args.get("output_attention", False):
                return _split_with_none(lang_feats, visn_feats, joint_feats), all_attention_weights
            return _split_with_none(lang_feats, visn_feats, joint_feats) #joint_feats[:, :lang_feats.size(1), :].contiguous(), joint_feats[:, lang_feats.size(1):, :].contiguous()

        # Run language layers
        if lang_feats is not None:
            for layer_module in self.layer:
                lang_feats = layer_module(lang_feats, lang_attention_mask)

        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)

        # Run cross-modality layers
        if lang_feats is not None:
            for layer_module in self.x_layers:
                lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask,
                                                      visn_feats, visn_attention_mask)

        return lang_feats, visn_feats


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class BertVisualObjHead(nn.Module):
    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Decide the use of visual losses
        visual_losses = visual_losses.split(",")
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses

        sizes = {key: VISUAL_CONFIG.visual_loss_config[key][0] for key in self.visual_losses}

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict({
            key: nn.Linear(config.hidden_size, sizes[key])
            for key in self.visual_losses
        })

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        if args.get("lxmert_style_nlvr", False):
            self.seq_relationship_new = nn.Linear(config.hidden_size * 2, 2)
        else:
            self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, calculate_seq_score = True):
        prediction_scores = self.predictions(sequence_output)

        if not calculate_seq_score:
            return prediction_scores, None
        if args.get("lxmert_style_nlvr", False):
            seq_relationship_score = self.seq_relationship_new(pooled_output)
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        cache_dir = args.get("cache_dir", "/local/harold/tmp/")
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path == 'bert-base-uncased':
                try:
                    print("The BERT-weight-downloading query to AWS was time-out;" 
                          "trying to download from UNC servers")
                    archive_file = "https://nlp.cs.unc.edu/data/bert/bert-base-uncased.tar.gz"
                    resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
                except EnvironmentError:
                    print("The weight-downloading still crashed with link: %s, "
                          "please check your network connection" % archive_file)
                    return None
            else:
                logger.error(
                        "Model name '{}' was not found in model name list ({}). "
                        "We assumed '{}' was a path or url but couldn't find any file "
                        "associated to this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                            archive_file))
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp(prefix="/local/harold/tmp/")
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
             logger.info("Weights of {} not initialized from pretrained model: {}".format(
                 model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
             logger.info("Weights from pretrained model not used in {}: {}".format(
                 model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model

from param import args

class LXRTModel(BertPreTrainedModel):
    """LXRT Model."""

    def __init__(self, config):
        super().__init__(config)
        if args.get("hybrid_embedding", False):
            self.embeddings = BertEmbeddingsWithVisualEmbedding(config)
        else:
            self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None, position_embeddings_visual=None,
                visual_tags=None, visual_tags_mask=None, visual_tags_box=None, visual_tags_type=None, visual_tags_segment_ids=None,
                visual_feats_seg_ids = None,
                ):
        if visual_attention_mask is None and visual_feats[0] is not None:
            if args.get("uneven_masks", False):
                visual_attention_mask = 1 - (visual_feats[0] == 0.0).all(-1).float().to(next(self.parameters()).device)
            else:  
                visual_attention_mask = torch.ones(visual_feats[0].size(0), visual_feats[0].size(1)).to(next(self.parameters()).device)
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)
        if visual_tags_mask is None and visual_tags is not None:
            visual_tags_mask = torch.ones_list(visual_tags)

        # Process masks
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None
        
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        if visual_tags_mask is not None:
            extended_visual_tags_mask = visual_tags_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_tags_mask = extended_visual_tags_mask.to(dtype=next(self.parameters()).dtype)
            extended_visual_tags_mask = (1.0 - extended_visual_tags_mask) * -10000.0
        else:
            extended_visual_tags_mask = None

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=None,
            position_ids=None,

            visual_embeddings=visual_feats[0],
            boxes=visual_feats[1],
            visual_embeddings_type=visual_feats_seg_ids,
            position_embeddings_visual=None,
            image_text_alignment=None,
            confidence=None,
        
            visual_tags=visual_tags,
            visual_tags_box=visual_tags_box,
            visual_tags_type=visual_tags_type,
            visual_tags_segment_ids = visual_tags_segment_ids
            )
        
        concated_mask = torch.cat([ i for i in [extended_attention_mask, extended_visual_tags_mask, extended_visual_attention_mask] if i is not None], dim=-1)
        
        # self.encoder will not distinguish between visual inputs, visual tag inputs or text inputs
        if args.get("output_attention", False):
            combined_feats, _, attention_weights = self.encoder(
            embedding_output,
            concated_mask,
            visn_feats=None,
            visn_attention_mask=None)
        else:
            combined_feats, _ = self.encoder(
                embedding_output,
                concated_mask,
                visn_feats=None,
                visn_attention_mask=None)
    
        if attention_mask is not None:
            lang_feats = combined_feats[:,:attention_mask.size(-1)]
        else:
            lang_feats = None
        
        if visual_tags_mask is not None:
            if attention_mask is None:
                tag_feats = combined_feats[:,:visual_tags_mask.size(-1)]
            else:
                tag_feats = combined_feats[:, attention_mask.size(-1): attention_mask.size(-1) + visual_tags_mask.size(-1)]
        else:
            tag_feats = None

        if visual_attention_mask is not None:
            visn_feats =combined_feats[:, -visual_attention_mask.size(-1):]
        else:
            visn_feats = None
        
        if lang_feats is not None:
            pooled_output = self.pooler(lang_feats)
            if args.get("output_attention", False):
                return (lang_feats, tag_feats, visn_feats), pooled_output, attention_weights
            return (lang_feats, tag_feats, visn_feats), pooled_output
        else:
            if args.get("output_attention", False):
                return (lang_feats, tag_feats, visn_feats), None, attention_weights
            return (lang_feats, tag_feats, visn_feats), None


class LXRTPretraining(BertPreTrainedModel):
    def __init__(self,
                 config,
                 args=None,
                 task_mask_lm=True,
                 task_matched=True,
                 task_obj_predict=True,
                 visual_losses='',
                 task_qa=True,
                 num_answers=2):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_answers = num_answers
        self.args = args

        # Use of pre-training tasks
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_matched = task_matched
        self.task_qa = task_qa

        # LXRT backbone
        self.bert = LXRTModel(config)

        # Pre-training heads
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = BertVisualObjHead(config, visual_losses)
        if self.task_qa:
            self.answer_head = BertVisualAnswerHead(config, self.num_answers)
        if args.get("use_tag_symbolic_embedding", False):
            self.symbolic_head = deepcopy(self.cls)

        # Weight initialization
        self.apply(self.init_bert_weights)
    
    def special_initialize_pretraining_head(self):
        self.symbolic_head.predictions.decoder.weight = self.bert.embeddings.symbolic_embedding.weight
        self.symbolic_head.predictions.bias = nn.Parameter(torch.zeros(self.symbolic_head.predictions.decoder.weight.size(0)))

    def forward(self,
                input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None,
                matched_label=None, ans=None,
                visual_tags=None, visual_tags_mask=None, visual_tags_box=None, visual_tags_type=None, visual_tags_objective=None, visual_tags_mismatch=None, visual_tags_segment_ids=None,
                visual_feats_seg_ids=None,
                return_cross_relationship_score = False
                ):
        (lang_output, tags_output, visn_output), pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            visual_feats=(visual_feats, pos), visual_feats_seg_ids = visual_feats_seg_ids,
            visual_tags=visual_tags, visual_tags_mask=visual_tags_mask, visual_tags_box=visual_tags_box, visual_tags_type = visual_tags_type, visual_tags_segment_ids = visual_tags_segment_ids
        )

        if input_ids is None:
            answer_score = None
            cross_relationship_score = None
        else:
            if args.get('lxmert_style_nlvr', False):
                pooled_output = pooled_output.view(pooled_output.size(0) // 2, 2 * pooled_output.size(-1))

            lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
            if self.task_qa:
                answer_score = self.answer_head(pooled_output)
            else:
                # This answer_score would not be used anywhere,
                # just to keep a constant return function signature.
                answer_score = pooled_output[0][0]

        total_loss = 0.
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        losses = ()
        losses_dict = {}
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
            losses += (masked_lm_loss.detach(),)
            if visual_feats is not None:
                losses_dict["Masked LM"] = masked_lm_loss.detach()
            else:
                losses_dict["Text Only Masked LM"] = masked_lm_loss.detach()
        if matched_label is not None and self.task_matched and cross_relationship_score is not None:
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            losses += (matched_loss.detach(),)
            losses_dict["Matches"] = matched_loss.detach()
        if obj_labels is not None and self.task_obj_predict and not args.get("disable_visual_and_tag_objective", False):
            loss_fcts = {
                'l2': SmoothL1Loss(reduction='none'),
                'ce': CrossEntropyLoss(ignore_index=-1, reduction='none'),
                "kl": torch.nn.KLDivLoss(reduction = "batchmean")
            }
            total_visn_loss = 0.
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in self.args.visual_losses.split(","):
                label, mask_conf = obj_labels[key]
                if key == "attr" or key == "obj":
                    label = label.long()
                elif key == "feat":
                    label = label.float()
                else:
                    assert(0)

                output_dim, loss_fct_name, label_shape, weight = VISUAL_CONFIG.visual_loss_config[key]
                visn_loss_fct = loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(
                    visn_prediction_scores.view(-1, output_dim),
                    label.view(*label_shape),
                )
                if visn_loss.dim() > 1:     # Regression Losses
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
                losses += (visn_loss.detach(),)
                losses_dict[key] = visn_loss.detach()

            total_loss += total_visn_loss
        if ans is not None and self.task_qa and input_ids is not None:
            answer_loss = loss_fct(
                answer_score.view(-1, self.num_answers),
                ans.view(-1)
            )  
            # Since this Github version pre-trains with QA loss from the beginning,
            # I exclude "*2" here to match the effect of QA losses.
            # Previous: (loss *0) for 6 epochs, (loss *2) for 6 epochs.   (Used 10 instead of 6 in EMNLP paper)
            # Now     : (loss *1) for 12 epochs
            #
            # * 2       # Multiply by 2 because > half of the data will not have label
            total_loss += answer_loss
            losses += (answer_loss.detach(),)
            losses_dict["qa"] = answer_loss.detach()
        if visual_tags_objective is not None and not args.get("disable_visual_and_tag_objective", False):
            if args.get("use_bert_input_for_tags", False):
                tags_output, _ = self.cls(tags_output, tags_output[:, 0], calculate_seq_score = False)
                masked_tag_loss = loss_fct(
                    tags_output.view(-1, self.config.vocab_size),
                    visual_tags_objective.view(-1)
                )
            else:
                tags_output, _ = self.symbolic_head(tags_output, tags_output[:, 0])
                masked_tag_loss = loss_fct(
                    tags_output.view(-1, 2003),
                    visual_tags_objective.view(-1)
                )      
            total_loss += masked_tag_loss
            losses_dict["Masked Tags"] = masked_tag_loss.detach()

            if visual_tags_mismatch is not None:
                matched_loss = loss_fct(
                    cross_relationship_score.view(-1, 2),
                    visual_tags_mismatch.view(-1)
                    )
                total_loss += matched_loss
                losses += (matched_loss.detach(),)
                losses_dict["Tag mismatch"] = matched_loss.detach()    
        
        if answer_score is None:
            return total_loss, torch.stack(losses).unsqueeze(0), answer_score, losses_dict
        
        return total_loss, torch.stack(losses).unsqueeze(0) if len(losses) is not None else (), answer_score.detach(), losses_dict


class LXRTFeatureExtraction(BertPreTrainedModel):
    """
    BERT model for classification.
    """
    def __init__(self, config, mode='lxr'):
        """
        :param config:
        :param mode:  Number of visual layers
        """
        super().__init__(config)
        self.config = config
        self.bert = LXRTModel(config)
        self.mode = mode
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, visual_feats=None,
                visual_attention_mask=None, return_both = False, visual_feats_seg_ids = None):
        feat_seq, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                            visual_feats=visual_feats,
                                            visual_attention_mask=visual_attention_mask,
                                            visual_feats_seg_ids = visual_feats_seg_ids)
        if return_both:
            return feat_seq, pooled_output

        if 'x' == self.mode:
            return pooled_output
        elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
            return feat_seq, pooled_output
        elif 'l' in self.mode or 'r' in self.mode:
            return feat_seq